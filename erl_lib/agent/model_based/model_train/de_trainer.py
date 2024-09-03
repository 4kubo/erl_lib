from typing import Dict, List
import time
import logging
from copy import deepcopy
import numpy as np
from scipy import stats
from tqdm import trange
import torch
from torch import optim as optim
from torch.distributions import Normal


from erl_lib.util.misc import TransitionIterator


class DETrainer:
    """Trainer for Deep Ensemble."""

    def __init__(
        self,
        model,
        lr: float = 1e-4,
        grad_clip: float = 0.0,
        z_test_improvement: bool = True,
        improvement_threshold: float = 0.01,
        keep_threshold: float = 0.5,
        logger=None,
        silent: bool = True,
    ):
        self.model = model
        self.num_members = model.num_members
        self.lr = lr
        self.dim_output = model.dim_output
        self._train_iteration = 0
        self.grad_clip = grad_clip
        self.z_test_improvement = z_test_improvement
        self.improvement_threshold = improvement_threshold
        self.keep_threshold = keep_threshold
        self.logger = logger
        self.silent = silent
        if logger is not None:
            self.detail_log = logger.log_level < logging.INFO
        else:
            self.detail_log = False

        # Target model
        self.old_states = {}
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                self.old_states[key] = value.clone()

        assert (
            improvement_threshold <= keep_threshold
        ), f"{improvement_threshold} > {keep_threshold}"
        self.early_stop = keep_threshold < 1

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_eval: TransitionIterator,
        env_step: int,
        num_max_epochs: int,
        keep_epochs: int = 1,
    ):
        """Trains the model for some number of epochs."""
        self.optimizer = optim.AdamW(self.model.optimized_parameters(), lr=self.lr)
        best_val_score, _ = self.evaluate(dataset_eval)

        if 1 <= self.keep_threshold:
            keep_epochs = num_max_epochs
        self.early_stopper = EarlyStopping(
            self.model,
            best_val_score,
            keep_epochs,
            z_test_improvement=self.z_test_improvement,
            improve_p_threshold=self.improvement_threshold,
            continue_p_threshold=self.keep_threshold,
        )

        train_loss_hist, val_score_hist = [], [best_val_score.mean(-1)]

        time_train, time_eval = 0.0, 0.0

        iterator = trange(
            num_max_epochs, mininterval=5, disable=self.silent, desc="[Model]"
        )
        with iterator as pbar:
            t_p = time.time()

            for epoch in pbar:
                # Train model
                batch_losses: List[torch.Tensor] = []
                log = True
                info = {}
                t_1 = time.time()
                self.model.train()
                for batch in dataset_train:
                    loss, info_ = self.model.update(batch, self.optimizer)
                    batch_losses.append(loss.detach())
                    if log:
                        info = info_
                    log = False

                # Evaluate model
                t_2 = time.time()

                scores_val, _ = self.evaluate(dataset_eval)

                with torch.no_grad():
                    train_loss = torch.hstack(batch_losses).mean() / self.num_members
                train_loss_hist.append(train_loss)
                val_score_hist.append(scores_val.mean())

                time_train += t_2 - t_1
                time_eval += time.time() - t_2

                # Stop model learning if enough
                should_stop, best_val_score, last_epoch = self.early_stopper.step(
                    scores_val
                )
                # Misc process for progress bar if needed
                if not self.silent and (2 < t_2 - t_p):
                    t_p = time.time()
                    pbar.set_postfix(
                        {
                            "Train": train_loss.item(),
                            "Val": scores_val.mean().item(),
                            "Best": best_val_score.item(),
                            "Last": last_epoch,
                        },
                        refresh=False,
                    )

                if should_stop or (epoch == num_max_epochs - 1):
                    break

        # saving the best models:
        best_model, best_epoch = self.early_stopper.post_process()
        if self.early_stop:
            self.model.load_state_dict(best_model)

        self.model.eval()
        # Log training metrics
        if self.logger:
            index = {
                "iteration": self._train_iteration,
                "env_step": env_step,
            }

            scores_val_last, info_val = self.evaluate(dataset_eval, log=True)
            score_val_last = scores_val_last.mean()
            info_val.update(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "train_dataset_size": dataset_train.num_stored,
                    "val_dataset_size": dataset_eval.num_stored,
                    "last_train_loss": train_loss,
                    "last_validation_score": score_val_last,
                    "val_loss_improvement": best_val_score - val_score_hist[0],
                    "best_val_score": best_val_score,
                    "decay_loss": self.model.decay_loss(),
                    "time_train": time_train,
                    "time_eval": time_eval,
                },
                **info,
            )
            # info.update(info_eval)

            self.logger.append("model_train", index, info_val)

        self._train_iteration += 1

        return train_loss_hist, val_score_hist

    def evaluate(self, dataset: TransitionIterator, log=False):
        """Evaluates the model on the validation dataset."""
        scores_list, errors_list, vars_list = [], [], []
        total_batch_size = 0
        info = {}

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(dataset):
                log_i = "detail" if log and (i == len(dataset) - 1) else False
                batch_score, error, variance, info_ = self.model.eval_score(
                    batch, log=log_i
                )
                scores_list.append(batch_score)
                errors_list.append(error)
                vars_list.append(variance)
                total_batch_size += len(batch)
                info.update(**info_)

            scores = torch.hstack(scores_list)

            if log:
                errors = torch.vstack(errors_list)
                errors = self.model.rescale_error(errors)
                variances = torch.vstack(vars_list)
                # MSE
                mse = errors.sum(1).mean()
                info["mse"] = mse
                # Calibration score
                i = variances.argsort(0)
                n_split = min(dataset.num_stored, 20)
                bin_idxes = torch.tensor_split(i, n_split)

                rmv = torch.vstack(
                    [torch.take(variances, idx).mean(0) for idx in bin_idxes]
                )
                rmse = torch.vstack(
                    [torch.take(errors, idx).mean(0) for idx in bin_idxes]
                )
                # The Expected Normalized Calibration Error (ENCE)fo
                # c.f. Evaluating and Calibrating Uncertainty Prediction in Regression Tasks
                calib_score = torch.mean(torch.abs(rmv - rmse) / rmv)
                info["calib_score"] = calib_score

        info = {f"eval/{key}": value for key, value in info.items()}
        return scores, info


class EarlyStopping:
    def __init__(
        self,
        model,
        scores_val,
        num_continue,
        z_test_improvement,
        improve_p_threshold=0.01,
        continue_p_threshold=1.0,
    ):
        self.model = model
        self.num_continue = num_continue

        self.n_models = model.num_members
        self.best_scores = scores_val
        self.best_score = scores_val.mean()

        self.best_state_dict = deepcopy(self.model.state_dict())
        if improve_p_threshold < 1.0:
            assert improve_p_threshold <= continue_p_threshold, (
                f"Expected: improve_p_threshold < continue_p_threshold, "
                f"But: {improve_p_threshold} > {continue_p_threshold}"
            )
        self.z_test_improvement = z_test_improvement
        self.improve_p_threshold = improve_p_threshold
        self.continue_p_threshold = continue_p_threshold

        self.iter_continued = 0
        self.epoch = 0
        self.epoch_best = 0

    def step(self, scores_val):
        self.epoch += 1
        if self.z_test_improvement:
            self._step_z_test(scores_val)
        else:
            self._step_mean(scores_val)

        if self.num_continue < self.iter_continued:
            stop = True
        else:
            stop = False
        return stop, self.best_score, self.epoch_best

    def _step_z_test(self, scores_val):
        pvalue = p_value(scores_val, self.best_scores)
        may_continue = pvalue < self.continue_p_threshold

        if may_continue:
            self.iter_continued = 0

            improved = pvalue < self.improve_p_threshold

            if improved:
                for key, value in self.model.state_dict().items():
                    if value.ndim == 3 and value.shape[0] == self.n_models:
                        self.best_state_dict[key] = value.clone()
                self.best_scores = scores_val
                self.best_score = scores_val.mean()
                self.epoch_best = self.epoch
        else:
            self.iter_continued += 1

    def _step_mean(self, scores_val):
        score = scores_val.mean()
        may_continue = score < self.best_score

        if may_continue:
            self.iter_continued = 0

            normalized_score = (self.best_score - score) / (self.best_score + 1e-8)
            improved = self.improve_p_threshold < normalized_score
            if improved:
                for key, value in self.model.state_dict().items():
                    if value.ndim == 3 and value.shape[0] == self.n_models:
                        self.best_state_dict[key] = value.clone()
                self.best_score = score
                self.epoch_best = self.epoch
        else:
            self.iter_continued += 1

    def post_process(self):
        return self.best_state_dict, self.epoch_best


def p_value(scores, best_scores):
    data_size = scores.numel()
    # Switching for computation efficiency
    if data_size < 1000:
        p_value = stats.ttest_rel(
            scores.cpu().numpy(), best_scores.cpu().numpy(), alternative="less"
        ).pvalue
    else:
        diff = scores - best_scores
        inv_var = np.sqrt(data_size) / diff.std()
        z_stat = diff.mean() * inv_var
        p_value = Normal(0.0, 1.0).cdf(z_stat)
    return p_value
