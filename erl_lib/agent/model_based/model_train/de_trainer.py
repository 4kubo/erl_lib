from typing import Dict, List
import time
from collections import defaultdict
from copy import deepcopy
import re
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

        # Target model
        self.old_states = {}
        with torch.no_grad():
            for key, value in self.model.state_dict().items():
                self.old_states[key] = value.clone()

        assert (
            improvement_threshold <= keep_threshold
        ), f"{improvement_threshold} > {keep_threshold}"
        self.early_stop = improvement_threshold < 1

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: TransitionIterator,
        env_step: int,
        num_max_epochs: int,
        dataset_eval: TransitionIterator = None,
        keep_epochs: int = 1,
        log_detail=False,
    ):
        """Trains the model for some number of epochs."""
        self.optimizer = optim.AdamW(self.model.optimized_parameters(), lr=self.lr)

        train_metric_hist, val_metric_hist, eval_metric_hist = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        best_val_score, info_val = self.evaluate(dataset_val, log=log_detail)
        for key, value in info_val.items():
            if not re.match(r"\d", key.split("/")[-1]):
                val_metric_hist[key].append(value)
        if dataset_eval:
            best_eval_score, info_eval = self.evaluate(dataset_eval, log=log_detail)
            for key, value in info_eval.items():
                if not re.match(r"\d", key.split("/")[-1]):
                    eval_metric_hist[key].append(value)

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

        time_train, time_eval = 0.0, 0.0

        iterator = trange(
            num_max_epochs, mininterval=5, disable=self.silent, desc="[Model]"
        )
        with iterator as pbar:
            t_p = time.time()

            for epoch in pbar:
                # Train model
                batch_metrics = defaultdict(list)
                log = True
                info = {}
                t_1 = time.time()
                self.model.train()
                for batch in dataset_train:
                    loss, info = self.model.update(batch, self.optimizer, log)
                    batch_metrics["train_loss"].append(loss.detach())
                    for key, value in info.items():
                        batch_metrics[key].append(value)
                    log = log_detail
                for key, value in batch_metrics.items():
                    self.agg_batch_metrics(key, value, train_metric_hist)

                # Evaluate model
                t_2 = time.time()

                scores_val, info_val = self.evaluate(dataset_val, log=log_detail)
                for key, value in info_val.items():
                    if not re.match(r"\d", key.split("/")[-1]):
                        val_metric_hist[key].append(value)
                if dataset_eval:
                    best_eval_score, info_eval = self.evaluate(
                        dataset_eval, log=log_detail
                    )
                    for key, value in info_eval.items():
                        if not re.match(r"\d", key.split("/")[-1]):
                            eval_metric_hist[key].append(value)

                time_train += t_2 - t_1
                time_eval += time.time() - t_2

                # Stop model learning if enough
                should_stop, best_val_score, last_epoch = self.early_stopper.step(
                    scores_val
                )
                # Misc process for progress bar if needed
                if not self.silent and (2 < t_2 - t_p):
                    t_p = time.time()
                    train_loss = train_metric_hist["train_loss"][-1]
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
            info_train = {key: value[-1] for key, value in train_metric_hist.items()}
            scores_val_last, info_val = self.evaluate(dataset_val, log=True)
            score_val_last = scores_val_last.mean()
            info_val.update(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "last_validation_score": score_val_last,
                    "best_val_score": best_val_score,
                    "decay_loss": self.model.decay_loss,
                    "time_train": time_train,
                    "time_eval": time_eval,
                },
                **info_train,
            )
            info_val = {
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in info_val.items()
            }
            self.logger.append("model_train", index, info_val)

        self._train_iteration += 1

        return train_metric_hist, val_metric_hist, eval_metric_hist

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
                variances = torch.vstack(vars_list)
                errors, variances = self.model.rescale_error(errors, variances)
                # Errors
                rmse = errors.sum(1).mean().sqrt()
                info["root_mean_squared_error"] = rmse
                root_median_se = errors.sum(1).median().sqrt()
                info["root_median_squared_error"] = root_median_se
                info["max_error"] = errors.sqrt().mean(1).max()
                # The Expected Normalized Calibration Error (ENCE)
                # c.f. Evaluating and Calibrating Uncertainty Prediction in Regression Tasks
                i = variances.argsort(0)
                n_split = min(dataset.num_stored, 20)
                bin_idxes = torch.tensor_split(i, n_split)

                rmv = torch.vstack(
                    [torch.take(variances, idx).mean(0) for idx in bin_idxes]
                ).sqrt()
                rmses = torch.vstack(
                    [torch.take(errors, idx).mean(0) for idx in bin_idxes]
                ).sqrt()
                calib_score = torch.mean(torch.abs(rmv - rmses) / rmv)
                info["ence"] = calib_score
                # NLL
                scale_log = variances.log() * 0.5
                nlls = 0.5 * errors / variances + scale_log
                info["nll"] = nlls.mean().sum()

        info = {f"eval/{key}": value for key, value in info.items()}
        return scores, info

    @staticmethod
    def agg_batch_metrics(key, value_batch, metric_dict):
        append = isinstance(metric_dict, defaultdict)
        metric = torch.hstack(value_batch)
        if "max" in key:
            metric = metric.max()
            if append:
                metric_dict[key].append(metric)
            else:
                metric_dict[key] = metric
        elif "squared_error" in key:
            # value = torch.hstack(value_batch)
            root_mean = metric.mean().sqrt()
            root_median = metric.median().sqrt()
            if append:
                metric_dict[f"root_mean_{key}"].append(root_mean)
                metric_dict[f"root_median_{key}"].append(root_median)
            else:
                metric_dict[f"root_mean_{key}"] = root_mean
                metric_dict[f"root_median_{key}"] = root_median
        else:
            metric = metric.mean()
            if append:
                metric_dict[key].append(metric)
            else:
                metric_dict[key] = metric


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
