from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from erl_lib.agent.model_based.modules.model import Model
from erl_lib.agent.module.layer import (
    NormalizedEnsembleLinear,
    EnsembleLinearLayer,
    ensemble_kaiming_normal,
)
from erl_lib.base.datatype import (
    TransitionBatch,
)

# Uncertainty propagation strategies
PS_MM = "moment_matching"
PS_MMD = "moment_matching_direct"
PS_TS1 = "ts_1"
PS_TS1_F = "ts_1_full"
PS_INF = "ts_infinity"
PS_NO_EPISTEMIC = "no_epi"

activation_fn_dict = {
    "SiLU": nn.SiLU,
    "ReLU": nn.ReLU,
    "Mish": nn.Mish,
    "LeakyReLU": nn.LeakyReLU,
}


class GaussianMLP(Model):
    """Wrapper class for 1-D dynamics models."""

    INTRINSIC_REWARD = "intrinsic_reward"

    def __init__(
        self,
        term_fn,
        input_normalizer,
        output_normalizer,
        batch_size: int,
        normalize_input: bool = True,
        normalize_po_input: bool = True,
        normalize_delta: bool = True,
        *args,
        num_members: int = 1,
        dim_hidden: int = 200,
        noise_wd: float = 0.0,
        residual: bool = False,
        lb_std: float = 1e-3,
        layer_norm: float = 1.0,
        drop_rate_base: float = 0.01,
        activation_fn: str = None,
        weight_decay_base: float = 0.001,
        weight_decay_ratios: Sequence[float] = None,
        # Prediction
        normalized_reward=False,
        delta_prediction: bool = True,
        priors_on_function_values: bool = False,
        prediction_strategy: str = PS_TS1,
        uncertainty_bonus: bool = True,
        no_epistemic_pred: bool = True,
        # Training
        normalized_target: bool = True,
        mse_score: bool = False,
        sum_over_data: bool = True,
        training_loss_fn: str = "nll",  # nll or gauss_adapt
        **kwargs,
    ):
        if (not normalized_target and normalize_delta) or (
            normalized_target and not normalize_input
        ):
            raise NotImplementedError
        assert 0 <= layer_norm

        super().__init__(*args, **kwargs)
        self.term_fn = term_fn
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.batch_size = batch_size
        self.normalize_input = normalize_input
        self.normalize_po_input = normalize_po_input
        self.normalize_delta = normalize_delta and output_normalizer is not None
        self.num_members = num_members
        self.noise_wd = noise_wd

        self.weight_decay_base = weight_decay_base
        self.normalized_reward = normalized_reward
        self.delta_prediction = delta_prediction
        self.priors_on_function_values = priors_on_function_values
        assert prediction_strategy in [
            PS_MM,
            PS_TS1,
            PS_TS1_F,
            PS_INF,
            PS_MMD,
            PS_NO_EPISTEMIC,
        ]
        self.prediction_strategy = prediction_strategy
        self.uncertainty_bonus = uncertainty_bonus
        self.no_epistemic_pred = no_epistemic_pred
        # Model training
        self.normalized_target = normalized_target
        self.mse_score = mse_score
        self.sum_over_data = sum_over_data

        self.training_loss_fn = training_loss_fn

        activation_fn = activation_fn_dict.get(activation_fn, nn.SiLU)
        depth = len(weight_decay_ratios)
        max_ratio = max(weight_decay_ratios)
        wd_ratio_i = weight_decay_ratios[0]
        wd_0 = wd_ratio_i * weight_decay_base
        dr_0 = (max_ratio - wd_ratio_i) * drop_rate_base
        hidden_layers = [
            NormalizedEnsembleLinear(
                num_members,
                self.dim_input,
                dim_output=dim_hidden,
                weight_decay=wd_0,
                dropout_rate=dr_0,
                normalize_eps=0,
                activation=activation_fn(),
            ),
        ]

        for i, wd_ratio_i in enumerate(weight_decay_ratios[1:-1]):
            wd_i = wd_ratio_i * weight_decay_base
            dr_i = (max_ratio - wd_ratio_i) * drop_rate_base
            eps_i = min(max(np.power(10.0, -i), 1e-5), layer_norm)
            hidden_layers.append(
                NormalizedEnsembleLinear(
                    num_members,
                    dim_hidden,
                    weight_decay=wd_i,
                    dropout_rate=dr_i,
                    activation=activation_fn(),
                    normalize_eps=eps_i,
                    residual=residual,
                )
            )

        ratio_last = weight_decay_ratios[-1]
        wd_last = ratio_last * weight_decay_base
        dr_last = (max_ratio - ratio_last) * drop_rate_base

        eps_last = min(max(np.power(10.0, -(depth - 2)), 1e-5), layer_norm)
        hidden_layers.append(
            NormalizedEnsembleLinear(
                num_members,
                dim_hidden,
                dim_output=self.dim_output,
                weight_decay=wd_last,
                dropout_rate=dr_last,
                normalize_eps=eps_last,
            )
        )

        self.layers = nn.Sequential(*hidden_layers)
        # Homo-scedastic noise
        self._log_noise = nn.Parameter(
            torch.zeros((num_members, 1, self.dim_output), dtype=torch.float32),
            requires_grad=True,
        )
        self.to(self.device)

        self.lb_std = lb_std or 0
        self.lb_log_scale = (
            np.log(lb_std) if (lb_std is not None and 0 < lb_std) else None
        )

        if isinstance(batch_size, int) and 0 < batch_size:
            index = np.arange(num_members).repeat(
                int(np.ceil(batch_size / num_members))
            )
            index = torch.as_tensor(index)
            self.base_index = (
                torch.nn.functional.one_hot(index)
                .T[..., None]
                .repeat(1, 1, self.dim_output)
                .to(device=self.device, dtype=torch.float32)
            )  # [M, B, D]

        self.eval()

    def base_forward(self, x: torch.Tensor, normalize_delta=None):
        """"""
        mus = self.layers(x)
        if normalize_delta is None:
            normalize_delta = self.normalize_delta

        if normalize_delta:
            sample_mean, sample_std = (
                self.output_normalizer.mean,
                self.output_normalizer.std,
            )
            mus = sample_mean + sample_std * mus
            log_std_ale = self._log_noise + sample_std.log()
        else:
            log_std_ale = self._log_noise
        return mus, log_std_ale

    def moment_dist(self, x, normalize_delta=None, log=False, post_process=None):
        """Mu and sigma in Gaussian distribution's parameter."""
        mus = self.layers(x)
        log_var_ale = self._log_noise.mean(0) * 2

        mu = mus.mean(0)
        var_epi = mus.var(0)
        log_var_epi = var_epi.log()

        # c.f. eq. (10) and (11) from "Augmenting Neural Networks with Priors on
        # Function Values"
        if post_process is None:
            post_process = self.priors_on_function_values
        if post_process == "pfv":
            mu /= 1 + var_epi
            log_var_epi -= torch.log1p(var_epi)

        if normalize_delta is None:
            normalize_delta = self.normalize_delta
        if normalize_delta:
            sample_mean, sample_std = (
                self.output_normalizer.mean,
                self.output_normalizer.std,
            )
            mu = sample_mean + sample_std * mu
            sample_log_var = sample_std.log() * 2

            log_var_epi += sample_log_var
            log_var_ale += sample_log_var

        if self.uncertainty_bonus or log:
            self.state_entropy = self.uncertainty(mus)
        return mu, log_var_epi, log_var_ale

    def forward(self, x: torch.Tensor, prediction_strategy=None, **kwargs):
        """Propagate uncertainty forward with specified `prediction_strategy`."""
        prediction_strategy = prediction_strategy or self.prediction_strategy
        if prediction_strategy in (PS_TS1, PS_INF):
            batch_size = x.shape[0]
            num_samples_per_member = batch_size // self.num_members
            x = x.view(self.num_members, num_samples_per_member, self.dim_input)
            if prediction_strategy == PS_TS1:
                shuffle_idx = torch.randperm(self.num_members, device=self.device)
                reshuffle_idx = shuffle_idx.argsort()
                log_std_ale = self._log_noise[shuffle_idx, ...]
                x = x[shuffle_idx, ...]
                mu = self.layers(x)
                mu = mu[reshuffle_idx, ...].view(batch_size, self.dim_output)
            else:
                log_std_ale = self._log_noise
                mu = self.layers(x).view(batch_size, self.dim_output)

            log_std_ale = log_std_ale.repeat(1, num_samples_per_member, 1).view(
                batch_size, self.dim_output
            )

            scale = log_std_ale.exp()
        elif prediction_strategy == PS_TS1_F:
            mu, scale = self.predict_ts1_full(x)

        elif prediction_strategy == PS_MMD:
            mu, scale = self.predict_mm_direct(x)

        elif prediction_strategy == PS_MM:
            mu, log_var_epi, log_var_ale = self.moment_dist(
                x, normalize_delta=False, **kwargs
            )
            variance = log_var_ale.exp() + log_var_epi.exp()
            scale = variance.sqrt()
        else:
            assert prediction_strategy == PS_NO_EPISTEMIC
            mu = self.layers(x)[0]
            scale = self._log_noise[0].exp()

        return mu, scale

    def predict_ts1_full(self, x, tile=None):
        shuffle_index = torch.randperm(self.batch_size, device=self.device)
        if tile:
            shuffle_index = shuffle_index.tile(tile)
        shuffle_mask = self.base_index[:, shuffle_index, :]
        mus = self.layers(x)
        mu = torch.sum(mus * shuffle_mask, 0)
        log_std_ale = self._log_noise[shuffle_mask[..., 0].max(0).indices, ...].squeeze(
            1
        )
        scale = log_std_ale.exp()
        if self.uncertainty_bonus:
            self.state_entropy = self.uncertainty(mus)
        return mu, scale

    def predict_mm_direct(self, x):
        gauss_noise = torch.randn((self.num_members, x.shape[0], 1), device=self.device)
        mus = self.layers(x)
        mean = mus.mean(0, keepdims=True)

        mu = 1 / np.sqrt(self.num_members) * ((mus - mean) * gauss_noise).sum(
            axis=0
        ) + mean.squeeze(0)

        log_std_ale = self._log_noise.mean(0)
        scale = log_std_ale.exp()
        if self.uncertainty_bonus:
            self.state_entropy = self.uncertainty(mus)
        # breakpoint()
        return mu, scale

    def uncertainty(self, mus):
        # Geometric Jensen Shannon entropy
        vars = (self._log_noise * 2).exp()
        var_c = vars[None, ...]
        var_r = vars[:, None, ...]
        var_mean = 2 * var_c * var_r / (var_c + var_r)

        mu_c = mus[None, ...]
        mu_r = mus[:, None, ...]
        mu_mean = var_mean * (mu_c / var_c + mu_r / var_r) * 0.5

        trace_term = (var_c + var_r) / var_mean * 0.5
        log_term = var_mean.log() - 0.5 * (var_c.log() + var_r.log())
        error_term = (
            0.5 * ((mu_mean - mu_c).square() + (mu_mean - mu_r).square()) / var_mean
        )
        gjs = torch.mean(trace_term + log_term + error_term - 1, dim=-1) * 0.5
        gjs = gjs.sum((0, 1)) / (self.num_members * (self.num_members - 1)) * 0.25
        return gjs

    def update(
        self,
        batch: TransitionBatch,
        optimizer: torch.optim.Optimizer,
        log: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Updates the model given a batch of transitions and an optimizer."""
        model_in, target, weight = self.process_batch(batch, training=True)

        self.train()
        optimizer.zero_grad()

        # Predict
        normalize_delta = self.normalize_delta and not self.normalized_target
        mus, log_std_ale = self.base_forward(model_in, normalize_delta=normalize_delta)
        weight = weight.t()
        # Calculate loss
        if (
            self.training_loss_fn == "nll"
        ):  # Temporary solution for switching between loss functions
            loss = self.nll_loss(mus, log_std_ale, target, weight)
        else:
            assert self.training_loss_fn == "gauss_adapt"
            loss = self.gauss_adapt_loss(mus, log_std_ale, target, weight)

        loss.backward()
        optimizer.step()

        if log:
            with torch.no_grad():
                mu = mus.mean(0)
                diff = mu - target.mean(0)
                if self.normalize_delta and self.normalized_target:
                    std = self.output_normalizer.std / self.output_normalizer.scale
                    diff *= std
                error = diff.square()
                se = error.sum(-1)
                max_error = diff.abs().mean(-1).max()
                max_l2_error = diff.square().mean(-1).sqrt().max()
                max_target = target.abs().mean((0, 2)).max()
            info = {
                "squared_error": se,
                "max_error": max_error,
                "max_l2_error": max_l2_error,
                "max_target": max_target,
            }
        else:
            info = {}

        return loss, info

    def nll_loss(
        self,
        pred_mus: torch.Tensor,
        pred_log_noise: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = 1.0,
        log=False,
    ) -> torch.Tensor:
        """Negative log-likelihood on mini-batch.

        B: Batch size
        D: Dimension of input data
        E: Ensemble size
        model_in: [(E,) B, D]
        target: [(E,) B, D]
        weight: [E, B] if ensemble
        """

        l2 = F.mse_loss(pred_mus, target, reduction="none")
        inv_var = (-2 * pred_log_noise).exp() * 0.5
        if self.lb_std:
            pred_logstd = pred_log_noise.clamp_min(self.lb_log_scale)
        else:
            pred_logstd = pred_log_noise
        losses = l2 * inv_var + pred_logstd
        loss = losses.sum(-1) if self.sum_over_data else losses.mean(-1)
        loss = (loss * weight).sum(0).mean()
        return loss

    def gauss_adapt_loss(  # self, y_pred, y, y_std, eps=1e-12):
        self,
        pred_mus: torch.Tensor,
        pred_log_noise: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = 1.0,
        log=False,
    ) -> torch.Tensor:
        """Negative log-likelihood on mini-batch.

        B: Batch size
        D: Dimension of input data
        E: Ensemble size
        model_in: [(E,) B, D]
        target: [(E,) B, D]
        weight: [E, B] if ensemble
        """

        l2_mean_loss = F.mse_loss(pred_mus, target, reduction="none")

        if self.lb_std:
            pred_logstd = pred_log_noise.clamp_min(self.lb_log_scale)
        else:
            pred_logstd = pred_log_noise

        mean_log = l2_mean_loss.detach().mean(dim=[0, 1], keepdims=True).log()
        mean_log = mean_log.expand(
            pred_logstd.shape[0], mean_log.shape[1], mean_log.shape[2]
        )
        pred_logvar = 2 * pred_logstd
        var_loss = F.mse_loss(pred_logvar, mean_log, reduction="none")

        losses = l2_mean_loss + var_loss
        if self.sum_over_data:
            loss = losses.sum([0, 2]).mean()
        else:
            loss = losses.sum(0).mean()
        return loss

    @torch.no_grad()
    def eval_score(self, batch, log=False):
        """Computes the loss on batch."""
        model_in, target, weight = self.process_batch(batch)
        assert model_in.ndim == 2 and target.ndim == 2
        normalize_delta = self.normalize_delta and not self.normalized_target

        mu, log_var_epi, log_var_ale = self.moment_dist(
            model_in,
            normalize_delta=normalize_delta,
            log=log,
            post_process=False,
        )
        error = torch.square(target - mu)  # [B, D]

        var_epi = log_var_epi.exp()
        variance = log_var_ale.exp() + var_epi

        if self.mse_score:
            eval_score = error.sum(-1)
        else:
            scale_log = variance.log() * 0.5

            eval_scores = 0.5 * error / variance + scale_log
            eval_score = eval_scores.sum(-1)  # [B, D] -> [B]

        # Logging
        info = {}
        if log:
            info["uncertainty"] = self.state_entropy.detach()

            if log == "detail":
                log_std_ale = log_var_ale * 0.5

                log_std_epi_ = var_epi.sqrt().log().mean(0)
                log_std_ale_ = log_std_ale.mean(0)

                for dim, (logstd_epi_d, logstd_ale_d) in enumerate(
                    zip(log_std_epi_, log_std_ale_)
                ):
                    info[f"{dim}_logstd_epi"] = logstd_epi_d
                    info[f"{dim}_logstd_ale"] = logstd_ale_d

        return eval_score, error, variance, info

    def rescale_error(self, error, variance):
        if self.normalize_delta is None and self.normalized_target:
            std = self.input_normalizer.std / self.input_normalizer.scale
            var = std.square()
            error[:, 1:] *= var
            variance[:, 1:] *= var
        elif self.normalize_delta and self.normalized_target:
            std = self.output_normalizer.std / self.output_normalizer.scale
            var = std.square()
            error *= var
            variance *= var
        return error, variance

    def mse(self, mus, log_std_ale, target, variance=False):
        mu = mus.mean(0)
        error = F.mse_loss(mu, target, reduction="none")
        mse = error.sum(-1)
        if variance:
            var_epi = mus.var(0)
            variance = var_epi + (2 * log_std_ale).exp()
            return error, mse, variance
        else:
            return error, mse

    @property
    @torch.no_grad()
    def decay_loss(self) -> torch.Tensor:
        decay_loss: torch.Tensor = 0.0  # type: ignore
        for layer in self.layers:
            if isinstance(
                layer, (nn.Linear, NormalizedEnsembleLinear, EnsembleLinearLayer)
            ):
                decay_loss_m = torch.sum(torch.square(layer.weight))
                decay_loss_m += torch.sum(torch.square(layer.bias))
                decay_loss += layer.weight_decay * decay_loss_m * 0.5

        decay_loss += torch.square(self._log_noise).sum() * self.noise_wd

        return decay_loss

    def process_batch(
        self,
        batch: TransitionBatch,
        training=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch.obs
        action = batch.action
        # Input
        if self.normalize_input:
            obs_input = self.input_normalizer.normalize(obs)
        else:
            obs_input = obs
        model_in = torch.cat([obs_input, action], dim=1)

        # Target
        if self.delta_prediction:
            target_obs = batch.next_obs - obs
        else:
            target_obs = batch.next_obs
        # State-normalization
        if not self.normalize_delta and self.normalized_target:
            input_scale = self.input_normalizer.std
            target_obs /= input_scale

        if self.learned_reward:
            target = torch.cat([batch.reward, target_obs], dim=obs.ndim - 1)
        else:
            target = target_obs

        if training:
            target = target[None, ...].repeat(self.num_members, 1, 1)
            target[..., 0] += (
                torch.randn(target.shape[:2], device=self.device) * self.lb_std
            )
        # Delta-normalization
        if self.normalize_delta and self.normalized_target:
            target = self.output_normalizer.normalize(target)
        return model_in, target.clone(), batch.weight

    def sample(
        self,
        act: torch.Tensor,
        obs: torch.Tensor,
        log: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, float]],
    ]:
        """Samples next observations and rewards from the underlying 1-D model."""
        if not self.normalize_po_input and self.normalize_input:
            obs_input = self.input_normalizer.normalize(obs)
        else:
            obs_input = obs
        model_in = torch.cat([obs_input, act], dim=obs.ndim - 1)
        mu, scale = self.forward(model_in, log=log, **kwargs)

        info = {}
        # Log before adding noises
        if log:
            with torch.no_grad():
                if self.learned_reward:
                    reward_mu, obs_mu = torch.tensor_split(mu.abs(), [1], 1)
                    reward_scale, obs_scale = torch.tensor_split(scale, [1], 1)
                    info.update(
                        {
                            # "{sample (std, max)}_{dimension (l1, l2)}_{obs / reward}_{mu / scale}"
                            # Observation
                            "std_obs_mu": obs_mu.std(0).mean(),
                            "mean_obs_scale": obs_scale.mean(),
                            "max_l1_obs_mu": obs_mu.abs().mean(-1).max(),
                            "max_l2_obs_mu": obs_mu.square().mean(-1).sqrt().max(),
                            # Reward
                            "mean_reward": reward_mu.mean(),
                            "max_l1_reward_mu": reward_mu.abs().max(),
                            "max_l2_reward_mu": reward_mu.square()
                            .mean(-1)
                            .sqrt()
                            .max(),
                            # "max_l2_reward_scal": reward_mu.square().mean(-1).std(),
                        }
                    )
                else:
                    info["max_obs_mu"] = mu.abs().max()
                    info["max_obs_scale"] = scale.max()
            if self.uncertainty_bonus:
                info.update(uncertainty_bonus=self.state_entropy.mean())

        epsilon = torch.randn_like(obs)
        if self.normalize_delta:
            mu[..., 1:] += scale[..., 1:] * epsilon

            if self.learned_reward:
                assert self.input_normalizer is not None
                input_std = self.input_normalizer.std
                output_mu, output_std = (
                    self.output_normalizer.mean,
                    self.output_normalizer.std,
                )
                if self.normalized_reward:
                    mu[..., 1:] = output_mu[..., 1:] + mu[..., 1:] * output_std[..., 1:]
                else:
                    mu = output_mu + mu * output_std
                reward, obs_diff = torch.tensor_split(mu, [1], dim=-1)
                if self.normalize_po_input:
                    obs_diff /= input_std
            else:
                reward = False
                obs_diff = self.output_normalizer.denormalize(mu)
            next_obs = obs + obs_diff

        else:
            # TODO: For the case self.normalize_target but not self.normalize_po_input
            assert self.normalize_po_input
            if self.learned_reward:
                reward, obs_mu = torch.tensor_split(mu, [1], dim=-1)
                obs_scale = scale[:, 1:]
            else:
                obs_mu = mu
                obs_scale = scale
                reward = None
            obs_delta = obs_mu + epsilon * obs_scale
            if not self.normalized_target and self.normalize_po_input:
                input_std = self.input_normalizer.std
                obs_delta /= input_std
            next_obs = obs + obs_delta

        if self.term_fn:
            obs_t = obs
            next_obs_t = next_obs
            # if self.normalized_target and self.normalize_po_input:
            if self.normalize_po_input:
                obs_t = self.input_normalizer.denormalize(obs_t)
                next_obs_t = self.input_normalizer.denormalize(next_obs_t)
            terminated = self.term_fn(obs_t, act, next_obs_t)
        else:
            terminated = False
        if isinstance(terminated, bool):
            terminated = torch.full_like(reward, terminated, dtype=torch.bool)

        return next_obs, reward, terminated, info

    def optimized_parameters(self, recurse: bool = True):
        params = []
        for layer in self.layers:
            if isinstance(
                layer, (nn.Linear, EnsembleLinearLayer, NormalizedEnsembleLinear)
            ):
                param = {
                    "params": layer.parameters(recurse),
                    "weight_decay": layer.weight_decay,
                }
                params.append(param)
            elif isinstance(layer, nn.LayerNorm) and layer.elementwise_affine:
                params.append({"params": layer.parameters()})

        params.append({"params": self._log_noise, "weight_decay": self.noise_wd})
        return params

    def print_noise(self):
        noises = self._log_noise.detach().exp().cpu().numpy()
        print("The model noise levels are: ", noises.T)
        return noises.T
