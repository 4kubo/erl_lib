from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from erl_lib.agent.model_based.modules.model import Model
from erl_lib.agent.module.layer import (
    NormalizedEnsembleLinear,
    EnsembleLinearLayer,
)
from erl_lib.base.datatype import (
    TransitionBatch,
)

# Uncertainty propagation strategies
PS_MM = "moment_matching"
PS_TS1 = "ts_1"
PS_INF = "ts_infinity"


class GaussianMLP(Model):
    """Wrapper class for 1-D dynamics models."""

    INTRINSIC_REWARD = "intrinsic_reward"

    def __init__(
        self,
        term_fn,
        input_normalizer,
        output_normalizer,
        *args,
        num_members: int = 1,
        dim_hidden: int = 200,
        noise_bias_fac: float = 0.1,
        noise_wd: float = 0.0,
        normalize_layer: bool = True,
        lb_std: float = 1e-3,
        weight_decay_base: float = 0.001,
        weight_decay_ratios: Sequence[float] = None,
        # Prediction
        normalize_io: bool = True,
        uncertainty_bonus: float = 1.0,
        normalized_reward=False,
        priors_on_function_values: bool = True,
        prediction_strategy: str = PS_TS1,
        sample_reward: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.term_fn = term_fn
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.normalize_io = input_normalizer is not None
        self.num_members = num_members
        self.noise_wd = noise_wd
        self.lb_std = lb_std or 0
        self.lb_log_scale = (
            np.log(lb_std) if (lb_std is not None and 0 < lb_std) else None
        )
        self.weight_decay_base = weight_decay_base
        self.uncertainty_bonus = uncertainty_bonus
        self.normalized_reward = normalized_reward
        self.priors_on_function_values = priors_on_function_values
        assert prediction_strategy in [PS_MM, PS_TS1, PS_INF]
        self.prediction_strategy = prediction_strategy
        self.sample_reward = sample_reward

        # Build neural network model
        weight_decays = np.asarray(weight_decay_ratios) * weight_decay_base
        num_layers = len(weight_decay_ratios)
        max_r = max(weight_decay_ratios)
        drop_rs = (max_r - np.asarray(weight_decay_ratios)) * 0.01
        layers = []
        dims = [self.dim_input] + [dim_hidden] * (num_layers - 1) + [self.dim_output]
        for i, (dim_input, dim_output, wd_i, dr_i) in enumerate(
            zip(dims[:-1], dims[1:], weight_decays, drop_rs)
        ):
            if normalize_layer and 0 < i:
                layers += [nn.LayerNorm(dim_input, elementwise_affine=False)]
            if 1 < num_members:
                layers += [
                    EnsembleLinearLayer(
                        num_members, dim_input, dim_output, weight_decay=wd_i
                    )
                ]
            else:
                linear = nn.Linear(dim_input, dim_output)
                linear.weight_decay = wd_i
                layers += [linear]

            if 0 < dr_i < 1:
                layers += [nn.Dropout(dr_i)]
            if i != num_layers - 1:
                layers += [nn.SiLU()]

        self.layers = nn.Sequential(*layers)

        # Homo-scedastic noise
        if 1 < num_members:
            shape = (num_members, 1, self.dim_output)
        else:
            shape = (1, self.dim_output)

        self.noise_bias_fac = noise_bias_fac

        log_noise_init = (
            np.log(noise_bias_fac).astype(np.float32).item()
            if 0 < noise_bias_fac
            else 0.0
        )
        self._log_noise = nn.Parameter(
            torch.full(shape, log_noise_init, dtype=torch.float32),
            requires_grad=True,
        )

        self.to(self.device)

    def base_forward(
        self,
        x: torch.Tensor,
        normalized_reward=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            x : [B, D]
            normalized_reward: If True, Predict normalized reward by running statistics
        """
        mus = self.layers(x)

        if self.normalize_io:
            sample_mean, sample_std = (
                self.output_normalizer.mean,
                self.output_normalizer.std,
            )
            mus = sample_mean + sample_std * mus
            log_std_ale = self._log_noise + sample_std.log()
        else:
            log_std_ale = self._log_noise
        return mus, log_std_ale, {}

    def base_dist(self, x, normalized_reward=False, uncertainty_bonus=0.0, **_):
        """Parameters for moment matching as prediction strategy."""
        mus, log_std_ale, info = self.base_forward(x, normalized_reward)

        if self.priors_on_function_values:
            # c.f. eq. (10) and (11) from "Augmenting Neural Networks with Priors on
            #  Function Values"
            var_epi = mus.var(0)
            std_prior = self.output_normalizer.std
            var_prior = torch.square(std_prior)
            var_sum = var_epi + var_prior
            ratio = var_prior / var_sum
            mu = ratio * mus.mean(0) + (1.0 - ratio) * self.output_normalizer.mean

            log_var_epi = var_epi.log()
            log_std_epi = (log_var_epi + ratio.log()) * 0.5
        else:
            mu = mus.mean(0)
            log_std_epi = mus.std(0).log()

        log_std_ale = log_std_ale.mean(0)

        # if 0 < uncertainty_bonus:
        #     normalized_std_epi = (
        #         log_std_epi.exp() / self.output_normalizer.std.clamp_min(1e-8)
        #     )
        #     # normalized_std_epi = torch.exp(log_std_epi - log_std_ale)
        #     epi_bonus = normalized_std_epi.mean(-1, keepdims=True) * uncertainty_bonus
        #     info[self.INTRINSIC_REWARD] = epi_bonus

        return mu, log_std_epi, log_std_ale, info

    def forward(
        self,
        x: torch.Tensor,
        normalized_reward=False,
        uncertainty_bonus=0.0,
        log: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """

        Args:
            x:
            log:
            **kwargs:

        Returns:
            mu: [B, D]
            scale: [B, D]
        """
        assert x.ndim == 2
        assert not self.layers.training
        if self.prediction_strategy in (PS_TS1, PS_INF):
            if self.prediction_strategy == PS_TS1:
                ensemble_idx = torch.randperm(self.num_members, device=self.device)
                for layer in self.layers:
                    if isinstance(layer, EnsembleLinearLayer):
                        layer.set_index(ensemble_idx)

                log_std_ale = self._log_noise[ensemble_idx, ...]
            else:
                log_std_ale = self._log_noise

            batch_size = x.shape[0]
            num_samples_per_member = batch_size // self.num_members
            x = x.view(self.num_members, num_samples_per_member, self.dim_input)
            mu = self.layers(x).view(batch_size, self.dim_output)

            # if self.normalize_io:
            #     mu, log_std_ale = self.rescale(mu, log_noise)

            log_std_ale = (
                log_std_ale.repeat(1, num_samples_per_member, 1)
                .contiguous()
                .view(-1, self.dim_output)
            )

            scale = log_std_ale.exp()
            info = {}

        elif 1 < self.num_members:
            mu, log_std_epi, log_std_ale, info = self.base_dist(
                x,
                normalized_reward=normalized_reward,
                log=log,
                uncertainty_bonus=uncertainty_bonus,
                **kwargs,
            )

            var_epi = (2 * log_std_epi).exp()
            var_ale = (2 * log_std_ale).exp()
            scale = torch.sqrt(var_epi + var_ale)
        else:
            mu, log_std_ale, info = self.base_forward(x)
            scale = log_std_ale.exp()

        # Logging
        if log == "detail":
            with torch.no_grad():
                if 1 < self.num_members:
                    log_std_epi_ = log_std_epi.mean(0)
                    log_std_ale_ = log_std_ale.mean(0)

                    for dim, (logstd_epi_d, logstd_ale_d) in enumerate(
                        zip(log_std_epi_, log_std_ale_)
                    ):
                        info[f"{dim}_logstd_epi"] = logstd_epi_d
                        info[f"{dim}_logstd_ale"] = logstd_ale_d
                else:
                    log_std_ale_ = log_std_ale.squeeze(0)

                    for dim, logstd_ale_d in enumerate(log_std_ale_):
                        info[f"{dim}_logstd_ale"] = logstd_ale_d
        return mu, scale, info

    # def rescale(self, mu, log_std_ale):
    #     mu_out, std_out = (
    #         self.output_normalizer.mean,
    #         self.output_normalizer.std,
    #     )
    #     mu = mu_out + std_out * mu
    #     log_std_ale = log_std_ale + std_out.log()
    #
    #     return mu, log_std_ale

    def _mse_loss(
        self,
        model_in: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = 1.0,
        log=False,
    ):
        """Mean-squared-error for ensemble of deterministic models.

        B: Batch size
        D: Dimension of input data
        E: Ensemble size
        model_in: [(1,) B, D]
        target: [B, D]
        weight: [B, E]
        """
        mus, log_std_ale, info = self.base_forward(model_in)
        errors = F.mse_loss(mus, target[None, :], reduce=False)
        mses = torch.mean(errors.sum(-1) * weight, dim=0)  # average over batch
        variance = (
            (log_std_ale.mean(0) * 2).exp().expand(model_in.shape[0], self.dim_output)
        )
        return mses, errors.mean(0), variance, info

    def _nll_loss(
        self,
        model_in: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = 1.0,
        log=False,
    ) -> Tuple[torch.Tensor, dict]:
        """Negative log-likelihood on mini-batch.

        B: Batch size
        D: Dimension of input data
        E: Ensemble size
        model_in: [(E,) B, D]
        target: [(E,) B, D]
        weight: [E, B] if ensemble
        """
        pred_mean, pred_logstd, info = self.base_forward(
            model_in, normalized_reward=False
        )
        l2 = F.mse_loss(pred_mean, target, reduction="none")
        inv_var = (-2 * pred_logstd).exp() * 0.5
        if self.lb_std:
            pred_logstd = pred_logstd.clamp_min(np.log(self.lb_std))
        nll = torch.sum(l2 * inv_var + pred_logstd, -1)
        nll = torch.mean(nll * weight, dim=-1)  # average over batch

        loss = nll.sum()  # sum over ensemble dimension
        return loss, info

    def update(
        self,
        batch: TransitionBatch,
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 0.0,
        log: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Updates the model given a batch of transitions and an optimizer."""
        model_in, target, weight = self.process_batch(batch)
        assert model_in.ndim == target.ndim

        self.train()
        optimizer.zero_grad()

        # Predict
        mus, log_std_ale, info = self.base_forward(model_in, normalized_reward=False)
        # if self.normalize_io:
        #     state_mu, state_std = self.input_normalizer.mean, self.input_normalizer.std
        #     reward_mu, reward_std = (
        #         self.output_normalizer.mean,
        #         self.output_normalizer.std,
        #     )
        #     target[:, 1:] = (target[:, 1:] - state_mu) / state_std
        #     target[:, :1] = (target[:, :1] - reward_mu) / reward_std
        # log_std_ale = log_std_ale + sample_std.log()
        # Preprocess on target values
        if 1 < self.num_members:
            target = target[None, ...].repeat(self.num_members, 1, 1)
            weight = weight.t()
        else:
            weight = None
        # Calculate loss
        loss = self.nll_loss(mus, log_std_ale, target, weight)

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), grad_clip)

        optimizer.step()
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
        # if self.lb_std:
        #     pred_log_noise = pred_log_noise.clamp_min(self.lb_log_scale)
        inv_var = (-2 * pred_log_noise).exp() * 0.5
        if self.lb_std:
            pred_logstd = pred_log_noise.clamp_min(self.lb_log_scale)
        else:
            pred_logstd = pred_log_noise
        nll = torch.sum(l2 * inv_var + pred_logstd, -1)
        nll = torch.mean(nll * weight, dim=-1)  # average over batch

        loss = nll.sum()  # sum over ensemble dimension
        return loss

    def eval_score(self, batch, log=False):
        """Computes the squared error for the model over the given input/target."""
        model_in, target, weight = self.process_batch(batch)
        assert model_in.ndim == 2 and target.ndim == 2
        if self.prediction_strategy in (PS_TS1, PS_INF):
            # return self._mse_loss(model_in, target)
            mus, log_std_ale, info = self.base_forward(model_in)
            # if self.normalize_io:
            #     sample_mean, sample_std = (
            #         self.output_normalizer.mean,
            #         self.output_normalizer.std,
            #     )
            #     mus = sample_mean + sample_std * mus
            #     log_std_ale = log_std_ale + sample_std.log()
            mu = mus.mean(0)
            var_ale = (2 * log_std_ale).mean(0).exp()
            var_epi = mus.var(0)
            variance = var_ale + var_epi
            scale_log = variance.log() * 0.5

            error = (target - mu) ** 2  # [B, D]

            # Logging
            if log == "detail":
                with torch.no_grad():
                    if 1 < self.num_members:
                        log_std_epi_ = var_epi.sqrt().log().mean(0)
                        log_std_ale_ = log_std_ale.mean((0, 1))

                        for dim, (logstd_epi_d, logstd_ale_d) in enumerate(
                            zip(log_std_epi_, log_std_ale_)
                        ):
                            info[f"{dim}_logstd_epi"] = logstd_epi_d
                            info[f"{dim}_logstd_ale"] = logstd_ale_d
                    else:
                        raise NotImplementedError

        else:
            mu, scale, info = self.forward(model_in, log=log, normalized_reward=False)
            variance = torch.square(scale)  # [B, D]

            error = F.mse_loss(target, mu, reduction="none")  # [B, D]
            # error = (target - mu) ** 2  # [B, D]
            scale_log = scale.log()

        # score = 0.5 * error / (variance + self.lb_std**2) + scale_log
        score = 0.5 * error / variance + scale_log

        eval_score = score.sum(-1)  # [B, D] -> [B]
        return eval_score, error, variance, info

    def decay_loss(self) -> torch.Tensor:
        decay_loss: torch.Tensor = 0.0  # type: ignore
        for layer in self.layers:
            if isinstance(layer, (nn.Linear, NormalizedEnsembleLinear)):
                decay_loss_m = torch.sum(torch.square(layer.weight))
                decay_loss_m += torch.sum(torch.square(layer.bias))
                decay_loss += layer.weight_decay * decay_loss_m * 0.5

        decay_loss += torch.square(self._log_noise).sum() * self.noise_wd

        return decay_loss

    def process_batch(
        self,
        batch: TransitionBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch.obs
        action = batch.action
        # Input
        if self.normalize_io:
            obs_input = self.input_normalizer.normalize(obs)
        else:
            obs_input = obs
        model_in = torch.cat([obs_input, action], dim=1)

        # Target
        target_obs = batch.next_obs - obs
        if self.learned_reward:
            target = torch.cat([batch.reward, target_obs], dim=obs.ndim - 1)
        else:
            target = target_obs
        # if 0 < target_noise_scale:
        #     target += torch.randn_like(target) * target_noise_scale
        # if self.normalize_io:
        #     target = self.output_normalizer.normalize(target)
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
        model_in = torch.cat([obs, act], dim=obs.ndim - 1)
        mu, scale, info = self.forward(model_in, log=log, **kwargs)

        epsilon = torch.randn_like(obs)
        if self.normalize_io:
            input_mu, input_std = self.input_normalizer.mean, self.input_normalizer.std
            mu[:, 1:] += scale[:, 1:] * epsilon
            diff = self.output_normalizer.denormalize(mu)
            if self.learned_reward:
                reward, obs_diff = torch.tensor_split(diff, [1], dim=1)
                obs_diff /= input_std
            else:
                reward = False
                obs_diff = diff
            next_obs = obs + obs_diff

            if self.term_fn:
                # std = self.input_normalizer.std
                obs_t = self.input_normalizer.denormalize(obs)
                next_obs_t = self.input_normalizer.denormalize(next_obs)
                terminated = self.term_fn(obs_t, act, next_obs_t)
            else:
                terminated = False
        else:
            if self.learned_reward:
                reward, obs_mu = torch.tensor_split(mu, [1], dim=1)
                obs_scale = scale[:, 1:]
            else:
                obs_mu = mu
                obs_scale = scale
                reward = None
            next_obs = obs + obs_mu + epsilon * obs_scale
            if self.term_fn:
                terminated = self.term_fn(obs, act, next_obs)
            else:
                terminated = False

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
