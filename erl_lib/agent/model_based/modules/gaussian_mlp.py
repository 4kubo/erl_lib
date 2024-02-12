from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from erl_lib.util.misc import soft_bound
from erl_lib.agent.model_based.modules.model import Model
from erl_lib.agent.module.layer import (
    NormalizedEnsembleLinear,
    EnsembleLinearLayer,
)
from erl_lib.base.datatype import (
    TransitionBatch,
)


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logstd: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True,
    min_log_noise: float = None,
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian distribution."""
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-2 * pred_logstd).exp() * 0.5
    if min_log_noise:
        pred_logstd = pred_logstd.clamp_min(np.log(1e-3))
    losses = l2 * inv_var + pred_logstd
    if reduce:
        return losses.sum(dim=1).mean()
    return losses


class GaussianMLP(Model):
    """Wrapper class for 1-D dynamics models."""

    INTRINSIC_REWARD = "intrinsic_reward"

    def __init__(
        self,
        input_normalizer,
        output_normalizer,
        *args,
        num_members: int = 1,
        dim_hidden: int = 200,
        layer_order: str = "NLDS",
        noise_bias_fac: float = 0.5,
        noise_wd: float = 0.0,
        lb_std: float = 1e-3,
        dropout_rate: float = 0.1,
        droprate_type: str = "logspace",
        weight_decay_base: float = 0.001,
        weight_decay_ratios: Sequence[float] = None,
        min_ale: float = 1e-3,
        uncertainty_bonus: float = 1.0,
        normalized_reward=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.num_members = num_members
        self.noise_wd = noise_wd
        self.lb_std = lb_std
        self.lb_log_scale = np.log(lb_std) if 0 < lb_std else None
        self.weight_decay_base = weight_decay_base
        self.min_ale = min_ale
        self.uncertainty_bonus = uncertainty_bonus
        self.normalized_reward = normalized_reward

        weight_decays = np.asarray(weight_decay_ratios) * weight_decay_base
        num_layers = len(weight_decay_ratios)
        if 0 < dropout_rate:
            if droprate_type == "logspace":
                droprate_log = np.log10(dropout_rate)
                drop_rs = np.logspace(droprate_log, droprate_log - 1, num_layers)
            else:
                max_r = max(weight_decay_ratios)
                drop_rs = (max_r - np.asarray(weight_decay_ratios)) * dropout_rate
        else:
            assert droprate_type in (None, "", False)
            drop_rs = np.zeros(num_layers - 1)

        layers = []
        dims = [self.dim_input] + [dim_hidden] * (num_layers - 1) + [self.dim_output]
        for i, (dim_input, dim_output, wd_i, dr_i) in enumerate(
            zip(dims[:-1], dims[1:], weight_decays, drop_rs)
        ):
            if 0 < i:
                layers += [nn.LayerNorm(dim_input, elementwise_affine=False)]
            if 1 < num_members:
                layers += [
                    EnsembleLinearLayer(
                        num_members, dim_input, dim_output, weight_decay=wd_i
                    )
                ]
            else:
                raise NotImplementedError
            if 0 < dr_i < 1:
                layers += [nn.Dropout(dr_i)]
            if i != num_layers - 1:
                layers += [nn.SiLU()]

        self.layers = nn.Sequential(*layers)

        # Homo-scedastic noise
        assert 0 < noise_bias_fac
        self._log_noise_bias = np.log(noise_bias_fac).astype(np.float32).item()
        if 1 < num_members:
            shape = (num_members, 1, self.dim_output)
        else:
            shape = (1, self.dim_output)
        self._log_noise_base = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=True
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
        sample_mean, sample_std = (
            self.output_normalizer.mean,
            self.output_normalizer.std,
        )
        # log_std_ale = (
        #     self._log_noise_base
        #     + self._log_noise_bias
        #     + sample_std.log().clamp_min(0.0)
        # )
        log_std_ale = self._log_noise_base + self._log_noise_bias + sample_std.log()
        if normalized_reward:
            assert self.learned_reward is True
            mus[:, :, 1:] += (
                sample_mean[:, 1:] + (sample_std[:, 1:] - 1) * mus[:, :, 1:]
            )
            # Do nothing about std since std is not used for sampling of reward
        else:
            mus = sample_mean + sample_std * mus
        return mus, log_std_ale, {}

    def base_dist(self, x, normalized_reward=False, uncertainty_bonus=0.0, **_):
        mus, log_std_ale, info = self.base_forward(x, normalized_reward)

        # if self.sample_prior:
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

        if 0 < uncertainty_bonus:
            normalized_std_epi = (
                log_std_epi.exp() / self.output_normalizer.std.clamp_min(1e-8)
            )
            epi_bonus = normalized_std_epi.mean(-1, keepdims=True) * uncertainty_bonus
            info[self.INTRINSIC_REWARD] = epi_bonus

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
        if 1 < self.num_members:
            mu, log_std_epi, log_std_ale, info = self.base_dist(
                x,
                normalized_reward=normalized_reward,
                log=log,
                uncertainty_bonus=uncertainty_bonus,
                **kwargs,
            )
            if self.lb_log_scale:
                log_std_epi = soft_bound(log_std_epi, lb=self.lb_log_scale)
                log_std_ale = soft_bound(log_std_ale, lb=self.lb_log_scale)

            var_epi = (2 * log_std_epi).exp()
            var_ale = (2 * log_std_ale).exp().mean(0)
            scale = torch.sqrt(var_epi + var_ale)
        else:
            mu, log_std_ale, info = self.base_forward(x)
            scale = log_std_ale.exp()

        # Logging
        if log == "detail":
            with torch.no_grad():
                if 1 < self.num_members:
                    log_std_epi_ = log_std_epi.mean(0)
                    log_std_ale_ = log_std_ale.mean((0, 1))

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
        nll = gaussian_nll(
            pred_mean,
            pred_logstd,
            target,
            min_log_noise=np.log(self.lb_std),
            reduce=False,
        ).sum(-1)
        nll = torch.mean(nll * weight, dim=-1)  # average over batch

        loss = nll.sum()  # sum over ensemble dimension
        return loss, info

    # def _mse_loss(
    #     self,
    #     model_in: torch.Tensor,
    #     target: torch.Tensor,
    #     weight: torch.Tensor = 1.0,
    #     log=False,
    # ) -> Tuple[torch.Tensor, dict]:
    #     """Mean-squared-error for ensemble of deterministic models.
    #
    #     B: Batch size
    #     D: Dimension of input data
    #     E: Ensemble size
    #     model_in: [(1,) B, D]
    #     target: [B, D]
    #     weight: [B, E]
    #     """
    #     pred_mean, _, info = self.base_forward(model_in, log=log)
    #     mse = F.mse_loss(pred_mean, target, reduce=False).sum(-1)
    #     mse = torch.mean(mse * weight.t(), dim=1)  # average over batch
    #
    #     loss = mse.sum()  # sum over ensemble dimension
    #     return loss, info

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

        if 1 < self.num_members:
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
            target = target.repeat(self.num_members, 1, 1)
            # if self.min_ale:
            #     target += torch.randn_like(target) * self.min_ale
            loss, info = self._nll_loss(model_in, target, weight.t())
        else:
            loss, info = self._nll_loss(model_in, target)

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), grad_clip)

        optimizer.step()
        return loss, info

    def eval_score(self, batch, log=False):
        """Computes the squared error for the model over the given input/target."""
        model_in, target, weight = self.process_batch(batch)
        assert model_in.ndim == 2 and target.ndim == 2
        mu, scale, info = self.forward(model_in, log=log, normalized_reward=False)
        variance = torch.square(scale)

        error = (target - mu) ** 2
        score = 0.5 * error / (variance + self.lb_std**2) + scale.log()

        eval_score = score.sum(-1)  # [B, D] -> [B]
        return eval_score, error, variance, info

    def decay_loss(self) -> torch.Tensor:
        decay_loss: torch.Tensor = 0.0  # type: ignore
        for layer in self.layers:
            if isinstance(layer, (nn.Linear, NormalizedEnsembleLinear)):
                decay_loss_m = torch.sum(torch.square(layer.weight))
                decay_loss_m += torch.sum(torch.square(layer.bias))
                decay_loss += layer.weight_decay * decay_loss_m * 0.5

        decay_loss += torch.square(self._log_noise_base).sum() * self.noise_wd

        return decay_loss

    def process_batch(
        self, batch: TransitionBatch, *args
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch.obs
        action = batch.action
        # Input
        obs_input = self.input_normalizer.normalize(obs)
        model_in = torch.cat([obs_input, action], dim=1)

        # Target
        target_obs = batch.next_obs - obs
        if self.learned_reward:
            target = torch.cat([batch.reward, target_obs], dim=obs.ndim - 1)
        else:
            target = target_obs
        return model_in, target, batch.weight

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
        obs_input = self.input_normalizer.normalize(obs)
        model_in = torch.cat([obs_input, act], dim=obs.ndim - 1)

        mu, scale, info = self.forward(
            model_in,
            log=log,
            normalized_reward=self.normalized_reward,
            uncertainty_bonus=self.uncertainty_bonus,
            **kwargs,
        )
        epsilon = torch.randn_like(obs)

        if self.learned_reward:
            rewards, obs_mu = torch.tensor_split(mu, [1], dim=1)
            obs_scale = scale[:, 1:, ...]
        else:
            obs_mu = mu
            obs_scale = scale
            rewards = None
        if self.INTRINSIC_REWARD in info:
            intrinsic_reward = info.pop(self.INTRINSIC_REWARD)
            rewards += intrinsic_reward
            if log:
                info[self.INTRINSIC_REWARD] = intrinsic_reward.detach().mean()

        next_obs = obs + obs_mu + epsilon * obs_scale

        return next_obs, rewards, None, info

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

        params.append({"params": self._log_noise_base, "weight_decay": self.noise_wd})
        return params
