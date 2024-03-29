import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Exponential
from tqdm import trange

from erl_lib.base import (
    OBS,
    ACTION,
    REWARD,
    MASK,
    NEXT_OBS,
    WEIGHT,
    KEY_ACTOR_LOSS,
    KEY_CRITIC_LOSS,
)
from erl_lib.base.agent import BaseAgent
from erl_lib.util.misc import (
    ReplayBuffer,
    Normalizer,
    soft_update_params,
    calc_grad_norm,
)


class SACAgent(BaseAgent):
    """SAC algorithm."""

    def __init__(
        self,
        buffer_size: int,
        buffer_device,
        normalize_input: bool,
        device,
        discount: float,
        # Optimization
        num_policy_opt_per_step: int,
        max_batch_size: int,
        lr: float,
        lr_loss_norm: float,
        batch_size: int,
        clip_grad_norm: float,
        split_validation: bool,
        num_sample_weights: int,
        modality: str,
        warm_start: bool,
        # Critic
        critic,
        critic_tau: float,
        critic_subset_size: int,
        num_critic_iter,
        critic_lr_ratio: float,
        reward_q_th_lb: float,
        normalized_reward: bool,
        # Actor
        actor,
        actor_reduction: str,
        actor_tau: float,
        # Alpha
        init_alpha: float,
        lr_alpha: float,
        entropy_balance: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device(device)

        self.num_policy_opt_per_step = num_policy_opt_per_step
        self.discount = discount
        self.batch_size = batch_size
        self.lr = lr
        self.lr_loss_norm = lr_loss_norm
        self.warm_start = warm_start
        # Critic
        assert 0 <= reward_q_th_lb <= 1
        self.reward_q_th = (reward_q_th_lb, 1- reward_q_th_lb)
        self.build_critics(critic)
        self.critic_tau = critic_tau
        self.num_critic_iter = num_critic_iter
        self.critic_lr_ratio = critic_lr_ratio
        self.critic_subset_size = critic_subset_size
        self.num_critic_ensemble = critic.num_members
        self.clip_grad_norm = clip_grad_norm
        self.normalized_reward = normalized_reward
        self.weight_rate = torch.ones(
            (self.batch_size, self.num_critic_ensemble), device=self.device
        )
        if self.critic_scaled:
            self.q_th = torch.tensor(self.reward_q_th, device=self.device)
            self._q_lb = None
            self._q_ub = None
        # Actor
        self.actor = hydra.utils.instantiate(actor).to(self.device)
        self.actor_reduction = actor_reduction
        # Alpha
        self.lr_alpha = lr_alpha
        self.learnable_alpha = 0 < lr_alpha
        self.init_raw_alpha = np.log(init_alpha)
        if self.learnable_alpha:
            self.raw_alpha = torch.as_tensor(
                self.init_raw_alpha, dtype=torch.float32
            ).to(self.device)
            self.raw_alpha.requires_grad = True
        else:
            self.raw_alpha = torch.tensor(self.init_raw_alpha).to(self.device)
        self.target_entropy = -self.dim_act * entropy_balance
        # optimizers
        self.init_optimizer()

        # Modules
        # obs_shape = [self.dim_obs] if modality == "state" else self.dim_obs
        field_shapes = {
            OBS: self.dim_obs,
            ACTION: self.dim_act,
            REWARD: 1,
            NEXT_OBS: self.dim_obs,
            MASK: 1,
            WEIGHT: num_sample_weights,
        }
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            buffer_device,
            max_batch_size=max_batch_size,
            split_section_dict=field_shapes,
            split_validation=split_validation,
            num_sample_weights=num_sample_weights,
        )

        self.normalize_input = normalize_input
        if self.normalize_input:
            self.input_normalizer = Normalizer(
                self.dim_obs, self.device, "input_normalizer"
            )
        else:
            self.input_normalizer = None

        self.num_updated = 0

    def build_critics(self, critic_cfg):
        self.critic_scale_factor = critic_cfg.bound_factor
        self.critic_scaled = critic_cfg.bounded_prediction

        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def init_optimizer(self):
        # self.raw_alpha.data.copy_(self.init_raw_alpha)
        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": [self.raw_alpha], "lr": self.lr_alpha},
            ],
            lr=self.lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr * self.critic_lr_ratio,
        )

    def _act(self, obs, sample):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            if self.normalize_input:
                obs = self.input_normalizer.normalize(obs)
            dist = self.actor(obs)
            if sample:
                action = dist.sample()
            else:
                action = dist.mean
            action = action.cpu().numpy()
        return action

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        super().observe(obs, action, reward, next_obs, terminated, truncated, info)
        # Policy optimization if necessary
        if (self.seed_iters <= self.total_iters) and self.step == 0:
            # Pre-update
            if self.input_normalizer is not None:
                self.input_normalizer.to()

            t1 = time.time()
            iterator = trange(
                self.num_opt_steps,
                **self.kwargs_trange,
                desc=f"[Policy@Ep{self.total_iters: >4}]",
            )
            # Main loop
            with iterator as pbar:
                for opt_step in pbar:
                    self.update(opt_step, opt_step == self.num_opt_steps - 1)
                    # Misc process for Std-output
                    t2 = time.time()
                    elapsed = t2 - t1
                    if 2 <= elapsed:  # Avoiding frequent update
                        t1 = t2
                        info_pbar = {
                            "Actor": self._info["actor_loss"].cpu().item(),
                            "Critic": self._info["critic_loss"].cpu().item(),
                        }
                        pbar.set_postfix(info_pbar)
            self.logger.append(
                "policy_optimization", {"iteration": self.total_iters}, self._info
            )

    def update(self, opt_step, log=False, buffer=None):
        """The main method of the algorithm to update actor and critic models."""
        buffer = buffer or self.replay_buffer
        obs, batch = self.update_critic(buffer, log)
        soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def update_critic(self, replay_buffer, log=False):
        for i in range(self.num_critic_iter):
            batch = replay_buffer.sample(self.batch_size)
            if self.input_normalizer:
                obs = self.input_normalizer.normalize(batch.obs)
                next_obs = self.input_normalizer.normalize(batch.next_obs)
            else:
                obs = batch.obs
                next_obs = batch.next_obs

            critic_loss, q_values = self.critic_loss(
                obs,
                batch.action,
                batch.reward,
                next_obs,
                batch.mask,
            )

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Preprocess before an update on the gradients
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.clip_grad_norm
                )
            self._info[KEY_CRITIC_LOSS] = critic_loss.detach()
            if log and i == self.num_critic_iter - 1:
                self._info.update(
                    **{
                        "gradient-norm": calc_grad_norm(self.critic),
                        "q_value-mean": q_values.detach().mean(),
                        "q_value-std": q_values.detach().std(-1).mean(),
                    }
                )

        # Step a SGD step
        self.critic_optimizer.step()

        return obs, batch

    def critic_loss(self, obs, action, reward, next_obs, mask):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            next_sa = torch.cat([next_obs, next_action], 1)
            q_values = self.critic_target(next_sa)
            q_value = self._reduce(q_values)
            v_value = q_value - self.alpha.detach() * log_prob
            target_value = reward + (mask * self.discount * v_value)

        sa = torch.cat([obs, action], 1)
        q_values = self.critic(sa)
        qf_loss = F.mse_loss(q_values, target_value, reduction="none")
        # qf_loss = qf_loss * self.critic_loss_weight
        qf_loss = qf_loss.sum(-1).mean()
        return qf_loss, q_values.detach()

    def update_actor(self, obs, log=False):
        dist = self.actor(obs)
        action = dist.rsample()
        log_pi = dist.log_prob(action).sum(-1, keepdim=True)
        sa = torch.cat([obs, action], 1)
        q_values = self.critic(sa)
        q_value = self._reduce(q_values, self.actor_reduction)

        actor_loss = self.alpha.detach() * log_pi - q_value
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss = actor_loss.mean()

        log_pi.detach_()
        mean_log_pi = log_pi.mean()
        actor_kl = mean_log_pi + self.target_entropy
        self._info.update(
            **{"entropy": -mean_log_pi, KEY_ACTOR_LOSS: actor_loss.detach()}
        )
        if self.learnable_alpha:
            alpha_loss = -self.alpha * actor_kl
            actor_loss += alpha_loss

            self._info.update(
                **{
                    "alpha_loss": alpha_loss.detach(),
                    "alpha_value": self.alpha.detach(),
                }
            )

        actor_loss.backward()

        if log:
            self._info.update()

        self.actor_optimizer.step()

        return log_pi

    def update_critic_bound(self, reward_lb, reward_ub):
        # q_width = (max_reward - min_reward) * self.critic_scale_factor
        scale = 1.0 / (1 - self.discount)
        q_ub = reward_ub * scale
        q_lb = reward_lb * scale
        w = (q_ub - q_lb) * .5
        c = (q_ub + q_lb) * .5
        q_lb = c - w * self.critic_scale_factor
        q_ub = c + w * self.critic_scale_factor
        if self._q_ub is None:
            self._q_lb = q_lb
            self._q_ub = q_ub
        else:
            self._q_lb.lerp_(q_lb, self.lr_loss_norm)
            self._q_ub.lerp_(q_ub, self.lr_loss_norm)

        self._info.update(**{"reward_ub": reward_ub, "reward_lb": reward_lb,
                             "q_ub": self._q_ub, "q_lb": self._q_lb})

    def _reduce(self, values, reduction="min"):
        if reduction in ("min", "max"):
            if self.critic_subset_size < self.num_critic_ensemble:
                # Randomized ensemble Q-learning
                # Xinyue Chen, Che Wang, Zijian Zhou, Keith W. Ross
                # c.f. https://openreview.net/forum?id=AY8zfZm0tDd
                idx = np.random.choice(
                    range(self.num_critic_ensemble),
                    self.critic_subset_size,
                    replace=False,
                )
                values = values[:, idx]
            if reduction == "min":
                value = torch.min(values, 1, keepdim=True).values
            else:
                value = torch.max(values, 1, keepdim=True).values
        else:
            value = torch.mean(values, 1, keepdim=True)
            if reduction == "ub":
                value += torch.std(values, 1, keepdim=True)

        return value

    @property
    def alpha(self):
        return self.raw_alpha.exp()

    def save(self, dir_checkpoint):
        super().save(dir_checkpoint)
        modules = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "alpha": self.raw_alpha.detach().cpu(),
        }
        torch.save(modules, f"{dir_checkpoint}/sac.pt")
        self.replay_buffer.save(dir_checkpoint)
        if self.normalize_io:
            self.input_normalizer.save(dir_checkpoint)

    def load(self, dir_checkpoint):
        super().load(dir_checkpoint)
        modules = torch.load(
            f"{dir_checkpoint}/sac.pt", map_location=torch.device(self.device)
        )
        self.actor.load_state_dict(modules["actor"])
        self.critic.load_state_dict(modules["critic"])
        self.critic_target.load_state_dict(modules["critic_target"])
        self.raw_alpha.data.copy_(modules["alpha"])
        self.replay_buffer.load(dir_checkpoint)
        if isinstance(self.input_normalizer, Normalizer):
            self.input_normalizer.load(dir_checkpoint)

    @property
    def num_opt_steps(self):
        last_steps = self.total_iters if self.warm_start and (self.total_iters == self.seed_iters) else 1
        return int(
            self.num_policy_opt_per_step
            # * self.steps_per_iter
            * last_steps
            * self.num_envs
        )
