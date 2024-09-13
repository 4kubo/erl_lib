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
    Q_MEAN,
    Q_STD,
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

    trange_kv: dict = {"Actor": KEY_ACTOR_LOSS, "Critic": KEY_CRITIC_LOSS}

    def __init__(
        self,
        buffer_size: int,
        buffer_device: torch.device,
        normalize_po_input: bool,
        device: torch.device,
        discount: float,
        # Optimization
        num_policy_opt_per_step: int,
        max_batch_size: int,
        lr: float,
        batch_size: int,
        clip_grad_norm: float,
        split_validation: bool,
        num_sample_weights: int,
        warm_start: bool,
        # Critic
        critic,
        critic_tau: float,
        num_critic_iter: int,
        critic_lr_ratio: float,
        scaled_critic: bool,
        bounded_critic: bool,
        weighted_critic: bool,
        reward_q_th_lb: float,
        normalized_reward: bool,
        # Actor
        actor,
        actor_reduction: str,
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
        self.warm_start = warm_start
        # Critic
        assert 0 <= reward_q_th_lb <= 1
        self.reward_q_th = (reward_q_th_lb, 1 - reward_q_th_lb)
        self.scaled_critic = scaled_critic
        self.bounded_critic = bounded_critic
        self.weighted_critic = weighted_critic
        self.critic_tau = critic_tau
        self.num_critic_iter = num_critic_iter
        self.critic_lr_ratio = critic_lr_ratio
        self.num_critic_ensemble = critic.num_members
        self.clip_grad_norm = clip_grad_norm
        self.normalized_reward = normalized_reward
        self.build_critics(critic)
        self.q_th = torch.tensor(self.reward_q_th, device=self.device)
        self._q_lb, self._q_ub, self._q_width, self._q_center = None, None, None, None
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

        self.normalize_po_input = normalize_po_input
        self.input_normalizer = Normalizer(
            self.dim_obs,
            self.device,
            name="input_normalizer",
        )

        self.num_updated = 0

    def build_critics(self, critic_cfg):
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.weighted_critic:
            weight_rate = torch.ones(
                (self.batch_size, self.num_critic_ensemble), device=self.device
            )
            self.critic_loss_weight = Exponential(weight_rate).sample()

    def init_optimizer(self):
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
            if self.normalize_po_input:
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
            if self.normalize_po_input:
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
                            key: self._info[value].cpu().item()
                            for key, value in self.trange_kv.items()
                        }
                        pbar.set_postfix(info_pbar)
            self.logger.append(
                "policy_optimization", {"iteration": self.total_iters}, self._info
            )

    def update(self, opt_step, log=False, buffer=None):
        """The main method of the algorithm to update actor and critic models."""
        buffer = buffer or self.replay_buffer
        obs, reward = self.update_critic(buffer, log)
        soft_update_params(self.critic, self.critic_target, self.critic_tau)
        self.update_actor(obs, reward, log=log)

    def update_critic(self, replay_buffer, log=False):
        for i in range(self.num_critic_iter):
            batch = replay_buffer.sample(self.batch_size)
            if self.normalize_po_input:
                obs = self.input_normalizer.normalize(batch.obs)
                next_obs = self.input_normalizer.normalize(batch.next_obs)
            else:
                obs = batch.obs
                next_obs = batch.next_obs

            action, reward = batch.action, batch.reward
            if self.bounded_critic or self.scaled_critic:
                with torch.no_grad():
                    pi = self.actor(obs)
                    action_s = pi.sample()
                    log_pi = pi.log_prob(action_s).sum(-1, keepdims=True)
                    reward_pi = reward - self.alpha * log_pi
                    reward_lb, reward_ub = torch.quantile(reward_pi, self.q_th)
                    (
                        self._q_lb,
                        self._q_ub,
                        self._q_center,
                        self._q_width,
                    ) = self.update_critic_bound(
                        self._q_lb, self._q_ub, reward_lb, reward_ub
                    )

            critic_loss, q_values = self.critic_loss(
                obs,
                action,
                reward,
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
                        "critic_grad_norm": calc_grad_norm(self.critic),
                        Q_MEAN: q_values.detach().mean(),
                        Q_STD: q_values.detach().std(-1).mean(),
                    }
                )

        # Step a SGD step
        self.critic_optimizer.step()

        return obs, reward

    def critic_loss(self, obs, action, reward, next_obs, mask):
        self.critic.train()
        self.critic_target.eval()
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            next_sa = torch.cat([next_obs, next_action], 1)
            target_q_value = self.pred_target_q_value(next_sa)
            v_value = target_q_value - self.alpha.detach() * log_prob
            target_value = reward + (mask * self.discount * v_value)

        sa = torch.cat([obs, action], 1)
        # q_values = self.critic(sa)
        q_values = self.pred_q_value(sa)
        qf_loss = F.mse_loss(q_values, target_value, reduction="none")

        if self.weighted_critic:
            qf_loss *= self.critic_loss_weight
        qf_loss = qf_loss.sum(-1).mean()
        return qf_loss, q_values.detach()

    def update_actor(self, obs, reward, log=False, **kwargs):
        self.critic.eval()
        dist = self.actor(obs)
        action = dist.rsample()
        log_pi = dist.log_prob(action).sum(-1, keepdim=True)
        sa = torch.cat([obs, action], 1)
        q_value = self.pred_terminal_q(sa)

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
            self._info.update(actor_grad_norm=calc_grad_norm(self.actor))

        self.actor_optimizer.step()

        return log_pi

    def pred_q_value(self, obs_action):
        """Infer the Q-value for critic learning."""
        self.critic.train(True)
        pred_q = self.critic(obs_action)
        if self.scaled_critic:
            pred_q = pred_q * self._q_width + self._q_center
        return pred_q

    def pred_target_q_value(self, obs_action):
        """Infer the Q value for the target value in critic learning with MVE."""
        self.critic_target.train(False)
        target_q = self.critic_target(obs_action)
        if self.scaled_critic:
            target_q = target_q * self._q_width + self._q_center
        target_q = self._reduce(target_q, "min")
        if self.bounded_critic:
            target_q = self._q_ub - torch.relu(self._q_ub - target_q)
            target_q = self._q_lb + torch.relu(target_q - self._q_lb)
        return target_q

    def pred_terminal_q(self, obs_action):
        """Infer the Q-value at the terminal of rollout for SVG."""
        self.critic.train(False)
        pred_qs = self.pred_q_value(obs_action)
        pred_qs = self._reduce(pred_qs, self.actor_reduction).t()
        return pred_qs

    def update_critic_bound(self, q_lb, q_ub, reward_lb, reward_ub):
        scale = 1.0 / (1 - self.discount)
        q_ub_new = reward_ub * scale
        q_lb_new = reward_lb * scale
        if q_ub is None:
            q_lb = q_lb_new
            q_ub = q_ub_new
        else:
            q_lb.lerp_(q_lb_new, self.lr)
            q_ub.lerp_(q_ub_new, self.lr)
        q_center = (q_ub + q_lb) * 0.5
        q_width = (q_ub - q_lb) * 0.5

        return q_lb, q_ub, q_center, q_width

    def _reduce(self, values, reduction="min", dim=1, keepdim=True):
        """
        Args:
            values: Ensemble of predicted values are assumed to be at `dim` dimension
            reduction: type of reduction

        """
        if reduction in ("min", "max"):
            if 2 < self.num_critic_ensemble:
                # Randomized ensemble Q-learning
                # Xinyue Chen, Che Wang, Zijian Zhou, Keith W. Ross
                # c.f. https://openreview.net/forum?id=AY8zfZm0tDd
                idx = np.random.choice(
                    range(self.num_critic_ensemble),
                    2,
                    replace=False,
                )
                values = values[..., idx]
            if reduction == "min":
                value = torch.min(values, dim, keepdim=keepdim).values
            else:
                value = torch.max(values, dim, keepdim=keepdim).values
        else:
            value = torch.mean(values, dim, keepdim=keepdim)
            if reduction == "ub":
                value += torch.std(values, dim, keepdim=keepdim)

        return value

    @property
    def alpha(self):
        return self.raw_alpha.exp()

    def save(self, dir_checkpoint, last=False):
        super().save(dir_checkpoint)
        modules = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "alpha": self.raw_alpha.detach().cpu(),
        }
        if self.bounded_critic or self.scaled_critic:
            modules.update(
                **{
                    "q_lb": self._q_lb,
                    "q_ub": self._q_ub,
                    "q_width": self._q_width,
                    "q_center": self._q_center,
                }
            )
        torch.save(modules, f"{dir_checkpoint}/sac.pt")
        if self.normalize_po_input:
            self.input_normalizer.save(dir_checkpoint)
        if last:
            self.replay_buffer.save(dir_checkpoint)

    def load(self, dir_checkpoint):
        super().load(dir_checkpoint)
        modules = torch.load(
            f"{dir_checkpoint}/sac.pt", map_location=torch.device(self.device)
        )
        self.actor.load_state_dict(modules["actor"])
        self.critic.load_state_dict(modules["critic"])
        self.critic_target.load_state_dict(modules["critic_target"])
        self.raw_alpha.data.copy_(modules["alpha"])
        if self.bounded_critic or self.scaled_critic:
            self._q_lb = modules["q_lb"]
            self._q_ub = modules["q_ub"]
            self._q_width = modules["q_width"]
            self._q_center = modules["q_center"]
        self.replay_buffer.load(dir_checkpoint)
        if self.normalize_po_input:
            self.input_normalizer.load(dir_checkpoint)

    @property
    def num_opt_steps(self):
        last_steps = (
            self.total_iters
            if self.warm_start and (self.total_iters == self.seed_iters)
            else 1
        )
        return int(self.num_policy_opt_per_step * last_steps * self.num_envs)
