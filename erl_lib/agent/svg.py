from abc import ABCMeta
from contextlib import contextmanager, nullcontext
from typing import Callable
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from tqdm import trange

from erl_lib.base.agent import BaseAgent
from erl_lib.agent.sac import SACAgent
from erl_lib.base import (
    OBS,
    ACTION,
    REWARD,
    MASK,
    NEXT_OBS,
    Q_MEAN,
    Q_STD,
    KEY_ACTOR_LOSS,
    KEY_CRITIC_LOSS,
)
from erl_lib.util.misc import (
    calc_grad_norm,
    soft_update_params,
    Normalizer,
    TransitionIterator,
    ReplayBuffer,
)
from erl_lib.agent.model_based.modules.gaussian_mlp import PS_TS1_F


class SVGAgent(SACAgent):
    trange_kv: dict = {"Actor": KEY_ACTOR_LOSS, "Critic": KEY_CRITIC_LOSS}

    def __init__(
        self,
        learned_reward,
        term_fn,
        dynamics_model,
        model_train,
        rollout_horizon,
        training_rollout_horizon,
        rollout_freq,
        mve_horizon,
        retain_model_buffer_iter: int = 10,
        normalize_input: bool = True,
        normalize_output: bool = True,
        normalize_delta: bool = True,
        denormalize_scale: float = 1.0,
        update_after_episode: bool = True,
        uara=True,
        zeta_quantile=0.95,
        uara_xi=10.0,
        lr_kappa: float = 0.01,
        **kwargs,
    ):
        if rollout_horizon <= 0:
            kwargs["buffer_device"] = "cuda"
            rollout_freq = 0
        if kwargs["num_critic_iter"] is None:
            kwargs["num_critic_iter"] = int(kwargs["steps_per_iter"] / 4)

        self.model_update_freq = kwargs["steps_per_iter"]
        super().__init__(**kwargs)

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.normalize_delta = (
            normalize_delta and self.normalize_input and self.normalize_output
        )
        if self.normalize_output:
            self.output_normalizer = Normalizer(
                self.dim_obs + 1,
                self.device,
                scale=denormalize_scale,
                name="output_normalizer",
            )
        else:
            self.output_normalizer = None

        # Dynamics model
        dynamics_model.dim_input = self.dim_obs + self.dim_act
        dynamics_model.dim_output = self.dim_obs + 1
        dynamics_model.normalized_reward = self.normalized_reward
        self.dynamics_model = instantiate(
            dynamics_model,
            term_fn,
            batch_size=self.batch_size,
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
            normalize_input=self.normalize_input,
            normalize_po_input=self.normalize_po_input,
            normalize_delta=self.normalize_delta,
        )
        self.logger.debug(f"{self.dynamics_model._get_name()} is built")
        # The termination function is assumed to be known
        self.rollout_horizon = rollout_horizon
        self.num_rollout_samples = self.batch_size * rollout_freq
        self.rollout_freq = rollout_freq
        # Policy optimization
        if training_rollout_horizon < mve_horizon:
            raise ValueError(
                f"Should 'mve_horizon' <= 'training_rollout_horizon', but {training_rollout_horizon} < {mve_horizon:}"
            )
        self.mve_horizon = mve_horizon
        self.training_rollout_horizon = training_rollout_horizon
        self.update_after_episode = update_after_episode
        # Model training
        self.model_batch_size = model_train.model_batch_size
        self.val_batch_size = model_train.val_batch_size
        self.max_batch_iter = model_train.max_batch_iter
        self.keep_epochs = max(
            model_train.base_epochs * np.log(self.dynamics_model.dim_output),
            1,
        )
        self.max_epochs = self.keep_epochs * 5
        self.max_val_batch_iter = model_train.val_batch_iter
        self.model_trainer = instantiate(
            model_train.init,
            model=self.dynamics_model,
            silent=self.silent,
            logger=self.logger,
        )
        # Replay/Rollout buffer
        self.normalized_buffer = True
        self.replay_buffer.poisson_weights = model_train.poisson_weights
        self.capacity = self.num_rollout_samples * retain_model_buffer_iter
        if 0 < self.training_rollout_horizon:
            split_section_dict = {OBS: self.dim_obs}
        else:
            split_section_dict = {
                OBS: self.dim_obs,
                ACTION: self.dim_act,
                REWARD: 1,
                NEXT_OBS: self.dim_obs,
                MASK: 1,
            }
        self.rollout_buffer = ReplayBuffer(
            self.capacity,
            self.device,
            split_section_dict=split_section_dict,
        )
        self.logger.debug(f"{self.rollout_buffer} is built")

        # Constants used in policy optimization
        self.done = torch.full(
            (self.batch_size,), False, device=self.device, dtype=torch.bool
        )
        if 0 < self.mve_horizon:
            discount_exps = torch.stack(
                [
                    torch.arange(-i, -i + self.training_rollout_horizon + 1)
                    for i in range(mve_horizon)
                ],
                dim=0,
            ).to(self.device)
            self.discount_mat = torch.triu(self.discount**discount_exps)
        # Adaptive rollout length
        # c.f. Frauenknecht, Bernd, et al. "Trust the Model Where It Trusts Itself
        # --Model-Based Actor-Critic with Uncertainty-Aware Rollout Adaption."
        # arXiv preprint arXiv:2405.19014 (2024).
        self.uara = uara
        self.uara_xi = uara_xi
        self.lr_kappa = lr_kappa
        self.kappa = None
        if uara:
            self.round_rollout = 1
            self.zeta_quantile = torch.tensor(zeta_quantile, device=self.device)

    def build_critics(self, critic_cfg):
        super().build_critics(critic_cfg)
        if self.weighted_critic:
            self.critic_loss_weight = self.critic_loss_weight.unsqueeze(0)

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        BaseAgent.observe(
            self, obs, action, reward, next_obs, terminated, truncated, info
        )
        # Update running statistics of observed samples
        target_obs = next_obs - obs
        output = np.concatenate([reward[:, None], target_obs], 1, dtype=np.float32)
        if self.normalize_output:
            self.output_normalizer.update_stats(output)

        if self.seed_iters <= self.total_iters:
            first_update = (self.seed_iters == self.total_iters) and self.step == 0
            # Pre-update
            if self.num_samples % self.steps_per_iter == 0 and self.step == 0:
                self.update_model(first_update)
                self.rollout_buffer.clear()
                self.init_optimizer()
                # Policy evaluation just after model learning
                if 0 < self.training_rollout_horizon and 0 < self.num_critic_iter:
                    if 0 < self.rollout_horizon:
                        self.distribution_rollout(
                            num_rollout_samples=self.num_critic_samples
                        )
                    critic_iter = trange(
                        self.num_critic_iter,
                        **self.kwargs_trange,
                        desc="[Critic]",
                    )
                    self.mb_update_critic(iterator=critic_iter)

                # --------------- Agent Training -----------------
                self.update(first_update)
                if self.is_epoch_done:
                    self.logger.append(
                        "policy_optimization",
                        {"iteration": self.total_iters},
                        self._info,
                    )

    def update_model(self, first_update=False):
        """Returns training/validation iterators for the data in the replay buffer."""

        if self.normalize_input:
            self.input_normalizer.to()
        if self.normalize_output:
            self.output_normalizer.to()

        train_data, val_data = self.replay_buffer.split_data()
        iterator_train = TransitionIterator(
            train_data,
            self.model_batch_size,
            shuffle_each_epoch=True,
            device=self.device,
            max_iter=self.max_batch_iter,
        )
        iterator_valid = TransitionIterator(
            val_data,
            self.val_batch_size,
            shuffle_each_epoch=False,
            device=self.device,
            max_iter=self.max_val_batch_iter,
        )

        factor = self.seed_iters if self.warm_start and first_update else 1
        keep_epochs = int(factor * self.keep_epochs)
        max_epochs = int(factor * self.max_epochs)
        self.model_trainer.train(
            iterator_train,
            dataset_val=iterator_valid,
            env_step=self.num_samples,
            keep_epochs=keep_epochs,
            num_max_epochs=max_epochs,
        )

    def distribution_rollout(self, num_rollout_samples=None, **rollout_kwargs):
        """Distribution rollout, which is detached."""
        num_rollout_samples = num_rollout_samples or self.num_rollout_samples
        num_sampled = 0
        while True:
            batch = self.replay_buffer.sample(self.batch_size)
            obs = batch.obs
            with torch.no_grad(), self.policy_evaluation_context(
                **rollout_kwargs
            ) as ctx_modules:
                # The case using replay buffer
                if self.normalize_po_input:
                    obs = self.input_normalizer.normalize(obs)

                action = self.sample_action(ctx_modules.actor, obs)[0]

                prediction_strategy = PS_TS1_F if self.uara else None
                rollouts = self.rollout(
                    ctx_modules,
                    obs,
                    action,
                    self.rollout_horizon,
                    prediction_strategy=prediction_strategy,
                )
                obss = rollouts[0]
                masks = rollouts[4]
                if 0 < self.training_rollout_horizon:
                    batch_obs = torch.cat(
                        [obs[mask.squeeze(-1)] for obs, mask in zip(obss, masks)]
                    )
                    ctx_modules.buffer.add_batch([batch_obs])
                    num_sampled += batch_obs.shape[0]
                else:
                    obss = torch.stack(obss)
                    actions = torch.stack(rollouts[1][:-1])
                    rewards = torch.stack(rollouts[3])
                    masks = torch.stack(masks[1:])

                    shift_idx = torch.cat(
                        [
                            torch.ones(
                                1, masks.size(1), dtype=torch.bool, device=self.device
                            ),
                            masks[:-1, :],
                        ],
                        dim=0,
                    )
                    done_time = (shift_idx == True) & (masks == False)
                    target_idx = masks.clone()
                    target_idx[done_time] = True

                    valid_xs = obss[:-1][target_idx]
                    valid_us = actions[target_idx]
                    valid_rs = rewards[target_idx].unsqueeze(1)
                    valid_xps = obss[1:][target_idx]
                    valid_nd = masks[target_idx].unsqueeze(1)
                    ctx_modules.buffer.add_batch(
                        [valid_xs, valid_us, valid_rs, valid_xps, valid_nd]
                    )

                    num_sampled += valid_nd.shape[0]

            if num_rollout_samples <= num_sampled:
                if self.uara:
                    self._info["uara_kappa"] = self.kappa
                    self._info["mean_rollout_length"] = (
                        torch.stack(rollouts[4]).sum(0).float().mean()
                    )
                break

    def sample_action(self, actor, obs, log=False, **kwargs):
        """Sample an action only used for model rollout."""
        dist = actor(obs, log=log)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdims=True)
        return action, log_prob, dist

    def mb_update_critic(self, iterator, log=False, **kwargs):
        for i in iterator:
            with self.policy_evaluation_context(detach=True) as ctx_modules:
                batch = ctx_modules.buffer.sample(self.batch_size)
                obs = batch.obs.clone()
                # When using replay buffer
                if self.rollout_horizon <= 0 and self.input_normalizer:
                    obs = self.input_normalizer.normalize(obs)

                # Model-based value expansion and calculate critic loss
                loss_critic = self.mb_policy_evaluation(ctx_modules, obs, log=log)[-1]
                ctx_modules.critic_optimizer.zero_grad()
                # Update the critics
                loss_critic.backward()

                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        ctx_modules.critic.parameters(), self.clip_grad_norm
                    )
                ctx_modules.critic_optimizer.step()
                soft_update_params(
                    ctx_modules.critic, ctx_modules.critic_target, self.critic_tau
                )

    def mb_policy_evaluation(
        self,
        ctx_modules,
        obs,
        done=None,
        log=False,
        prediction_strategy=None,
    ):
        """Model-based value expansion and critic loss.

        Returned values are used in actor updated.
        """
        ctx = torch.no_grad if ctx_modules.detach else nullcontext
        with ctx():
            action, log_pi, _ = self.sample_action(ctx_modules.actor, obs, log=log)
            (
                obss,
                actions,
                log_pis,
                rewards,
                masks,
                infos,
            ) = self.rollout(
                ctx_modules,
                obs,
                action,
                self.training_rollout_horizon,
                done=done,
                log_pi=log_pi.squeeze(-1),
                log=log,
                prediction_strategy=prediction_strategy,
            )
        batch_mask = torch.stack(masks).float()
        batch_sa = torch.cat([torch.cat(obss, -2), torch.cat(actions, -2)], -1)
        log_pis = torch.stack(log_pis)
        rewards = torch.stack(rewards)

        target_value, pred_values = self.eval_rollout(
            ctx_modules=ctx_modules,
            batch_sa=batch_sa[..., : self.mve_horizon * self.batch_size, :],
            log_pis=log_pis,
            batch_masks=batch_mask,
            rewards=rewards,
            last_sa=batch_sa[..., -self.batch_size :, :],
            discounts=self.discount_mat,
        )
        loss_critic = F.mse_loss(pred_values, target_value, reduction="none")
        if self.weighted_critic:
            loss_critic *= self.critic_loss_weight

        loss_critic = loss_critic.mean((0, 1)).sum()

        # Collect learning metrics
        self._info[KEY_CRITIC_LOSS] = loss_critic.detach() / self.num_critic_ensemble
        # if log:
        with torch.no_grad():
            q_mean = pred_values.mean()
            q_std = pred_values.std(-1).mean()

            self._info.update(**{Q_MEAN: q_mean, Q_STD: q_std}, **infos[-1])

        return (
            batch_sa,
            batch_mask,
            log_pis,
            rewards,
            target_value,
            pred_values,
            loss_critic,
        )

    def rollout(
        self,
        ctx_modules,
        obs,
        action,
        rollout_horizon: int,
        done=None,
        log_pi=None,
        log=False,
        prediction_strategy=None,
        **kwargs,
    ):
        done = self.done.clone() if done is None else done

        obss, actions, log_pis, rewards, masks = (
            [obs],
            [action],
            [log_pi],
            [],
            [~done],
        )

        infos = []
        for step in range(rollout_horizon):
            # Sample action
            if 0 < step:
                action, log_pi, pi = self.sample_action(
                    ctx_modules.actor, obs, **kwargs
                )

                log_pis.append(log_pi.squeeze(-1))
                obss.append(obs)
                actions.append(action)

            obs, rewards_i, done_i, info_s = ctx_modules.model_step(
                action,
                obs,
                prediction_strategy=prediction_strategy,
                log=log,
                step=step,
            )

            rewards.append(rewards_i.squeeze(-1))
            done |= done_i.squeeze(-1)
            masks.append(~done)
            infos.append(info_s)

        obss.append(obs)

        # Terminal condition
        action, log_pi, pi = self.sample_action(ctx_modules.actor, obs, **kwargs)
        actions.append(action)

        log_pis.append(log_pi.squeeze(-1))

        return obss, actions, log_pis, rewards, masks, infos

    def eval_rollout(
        self,
        ctx_modules,
        batch_sa,
        log_pis,
        batch_masks,
        rewards,
        last_sa,
        discounts,
    ):
        with torch.no_grad():
            if self.bounded_critic or self.scaled_critic:
                reward_pi = rewards - ctx_modules.alpha * log_pis[:-1, :]
                reward_lb, reward_ub = torch.quantile(reward_pi, self.q_th)
                (
                    self._q_lb,
                    self._q_ub,
                    self._q_center,
                    self._q_width,
                ) = self.update_critic_bound(
                    self._q_lb, self._q_ub, reward_lb, reward_ub
                )
                self._info.update(
                    **{
                        "reward_ub": reward_ub,
                        "reward_lb": reward_lb,
                        "q_ub": self._q_ub,
                        "q_lb": self._q_lb,
                    }
                )
            q_values = ctx_modules.pred_target_q(last_sa)
            target_rewards = torch.cat([rewards, q_values[None, ..., 0]])
            target_rewards[1:, ...].sub_(ctx_modules.alpha.detach() * log_pis[1:, ...])
            target_values = discounts.mm(target_rewards * batch_masks).unsqueeze(-1)

        pred_values = ctx_modules.pred_q(batch_sa.detach())
        pred_values = pred_values.view(
            self.mve_horizon, self.batch_size, self.num_critic_ensemble
        )
        deviation = discounts.shape[1] - discounts.shape[0]
        pred_values.mul_(batch_masks[:-deviation, :, None])
        return target_values, pred_values

    def update(self, first_update=False, **ctx_kwargs):
        """Main loop which alternates critic update and actor update."""
        factor = self.total_iters if self.warm_start and first_update else 1
        num_po_iter = int(self.num_policy_opt_per_step * factor * self.steps_per_iter)
        # tqdm
        disable = (
            self.kwargs_trange["disable"]
            and (not first_update and self.update_after_episode)
            or self.silent
        )
        t1 = time.time()
        iterator = trange(
            num_po_iter,
            **dict(self.kwargs_trange, **{"disable": disable}),
            desc=f"[Policy@Ep{self.total_iters: >4}]",
        )
        # Main loop
        with iterator as pbar:
            for opt_step in pbar:
                if (
                    0 < self.rollout_horizon
                    and 0 < self.rollout_freq
                    and opt_step % self.rollout_freq == 0
                ):
                    self.distribution_rollout(**ctx_kwargs)
                last_step = opt_step == num_po_iter - 1
                if self.training_rollout_horizon <= 0:
                    super().update(opt_step, log=last_step, buffer=self.rollout_buffer)
                else:
                    self._update(last_step, **ctx_kwargs)
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

    def _update(self, last_step=False, **ctx_kwargs):
        """One step of main logic."""
        with self.policy_evaluation_context(**ctx_kwargs) as ctx_modules:
            batch = ctx_modules.buffer.sample(self.batch_size)
            obs = batch.obs.clone()
            # The case using replay buffer and normalization
            if self.rollout_horizon <= 0 and self.normalize_po_input:
                obs = self.input_normalizer.normalize(obs)

            # Critics
            (
                batch_sa,
                batch_mask,
                log_pis,
                rewards,
                target_q,
                pred_q,
                loss_critic,
            ) = self.mb_policy_evaluation(ctx_modules, obs, log=last_step)
            # Update the critics
            ctx_modules.critic_optimizer.zero_grad()
            loss_critic.backward()

            if last_step:
                # Actor
                self._info.update(**ctx_modules.actor.info)
                # Critic
                self._info.update(
                    critic_grad_norm=calc_grad_norm(ctx_modules.critic),
                    **ctx_modules.critic.info,
                )

            ctx_modules.critic_optimizer.step()
            soft_update_params(
                ctx_modules.critic, ctx_modules.critic_target, self.critic_tau
            )

            # Update the actor
            self.mb_update_actor(
                ctx_modules,
                batch_sa,
                batch_mask,
                log_pis,
                rewards,
                log=last_step,
            )

        self.num_updated += 1
        self._info["num_updated"] = self.num_updated
        return self._info

    def _actor_loss(
        self, ctx_modules, log_pis, batch_sa, rewards, masks, discounts=None, log=False
    ):
        if discounts is None:
            discounts = self.discount_mat
        pred_qs = ctx_modules.pred_terminal_q(batch_sa[..., -self.batch_size :, :])
        mc_q_pred = torch.cat([rewards, pred_qs])
        mc_q_pred.sub_(ctx_modules.alpha.detach() * log_pis)
        mc_v_pred = discounts[:1, :].mm(mc_q_pred * masks).t()
        return -mc_v_pred.mean()

    def mb_update_actor(
        self,
        ctx_modules,
        batch_sa,
        batch_mask,
        log_pis,
        rewards,
        log=False,
    ):
        """One step of Stochastic Value Gradient (SVG)."""
        loss_actor = self._actor_loss(
            ctx_modules, log_pis, batch_sa, rewards, batch_mask, log=log
        )

        entropy = -log_pis[0].detach().mean()
        self._info.update(**{KEY_ACTOR_LOSS: loss_actor.detach(), "entropy": entropy})

        if self.learnable_alpha:
            alpha_loss = -ctx_modules.alpha * (self.target_entropy - entropy)
            loss_actor += alpha_loss

            self._info.update(
                **{
                    "alpha_loss": alpha_loss.detach(),
                    "alpha_value": ctx_modules.alpha.detach(),
                }
            )

        # Take a SGD step
        ctx_modules.actor_optimizer.zero_grad()
        loss_actor.backward()

        if log:
            self._info["actor_grad_norm"] = calc_grad_norm(ctx_modules.actor)

        ctx_modules.actor_optimizer.step()

    @contextmanager
    def policy_evaluation_context(self, detach=False, **kwargs):
        if 0 < self.rollout_horizon:
            buffer = self.rollout_buffer
        else:
            buffer = self.replay_buffer

        context_modules = ContextModules(
            self.actor,
            self.actor_optimizer,
            self.pred_q_value,
            self.pred_target_q_value,
            self.pred_terminal_q,
            self.critic,
            self.critic_target,
            self.critic_optimizer,
            self.alpha,
            self.model_step_context,
            buffer,
            detach,
        )
        try:
            yield context_modules
        finally:
            pass

    def model_step_context(
        self, action, model_state, step=0, prediction_strategy=None, **kwargs
    ):
        next_obs, reward, done, info = self.dynamics_model.sample(
            action, model_state, prediction_strategy=prediction_strategy, **kwargs
        )
        with torch.no_grad():
            if self.uara and prediction_strategy == PS_TS1_F:
                uncertainty = self.dynamics_model.state_entropy
                if step == 0:
                    base_uncertainty = torch.quantile(uncertainty, self.zeta_quantile)
                    if self.kappa is None:
                        assert self.round_rollout == 1
                        self.kappa = base_uncertainty
                    else:
                        # self.kappa = (
                        #     base_uncertainty + self.round_rollout * self.kappa
                        # ) / (self.round_rollout + 1)
                        # self.round_rollout += 1
                        self.kappa.lerp_(base_uncertainty, self.lr_kappa)
                done |= (self.kappa * self.uara_xi < uncertainty)[:, None]
        return next_obs, reward, done, info

    def save(self, dir_checkpoint, last=False):
        super().save(dir_checkpoint, last)
        self.dynamics_model.save(dir_checkpoint)
        if self.normalize_output:
            self.output_normalizer.save(dir_checkpoint)
        if self.uara:
            torch.save({"kappa": self.kappa}, f"{dir_checkpoint}/svg.pt")

    def load(self, dir_checkpoint):
        super().load(dir_checkpoint)
        self.dynamics_model.load(dir_checkpoint)
        if self.normalize_output:
            self.output_normalizer.load(dir_checkpoint)
        if self.uara:
            modules = torch.load(f"{dir_checkpoint}/svg.pt")
            self.kappa = modules["kappa"]

    @property
    def num_critic_samples(self):
        return max(
            int(self.batch_size * self.num_critic_iter / self.rollout_horizon),
            self.batch_size,
        )

    @property
    def description(self) -> str:
        # Print model's architecture
        num_total_params = 0
        print_str = ""
        for m in (self.dynamics_model, self.critic, self.actor):
            num_params = sum(np.prod(p.shape) for p in m.parameters())
            num_total_params += num_params
            print_str += f"{str(m)}\n"
            print_str += f"#Params for {m._get_name()}: {num_params}\n"

        print_str += f"#Params in total: {num_total_params}\n"
        return print_str


@dataclass
class ContextModules(metaclass=ABCMeta):
    actor: torch.nn.Module = None
    actor_optimizer: torch.optim.Optimizer = None
    pred_q: Callable = None
    pred_target_q: Callable = None
    pred_terminal_q: Callable = None
    critic: torch.nn.Module = None
    critic_target: torch.nn.Module = None
    critic_optimizer: torch.optim.Optimizer = None
    alpha: torch.Tensor = None
    model_step: Callable = None
    buffer: ReplayBuffer = None
    detach: bool = None

    def __new__(cls, *args, **kwargs):
        dataclass(cls)
        return super().__new__(cls)
