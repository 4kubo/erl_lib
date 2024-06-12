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


class SVGAgent(SACAgent):
    trange_kv: dict = {"Actor": KEY_ACTOR_LOSS, "Critic": KEY_CRITIC_LOSS}

    def __init__(
        self,
        learned_reward,
        term_fn,
        dynamics_model,
        model_train,
        update_after_episode,
        rollout_horizon,
        training_rollout_horizon,
        rollout_freq,
        retain_model_buffer_iter,
        mve_horizon,
        normalize_output: bool = True,
        **kwargs,
    ):
        if rollout_horizon <= 0:
            kwargs["buffer_device"] = "cuda"
            rollout_freq = 0
        if kwargs["num_critic_iter"] is None:
            kwargs["num_critic_iter"] = int(kwargs["steps_per_iter"] / 4)

        self.model_update_freq = kwargs["steps_per_iter"]
        super().__init__(**kwargs)

        self.normalize_output = normalize_output
        if self.normalize_output:
            self.output_normalizer = Normalizer(
                self.dim_obs + 1, self.device, "output_normalizer"
            )
        else:
            self.output_normalizer = None
        self.normalize_io = self.normalize_input and self.normalize_output

        # Building model
        dynamics_model.dim_input = self.dim_obs + self.dim_act
        dynamics_model.dim_output = self.dim_obs + 1
        dynamics_model.normalized_reward = self.normalized_reward

        # Now instantiate the model
        self.dynamics_model = instantiate(
            dynamics_model,
            term_fn,
            input_normalizer=self.input_normalizer,
            output_normalizer=self.output_normalizer,
            normalize_input=self.normalize_input,
            normalize_io=self.normalize_io,
        )
        self.logger.debug(f"{self.dynamics_model._get_name()} is built")
        # The termination function is assumed to be known
        self.rollout_horizon = rollout_horizon
        self.num_rollout_samples = self.batch_size * self.rollout_horizon
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
        self.replay_buffer.poisson_weights = model_train.poisson_weights
        self.capacity = self.num_rollout_samples * retain_model_buffer_iter
        split_section_dict = {OBS: self.dim_obs}
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
                if 0 < self.num_critic_iter:
                    if 0 < self.rollout_horizon:
                        self.distribution_rollout(
                            num_rollout_samples=self.num_critic_samples
                        )
                    critic_iter = trange(
                        self.num_critic_iter,
                        **self.kwargs_trange,
                        desc="[Critic]",
                    )
                    self.update_critic(iterator=critic_iter)

                # --------------- Agent Training -----------------
                log = self.is_epoch_done
                self.update(first_update, log=log)
                if log:
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
            dataset_eval=iterator_valid,
            env_step=self.num_samples,
            keep_epochs=keep_epochs,
            num_max_epochs=max_epochs,
        )

    def distribution_rollout(self, num_rollout_samples=None, **rollout_kwargs):
        num_rollout_samples = num_rollout_samples or self.num_rollout_samples
        num_sampled = 0
        while True:
            batch = self.replay_buffer.sample(self.batch_size)
            obs = batch.obs
            with torch.no_grad(), self.policy_evaluation_context(
                **rollout_kwargs
            ) as ctx_modules:
                # The case using replay buffer
                if self.input_normalizer:
                    obs = self.input_normalizer.normalize(obs)

                action = self.sample_action(ctx_modules.actor, obs)[0]

                rollouts = self.rollout(ctx_modules, obs, action, self.rollout_horizon)
                obss = rollouts[0]
                masks = rollouts[4]
                batch_obs = torch.cat(
                    [obs[mask.squeeze(-1)] for obs, mask in zip(obss, masks)]
                )
                ctx_modules.buffer.add_batch([batch_obs])
                num_sampled += batch_obs.shape[0]

            if num_rollout_samples <= num_sampled:
                break

    def sample_action(self, actor, obs, log=False, **kwargs):
        dist = actor(obs, log=log)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdims=True)
        return action, log_prob, dist

    def update_critic(self, iterator, log=False, **kwargs):
        for i in iterator:
            with self.policy_evaluation_context(detach=True) as ctx_modules:
                batch = ctx_modules.buffer.sample(self.batch_size)
                obs = batch.obs.clone()
                # The case using replay buffer
                if self.rollout_horizon <= 0 and self.input_normalizer:
                    obs = self.input_normalizer.normalize(obs)

                # MC approximation of state value by model rollout
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
        ctx = torch.no_grad if ctx_modules.detach else nullcontext
        with ctx():
            action, log_pi, _ = self.sample_action(ctx_modules.actor, obs, log=log)
            (
                obss,
                actions,
                log_pis,
                rewards,
                masks,
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
        if self.bounded_critic or self.scaled_critic:
            with torch.no_grad():
                reward_pi = rewards - ctx_modules.alpha * log_pis[:-1, :]
                reward_lb, reward_ub = torch.quantile(reward_pi, self.q_th)
                self.update_critic_bound(reward_lb, reward_ub)

        target_value, pred_values = self.eval_rollout(
            batch_sa=batch_sa[..., : self.mve_horizon * self.batch_size, :],
            log_pis=log_pis,
            batch_masks=batch_mask,
            rewards=rewards,
            alpha=ctx_modules.alpha,
            last_sa=batch_sa[..., -self.batch_size :, :],
            discounts=self.discount_mat,
            critic=ctx_modules.pred_q,
            critic_target=ctx_modules.pred_target_q,
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

            self._info.update(**{Q_MEAN: q_mean, Q_STD: q_std})

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
            )

            rewards.append(rewards_i.squeeze(-1))
            done |= done_i.squeeze(-1)
            masks.append(~done)
            self._info.update(**info_s)

        obss.append(obs)

        # Terminal condition
        action, log_pi, pi = self.sample_action(ctx_modules.actor, obs, **kwargs)
        actions.append(action)

        log_pis.append(log_pi.squeeze(-1))

        return obss, actions, log_pis, rewards, masks

    def eval_rollout(
        self,
        batch_sa,
        log_pis,
        batch_masks,
        rewards,
        alpha,
        last_sa,
        discounts,
        critic,
        critic_target,
    ):
        with torch.no_grad():
            q_values = critic_target(last_sa)
            target_rewards = torch.cat([rewards, q_values[None, ..., 0]])
            target_rewards[1:, ...].sub_(alpha.detach() * log_pis[1:, ...])
            target_values = discounts.mm(target_rewards * batch_masks).unsqueeze(-1)

        pred_values = critic(batch_sa.detach())
        pred_values = pred_values.view(
            self.mve_horizon, self.batch_size, self.num_critic_ensemble
        )
        deviation = discounts.shape[1] - discounts.shape[0]
        pred_values.mul_(batch_masks[:-deviation, :, None])
        return target_values, pred_values

    def update(self, first_update=False, log=False, **kwargs):
        # log = self.is_epoch_done

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
                    self.distribution_rollout()
                self._update(log and (opt_step == num_po_iter - 1), **kwargs)
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

    def _update(self, log=False, **ctx_kwargs):
        with self.policy_evaluation_context(**ctx_kwargs) as ctx_modules:
            batch = ctx_modules.buffer.sample(self.batch_size)
            obs = batch.obs.clone()
            # The case using replay buffer and normalization
            if self.rollout_horizon <= 0 and self.input_normalizer:
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
            ) = self.mb_policy_evaluation(ctx_modules, obs, log=log)
            # Update the critics
            ctx_modules.critic_optimizer.zero_grad()
            loss_critic.backward()

            if log:
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
            self.update_actor(
                ctx_modules,
                batch_sa,
                batch_mask,
                log_pis,
                rewards,
                log=log,
            )

        self.num_updated += 1
        self._info["num_updated"] = self.num_updated
        return self._info

    def _actor_loss(self, ctx_modules, log_pis, batch_sa, rewards, masks, log=False):
        pred_qs = ctx_modules.pred_terminal_q(batch_sa[..., -self.batch_size :, :])
        mc_q_pred = torch.cat([rewards, pred_qs])
        mc_q_pred.sub_(ctx_modules.alpha.detach() * log_pis)
        mc_v_pred = self.discount_mat[:1, :].mm(mc_q_pred * masks).t()
        return -mc_v_pred.mean()

    def update_actor(
        self,
        ctx_modules,
        batch_sa,
        batch_mask,
        log_pis,
        rewards,
        log=False,
    ):
        # Model-based value expansion
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

    def model_step_context(self, action, model_state, **kwargs):
        return self.dynamics_model.sample(action, model_state, **kwargs)

    def save(self, dir_checkpoint):
        super().save(dir_checkpoint)
        self.dynamics_model.save(dir_checkpoint)
        if self.normalize_output:
            self.output_normalizer.save(dir_checkpoint)

    def load(self, dir_checkpoint):
        super().load(dir_checkpoint)
        self.dynamics_model.load(dir_checkpoint)
        if self.normalize_output:
            self.output_normalizer.load(dir_checkpoint)

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
class ContextModules:
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
