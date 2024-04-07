from contextlib import contextmanager, nullcontext
from typing import NamedTuple, Callable
import time

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.distributions import Exponential
from tqdm import trange

from erl_lib.base.agent import BaseAgent
from erl_lib.agent.sac import SACAgent
from erl_lib.base import (
    OBS,
    ACTION,
    REWARD,
    NEXT_OBS,
    MASK,
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
        gae_lambda=0.9,
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
        # split_section_dict = {OBS: self.dim_obs, ACTION: self.dim_act, REWARD: 1}
        split_section_dict = {OBS: self.dim_obs}
        self.rollout_buffer = ReplayBuffer(
            self.capacity,
            self.device,
            split_section_dict=split_section_dict,
        )
        self.logger.debug(f"{self.rollout_buffer} is built")

        # Constants used in policy optimization
        size = (self.batch_size, 1)
        self.done = torch.full(size, False, device=self.device, dtype=torch.bool)
        if 0 < self.mve_horizon:
            discount_exps = torch.stack(
                [
                    torch.arange(-i, -i + self.training_rollout_horizon + 1)
                    for i in range(mve_horizon)
                ],
                dim=0,
            ).to(self.device)
            self.discount_mat = torch.triu(self.discount**discount_exps)

        self.gae_lambda = gae_lambda
        # self.gae_lambda = gae_lambda = 0.99
        norm_factor = 1.0 / sum([gae_lambda**h for h in range(self.mve_horizon)])
        discount_reward = [
            np.sum([gae_lambda ** (k - h) for k in np.arange(h, self.mve_horizon)])
            * (self.discount * gae_lambda) ** h
            for h in np.arange(self.mve_horizon)
        ]
        discount_value = [
            (self.discount * gae_lambda) ** h for h in np.arange(self.mve_horizon)
        ]
        self.discount_reward = (
            torch.as_tensor(discount_reward, device=self.device, dtype=torch.float32)
            * norm_factor
        )[:, None]
        self.discount_value = (
            torch.as_tensor(discount_value, device=self.device, dtype=torch.float32)
            * norm_factor
            * self.discount
        )[:, None]

        # self.gae_lambda = gae_lambda = 1 - gae_lambda
        # lambda_discount = [
        #     (self.discount * gae_lambda) ** h for h in range(self.mve_horizon)
        # ]
        # self.discount_reward = (
        #     torch.as_tensor(lambda_discount, device=self.device, dtype=torch.float32)
        # )[:, None]
        #
        # self.discount_value = (
        #     torch.as_tensor(
        #         [ld * self.discount * gae_lambda if i == (self.mve_horizon - 1) else ld * self.discount * (1- gae_lambda) for i, ld in enumerate(lambda_discount)],
        #         device=self.device,
        #         dtype=torch.float32,
        #     )
        # )[:, None]

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
            with torch.no_grad(), self.policy_evaluation_context() as ctx_modules:
                # The case using replay buffer
                if self.input_normalizer:
                    obs = self.input_normalizer.normalize(obs)

                action = self.sample_action(ctx_modules.actor, obs)[0]

                rollouts = self.rollout(ctx_modules, obs, action, self.rollout_horizon)
                obss = rollouts[0]
                masks = rollouts[4]
                batch_obs = torch.cat(
                    [obs[mask[0, :]] for obs, mask in zip(obss, masks)]
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
            with self.policy_evaluation_context() as ctx_modules:
                batch = ctx_modules.buffer.sample(self.batch_size)
                obs = batch.obs.clone()
                ctx_modules.critic.train()
                ctx_modules.critic_target.eval()
                # The case using replay buffer
                if self.rollout_horizon <= 0 and self.input_normalizer:
                    obs = self.input_normalizer.normalize(obs)

                # MC approximation of state value by model rollout
                loss_critic = self.mb_policy_evaluation(ctx_modules, obs, log=log)[-2]
                # Update the critics
                ctx_modules.critic_optimizer.zero_grad()
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
        log=False,
    ):
        action, log_pi, _ = self.sample_action(ctx_modules.actor, obs, log=log)

        (
            obss,
            actions,
            log_pis,
            rewards,
            masks,
            info,
        ) = self.rollout(
            ctx_modules,
            obs,
            action,
            self.training_rollout_horizon,
            mve_horizon=self.mve_horizon,
            log_pi=log_pi,
            log=log,
        )
        batch_mask = torch.vstack(masks).float()
        batch_sa = torch.hstack([torch.vstack(obss[:-1]), torch.vstack(actions[:-1])])
        last_sa = torch.cat([obss[-1], actions[-1]], 1)
        log_pis = torch.hstack(log_pis).t()
        rewards = torch.stack(rewards)

        target_value, pred_values = self.eval_rollout(
            batch_sa=batch_sa,
            log_pis=log_pis,
            batch_masks=batch_mask,
            rewards=rewards,
            alpha=ctx_modules.alpha,
            last_sa=last_sa,
            discounts=self.discount_mat,
            critic=ctx_modules.critic,
            critic_target=ctx_modules.critic_target,
        )
        loss_critic = F.mse_loss(pred_values, target_value, reduction="none")

        loss_critic = loss_critic.sum(2).mean()

        # Collect learning metrics
        self._info[KEY_CRITIC_LOSS] = loss_critic.detach() / self.num_critic_ensemble
        if log:
            with torch.no_grad():
                q_mean = pred_values.mean()
                q_std = pred_values.std(-1).mean()

                self._info.update(**{"q_value-mean": q_mean, "q_value-std": q_std})

        return (
            batch_sa,
            batch_mask,
            log_pis,
            rewards,
            masks,
            last_sa,
            pred_values,
            loss_critic,
            info,
        )

    def rollout(
        self,
        ctx_modules,
        obs,
        action,
        rollout_horizon: int,
        mve_horizon=None,
        log_pi=None,
        log=False,
        prediction_strategy=None,
        **kwargs,
    ):
        done = (
            self.done.clone()
            if obs.shape[0] == self.batch_size
            else torch.full(
                (obs.shape[0], 1), False, device=self.device, dtype=torch.bool
            )
        )

        obss, actions, log_pis, rewards, masks = (
            [obs],
            [action],
            [log_pi],
            [],
            [~done.clone().t()],
        )
        mve_horizon = mve_horizon or rollout_horizon
        info = {}

        for step in range(rollout_horizon):
            # Sample action
            if 0 < step:
                action, log_pi, pi = self.sample_action(
                    ctx_modules.actor, obs, **kwargs
                )

                log_pis.append(log_pi)
                if step < mve_horizon:
                    obss.append(obs)
                    actions.append(action)

            obs, rewards_i, done_i, info_s = ctx_modules.model_step(
                action, obs, prediction_strategy=prediction_strategy
            )

            rewards.append(rewards_i[:, 0])
            done |= done_i
            masks.append(~done.clone().t())
            self._info.update(**info_s)

        # Convert a stack of dict into a dict of stacked model states
        obss.append(obs)

        # Terminal condition
        action, log_pi, pi = self.sample_action(ctx_modules.actor, obs, **kwargs)
        actions.append(action)

        log_pis.append(log_pi)

        return obss, actions, log_pis, rewards, masks, info

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
            q_values = critic_target(last_sa).t()
            q_values = self._reduce(q_values, "min")
            if self.critic_scaled:
                reward_lb, reward_ub = torch.quantile(rewards, self.q_th)
                self.update_critic_bound(reward_lb, reward_ub)
                q_values = self._q_ub - torch.relu(self._q_ub - q_values)
                q_values = self._q_lb + torch.relu(q_values - self._q_lb)
            target_rewards = torch.cat([rewards, q_values[None, :, 0]])
            target_rewards[1:, ...].sub_(alpha.detach() * log_pis[1:, ...])
            target_values = discounts.mm(target_rewards * batch_masks)

        pred_values = critic(batch_sa.detach()).t()
        pred_values = pred_values.view(
            -1, self.batch_size, self.num_critic_ensemble
        ).contiguous()
        deviation = discounts.shape[1] - discounts.shape[0]
        pred_values.mul_(batch_masks[:-deviation, :, None])
        return target_values[..., None], pred_values

    def update(self, first_update=False, **kwargs):
        log = self.is_epoch_done

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
                self._update(log and (opt_step == num_po_iter - 1))
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

    def _update(self, log=False):
        with self.policy_evaluation_context() as ctx_modules:
            batch = ctx_modules.buffer.sample(self.batch_size)
            obs = batch.obs.clone()
            ctx_modules.critic.train()
            ctx_modules.critic_target.eval()
            # The case using replay buffer
            if self.rollout_horizon <= 0 and self.input_normalizer:
                obs = self.input_normalizer.normalize(obs)

            # MC approximation of state value by model rollout
            (
                batch_sa,
                batch_mask,
                log_pis,
                rewards,
                masks,
                last_sa,
                pred_q,
                loss_critic,
                info,
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
            self.actor_loss(
                ctx_modules,
                batch_sa,
                batch_mask,
                log_pis,
                rewards,
                last_sa,
                info,
                log=log,
            )

        self.num_updated += 1
        self._info["num_updated"] = self.num_updated
        return self._info

    def _actor_loss(self, ctx_modules, log_pis, last_sa, rewards, masks, log=False):
        q_preds = ctx_modules.critic_svg(last_sa).t()
        q_pred = self._reduce(q_preds, self.actor_reduction)
        mc_q_pred = torch.cat([rewards, q_pred[None, :, 0]])
        mc_q_pred.sub_(ctx_modules.alpha.detach() * log_pis)
        mc_v_pred = self.discount_mat[:1, : len(rewards) + 1].mm(mc_q_pred * masks).t()
        return mc_v_pred

    def _actor_gae(
        self,
        ctx_modules,
        batch_sa,
        batch_mask,
        log_pis,
        rewards,
        last_sa,
    ):
        alpha = ctx_modules.alpha.detach()
        rewards -= alpha * log_pis[:-1, :]
        next_sas = torch.vstack([batch_sa[self.batch_size :, ...], last_sa])
        next_q = ctx_modules.critic_svg(next_sas)
        next_q = self._reduce(next_q, self.actor_reduction)
        next_q = next_q.view(self.mve_horizon, self.batch_size)
        next_v = next_q - alpha * log_pis[1:, :]

        lambda_return = (self.discount_reward * rewards) * batch_mask[:-1, :]
        lambda_return += (self.discount_value * next_v) * batch_mask[1:, :]
        lambda_return = lambda_return.sum(0)

        # alpha = ctx_modules.alpha.detach()
        # rewards -= alpha * log_pis[:-1, :]
        #
        # next_sas = torch.vstack([batch_sa[self.batch_size :, ...], last_sa])
        # next_q = ctx_modules.critic_svg(next_sas)
        # next_q = self._reduce(next_q, self.actor_reduction)
        # next_q = next_q.view(self.mve_horizon, self.batch_size)
        #
        # next_v = next_q - alpha * log_pis[1:, :]
        # lambda_return = (self.discount_reward * rewards) * batch_mask[:-1, :]
        # lambda_return += (self.discount_value * next_v) * batch_mask[1:, :]
        # lambda_return = torch.sum(lambda_return, 0)
        return lambda_return

    def actor_loss(
        self,
        ctx_modules,
        batch_sa,
        batch_mask,
        log_pis,
        rewards,
        last_sa,
        info,
        log=False,
    ):
        # Model-based value expansion
        if self.gae_lambda:
            mc_v_target = self._actor_gae(
                ctx_modules,
                batch_sa,
                batch_mask,
                log_pis,
                rewards,
                last_sa,
                log,
            )
        else:
            mc_v_target = self._actor_loss(
                ctx_modules, log_pis, last_sa, rewards, batch_mask, log
            )
        # Stochastic Value Gradient
        loss_actor = -mc_v_target.mean()

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

        return info

    @contextmanager
    def policy_evaluation_context(self, **kwargs):
        if 0 < self.rollout_horizon:
            buffer = self.rollout_buffer
        else:
            buffer = self.replay_buffer

        context_modules = ContextModules(
            self.actor,
            self.actor_optimizer,
            self.critic,
            self.critic_target,
            self.critic,
            self.critic_optimizer,
            self.alpha,
            self.model_step_context,
            buffer,
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
        self.output_normalizer.save(dir_checkpoint)

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


class ContextModules(NamedTuple):
    actor: torch.nn.Module
    actor_optimizer: torch.optim.Optimizer
    critic: torch.nn.Module
    critic_target: torch.nn.Module
    critic_svg: torch.nn.Module
    critic_optimizer: torch.optim.Optimizer
    alpha: torch.Tensor
    model_step: Callable
    buffer: ReplayBuffer
