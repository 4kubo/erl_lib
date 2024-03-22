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
from erl_lib.agent.model_based.model_env import ModelEnv
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
        **kwargs,
    ):
        if rollout_horizon <= 0:
            kwargs["buffer_device"] = "cuda"
            rollout_freq = 0
        if kwargs["num_critic_iter"] is None:
            kwargs["num_critic_iter"] = int(kwargs["steps_per_iter"] / 4)

        self.model_update_freq = kwargs["steps_per_iter"]
        super().__init__(**kwargs)

        if self.normalize_io:
            self.output_normalizer = Normalizer(
                self.dim_obs + 1, self.device, "output_normalizer"
            )
        else:
            self.output_normalizer = None

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
        )
        self.logger.debug(f"{self.dynamics_model._get_name()} is built")
        # The termination function is assumed to be known
        self.model_env = ModelEnv(self.dynamics_model, term_fn)
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
        split_section_dict = {OBS: self.dim_obs, ACTION: self.dim_act, REWARD: 1}
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

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        BaseAgent.observe(
            self, obs, action, reward, next_obs, terminated, truncated, info
        )
        # Update running statistics of observed samples
        target_obs = next_obs - obs
        output = np.concatenate([reward[:, None], target_obs], 1, dtype=np.float32)
        if self.normalize_io:
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
                    self.distribution_rollout(
                        num_rollout_samples=self.num_critic_samples
                    )
                    critic_iter = trange(
                        self.num_critic_iter,
                        **self.kwargs_trange,
                        desc="[Critic]",
                    )
                    self.update_critic(iterator=critic_iter)
            if (0 < self.rollout_freq) and (self.num_samples % self.rollout_freq == 0):
                self.distribution_rollout()

            # --------------- Agent Training -----------------
            if (
                self.update_after_episode and self.step == 0
            ) or not self.update_after_episode:
                self.update(first_update)
                if self.is_epoch_done:
                    self.logger.append(
                        "policy_optimization",
                        {"iteration": self.total_iters},
                        self._info,
                    )

    # def pre_update(self):
    #     # --------------- Model Training -----------------
    #     if self.iter % self.model_update_freq == 0:
    #         self.update_model()
    #     self.rollout_buffer.clear()
    #     # --------------- Agent Training -----------------
    #     self.init_optimizer()
    #
    #     # Policy evaluation just after model learning
    #     if 0 < self.num_critic_iter:
    #         num_samples = max(
    #             int(self.batch_size * self.num_critic_iter / self.rollout_horizon),
    #             self.batch_size,
    #         )
    #         self.distribution_rollout(num_rollout_samples=num_samples)
    #
    #         critic_iter = trange(
    #             self.num_critic_iter,
    #             **self.kwargs_trange,
    #             desc="[Critic]",
    #         )
    #         self.update_critic(iterator=critic_iter)

    def update_model(self, first_update=False):
        """Returns training/validation iterators for the data in the replay buffer."""
        if self.normalize_io:
            self.input_normalizer.to()
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
            if self.normalize_io:
                obs = self.input_normalizer.normalize(obs)

            with torch.no_grad(), self.policy_evaluation_context(
                **rollout_kwargs
            ) as ctx_modules:
                accum_dones = self.done.clone()

                for i in range(self.rollout_horizon):
                    action, _, _ = self.sample_action(ctx_modules.actor, obs)

                    next_obs, reward, done, info = ctx_modules.model_step(
                        action,
                        obs,
                        sample=True,
                    )

                    accum_dones = accum_dones | done
                    continuing = ~accum_dones.squeeze(-1)
                    nnz = continuing.count_nonzero()
                    num_sampled += nnz
                    # Filter out done samples if needed
                    if nnz != self.batch_size:
                        batch_obs = obs[continuing]
                        batch_action = action[continuing]
                        batch_reward = reward[continuing]
                        batch = [batch_obs, batch_action, batch_reward]
                        # if self.training_rollout_horizon <= 0:
                        #     batch += [next_obs[continuing], ~accum_dones]
                    else:
                        batch_obs = obs
                        batch_action = action
                        batch_reward = reward
                        batch = [batch_obs, batch_action, batch_reward]
                        # if self.training_rollout_horizon <= 0:
                        #     batch += [next_obs, ~done]

                    ctx_modules.buffer.add_batch(batch)

                    if not continuing.any() or num_rollout_samples <= num_sampled:
                        break

                    obs = next_obs

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
                obs = batch.obs
                ctx_modules.critic.train()
                ctx_modules.critic_target.eval()
                # The case using replay buffer
                if self.rollout_horizon <= 0 and self.normalize_io:
                    obs = self.input_normalizer.normalize(obs)

                # MC approximation of state value by model rollout
                loss_critic = self.mb_policy_evaluation(
                    ctx_modules,
                    obs,
                    detach_target=False,
                    log=log,
                )[-2]
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
        detach_target,
        action=None,
        log=False,
    ):
        if action is None:
            action, log_pi, _ = self.sample_action(ctx_modules.actor, obs, log=log)
        else:
            log_pi = None

        (
            states,
            actions,
            _,
            masks,
            rewards,
            last_state,
            last_action,
            last_log_pi,
            info,
        ) = self.rollout(
            ctx_modules.alpha,
            obs,
            action,
            ctx_modules.model_step,
            actor=ctx_modules.actor,
            mve_horizon=self.mve_horizon,
            detach_target=detach_target,
            log=log,
        )
        sas = torch.cat([states, actions], 1)
        last_sa = torch.cat([last_state, last_action], 1)
        target_value, pred_values = self.eval_rollout(
            sas=sas,
            masks=masks,
            rewards=rewards,
            alpha=ctx_modules.alpha,
            last_sa=last_sa,
            log_pi=last_log_pi,
            discounts=self.discount_mat,
            critic=ctx_modules.critic,
            critic_target=ctx_modules.critic_target,
            info=info,
        )
        loss_critic = F.mse_loss(pred_values, target_value[..., None], reduction="none")

        loss_critic = loss_critic.sum(2).mean()

        # Collect learning metrics
        self._info[KEY_CRITIC_LOSS] = loss_critic.detach() / self.num_critic_ensemble
        if log:
            with torch.no_grad():
                q_mean = pred_values.mean()
                q_std = pred_values.std(-1).mean()

                self._info.update(**{"q_value-mean": q_mean, "q_value-std": q_std})

        return (
            log_pi,
            rewards,
            masks,
            last_sa,
            last_log_pi,
            pred_values,
            loss_critic,
            info,
        )

    def rollout(
        self,
        alpha,
        obs,
        action,
        model_step,
        mve_horizon: int,
        # actions=None,
        actor=None,
        detach_target=False,
        regularized=True,
        scales=None,
        log=False,
        **kwargs,
    ):
        target_ctx = torch.no_grad if detach_target else nullcontext

        done = (
            self.done.clone()
            if obs.shape[0] == self.batch_size
            else torch.full(
                (obs.shape[0], 1), False, device=self.device, dtype=torch.bool
            )
        )

        obs_input = obs

        obss, rewards, dones = (
            [obs_input.detach()],
            [],
            [done.clone()],
        )
        if actor is None:
            actions = action
        else:
            actions = [action.detach()]
        info = {}

        for step in range(self.training_rollout_horizon):
            # Sample action
            if actor is None:
                action = actions[step]
                log_pi = None
            elif 0 < step:
                with target_ctx():
                    action, log_pi, pi = self.sample_action(actor, obs_input, **kwargs)

                if step < mve_horizon:
                    obss.append(obs_input.detach())
                    actions.append(action.detach())
                    if scales is not None:
                        scales.append(pi.scale.detach())
            else:
                log_pi = None

            # Predict observation
            with target_ctx():
                obs, rewards_i, done_i, info_s = self.predict_obs(
                    model_step,
                    obs,
                    action,
                    done,
                    alpha,
                    log_pi=log_pi,
                    log=log,
                    regularized=regularized,
                )
                obs_input = obs

            # Increment process
            rewards.append(rewards_i[:, 0])
            done |= done_i
            dones.append(done.clone())
            self._info.update(**info_s)

        # Convert a stack of dict into a dict of stacked model states
        obss = torch.vstack(obss)
        masks = (~torch.hstack(dones).t()).float()

        # Terminal condition
        with target_ctx():
            if actor is None:
                action = actions[-1]
            else:
                actions = torch.vstack(actions)
                action, log_pi, pi = self.sample_action(actor, obs_input, **kwargs)

            if scales is not None:
                scales.append(pi.scale.detach())
                scales = torch.vstack(scales)

        return (
            obss,
            actions,
            scales,
            masks,
            rewards,
            obs_input,
            action,
            log_pi,
            info,
        )

    def eval_rollout(
        self,
        sas,
        masks,
        rewards,
        alpha,
        last_sa,
        log_pi,
        discounts,
        critic,
        critic_target,
        info: dict,
    ):
        with torch.no_grad():
            q_values = critic_target(last_sa)
            q_values = self._reduce(q_values, "min")
            q_values = self._regularize_reward(q_values, log_pi, None, alpha)
            if self.critic_scaled:
                reward_lb, reward_ub = torch.quantile(torch.stack(rewards), self.q_th)
                self.update_critic_bound(reward_lb, reward_ub)
                q_values = self._q_ub - torch.relu(self._q_ub - q_values)
                q_values = self._q_lb + torch.relu(q_values - self._q_lb)
            target_rewards = torch.stack(rewards + [q_values[:, 0]], 0)  # [H+1,B]
            target_values = discounts.mm(target_rewards * masks)

        pred_values = critic(sas)
        pred_values = pred_values.view(
            -1, self.batch_size, self.num_critic_ensemble
        ).contiguous()
        deviation = discounts.shape[1] - discounts.shape[0]
        pred_values.mul_(masks[:-deviation, :, None])
        return target_values, pred_values

    def update(self, first_update=False, **kwargs):
        log = self.is_epoch_done
        if 0 < self.training_rollout_horizon:
            if first_update or self.update_after_episode:
                factor = self.total_iters if self.warm_start and first_update else 1
                num_po_iter = int(
                    self.num_policy_opt_per_step * factor * self.steps_per_iter
                )
                disable = self.kwargs_trange["disable"] and (not first_update and self.update_after_episode) or self.silent
                t1 = time.time()
                iterator = trange(
                    num_po_iter,
                    **dict(self.kwargs_trange, **{"disable": disable}),
                    desc=f"[Policy@Ep{self.total_iters: >4}]",
                )
                # Main loop
                with iterator as pbar:
                    for opt_step in pbar:
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
            else:
                self._update(log)
        else:
            super().update(self.num_samples, log=log, buffer=self.rollout_buffer)

    def _update(self, log=False):
        with self.policy_evaluation_context() as ctx_modules:
            batch = ctx_modules.buffer.sample(self.batch_size)
            obs = batch.obs
            ctx_modules.critic.train()
            ctx_modules.critic_target.eval()
            # The case using replay buffer
            if self.rollout_horizon <= 0 and self.normalize_io:
                obs = self.input_normalizer.normalize(obs)

            # MC approximation of state value by model rollout
            (
                log_pi,
                rewards,
                masks,
                last_sa,
                last_log_pi,
                pred_values,
                loss_critic,
                info,
            ) = self.mb_policy_evaluation(
                ctx_modules,
                obs,
                detach_target=False,
                log=log,
            )
            # Update the critics
            ctx_modules.critic_optimizer.zero_grad()
            loss_critic.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    ctx_modules.critic.parameters(), self.clip_grad_norm
                )
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
                last_sa,
                last_log_pi,
                rewards,
                masks,
                log_pi,
                info,
                None,
                log=log,
            )

        self.num_updated += 1
        self._info["num_updated"] = self.num_updated
        return self._info

    def _actor_loss(
        self, ctx_modules, last_sa, last_log_pi, rewards, masks, log_pi, info, log=False
    ):
        q_preds = ctx_modules.critic_svg(last_sa)
        q_pred = self._reduce(q_preds, self.actor_reduction)
        q_pred = self._regularize_reward(q_pred, last_log_pi, None, ctx_modules.alpha)
        mc_q_pred = torch.stack(rewards + [q_pred[:, 0]], 0)
        mc_q_target = (
            self.discount_mat[:1, : len(rewards) + 1].mm(mc_q_pred * masks).t()
        )
        mc_v_target = mc_q_target - ctx_modules.alpha.detach() * log_pi
        return mc_v_target

    def actor_loss(
        self,
        ctx_modules,
        last_sa,
        last_log_pi,
        rewards,
        masks,
        log_pi,
        info,
        loss_scale=None,
        log=False,
    ):
        # Model-based value expansion
        mc_v_target = self._actor_loss(
            ctx_modules, last_sa, last_log_pi, rewards, masks, log_pi, info, log
        )
        # Stochastic Value Gradient
        loss_actor = -mc_v_target.mean()

        entropy = -log_pi.detach().mean()
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
        if loss_scale:
            loss_actor.register_hook(lambda grad: grad / loss_scale)
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

    def predict_obs(
        self,
        model_step,
        obs,
        action,
        done,
        alpha,
        regularized=True,
        log_pi=None,
        log=False,
    ):
        next_obs, rewards, done_i, info = model_step(
            action,
            obs,
            sample=True,
            log=log,
        )
        rewards = self._regularize_reward(rewards, log_pi, done, alpha, regularized)

        return next_obs, rewards, done_i, info

    def _regularize_reward(self, reward, log_pi, done, alpha, regularized=True):
        if regularized and (log_pi is not None):
            reward.add_(-alpha.detach() * log_pi)

        return reward

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
