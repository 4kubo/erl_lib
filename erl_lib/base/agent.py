import json
from typing import Optional
import time
from collections import defaultdict
import numpy as np
import torch.nn as nn


from erl_lib.base import (
    OBS,
    ACTION,
    REWARD,
    NEXT_OBS,
    MASK,
    TIMESTEPS_TOTAL,
    EPOCH,
    EVAL,
    GYM_KEY_OBS,
    GYM_KEY_FINAL,
    GYM_KEY_EPISODE,
    GYM_KEY_RETURN,
)
from erl_lib.util.logger import Logger
from erl_lib.util.misc import ReplayBuffer, Normalizer


class BaseAgent:
    actor: nn.Module
    critic: Optional[nn.Module]
    replay_buffer: ReplayBuffer = None
    input_normalizer: Normalizer = None

    def __init__(
        self,
        dim_obs: int,
        dim_act: int,
        logger: Logger,
        num_envs: int = 1,
        step_multiplier: int = 1,
        seed_iters: int = 10,
        steps_per_iter: int = 0,
        iters_per_epoch: int = 1,
        silent: bool = True,
        **kwargs,
    ):
        self.logger = logger
        if len(kwargs) != 0:
            self.logger.warning(f"Unused kwargs: {kwargs}")

        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.num_envs = num_envs
        self.step_multiplier = step_multiplier
        self.seed_iters = seed_iters
        self.steps_per_iter = steps_per_iter
        self.iters_per_epoch = iters_per_epoch
        self.silent = silent

        # Counters
        #  Iteration
        self._last_iter = 0
        self.iter = 0
        #  Step
        self.step = 0
        #  Episode
        self._num_last_episodes = 0
        self.num_episodes_this_iter = 0

        # Misc
        self.kwargs_trange = {
            "smoothing": 0.1,
            "mininterval": 1,
            "maxinterval": 10,
            "disable": self.silent,
        }
        self._info = {}
        self._obs_stack = []

    def reset(self):
        self._last_iter += self.iter
        self._num_last_episodes += self.num_episodes_this_iter
        self.iter = 0
        self.step = 0
        self.num_episodes_this_iter = 0

        self.time_last_epoch = time.time()
        self._obs_stack = []

    def act(self, obs, sample=True, stack_obs=False):
        num_envs = obs.shape[0]
        if stack_obs:
            self._obs_stack.append(obs)

        if self.total_iters < self.seed_iters:
            return np.random.uniform(-1.0, 1.0, size=(num_envs, self.dim_act))
        else:
            return self._act(obs, sample)

    def _act(self, obs, sample):
        num_envs = obs.shape[0]
        return np.random.uniform(-1.0, 1.0, size=(num_envs, self.dim_act))

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        """Update observation and update running statistics."""
        # Step and iteration
        self.step += 1
        if self.steps_per_iter <= self.step:
            self.iter += 1
            self.step = 0
        # Count up the number of ended episodes
        auto_rest_envs = np.logical_or(truncated, terminated)
        if auto_rest_envs.any():
            # Retrieve last_obs caused by auto-reset
            next_obs = next_obs.copy()
            self.num_episodes_this_iter += int(auto_rest_envs.sum())
            next_obs[auto_rest_envs] = np.vstack(
                [
                    info[i][GYM_KEY_OBS]
                    for i, is_final in enumerate(auto_rest_envs)
                    if is_final
                ]
            )
        # Pool the sample to buffer if necessary
        if self.replay_buffer is not None:
            # Store observations to buffer
            mask = np.logical_not(terminated)
            data = [obs, action, reward[:, None], next_obs, mask[:, None]]
            self.replay_buffer.add_batch(data)
        if self.input_normalizer is not None:
            self.input_normalizer.update_stats(obs)

        return next_obs

    def evaluate(
        self,
        envs,
        num_episodes_needed,
        mean_training_return,
        stack_obs=False,
        callbacks=None,
    ):
        time_start = time.time()
        accum_dones = np.array([False] * num_episodes_needed)

        returns = {}
        infos = defaultdict(lambda: np.zeros(num_episodes_needed, dtype=np.float32))

        seed = np.random.random_integers(int(1e5))
        envs.envs[0].step_id = 0
        envs.envs[0].episode_id = self.time_steps_total
        obs, info = envs.reset(seed=int(seed))
        # Starts envs' steps
        while True:
            action = self.act(obs, sample=False, stack_obs=stack_obs)
            obs, reward, terminated, truncated, info = envs.step(action)

            dones = np.logical_or(terminated, truncated)

            # Aggregate episode infos if necessary
            for env_id, (done, acc_done, info_i) in enumerate(
                zip(dones, accum_dones, info)
            ):
                undone = ~acc_done
                # The first time to be done during the evaluation for `env_id`-th env
                if undone and done:
                    assert GYM_KEY_FINAL in info_i
                    assert not (env_id in returns)

                    info_episode = info_i[GYM_KEY_EPISODE]
                    returns[env_id] = float(info_episode[GYM_KEY_RETURN])
                    infos["elapsed_step"][env_id] = int(info_episode.get("l", 0))

                    info_final = info_i[GYM_KEY_FINAL]
                    for key, value in info_final.items():
                        # Accumulated metrics
                        if key.startswith("reward"):
                            infos[key][env_id] += value
                        elif not key.startswith("_"):
                            infos[key][env_id] = value
                elif undone:
                    for key, value in info_i.items():
                        # Accumulated metrics
                        if key.startswith("reward"):
                            infos[key][env_id] += value

            accum_dones |= dones

            # The case the necessary number of episodes were collected
            if accum_dones.all():
                assert len(returns) == num_episodes_needed
                break

        if callbacks:
            for callback in callbacks:
                callback(self, infos, returns)

        # Return
        for i, return_i in returns.items():
            value_dict = {"return": return_i}
            for key, value in infos.items():
                if key.startswith("reward"):
                    value_dict[key] = value[i]
            self.logger.append(
                EVAL,
                index={TIMESTEPS_TOTAL: self.time_steps_total, "id": i},
                value_dict=value_dict,
            )
        # Other infos
        infos = {
            key: np.mean(values)
            for key, values in infos.items()
            if not key.startswith("reward")
        }
        infos.update(
            **{
                "time_eval": time.time() - time_start,
                "mean_last_training_return": mean_training_return,
                "num_samples_total": float(self.num_samples),
                TIMESTEPS_TOTAL: float(self.time_steps_total),
                "num_episodes_total": float(self.num_episodes),
                "num_episodes_this_epoch": float(self.num_episodes_this_iter),
            }
        )
        self.logger.append(
            EPOCH, index={TIMESTEPS_TOTAL: self.time_steps_total}, value_dict=infos
        )
        # Finalize logging at this epoch
        result = self.logger.end_epoch(self.time_steps_total)

        if self.silent:
            print_str = (
                f"Steps: {self.time_steps_total: >10,d}, "
                f"Episodes: {self.num_episodes: >6,d}, "
                f"EvalReturn: {result[EVAL + '/return']:7.2f}"
            )
            self.logger.info(print_str)
        else:
            print_str = ""
            for key, value in result.items():
                print_str += f"{key}: {value:.3f}\n"
            self.logger.info(print_str)

        return result

    def save(self, dir_checkpoint):
        self.logger.info(f"Checkpointing to {dir_checkpoint}")
        state = {
            "last_iter": self._last_iter,
            "last_episodes": self._num_last_episodes,
        }
        name_checkpoint = f"{dir_checkpoint}/base.yaml"
        with open(name_checkpoint, "w") as f:
            json.dump(state, f)

    def load(self, dir_checkpoint):
        self.logger.info(f"Loading from check point {dir_checkpoint}")
        name_checkpoint = f"{dir_checkpoint}/base.yaml"
        with open(name_checkpoint, "rb") as f:
            state = json.load(f)
        self._last_iter = state["last_iter"]
        self._num_last_episodes = state["last_episodes"]

    @property
    def is_epoch_done(self):
        if self.total_iters < self.seed_iters:
            return False
        else:
            # is_done = self.iters_per_epoch <= self.iter
            is_done = 0 < self.iter
            is_done &= self.total_iters == self.seed_iters or (
                self.total_iters % self.iters_per_epoch == 0
            )
            if is_done:
                time_done = time.time()
                time_epoch = time_done - self.time_last_epoch
                self.time_last_epoch = time_done
                self.logger.update({f"{EPOCH}/time_epoch": time_epoch})
            return is_done

    @property
    def iters_this_epoch(self):
        if self.total_iters < self.seed_iters:
            return max(self.iters_per_epoch, self.seed_iters)
        else:
            return self.iters_per_epoch

    @property
    def total_iters(self):
        return self._last_iter + self.iter

    @property
    def num_steps(self):
        return self.step + self.total_iters * self.steps_per_iter

    @property
    def num_samples(self):
        return self.num_steps * self.num_envs

    @property
    def time_steps_total(self):
        """The total number of steps before applying action repeats."""
        return self.num_samples * self.step_multiplier

    @property
    def num_episodes(self):
        return self.num_episodes_this_iter + self._num_last_episodes

    # def __str__(self):
    #     return (
    #         f"@{self.total_iters}\n"
    #         f"Total samples observed: {self.num_samples}\n"
    #         f"Total episodes observed: {self.num_episodes}"
    #     )

    @property
    def description(self) -> str:
        return ""
