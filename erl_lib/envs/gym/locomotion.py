"""Base class for mujoco-based simulation tasks."""
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle

from erl_lib.base.env import BaseEnv
from erl_lib.envs.terminataion import LargeStateTermination
from erl_lib.base import REW_STATE, REW_CTRL


class LocomotionEnv(BaseEnv, MujocoEnv, EzPickle):
    """Base Locomotion environment. Is a hack to avoid repeated code."""

    def __init__(
        self,
        xml_file,
        frame_skip,
        observation_space,
        dim_pos,
        dim_action,
        ctrl_cost_weight,
        forward_reward_weight=1.0,
        healthy_reward=0.0,
        sparse=False,
        reset_noise_scale=1e-2,
        health_checker=None,
        # rendering
        render_mode="rgb_array",
        width=128,
        height=128,
        camera_id=None,
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            dim_pos,
            dim_action,
            ctrl_cost_weight,
            forward_reward_weight,
            healthy_reward,
            reset_noise_scale,
            health_checker,
            **kwargs,
        )

        self.dim_pos = dim_pos
        self.prev_pos = np.zeros(dim_pos)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._forward_reward_weight = forward_reward_weight
        self._health_checker = health_checker
        self._healthy_reward = healthy_reward
        self._reset_noise_scale = reset_noise_scale

        self.reward_range = ()
        self._termination_model = LargeStateTermination(health_checker)
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_id=camera_id,
            **kwargs,
        )

    def step(self, action):
        """See gym.Env.step()."""
        obs = self._get_obs()
        self.prev_pos = self._get_position(self.data.qpos.copy())
        self.do_simulation(action, self.frame_skip)

        next_obs = self._get_obs()

        reward, info = self._reward(action)
        if self._health_checker:
            reward += self._healthy_reward * self._health_checker.check(next_obs)

        done = self._termination_model(obs, action, next_obs)

        # info = self._reward_model.info
        info.update(x_position=self.prev_pos[0])
        if self.dim_pos == 2:
            info.update(y_poisition=self.prev_pos[1])

        if self.render_mode == "human":
            self.render()

        return next_obs, reward.item(), done, False, info

    def reset(self, seed: int = None, options: dict = None):
        return MujocoEnv.reset(self, seed=seed, options=options)

    def render(self, **kwargs):
        return MujocoEnv.render(self, **kwargs)

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        obs_list = (position[self.dim_pos :], velocity)

        return np.concatenate(obs_list).ravel()

    def _get_position(
        self, q,
    ):
        return q[: self.dim_pos]

    def reset_model(self):
        """Reset model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()

        return observation

    def _reward(self, action):
        lin_position = self._get_position(self.data.qpos.copy())
        forward_vel = (lin_position - self.prev_pos) / self.dt
        reward_state = self._forward_reward_weight * forward_vel[:1]
        reward_ctrl = -self._ctrl_cost_weight * (action ** 2).sum(-1, keepdims=True)
        reward = reward_state + reward_ctrl

        info = {REW_STATE: reward_state, REW_CTRL: reward_ctrl}
        return reward, info

    @property
    def max_episode_steps(self):
        return self.spec.max_episode_steps

    @property
    def termination_model(self):
        return self._termination_model
