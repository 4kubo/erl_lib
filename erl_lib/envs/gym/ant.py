"""Ant Environment with full observation."""
import numpy as np
from gymnasium.spaces import Box

from erl_lib.base import REW_STATE, REW_CTRL
from erl_lib.envs.gym.locomotion import LocomotionEnv
from erl_lib.envs.terminataion import HealthCheck


class MBAntEnv(LocomotionEnv):
    """Ant Environment."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array",],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        truncate_obs=True,
        **kwargs,
    ):
        self._contact_cost_weight = contact_cost_weight
        self._contact_force_range = contact_force_range
        self._truncate_obs = truncate_obs

        dim_action = (8,)
        dim_pos = 2
        healthy_z_range = (0.2, 1.0)
        healthy_reward = 1.0

        dim_obs = 27 if truncate_obs else 84
        z_dim = 0
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float64
        )
        health_checker = HealthCheck(
            z_dim=z_dim,
            healthy_z_range=healthy_z_range,
            healthy_state_range=(-np.inf, np.inf),
        )
        LocomotionEnv.__init__(
            self,
            "ant.xml",
            5,
            observation_space,
            dim_pos=dim_pos,
            dim_action=dim_action,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            healthy_reward=healthy_reward,
            health_checker=health_checker,
            **kwargs,
        )

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        obs_list = (position[self.dim_pos :], velocity)
        if not self._truncate_obs:
            obs_list += (self.contact_forces.flat.copy(),)

        obs = np.concatenate(obs_list).ravel()
        return obs

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_position(self, q, reset=False):
        return self.get_body_com("torso")[: self.dim_pos].copy()

    def _reward(self, action):
        lin_position = self._get_position(self.data.qpos.copy())
        forward_vel = (lin_position - self.prev_pos) / self.dt
        reward_state = self._forward_reward_weight * forward_vel[:1]
        ctrl_cost = self._ctrl_cost_weight * (action ** 2).sum(-1, keepdims=True)
        contact_cost = self.contact_cost
        reward = reward_state - ctrl_cost - contact_cost

        info = {REW_STATE: reward_state, REW_CTRL: -ctrl_cost, "reward_contact": -contact_cost}
        return reward, info

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost
