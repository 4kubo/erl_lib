"""Humanoid Environment with full observation."""
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.humanoid_v4 import mass_center, DEFAULT_CAMERA_CONFIG
from gymnasium.envs.mujoco.humanoidstandup_v4 import (
    DEFAULT_CAMERA_CONFIG as STAND_CAMERA_CONFIG,
)

from erl_lib.base import REW_STATE, REW_CTRL
from erl_lib.envs.gym.locomotion import LocomotionEnv
from erl_lib.envs.terminataion import HealthCheck


class MBHumanoidBase(LocomotionEnv):
    """Humanoid Environment."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array",],
        "render_fps": 67,
    }

    def __init__(
        self,
        xml_file,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        truncate_obs=True,
        **kwargs,
    ):
        self._forward_reward_weight = forward_reward_weight
        self._contact_cost_weight = contact_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._truncate_obs = truncate_obs

        dim_obs = 45 if truncate_obs else 376
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float64
        )
        dim_action = (17,)
        dim_pos = 2

        LocomotionEnv.__init__(
            self,
            xml_file,
            5,
            observation_space,
            dim_pos=dim_pos,
            dim_action=dim_action,
            ctrl_cost_weight=0.0,
            forward_reward_weight=1.0,
            **kwargs,
        )

    def _get_position(self, *_):
        return mass_center(self.model, self.data)

    @property
    def control_cost(self):
        return self._ctrl_cost_weight * np.square(self.data.ctrl).sum()

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        obs = (position[self.dim_pos :], velocity)
        if not self._truncate_obs:
            obs += (
                self.data.cinert.flat.copy(),
                self.data.cvel.flat.copy(),
                self.data.qfrc_actuator.flat.copy(),
                self.data.cfrc_ext.flat.copy(),
            )
        return np.concatenate(obs).ravel()


class MBHumanoidEnv(MBHumanoidBase):
    def __init__(self, **kwargs):
        healthy_z_range = (1.0, 2.0)
        healthy_state_range = (-1000, 1000)
        healthy_reward = 5.0
        z_dim = 0

        health_checker = HealthCheck(
            z_dim=z_dim,
            healthy_z_range=healthy_z_range,
            healthy_state_range=healthy_state_range,
        )

        super().__init__(
            xml_file="humanoid.xml",
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            health_checker=health_checker,
            healthy_reward=healthy_reward,
            **kwargs,
        )

    def _reward(self, *_):
        lin_position = self._get_position(None)
        forward_vel = (lin_position - self.prev_pos) / self.dt
        reward_ctrl = -self.control_cost
        reward_state = self._forward_reward_weight * forward_vel[:1]
        reward = reward_state + reward_ctrl

        info = {REW_STATE: reward_state, REW_CTRL: reward_ctrl}
        return reward, info


class MBHumanoidStandupEnv(MBHumanoidBase):
    def __init__(self, **kwargs):
        super().__init__(
            xml_file="humanoidstandup.xml",
            default_camera_config=STAND_CAMERA_CONFIG,
            healthy_reward=1.0,
            reset_noise_scale=0.01,
            **kwargs,
        )

    def _reward(self, *_):
        pos_after = self.data.qpos[2:3]
        reward_state = pos_after / self.model.opt.timestep
        quad_ctrl_cost = self.control_cost
        quad_impact_cost = min(
            self._contact_cost_weight * np.square(self.data.cfrc_ext).sum(), 10
        )
        reward_ctrl = -quad_impact_cost - quad_ctrl_cost
        reward = reward_state + reward_ctrl

        info = {REW_STATE: reward_state, REW_CTRL: reward_ctrl}
        return reward, info
