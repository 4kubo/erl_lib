"""Hopper Environment with full observation."""
import numpy as np
from gymnasium.spaces import Box

from erl_lib.envs.gym.locomotion import LocomotionEnv
from erl_lib.envs.terminataion import HealthCheck


class MBHopperEnv(LocomotionEnv):
    """Hopper Environment."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array",],
        "render_fps": 125,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        reset_noise_scale=5e-3,
        **kwargs
    ):
        dim_obs = 11
        z_dim = 0

        dim_pos = 1
        healthy_state_range = (-100.0, 100.0)
        healthy_z_range = (0.7, float("inf"))
        healthy_angle_range = (-0.2, 0.2)
        healthy_reward = 1.0
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float64
        )

        health_checker = HealthCheck(
            z_dim=z_dim,
            healthy_angle_range=healthy_angle_range,
            healthy_z_range=healthy_z_range,
            healthy_state_range=healthy_state_range,
        )
        LocomotionEnv.__init__(
            self,
            "hopper.xml",
            4,
            observation_space,
            dim_pos=dim_pos,
            dim_action=(3,),
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            healthy_reward=healthy_reward,
            health_checker=health_checker,
            **kwargs,
        )
