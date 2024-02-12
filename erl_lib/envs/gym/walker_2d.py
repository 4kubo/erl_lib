"""Walker2d Environment with full observation."""
import numpy as np
from gymnasium.spaces import Box

from erl_lib.envs.gym.locomotion import LocomotionEnv
from erl_lib.envs.terminataion import HealthCheck


class MBWalker2dEnv(LocomotionEnv):
    """Walker2d Environment."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array",],
        "render_fps": 125,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        reset_noise_scale=5e-3,
        **kwargs,
    ):
        dim_obs = 17
        z_dim = 0

        dim_pos = 1
        healthy_z_range = (0.8, 2.0)
        healthy_angle_range = (-1.0, 1.0)
        healthy_reward = 1.0

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float64
        )
        health_checker = HealthCheck(
            z_dim=z_dim,
            healthy_angle_range=healthy_angle_range,
            healthy_z_range=healthy_z_range,
        )
        LocomotionEnv.__init__(
            self,
            "walker2d.xml",
            4,
            observation_space,
            dim_pos=dim_pos,
            dim_action=(6,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=forward_reward_weight,
            healthy_reward=healthy_reward,
            health_checker=health_checker,
            reset_noise_scale=reset_noise_scale,
            **kwargs,
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
