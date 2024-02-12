"""Half-Cheetah Environment with full observation."""
import numpy as np
from gymnasium.spaces import Box

from .locomotion import LocomotionEnv


class MBHalfCheetahEnv(LocomotionEnv):
    """Half-Cheetah Environment."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array",],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        reset_noise_scale=0.1,
        ctrl_cost_weight=0.1,
        **kwargs,
    ):
        dim_obs = 17
        dim_pos = 1
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(dim_obs,), dtype=np.float64
        )
        LocomotionEnv.__init__(
            self,
            "half_cheetah.xml",
            5,
            observation_space,
            dim_pos=dim_pos,
            dim_action=(6,),
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            **kwargs,
        )

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
        qpos[: self.dim_pos] = np.zeros(self.dim_pos).copy()
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
