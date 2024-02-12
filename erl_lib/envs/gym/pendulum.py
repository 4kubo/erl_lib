import numpy as np


from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from erl_lib.base import REW_STATE, REW_CTRL
from erl_lib.base.env import BaseEnv
from erl_lib.envs.gym.reward.state_action_reward import StateActionReward, tolerance


class PendulumReward(StateActionReward):
    dim_action = (1,)

    def __init__(
        self, ctrl_cost_weight=0.004, sparse=False, same_scale_state_reward=True,
    ):
        super().__init__(
            ctrl_cost_weight=ctrl_cost_weight, sparse=sparse,
        )
        if same_scale_state_reward:
            self.compatible_scale = np.pi ** 2 + 0.1 * 64
        else:
            self.compatible_scale = 1.0
        if sparse:
            self.state_reward = self.state_sparse_reward
        else:
            self.state_reward = self.state_dense_reward

    def state_sparse_reward(self, state, next_state):
        """Get sparse reward that corresponds to the states."""
        if isinstance(state, np.ndarray):
            angle_tolerance = self._info["reward_angle"]
            velocity_tolerance = self._info["reward_velocity"]
            state_reward = (
                -(1 - angle_tolerance * velocity_tolerance) * self.compatible_scale
            )
            return np.asarray(state_reward)[None]
        else:
            raise NotImplementedError

    def state_dense_reward(self, state, next_state):
        """Get reward that corresponds to the states."""
        if isinstance(state, np.ndarray):
            return self._info[f"{REW_STATE}_dense"]
        else:
            raise NotImplementedError

    def forward(self, state, action, next_state):
        theta, theta_dot = state
        # Dense reward
        theta = angle_normalize(theta)
        reward_state = -(theta ** 2) - 0.1 * theta_dot ** 2
        reward_state = np.asanyarray(reward_state)[None]
        self._info[f"{REW_STATE}_dense"] = reward_state
        # Sparse reward
        angle_tolerance = tolerance(np.cos(theta), lower=0.95, upper=1, margin=0.1)
        velocity_tolerance = tolerance(theta_dot, lower=-0.5, upper=0.5, margin=0.5)
        self._info["reward_angle"] = angle_tolerance
        self._info["reward_velocity"] = velocity_tolerance
        # Just call value according to the configuration
        reward, info = super().forward(state, action, next_state)
        if self.sparse:
            self._info["reward_dense"] = (
                info[f"{REW_STATE}_dense"] + info[f"{REW_CTRL}_dense"]
            )
        return reward, self._info


class Pendulum(BaseEnv, PendulumEnv):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        ctrl_cost_weight=0.004,
        height=500,
        width=500,
        max_torque=2.0,
        sparse=False,
        ube_model=False,
        same_scale_state_reward=True,
        **kwargs,
    ):
        PendulumEnv.__init__(self, **kwargs)
        if ube_model:
            self.friction = 0.005
            self.m = 0.3
            self.l = 0.5

            self.dt = 1 / 80
            self.inertia = self.m * self.l ** 2
            self.g = 9.81
            self.metadata["render_fps"] = 80

        self.ube_model = ube_model
        self.max_torque = max_torque

        self.reward_model = PendulumReward(
            ctrl_cost_weight,
            sparse=sparse,
            same_scale_state_reward=same_scale_state_reward,
        )

        # overrides
        self.screen_dim = height or width
        self.action_space = Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

    def ube_step(self, action):
        """Taken from the following:
        https://github.com/boschresearch/ube-mbrl/blob/main/ube_mbrl/envs/sparse_pendulum.py
        """
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        state = self.state
        th, th_dot = state
        # Step forward the system dynamics
        th_ddot = (
            (self.g / self.l) * np.sin(th)
            + u * (1 / self.inertia)
            - (self.friction / self.inertia) * th_dot
        )
        th = th + self.dt * th_dot
        th_dot = th_dot + self.dt * th_ddot
        self.state = next_state = np.array([th, th_dot])

        # done = self.termination_model(state, action, next_state)
        reward, info_reward = self.reward_model(state, action, next_state)
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()
        return obs, reward, False, False, info_reward

    def gym_step(self, action):
        state = self.state
        obs, _, done, term, info = PendulumEnv.step(self, action)
        next_state = self.state
        reward, info_reward = self.reward_model(state, action, next_state)
        info.update(**info_reward)
        return obs, reward, done, term, info

    def step(self, action):
        if self.ube_model:
            return self.ube_step(action)
        else:
            return self.gym_step(action)

    def reset(self, seed: int = None, options: dict = None):
        return PendulumEnv.reset(self, seed=seed, options=options)

    def render(self):
        return PendulumEnv.render(self)

    @property
    def max_episode_steps(self):
        return self.spec.max_episode_steps


class PendulumV1(Pendulum):
    def __init__(
        self, reset_noise_scale=1e-2, **kwargs,
    ):
        super().__init__(**kwargs)
        self.reset_noise_scale = reset_noise_scale

    def reset(self, seed: int = None, **_):
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.state = (
            np.asarray([np.pi, 0.0])
            + self.np_random.normal(size=2) * self.reset_noise_scale
        )
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
