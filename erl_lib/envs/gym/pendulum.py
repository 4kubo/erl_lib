import numpy as np
import torch

from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from erl_lib.base import REW_STATE, REW_CTRL
from erl_lib.base.env import BaseEnv


class StateActionReward:
    r"""Base class for state-action reward functions."""

    def __init__(self, ctrl_cost_weight=0.1, sparse=False, action_scale=1.0):
        self.ctrl_cost_weight = ctrl_cost_weight
        self.action_scale = action_scale
        self.sparse = sparse
        self._info = {}

    def forward(self, state, action, next_state):
        """Get reward distribution for state, action, next_state."""
        reward_ctrl = self.action_reward(action)
        reward_ctrl *= self.ctrl_cost_weight

        reward_state = self.state_reward(state, next_state)
        reward = reward_state + reward_ctrl

        if isinstance(reward_state, torch.Tensor):
            reward_state = reward_state.detach().cpu().numpy()
            reward_ctrl = reward_ctrl.detach().cpu().numpy()
        self._info.update(**{REW_STATE: reward_state, REW_CTRL: reward_ctrl})
        return reward, self._info

    def __call__(self, state, action, next_state):
        return self.forward(state, action, next_state)

    def action_sparse_reward(self, action):
        """Get action sparse reward."""
        rewards_action_sparse = tolerance(action, lower=-0.1, upper=0.1, margin=0.1)
        self.action_dense_reward(action)
        return rewards_action_sparse.prod(-1, keepdims=True)

    def action_dense_reward(self, action):
        """Get action non-sparse rewards."""
        reward_action = -(action**2).sum(-1, keepdims=True)
        self._info[f"{REW_CTRL}_dense"] = reward_action
        return reward_action

    def action_reward(self, action):
        """Get reward that corresponds to action."""
        if self.sparse:
            return self.action_sparse_reward(action)
        else:
            return self.action_dense_reward(action / self.action_scale)


class PendulumReward(StateActionReward):
    dim_action = (1,)

    def __init__(
        self,
        ctrl_cost_weight=0.004,
        sparse=False,
        same_scale_state_reward=True,
    ):
        super().__init__(
            ctrl_cost_weight=ctrl_cost_weight,
            sparse=sparse,
        )
        self.same_scale_state_reward = same_scale_state_reward
        if same_scale_state_reward:
            self.compatible_scale = np.pi**2 + 0.1 * 64
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
            if self.same_scale_state_reward:
                state_reward = (
                    -(1 - angle_tolerance * velocity_tolerance) * self.compatible_scale
                )
            else:
                state_reward = angle_tolerance * velocity_tolerance

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
        reward_state = -(theta**2) - 0.1 * theta_dot**2
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
            self.inertia = self.m * self.l**2
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
        self,
        reset_noise_scale=1e-2,
        **kwargs,
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


def gaussian(x, value_at_1):
    """Apply an un-normalized Gaussian function with zero mean and scaled variance.

    Parameters
    ----------
    x : The points at which to evaluate_agent the Gaussian
    value_at_1: The reward magnitude when x=1. Needs to be 0 < value_at_1 < 1.
    """
    if type(x) is torch.Tensor:
        scale = torch.sqrt(-2 * torch.log(value_at_1))
        return torch.exp(-0.5 * (x * scale) ** 2)
    else:
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)


def tolerance(x, lower, upper, margin=None):
    """Apply a tolerance function with optional smoothing.

    Can be used to design (smoothed) box-constrained reward functions.

    A tolerance function is returns 1 if x is in [lower, upper].
    If it is outside, it decays exponentially according to a margin.

    Parameters
    ----------
    x : the value at which to evaluate_agent the sparse reward.
    lower: The lower bound of the tolerance function.
    upper: The upper bound of the tolerance function.
    margin: A margin over which to smooth out the box-reward.
        If a positive margin is provided, uses a `gaussian` smoothing on the boundary.
    """
    if margin is None or margin == 0.0:
        in_bounds = (lower <= x) & (x <= upper)
        return in_bounds
    else:
        assert margin > 0
        diff = 0.5 * (upper - lower)
        mid = lower + diff

        if type(x) is torch.Tensor:
            # Distance is positive only outside the bounds
            distance = torch.abs(x - mid) - diff
            return gaussian(torch.relu(distance * (1 / margin)), value_at_1=0.1)
        else:
            distance = np.abs(x - mid) - diff
            return gaussian(np.maximum(0, (distance / margin)), value_at_1=0.1)
