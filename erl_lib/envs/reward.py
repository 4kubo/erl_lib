"""State Action Reward Functions."""
import torch
import numpy as np

from erl_lib.base import REW_STATE, REW_CTRL


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
        self._info = {}
        reward, info = self.forward(state, action, next_state)
        return reward, info

    def action_sparse_reward(self, action):
        """Get action sparse reward."""
        rewards_action_sparse = tolerance(action, lower=-0.1, upper=0.1, margin=0.1) - 1
        self.action_dense_reward(action)
        return rewards_action_sparse.prod(-1, keepdims=True)

    def action_dense_reward(self, action):
        """Get action non-sparse rewards."""
        reward_action = -(action ** 2).sum(-1, keepdims=True)
        self._info[f"{REW_CTRL}_dense"] = reward_action
        return reward_action

    def action_reward(self, action):
        """Get reward that corresponds to action."""
        if self.sparse:
            return self.action_sparse_reward(action)
        else:
            return self.action_dense_reward(action / self.action_scale)

    def state_reward(self, state, next_state):
        """Get reward that corresponds to the states."""
        raise NotImplementedError


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
