import numpy as np
import torch

from erl_lib.envs.terminataion import LargeStateTermination
# from erl_lib.model_based.modules.model import Model


class ModelEnv:
    """Wraps a model to be used as gym-like environment."""

    def __init__(
        self,
        model,
        termination_fn=None,
        reward_fn=None,
    ):
        self.model = model
        self.termination_fn = termination_fn or LargeStateTermination()
        self.reward_fn = reward_fn
        self.device = model.device

    def step(
        self,
        actions,
        obs: torch.Tensor,
        log: bool = False,
        **kwargs,
    ):
        """Steps the model environment with the given batch of actions."""
        assert len(actions.shape) == 2  # batch, action_dim
        # if actions is tensor, code assumes it's already on self.device
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        (
            next_obs,
            pred_rewards,
            pred_terminals,
            info,
        ) = self.model.sample(actions, obs, log=log, **kwargs)
        if self.reward_fn is None:
            rewards, reward_info = pred_rewards, {}
        else:
            rewards, reward_info = self.reward_fn(obs, actions, next_obs)

        if self.termination_fn:
            dones = self.termination_fn(obs, actions, next_obs)
        else:
            dones = False

        if log:
            reward_info = {
                f"reward/{k}": v.mean() if hasattr(v, "ndim") and 0 < v.ndim else v
                for k, v in reward_info.items()
            }
            info.update(**reward_info)
        return next_obs, rewards, dones, info
