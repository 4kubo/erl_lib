import abc
from typing import Dict, Optional, Tuple
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

from erl_lib.envs.terminataion import LargeStateTermination
from erl_lib.envs.reward import StateActionReward


class BaseEnv(abc.ABC):
  observation_space: Box
  action_space: Box
  metadata: {}
  render_mode: None

  reward_model: Optional[StateActionReward] = None
  termination_model: Optional[LargeStateTermination] = None

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    check_env(self)

  @abc.abstractmethod
  def step(
      self, action: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    raise NotImplementedError

  @abc.abstractmethod
  def reset(self, seed: int = None, options: dict = None):
    raise NotImplementedError

  @abc.abstractmethod
  def render(self):
    raise NotImplementedError

  @abc.abstractproperty
  def max_episode_steps(self):
    raise NotImplementedError

  @property
  def unwrapped(self):
    return self
