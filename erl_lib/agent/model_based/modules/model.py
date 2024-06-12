import abc
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn as nn


class Model(nn.Module):
    """Base class for all dynamics models.

    All classes derived from `Model` must implement the following methods:

        - ``loss``: computes a loss tensor that can be used for backpropagation.
        - ``eval_score``: computes a non-reduced tensor that gives an evaluation score
          for the model on the input data (e.g., squared error per element).
        - ``save``: saves the model to a given path.
        - ``load``: loads the model from a given path.

    Subclasses may also want to overrides :meth:`sample` and :meth:`reset`.
    """

    def __init__(self, dim_input: int, dim_output: int, device, learned_reward=True):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.device = torch.device(device)
        self.learned_reward = learned_reward

    @abc.abstractmethod
    def eval_score(
        self,
        batch,
        log: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Computes an evaluation score for the model over the given input/target."""

    def reset(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Initializes the model to start a new simulated trajectory."""
        raise NotImplementedError(
            "ModelEnv requires that model has a reset() method defined."
        )

    @abc.abstractmethod
    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        log: Optional[bool] = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
        Optional[Dict[str, float]],
    ]:
        """Samples a simulated transition from the dynamics model."""
        raise NotImplementedError(
            "ModelEnv requires that model has a sample() method defined."
        )

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        torch.save(self.state_dict(), pathlib.Path(save_dir) / "model.pt")

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        self.load_state_dict(torch.load(pathlib.Path(load_dir) / "model.pt"))
