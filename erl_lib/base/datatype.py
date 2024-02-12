from typing import Any, List, Optional, Sequence, Sized, Tuple, Type, Union
from dataclasses import dataclass

import numpy as np
import torch

# from tensordict import TensorDict


TensorType = Union[torch.Tensor, np.ndarray]
Transition = Tuple[
  TensorType, TensorType, TensorType, TensorType, TensorType, TensorType
]


# @dataclass
# class TransitionBatch:
#   """Represents a batch of transitions"""
#   #
#   # obs: Optional[TensorType]
#   # action: Optional[TensorType]
#   # reward: Optional[TensorType]
#   # next_obs: Optional[TensorType]
#   # done: Optional[TensorType]
#   # weight: Optional[TensorType]
#
#   def __init__(self, data, split_section_dict):
#     # if data.device != torch.device("cuda"):
#     #   data = data.to(torch.device("cuda"))
#     # self._data = data
#     self.split_section_dict = split_section_dict
#     # Because output of torch.split is a view of data, it should be efficient
#     split_sections = list(split_section_dict.values())
#     split_data = torch.split(data, split_sections, dim=-1)
#
#     self._data = TensorDict({key: value for key, value in zip(split_section_dict.keys(), split_data)}, [data.shape[0]], device="cuda")
#
#   def astuple(self):
#     return tuple(getattr(self, key) for key in self.split_section_dict.keys())
#
#   def __len__(self):
#     return self._data.shape[0]
#
#   def __getitem__(self, item):
#     return self._data[item]
#
#   def __setitem__(self, key, value):
#     self._data[key] = value
#
#   @property
#   def data(self):
#     return self._data
#
#   def size(self):
#     return len(self)

@dataclass
class TransitionBatch:
  """Represents a batch of transitions"""

  obs: Optional[TensorType]
  action: Optional[TensorType]
  reward: Optional[TensorType]
  next_obs: Optional[TensorType]
  done: Optional[TensorType]
  weight: Optional[TensorType]

  def __init__(self, data, split_section_dict):
    if data.device != torch.device("cuda"):
      data = data.to(torch.device("cuda"))
    self._data = data
    self.split_section_dict = split_section_dict
    # Because output of torch.split is a view of data, it should be efficient
    split_sections = list(split_section_dict.values())
    split_data = torch.split(self._data, split_sections, dim=-1)
    for key, value in zip(split_section_dict.keys(), split_data):
      setattr(self, key, value)

  def astuple(self):
    return tuple(getattr(self, key) for key in self.split_section_dict.keys())


  def __len__(self):
    return self._data.shape[0]


  def __getitem__(self, item):
    return TransitionBatch(self._data[item], self.split_section_dict)


  @property
  def data(self):
    return self._data


  def size(self):
    return len(self)


@dataclass
class ColumnIndex:
  index: tuple
  values: list

  def __init__(self, index, value):
    self.index = index
    self.values = [value]
    self.n_values = len(value)

  def __add__(self, other):
    assert self.index == other.index
    assert self.n_values == other.n_values
    self.values.extend(other.values)
    return self


@dataclass
class Metrics:
  index: ColumnIndex
  values: ColumnIndex

  def append(self, index, values):
    # new_index =
    self.index += self._make_index(index)
    self.values += self._make_index(values)

  @staticmethod
  def _make_index(index_kwargs: dict):
    index, value = list(zip(*index_kwargs.items()))
    return ColumnIndex(index, value)

  def __init__(self, index, values):
    self.index = self._make_index(index)
    self.values = self._make_index(values)
