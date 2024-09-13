import numpy as np
import random
import json
import pathlib
import torch
from torch import nn

from erl_lib.base.datatype import TransitionBatch


def calc_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm += torch.sum(p.grad.data**2)
    return torch.sqrt(total_norm)


def soft_update_params(net, target_net, tau):
    with torch.no_grad():
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def soft_bound(x, ub=None, lb=None, beta=1):
    if ub is not None:
        x = ub - nn.functional.softplus(ub - x, beta=beta)
    if lb is not None:
        x = x.sub(lb)
        x = lb + nn.functional.softplus(x, beta=beta)
    return x


class SymLog(nn.Module):
    def forward(self, input):
        return torch.sign(input) * torch.log1p(torch.abs(input))


class ReplayBuffer:
    TRAIN = 1.0
    VALID = -1.0

    def __init__(
        self,
        capacity: int,
        device,
        device_to: torch.device = "cuda",
        max_batch_size: int = None,
        valid_ratio: float = 0.2,
        split_validation: bool = False,
        split_section_dict=None,
        num_sample_weights=0,
        poisson_weights=False,
    ):
        self.capacity = int(capacity)
        self.max_batch_size = max_batch_size
        self.num_sample_weights = num_sample_weights
        self.poisson_weights = poisson_weights
        self.dtype = torch.float32
        self.device = device
        self.device_to = device_to
        self.valid_ratio = valid_ratio
        self.split_validation = split_validation

        if num_sample_weights:
            split_section_dict["weight"] = num_sample_weights
        self.split_section_dict = split_section_dict
        split_sections = list(split_section_dict.values())
        self.dim_data = sum(split_sections)
        dim_data = self.dim_data
        if split_validation:
            dim_data += 1
        self._buffer = torch.empty(
            (self.capacity, dim_data), dtype=self.dtype, device=self.device
        )
        self.cur_idx = 0
        self.num_stored = 0

    def add(self, data_list: list):
        """Adds a transition compatible with 'self.split_section_dict'.

        Args:
            data_list: elements should correspond to self.split_section_dict.
        """
        if self.split_validation:
            label = self.sample_label()
            data_list.append(label)
        data = np.hstack(data_list)

        self._buffer[self.cur_idx] = torch.as_tensor(
            data, dtype=self.dtype, device=self.device
        )

        self.cur_idx = (self.cur_idx + 1) % self.capacity
        self.num_stored = min(self.num_stored + 1, self.capacity)

    def add_batch(self, data_list: list):
        batch_size = data_list[0].shape[0]
        # Sample weights for deep ensemble model
        if self.num_sample_weights:
            sample_weight = self.sample_weight(batch_size)
            data_list.append(sample_weight)
        # Train / Validation label
        if self.split_validation:
            label = self.sample_label(batch_size)
            data_list.append(label)

        if isinstance(data_list[0], np.ndarray):
            data = np.hstack(data_list)
            data = torch.as_tensor(data, dtype=self.dtype, device=self.device)
        else:
            data = torch.hstack(data_list)

        idxes = np.arange(self.cur_idx, self.cur_idx + batch_size) % self.capacity
        self._buffer[idxes, ...] = data

        self.cur_idx = (self.cur_idx + batch_size) % self.capacity
        self.num_stored = min(self.num_stored + batch_size, self.capacity)

    def sample_weight(self, data_size):
        size = (data_size, self.num_sample_weights)
        if self.poisson_weights:
            return np.random.poisson(1, size=size)
        else:
            return np.random.exponential(1, size=size)

    def sample(
        self,
        batch_size: int,
    ) -> TransitionBatch:
        """Samples a batch of transitions from the replay buffer."""
        indices = torch.randint(
            high=self.num_stored, size=(batch_size,), device=self.device
        )
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices) -> TransitionBatch:
        selected_data = self._buffer[indices, : self.dim_data]
        return TransitionBatch(
            selected_data, self.split_section_dict, device=self.device_to
        )

    def sample_label(self, num_sample=1):
        label = np.full((num_sample, 1), self.TRAIN)
        valid_idx = np.random.random(num_sample) < self.valid_ratio
        label[valid_idx] = self.VALID
        return label

    def shuffle(self):
        pass

    def clear(self):
        pass

    def split_data(self):
        assert self.split_validation
        if self.max_batch_size and self.max_batch_size < self.num_stored:
            idx = np.random.choice(
                self.num_stored, size=self.max_batch_size, replace=False
            )
            idx = torch.as_tensor(idx, device=self.device)
        else:
            idx = torch.arange(self.num_stored, device=self.device)

        label = self._buffer[idx, -1]
        train_indices = idx[label[: self.num_stored] == self.TRAIN]
        valid_indices = idx[label[: self.num_stored] == self.VALID]
        num_valid = len(valid_indices)
        # For the case of very small data size
        if num_valid < 2:
            num_need = 2 - num_valid
            rand_train_indices = torch.randperm(len(train_indices))
            tmp_idx = rand_train_indices[:num_need]
            valid_indices = torch.hstack([valid_indices, train_indices[tmp_idx]])
            train_indices = train_indices[rand_train_indices[num_need:]]
        return (
            self._batch_from_indices(train_indices),
            self._batch_from_indices(valid_indices),
        )

    def save(self, dir_checkpoint):
        torch.save(
            self._buffer.cpu()[: self.num_stored, :],
            f"{dir_checkpoint}/replay_buffer.pt",
        )
        saved_params = {"cur_idx": self.cur_idx, "num_stored": self.num_stored}
        with open(f"{dir_checkpoint}/replay_buffer.json", "w") as f:
            json.dump(saved_params, f)

    def load(self, dir_checkpoint):
        with open(f"{dir_checkpoint}/replay_buffer.json", "r") as f:
            saved_params = json.load(f)
        self.num_stored = saved_params["num_stored"]
        self.cur_idx = saved_params["cur_idx"]
        self._buffer[: self.num_stored, :] = torch.load(
            f"{dir_checkpoint}/replay_buffer.pt", map_location=torch.device(self.device)
        )

    @property
    def all_data(self):
        data = self._buffer[: self.num_stored, : self.dim_data]
        return TransitionBatch(data, self.split_section_dict, device=self.device_to)

    def __len__(self):
        return self.num_stored


class WithoutReplacementBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle()

    def shuffle(self):
        self.empty_idx = torch.randperm(self.num_stored, device=self.device)

    def sample(self, batch_size: int):
        if len(self.empty_idx) < batch_size:
            self.shuffle()
        size_empty = len(self.empty_idx)
        size_rest = size_empty - batch_size
        indices, self.empty_idx = torch.split(self.empty_idx, [batch_size, size_rest])
        return self._batch_from_indices(indices)

    def clear(self):
        self.cur_idx = 0
        self.num_stored = 0

    def __len__(self):
        return len(self.empty_idx)


class TransitionIterator:
    """An iterator for batches of transitions."""

    def __init__(
        self,
        transitions: TransitionBatch,
        batch_size: int,
        device,
        shuffle_each_epoch: bool = False,
        max_iter: int = None,
    ):
        self.transitions = transitions
        self.device = device
        self.shuffle_each_epoch = shuffle_each_epoch
        self.num_stored = len(transitions)
        self.batch_size = min(batch_size, self.num_stored)

        if shuffle_each_epoch:
            self.max_ptr = int(np.floor(self.num_stored / self.batch_size)) - 1
        else:
            self.max_ptr = int(np.ceil(self.num_stored / self.batch_size)) - 1
        self.max_iter = (
            min(max_iter, self.max_ptr) if max_iter is not None else self.max_ptr
        )

        self.batch_ptr = 0
        self.batch_iter = 0

        self._orders = torch.randperm(
            self.num_stored, generator=None, device=device
        ).split(self.batch_size)

    def _get_indices_next_batch(self):
        if self.max_ptr < self.batch_ptr:
            self.batch_ptr = 0
            if self.shuffle_each_epoch:
                self._orders = torch.randperm(
                    self.num_stored, generator=None, device=self.device
                ).split(self.batch_size)
            else:
                self.batch_iter = 0
                raise StopIteration
        if self.max_iter < self.batch_iter:
            self.batch_iter = 0
            if not self.shuffle_each_epoch:
                self.batch_ptr = 0
            raise StopIteration
        indices = self._orders[self.batch_ptr]
        self.batch_ptr += 1
        self.batch_iter += 1
        return indices

    def __iter__(self):
        return self

    def __next__(self):
        return self[self._get_indices_next_batch()]

    def __len__(self):
        return self.max_iter + 1

    def __getitem__(self, item):
        return self.transitions[item]


class Normalizer:
    """Class that keeps a running mean and variance and normalizes data accordingly."""

    def __init__(
        self, size: int, device: torch.device, scale: float = 1.0, name="normalizer"
    ):
        self.size = size
        self.mean_np = np.zeros((1, size), dtype=np.float32)
        self.var_np = np.ones((1, size), dtype=np.float32)
        self.mean = torch.as_tensor(self.mean_np, device=device)
        self.std = torch.as_tensor(self.var_np, device=device)
        self.scale = scale
        self.eps = 1e-4
        self.device = device
        self.name = name

        self.num_old = 0

    def update_stats(self, data: np.ndarray):
        """Updates the stored statistics using the given data."""
        assert data.ndim == 2 and data.shape[1] == self.size
        num_new = data.shape[0]
        if self.num_old == 0:
            self.mean_np = data.mean(0, keepdims=True)
            self.var_np = data.var(0, keepdims=True)
            self.num_old = num_new
        else:
            num_total = self.num_old + num_new

            sum_total = self.num_old * self.mean_np + data.sum(0, keepdims=True)
            sqr_total = self.num_old * (
                self.var_np + np.square(self.mean_np)
            ) + np.square(data).sum(0, keepdims=True)

            self.mean_np = sum_total / num_total
            self.var_np = sqr_total / num_total - np.square(self.mean_np)
            self.num_old += num_new

    def to(self):
        std_np = np.sqrt(self.var_np.clip(min=self.eps**2)) * self.scale
        self.mean.copy_(
            torch.as_tensor(self.mean_np, device=self.device, dtype=torch.float32)
        )
        self.std.copy_(torch.as_tensor(std_np, device=self.device, dtype=torch.float32))

    def normalize(self, val) -> torch.Tensor:
        """Normalizes the value according to the stored statistics."""
        return (val - self.mean) / self.std

    def denormalize(self, val) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics."""
        return self.std * val + self.mean

    def save(self, dir_checkpoint):
        """Saves stored statistics to the given path."""
        name_checkpoint = (pathlib.Path(dir_checkpoint) / self.name).with_suffix(".npz")
        stats = {"mean": self.mean_np, "var_np": self.var_np, "num_old": self.num_old}
        np.savez(name_checkpoint, **stats)

    def load(self, dir_checkpoint):
        """Loads saved statistics from the given path."""
        name_checkpoint = (pathlib.Path(dir_checkpoint) / self.name).with_suffix(".npz")
        stats = np.load(name_checkpoint)
        self.mean_np = stats["mean"]
        self.var_np = stats["var_np"]
        self.num_old = stats["num_old"]
        self.to()
