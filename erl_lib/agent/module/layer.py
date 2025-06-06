import numpy as np
import torch
from torch import nn as nn


def ensemble_kaiming_normal(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            nn.init.kaiming_normal_(
                m.weight[i, ...].transpose(0, 1), nonlinearity="leaky_relu"
            )
        nn.init.normal_(m.bias, std=stddev)


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self,
        num_members: int,
        in_size: int,
        out_size: int,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight_decay = weight_decay
        self.weight = nn.Parameter(
            torch.zeros(self.num_members, self.in_size, self.out_size)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_members, 1, self.out_size))
        self._index = None

        self.apply(ensemble_kaiming_normal)

    def forward(self, x):
        h = x.matmul(self.weight).add(self.bias)
        self._index = None
        return h

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, decay={self.weight_decay}"
        )


class NormalizedEnsembleLinear(nn.Module):
    def __init__(
        self,
        num_members,
        dim_input,
        weight_decay=1e-3,
        dropout_rate=0.0,
        normalize_eps: float = 0.0,
        dim_output=None,
        residual=False,
        activation=None,
    ):
        super().__init__()
        # self.residual = dim_output is None
        self.weight_decay = weight_decay

        if dim_output is None:
            dim_output = dim_input
            self.residual = residual
        else:
            self.residual = False

        layers = []
        if isinstance(normalize_eps, float) and 0 < normalize_eps:
            layers.append(
                nn.LayerNorm(dim_input, elementwise_affine=False, eps=normalize_eps)
            )
            self.idx_linear = 1
        else:
            self.idx_linear = 0

        linear = EnsembleLinearLayer(
            num_members, dim_input, dim_output, weight_decay=weight_decay
        )
        layers.append(linear)
        if 0 < dropout_rate:
            layers.append(nn.Dropout(p=dropout_rate))

        if activation:
            layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        h = self.layers(x)
        if self.residual:
            h += x
        return h

    def set_index(self, index):
        self.layers[self.idx_linear].set_index(index)

    @property
    def weight(self):
        # return self.get_parameter(f"layers.{self.idx_linear}.weight")
        return self.layers[self.idx_linear].weight

    @property
    def bias(self):
        # return self.get_parameter(f"layers.{self.idx_linear}.bias")
        return self.layers[self.idx_linear].bias
