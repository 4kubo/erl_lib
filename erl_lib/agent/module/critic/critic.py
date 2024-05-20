import torch
from torch import nn

from erl_lib.util.misc import weight_init
from erl_lib.agent.module.layer import EnsembleLinearLayer


class EnsembleCriticNetwork(nn.Module):
    scale_params = ["max_reward", "min_reward", "q_ub", "q_lb"]

    def __init__(
        self,
        dim_obs,
        dim_act,
        dim_hidden,
        num_hidden,
        num_members,
        dim_output: int = 1,
        **kwargs,
    ):
        super(EnsembleCriticNetwork, self).__init__()
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_hidden = dim_hidden
        self.num_hidden = num_hidden
        self.num_members = num_members
        self.info = {}

        layers = [
            EnsembleLinearLayer(num_members, dim_obs + dim_act, dim_hidden),
            nn.SiLU(),
        ]
        for i in range(num_hidden - 1):
            layers.extend(
                (
                    EnsembleLinearLayer(num_members, dim_hidden, dim_hidden),
                    nn.SiLU(),
                )
            )

        layers.append(EnsembleLinearLayer(num_members, dim_hidden, dim_output))
        self.hidden_layers = nn.Sequential(*layers)
        # Weight init
        self.hidden_layers[-1].weight.data.fill_(0)
        self.hidden_layers[-1].bias.data.fill_(0)

    def forward(self, xu):
        pred = self.hidden_layers(xu)
        return pred.squeeze(-1).t()
