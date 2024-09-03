import torch
from torch import nn

from erl_lib.util.misc import weight_init
from erl_lib.agent.module.layer import EnsembleLinearLayer, NormalizedEnsembleLinear


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
        dropout_rate: float = 0.0,
        norm_eps: float = 0.0,
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
            NormalizedEnsembleLinear(
                num_members,
                dim_obs + dim_act,
                dim_output=dim_hidden,
                activation=nn.SiLU(inplace=True),
                dropout_rate=dropout_rate,
                normalize_eps=0.0,
            )
        ]
        for i in range(num_hidden - 1):
            layers.append(
                NormalizedEnsembleLinear(
                    num_members,
                    dim_hidden,
                    dim_output=dim_hidden,
                    activation=nn.SiLU(inplace=True),
                    dropout_rate=dropout_rate,
                    normalize_eps=norm_eps,
                )
            )
        layers.append(
            NormalizedEnsembleLinear(
                num_members,
                dim_hidden,
                dim_output=dim_output,
                normalize_eps=norm_eps,
            )
        )

        self.hidden_layers = nn.Sequential(*layers)
        # Weight init
        self.hidden_layers[-1].weight.data.fill_(0)
        self.hidden_layers[-1].bias.data.fill_(0)

    def forward(self, xu):
        pred = self.hidden_layers(xu)
        return pred.squeeze(-1).t()
