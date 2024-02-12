import torch
from torch import nn

from erl_lib.util.misc import weight_init
from erl_lib.agent.module.layer import EnsembleLinearLayer


class EnsembleCriticNetwork(nn.Module):
    scale_params = ["max_reward", "min_reward", "q_ub", "q_lb", "q_center", "q_width"]

    def __init__(
        self,
        dim_obs,
        dim_act,
        dim_hidden,
        num_hidden,
        num_members,
        dim_output: int = 1,
        norm_eps=0,
        bound_factor: float = 1.0,
        bounded_prediction: bool = False,
        **kwargs,
    ):
        super(EnsembleCriticNetwork, self).__init__()
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_hidden = dim_hidden
        self.num_hidden = num_hidden
        self.num_members = num_members
        self.bound_factor = bound_factor
        self.bounded_prediction = bounded_prediction
        self.info = {}

        for param_name in self.scale_params:
            self.register_buffer(param_name, torch.as_tensor([torch.nan]))

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
            if 0 < norm_eps:
                layers.append(nn.LayerNorm(dim_hidden, eps=norm_eps, elementwise_affine=False))

        layers.append(EnsembleLinearLayer(num_members, dim_hidden, dim_output))
        self.hidden_layers = nn.Sequential(*layers)
        # Weight init
        self.hidden_layers[-1].weight.data.fill_(0)
        self.hidden_layers[-1].bias.data.fill_(0)

    def forward(self, xu, hard_bound=False, log=False):
        pred = self.hidden_layers(xu)
        pred = pred[..., 0].t()
        if log:
            with torch.no_grad():
                self.info.update(
                    critic_raw_mean=pred.mean(),
                    critic_raw_ensemble_std=pred.std(1).mean(),
                    critic_raw_sample_std=pred.mean(1).std(),
                )

        if self.bounded_prediction:
            pred = self.scale(
                pred, self.q_width, self.q_center, self.q_ub, self.q_lb, hard_bound
            )
        return pred

    def scale(self, pred, q_width, q_center, q_ub, q_lb, hard_bound=False):
        assert q_width is not None
        pred = pred * q_width / self.bound_factor * 0.1 + q_center
        if hard_bound:
            assert q_ub is not None
            pred = q_ub - torch.relu(q_ub - pred)
            pred = q_lb + torch.relu(pred - q_lb)
        return pred

    def set_stats(self, q_center, q_width, q_ub, q_lb):
        self.q_center.copy_(q_center)
        self.q_width.copy_(q_width)
        self.q_ub.copy_(q_ub)
        self.q_lb.copy_(q_lb)

    def get_stats(self):
        return self.q_center, self.q_width, self.q_ub, self.q_lb
