import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from erl_lib.util.misc import weight_init


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance
        # of certain algorithms.
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following
        # link:
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    deterministic = False

    def __init__(self, dim_obs, dim_act, dim_hidden, num_hidden):
        super().__init__()
        layers = [nn.Linear(dim_obs, dim_hidden), nn.SiLU()]
        for i in range(num_hidden - 1):
            layers += [nn.Linear(dim_hidden, dim_hidden), nn.SiLU()]
        layers.append(nn.Linear(dim_hidden, dim_act * 2))
        self.hidden_layers = nn.Sequential(*layers)

        self.reset()

        self.info = {}

    def forward(self, obs, log=False):
        mu, raw_log_std = self.hidden_layers(obs).chunk(2, dim=-1)

        std = F.softplus(raw_log_std) + 1e-5
        dist = SquashedNormal(mu, std)

        if log:
            with torch.no_grad():
                q005, q095 = torch.quantile(
                    raw_log_std,
                    torch.tensor([0.05, 0.95], device=obs.device),
                )
                self.info.update(
                    actor_raw_logstd_005=q005,
                    actor_raw_logstd_095=q095,
                    actor_raw_scale=std.mean(),
                )
        return dist

    def reset(self):
        self.apply(weight_init)
