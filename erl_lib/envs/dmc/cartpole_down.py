import numpy as np

from dm_control.rl import control
from dm_control.suite.cartpole import (
    _DEFAULT_TIME_LIMIT,
    SUITE,
    Physics,
    Balance,
    get_model_and_assets,
)


@SUITE.add("benchmarking")
def swingup_down(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Cartpole Swing-Up task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = BalanceDown(swing_up=True, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def swingup_sparse_down(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the sparse reward variant of the Cartpole Swing-Up task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = BalanceDown(swing_up=True, sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class BalanceDown(Balance):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Initializes the cart and pole according to `swing_up`, and in both cases
        adds a small random initial velocity to break symmetry.

        Args:
          physics: An instance of `Physics`.
        """
        nv = physics.model.nv
        physics.named.data.qpos["slider"] = 0.001 * self.random.randn()
        physics.named.data.qpos["hinge_1"] = np.pi + 0.001 * self.random.randn()
        physics.named.data.qpos[2:] = 0.01 * self.random.randn(nv - 2)
        physics.named.data.qvel[:] = 0.001 * self.random.randn(physics.model.nv)
        super(Balance, self).initialize_episode(physics)
