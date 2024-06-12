from dm_control.rl import control
import numpy as np


from dm_control.suite.acrobot import (
    _DEFAULT_TIME_LIMIT,
    SUITE,
    Physics,
    Balance,
    get_model_and_assets,
)


@SUITE.add("benchmarking")
def swingup_down(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns Acrobot balance task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = BalanceDown(sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def swingup_sparse_down(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns Acrobot sparse balance."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = BalanceDown(sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class BalanceDown(Balance):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Shoulder and elbow are set to a random position between [-pi, pi).

        Args:
          physics: An instance of `Physics`.
        """
        # physics.named.data.qpos[["shoulder", "elbow"]] = (
        #     np.pi + self.random.randn(2) * 0.01
        # )
        physics.named.data.qpos["shoulder"] = np.pi + self.random.randn() * 0.1
        physics.named.data.qpos["elbow"] = self.random.randn() * 0.1

        self.after_step(physics)
        # super(Balance, self).initialize_episode(physics)
