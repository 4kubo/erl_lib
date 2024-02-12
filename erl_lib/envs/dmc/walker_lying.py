from dm_control.rl import control
import numpy as np

from dm_control.utils import rewards
from dm_control.suite.walker import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    _WALK_SPEED,
    _RUN_SPEED,
    _STAND_HEIGHT,
    SUITE,
    Physics,
    PlanarWalker,
    get_model_and_assets,
)


@SUITE.add("benchmarking")
def stand_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalkerLying(move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


@SUITE.add("benchmarking")
def walk_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalkerLying(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


@SUITE.add("benchmarking")
def run_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalkerLying(move_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class PlanarWalkerLying(PlanarWalker):
    def initialize_episode(self, physics):
        qpos = physics.named.data.qpos
        for joint_id in range(physics.model.njnt):
            joint_name = physics.model.id2name(joint_id, "joint")
            if joint_name == "rootz":
                qpos[joint_name] = -0.8 + self.random.randn() * 0.01
            elif joint_name == "rooty":
                qpos[joint_name] = -np.pi * 0.5 - self.random.rand() * 0.5
            else:
                qpos[joint_name] = self.random.randn() * 0.01

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        stand_reward = rewards.tolerance(
            physics.torso_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 5,
        )
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(
                physics.horizontal_velocity(),
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            return stand_reward * (5 * move_reward + 1) / 6
