import numpy as np
from dm_control.rl import control
from dm_control.suite.hopper import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    SUITE,
    Physics as HopperPhysics,
    Hopper,
    get_model_and_assets,
)
from dm_control.utils import rewards

# Angular momentum above which reward is 1.
_SPIN_SPEED = 5


@SUITE.add("custom")
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Flip task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopper(goal="flip", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("custom")
def flip_backwards(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Flip Backwards task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopper(goal="flip-backwards", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def stand_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = HopperLying(hopping=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def hop_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = HopperLying(hopping=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("custom")
def flip_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Flip task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopperLying(goal="flip", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("custom")
def flip_backwards_lying(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Flip Backwards task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomHopperLying(goal="flip-backwards", random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


class Physics(HopperPhysics):
    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom["torso"][1]


class CustomHopper(Hopper):
    """Custom Hopper tasks."""

    def __init__(self, goal="hop-backwards", random=None):
        super().__init__(None, random)
        self._goal = goal

    def _flip_reward(self, physics, forward=True):
        reward = rewards.tolerance(
            (1.0 if forward else -1.0) * physics.angmomentum(),
            bounds=(_SPIN_SPEED, float("inf")),
            margin=_SPIN_SPEED / 2,
            value_at_margin=0,
            sigmoid="linear",
        )
        return reward

    def get_reward(self, physics):
        if self._goal == "flip":
            return self._flip_reward(physics, forward=True)
        elif self._goal == "flip-backwards":
            return self._flip_reward(physics, forward=False)
        else:
            raise NotImplementedError(f"Goal {self._goal} is not implemented.")


class HopperLying(Hopper):
    def initialize_episode(self, physics):
        qpos = physics.named.data.qpos
        for joint_id in range(physics.model.njnt):
            joint_name = physics.model.id2name(joint_id, "joint")
            # if joint_name == "rootz":
            #     qpos[joint_name] = -0.9 + self.random.randn() * 0.01
            # elif joint_name == "rooty":
            #     qpos[joint_name] = -np.pi * 0.5 - self.random.rand() * 0.5
            # else:
            #     qpos[joint_name] = self.random.randn() * 0.01
            if joint_name == "rootz":
                qpos[joint_name] = -0.8
            elif joint_name == "rooty":
                qpos[joint_name] = -np.pi * 0.5
            else:
                qpos[joint_name] = self.random.randn() * 0.01

        self.after_step(physics)


class CustomHopperLying(CustomHopper, HopperLying):
    pass
