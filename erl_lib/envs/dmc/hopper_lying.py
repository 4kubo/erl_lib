from dm_control.rl import control
import numpy as np


from dm_control.suite.hopper import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    SUITE,
    Physics,
    Hopper,
    get_model_and_assets,
)


@SUITE.add("benchmarking")
def stand_lying(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = HopperLying(hopping=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


@SUITE.add("benchmarking")
def hop_lying(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = HopperLying(hopping=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class HopperLying(Hopper):
    def initialize_episode(self, physics):
        qpos = physics.named.data.qpos
        for joint_id in range(physics.model.njnt):
            joint_name = physics.model.id2name(joint_id, "joint")
            if joint_name == "rootz":
                qpos[joint_name] = -0.8 + self.random.randn() * 0.01
            elif joint_name == "rooty":
                qpos[joint_name] = -np.pi * .5 - self.random.rand() * 0.5
            else:
                qpos[joint_name] = self.random.randn() * 0.01
