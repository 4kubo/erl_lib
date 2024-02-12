import numpy as np

from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.utils import rewards


from dm_control.suite.humanoid import (
    _CONTROL_TIMESTEP,
    _DEFAULT_TIME_LIMIT,
    _WALK_SPEED,
    _STAND_HEIGHT,
    SUITE,
    Physics,
    Humanoid,
    get_model_and_assets,
)


def randomize_limited_and_rotational_joints(physics, random=None):
    """Randomizes the positions of joints defined in the physics body."""
    random = random or np.random

    hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
    slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
    free = mjbindings.enums.mjtJoint.mjJNT_FREE

    qpos = physics.named.data.qpos

    for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, "joint")
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]

        if is_limited:
            if joint_type == hinge:
                # if joint_name == "right_shoulder1":
                #     qpos[joint_name] = 0.7 * random.randn() * 0.01
                # elif joint_name == "right_shoulder2":
                #     qpos[joint_name] = -0.6 * random.randn() * 0.01
                # elif joint_name == "right_elbow":
                #     qpos[joint_name] = -1.56 * random.randn() * 0.01
                # elif joint_name == "left_shoulder1":
                #     qpos[joint_name] = -0.7 * random.randn() * 0.01
                # elif joint_name == "left_shoulder2":
                #     qpos[joint_name] = 0.6 * random.randn() * 0.01
                # elif joint_name == "left_elbow":
                #     qpos[joint_name] = -1.56 * random.randn() * 0.01
                # else:
                range_min, range_max = physics.model.jnt_range[joint_id]
                center = (range_max + range_min) * 0.5
                width = (range_max - range_min) * 0.5 * 0.1  # Changed
                lo = center - width
                high = center + width
                qpos[joint_name] = random.uniform(lo, high)
        elif joint_type == free:
            if joint_name == "root":
                # quaternion of ratating pi/2 rad@Z-axis and pi/2 rad@Y-axis
                lying_pos = [0.5, 0.5, 0.5, -0.5] + np.random.randn(4) * 0.01
                qpos[joint_name][2] = 0.5
                qpos[joint_name][3:] = lying_pos


@SUITE.add("benchmarking")
def stand_lying(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = HumanoidLying(move_speed=0, pure_state=False, random=random)
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
    task = HumanoidLying(move_speed=_WALK_SPEED, pure_state=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class HumanoidLying(Humanoid):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Find a collision-free random initial configuration.
        penetrating = True
        while penetrating:
            randomize_limited_and_rotational_joints(physics, self.random)
            # Check for collisions.
            physics.after_reset()
            penetrating = physics.data.ncon > 0
        super(Humanoid, self).initialize_episode(physics)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(
            physics.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            sigmoid="linear",
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            physics.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=0.1,  # Changed
            value_at_margin=0,
        )
        # vertical_velocity = physics.center_of_mass_velocity()[2]
        # vertical_static = rewards.tolerance(
        #     vertical_velocity,
        #     bounds=(-0.1, 0.1),
        #     margin=0.1,
        # )
        # stand_reward = standing * vertical_static * upright
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
        ).mean()
        if self._move_speed == 0:
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.1).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            # move = (5 * move + 1) / 6
            return small_control * stand_reward * move
