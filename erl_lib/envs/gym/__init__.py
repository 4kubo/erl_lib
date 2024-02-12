from gymnasium.logger import set_level, ERROR
set_level(ERROR)
from gymnasium.envs.registration import register

base = "erl_lib.envs.gym"

register(
    id="MBAnt-v0",
    entry_point=f"{base}.ant:MBAntEnv",
    max_episode_steps=1000,
)
register(
    id="MBHalfCheetah-v0",
    entry_point=f"{base}.half_cheetah:MBHalfCheetahEnv",
    max_episode_steps=1000,
)
register(
    id="MBHopper-v0",
    entry_point=f"{base}.hopper:MBHopperEnv",
    max_episode_steps=1000,
)
register(
    id="MBHumanoid-v0",
    entry_point=f"{base}.humanoid:MBHumanoidEnv",
    max_episode_steps=1000,
)
register(
    id="MBHumanoidStandup-v0",
    entry_point=f"{base}.humanoid:MBHumanoidStandupEnv",
    max_episode_steps=1000,
)
register(
    id="MBWalker2d-v0",
    entry_point=f"{base}.walker_2d:MBWalker2dEnv",
    max_episode_steps=1000,
)
# register(
#     id="MBCartPole-v0",
#     entry_point=f"{base}.cart_pole:CartPoleSwingUpEnv",
#     max_episode_steps=25,
# )
# register(
#     id="MBDoubleCartPole-v0",
#     entry_point=f"{base}.double_cart_pole:DoubleCartPoleSwingaUpEnv",
#     max_episode_steps=30,
# )
register(
    id="MBPendulum-v0",
    entry_point=f"{base}.pendulum:Pendulum",
    max_episode_steps=200,
)
register(
    id="MBPendulum-v1",
    entry_point=f"{base}.pendulum:PendulumV1",
    max_episode_steps=400,
)
#
# register(
#     id="MBMountainCar-v0",
#     entry_point=f"{base}.mountain_car:MountainCar",
#     max_episode_steps=200,
# )
