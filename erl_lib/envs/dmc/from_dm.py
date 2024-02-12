import cv2
import numpy as np
from dm_control import suite
from gymnasium.spaces import Box

from erl_lib.base.env import BaseEnv


class DmControlEnv(BaseEnv):
    DEFAULT_CAMERAS = dict(
        locom_rodent=1,
        quadruped=2,
    )

    def __init__(
        self,
        domain_name,
        task_name,
        render_mode=None,
        visualize_reward=True,
        height=128,
        width=128,
        camera_id=-1,
        **kwargs,
    ):
        self.env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            visualize_reward=visualize_reward,
            environment_kwargs=kwargs,
        )
        # dm_control specs
        self.observation_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        # check observation shape
        observation_size = sum(
            np.prod(v.shape) if 0 < len(v.shape) else 1
            for v in self.observation_spec.values()
        )

        # gym spaces
        self.observation_space = Box(low=-1.0, high=1.0, shape=(observation_size,))
        self.action_space = Box(
            low=self.action_spec.minimum,
            high=self.action_spec.maximum,
            shape=self.action_spec.shape,
        )
        # Rendering
        self.render_mode = render_mode
        self.height = height
        self.width = width
        self.camera_id = self.DEFAULT_CAMERAS.get(domain_name, 0) if camera_id < 0 else camera_id
        self.metadata = {"render_fps": 1 / self.env.control_timestep()}

    def render(self, render_mode=None):
        render_mode = render_mode or self.render_mode
        image = self.env.physics.render(
            height=self.height, width=self.width, camera_id=self.camera_id
        )
        if render_mode == "human":
            # BGR to RGB conversion
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("render", image)
            cv2.waitKey(int(self.env.control_timestep() * 1000))
        else:
            return image

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._get_observation(time_step)

        reward = 0.0 if time_step.first() else time_step.reward

        terminated = False if time_step.first() else time_step.discount == 0
        truncated = time_step.last()

        # mujoco physics state to reproduce identical states later
        state = self.env.physics.get_state()
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"_state": state.copy()}

    def reset(self, seed: int = None, **kwargs):
        self.env.task.random.seed(seed)
        time_step = self.env.reset()
        state = self.env.physics.get_state()
        return self._get_observation(time_step), {"_state": state.copy()}

    def close(self):
        pass

    @staticmethod
    def _get_observation(time_steps):
        return np.hstack([v.flatten() for v in time_steps.observation.values()])

    @property
    def max_episode_steps(self):
        return self.env._step_limit
