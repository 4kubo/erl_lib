from gymnasium.spaces import Box


from erl_lib.base.env import BaseEnv


class MyoSuiteEnv(BaseEnv):
    def __init__(self, env_id: str, width=128, height=128, render_mode=None):
        import myosuite
        import gym

        self._max_episode_steps = 100
        # Remove timelimit wrapper
        self.env = gym.make(env_id).unwrapped

        self.observation_space = Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
        )
        self.action_space = Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
        )

        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.metadata = {"render_fps": int(1 / self.env.unwrapped.dt)}

    def step(self, action):
        obs, reward, _, raw_info = self.env.step(action)
        terminated = False
        truncated = "TimeLimit.truncated" in raw_info
        info = {
            "reward_dense": raw_info["rwd_dense"],
            "reward_sparse": raw_info["rwd_sparse"],
            "success": raw_info["solved"],
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, **kwargs):
        self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def render(self, render_mode=None):
        render_mode = render_mode or self.render_mode

        if render_mode == "human":
            # BGR to RGB conversion
            self.env.render()
        else:
            image = self.env.sim.renderer.render_offscreen(
                width=self.width,
                height=self.height,
                camera_id="hand_side_inter",
            )
            return image
