import os
from typing import Optional
import numpy as np
import gymnasium
from gymnasium import error, logger
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers import (
    FlattenObservation,
    RescaleAction,
    TimeLimit,
    RecordEpisodeStatistics,
    RecordVideo,
    VectorListInfo,
)
import imageio

from erl_lib.base.env import BaseEnv
from erl_lib.envs.dmc.from_dm import DmControlEnv
from erl_lib.envs.myosuite.from_myosuite import MyoSuiteEnv


class VectorEnv(SyncVectorEnv, BaseEnv):
    @property
    def max_episode_steps(self):
        max_episode_steps = None
        if hasattr(self.envs[0], "spec"):
            max_episode_steps = max_episode_steps or self.envs[0].spec.max_episode_steps
        elif hasattr(self.envs[0], "max_episode_steps"):
            max_episode_steps = self.envs[0].get_wrapper_attr("max_episode_steps")
        elif hasattr(self.envs[0].unwrapped, "max_episode_steps"):
            max_episode_steps = self.envs[0].unwrapped.max_episode_steps
        if max_episode_steps is None:
            raise AttributeError(
                f"max_episode_steps is not defined for the environment {self.envs[0]}"
            )
        return max_episode_steps

    @property
    def reward_model(self):
        return getattr(self.envs[0].unwrapped, "reward_model", None)

    @property
    def termination_model(self):
        return getattr(self.envs[0].unwrapped, "termination_model", None)

    def render(self):
        return self.envs[0].render()


class CustomVideoRecorder(VideoRecorder):
    def __init__(
        self,
        env,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
        disable_logger: bool = False,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Args:
            env (Env): Environment to take video of.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
            disable_logger (bool): Whether to disable moviepy logger or not.

        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        self._async = env.metadata.get("semantics.async")
        self.enabled = enabled
        self.disable_logger = disable_logger
        self._closed = False

        self.render_history = []
        self.env = env

        self.render_mode = env.render_mode

        if self.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {self.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        required_ext = ".mp4"
        self.path = base_path + required_ext

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension {required_ext}."
            )

        self.frames_per_sec = env.metadata.get("render_fps", 30)

        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = "video/mp4"
        self.metadata_path = f"{path_base}.meta.json"
        self.write_metadata()

        logger.info(f"Starting new video recorder writing to {self.path}")
        self.recorded_frames = []

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        # Create a video
        if len(self.recorded_frames) > 0:
            imageio.mimsave(self.path, self.recorded_frames, fps=self.frames_per_sec)

        # Stop tracking this for autoclose
        self._closed = True

    def write_metadata(self):
        pass


class CustomRecordVideo(RecordVideo):
    def __init__(self, *args, wandb=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._wandb = wandb

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}_{self.episode_id}steps"

        base_path = os.path.join(self.video_folder, video_name)
        if self._wandb:
            pass
        else:
            self.video_recorder = CustomVideoRecorder(
                env=self.env,
                base_path=base_path,
                metadata={},
                disable_logger=False,
            )

            self.video_recorder.capture_frame()
            self.recorded_frames = 1

        self.recording = True


class VideoRecorder(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """Utility class for logging evaluation videos."""

    def __init__(self, env, root_dir, wandb=None, render_size=384, fps=15):
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)

        self.save_dir = f"{root_dir}/video" if root_dir else None
        self._wandb = wandb
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and self._wandb and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.render_size,
                width=self.render_size,
                camera_id=0,
            )
            self.frames.append(frame)

    def save(self, step):
        if self.enabled:
            frames = np.stack(self.frames).transpose(0, 3, 1, 2)
            self._wandb.log(
                {"eval_video": self._wandb.Video(frames, fps=self.fps, format="mp4")},
                step=step,
            )


class RepeatAction(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(self, env: gymnasium.Env, action_repeat=1):
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)

        self.action_repeat = action_repeat
        self.metadata["render_fps"] /= action_repeat

    # noinspection PyUnboundLocalVariable
    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            obs, reward_i, terminated, truncated, info = self.env.step(action)
            reward += reward_i
            if terminated or truncated:
                break
        return obs, reward, terminated, truncated, info

    @property
    def max_episode_steps(self):
        return int(np.floor(self.env.max_episode_steps / self.action_repeat))


def make_envs(config_env, num_envs, log_dir=None):
    suite = config_env.suite
    task_id = config_env.task_id
    action_repeat = config_env.get("action_repeat", 1)
    max_episode_steps = config_env.get("max_episode_steps", 1000)
    env_kwargs = {} if config_env.kwargs is None else dict(config_env.kwargs)

    if suite == "dmc":
        domain_name, task_name = task_id.split("-", 1)

        def make(**kwargs):
            env = DmControlEnv(domain_name, task_name, **kwargs)
            return env

    elif suite in ["gym", "gym-robo"]:

        def make(**kwargs):
            try:
                env = gymnasium.envs.registration.make(
                    task_id, disable_env_checker=True, **kwargs
                ).unwrapped
            except TypeError:
                kwargs.pop("height")
                kwargs.pop("width")
                env = gymnasium.envs.registration.make(
                    task_id, disable_env_checker=True, **kwargs
                ).unwrapped
            return env

    elif suite == "myosuite":

        def make(**kwargs):
            return MyoSuiteEnv(task_id, **kwargs)

    else:
        raise NotImplementedError(f"Task {task_id} is invalid as of suite {suite}")

    wrappers = [
        (FlattenObservation, {}),
        (RepeatAction, {"action_repeat": action_repeat}),
        (RescaleAction, {"min_action": -1.0, "max_action": 1.0}),
        (TimeLimit, {"max_episode_steps": max_episode_steps}),
    ]

    def create_env(instance_id):
        """Creates an environment that can enable or disable the environment checker."""

        def _make_env():
            kwargs = env_kwargs.copy()
            if log_dir is not None and instance_id == 0:
                kwargs.update(
                    **{"height": 128, "width": 128, "render_mode": "rgb_array"}
                )

            env = make(**kwargs)

            for wrapper, kwargs in wrappers:
                env = wrapper(env, **kwargs)

            # Record as video for the 1-st env if necessary
            if log_dir is not None and instance_id == 0:
                env = CustomRecordVideo(
                    env,
                    log_dir,
                    step_trigger=lambda x: x == 0,
                    name_prefix=task_id,
                    disable_logger=True,
                )

            return env

        return _make_env

    env_fns = [create_env(instance_id) for instance_id in range(num_envs)]
    envs = VectorEnv(env_fns)

    # max_episode_steps = envs.max_episode_steps
    dim_obs = envs.single_observation_space.shape[0]
    dim_act = envs.single_action_space.shape[0]

    envs = RecordEpisodeStatistics(envs)
    envs = VectorListInfo(envs)
    return envs, dim_obs, dim_act, max_episode_steps, env_kwargs, action_repeat
