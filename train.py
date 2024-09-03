import logging
import os
from collections.abc import Iterable
from tqdm import trange
from importlib import import_module
from omegaconf import OmegaConf
import hydra
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from erl_lib.util.env import make_envs
from erl_lib.util.misc import set_seed_everywhere
from erl_lib.util.logger import Logger


def callbacks_if_need(config):
    callback = config.common.eval.callback
    if callback is not None:

        def import_callback(call_bacK_str):
            name_module, name_callable = call_bacK_str.split(":")
            module = import_module(name_module)
            if not hasattr(module, name_callable):
                msg = f"No callable object '{name_callable}' is found in module '{module}'"
                raise AttributeError(msg)
            callback = getattr(module, name_callable)
            return callback

        if isinstance(callback, str):
            callbacks = [import_callback(callback)]
        elif isinstance(callback, Iterable):
            callbacks = [import_callback(callback_i) for callback_i in callback]
        return callbacks


@hydra.main(config_path="config", config_name="train")
def run(cfg):
    set_seed_everywhere(cfg.common.seed)
    log_dir = cfg.log.log_dir

    env_seed, test_env_seed = np.random.randint(2**16, size=(2,))
    num_training_envs = cfg.common.num_training_envs
    # Environments for behavior
    (
        envs,
        dim_obs,
        dim_act,
        max_episode_steps,
        env_kwargs_train,
        action_repeat,
    ) = make_envs(cfg.env, num_training_envs)
    envs.reset(seed=int(env_seed))
    # Environments for test
    num_test_envs = cfg.common.num_eval_episodes
    test_env_seeds = np.random.randint(2**16, size=(num_test_envs,))
    cfg.env.kwargs = dict(env_kwargs_train, **(cfg.env_eval.kwargs or {}))  # Overriding
    dir_video = f"{log_dir}/videos" if cfg.log.record else None
    envs_test, _, _, length_eval_iter, _, _ = make_envs(
        cfg.env, num_test_envs, log_dir=dir_video
    )
    envs_test.reset(seed=int(test_env_seed))
    # Environment dependent parameters
    cfg.agent.steps_per_iter = cfg.agent.steps_per_iter or max_episode_steps
    cfg.agent.dim_obs = dim_obs
    cfg.agent.dim_act = dim_act

    logger = Logger(cfg)

    # Use same heuristics as the TD-MPC2 paper
    if cfg.agent.discount in (0, None):
        raw_discount = (max_episode_steps * 0.2 - 1) / (max_episode_steps * 0.2)
        cfg.agent.discount = float(np.clip(raw_discount, 0.95, 0.995))
    if cfg.agent.seed_iters in (0.0, None):
        cfg.agent.seed_iters = max(5, int(1000 / max_episode_steps))

    logger.info(OmegaConf.to_yaml(cfg))

    # Agent
    term_fn = envs.get_wrapper_attr("termination_model")
    agent = hydra.utils.instantiate(
        cfg.agent,
        num_envs=num_training_envs,
        step_multiplier=action_repeat,
        term_fn=term_fn,
        device=cfg.common.device,
        silent=cfg.log.silent,
        logger=logger,
        _recursive_=False,
    )
    if cfg.common.load_model:
        if not os.path.exists(cfg.common.load_model):
            logging.error(f"Not found: {cfg.common.load_model}")
        agent.load(cfg.common.load_model)

    logger.info(agent.description)

    # Configuring progress bar if necessary
    disable_outer = cfg.agent.iters_per_epoch < cfg.agent.steps_per_iter
    kwargs_outer_trange = agent.kwargs_trange.copy()
    kwargs_outer_trange["disable"] |= disable_outer
    agent.kwargs_trange["disable"] |= np.logical_not(disable_outer)

    # Initial evaluation
    agent.reset()
    agent.evaluate(
        envs_test,
        num_test_envs,
        None,
        stack_obs=cfg.common.eval.stack_obs,
        callbacks=callbacks_if_need(cfg),
    )

    # Main outer loop
    while agent.time_steps_total < cfg.common.max_time_steps:
        obs, info = envs.reset()
        agent.reset()

        desc = f"@{agent.time_steps_total: >10,}"
        pbar = trange(agent.iters_this_epoch, desc=desc, **kwargs_outer_trange)
        # Main inner loop for interaction with the environment and the agent's update
        while not agent.is_epoch_done:
            action = agent.act(obs, sample=True)
            next_obs, reward, terminated, truncated, info = envs.step(action)
            agent.observe(obs, action, reward, next_obs, terminated, truncated, info)
            obs = next_obs
            pbar.update()
        pbar.close()

        # Performance of the behavior policy
        mean_training_return = np.array(envs.env.return_queue)[-10:].mean()
        # Evaluate
        agent.evaluate(
            envs_test,
            num_test_envs,
            mean_training_return,
            stack_obs=cfg.common.eval.stack_obs,
            callbacks=callbacks_if_need(cfg),
        )
        # Checkpoint
        if cfg.log.checkpoint and not cfg.log.only_last_checkpoint:
            dir_ckpt = f"{log_dir}/checkpoint/{agent.time_steps_total:0>10}"
            os.makedirs(dir_ckpt, exist_ok=True)
            last = cfg.common.max_time_steps <= agent.time_steps_total
            agent.save(dir_ckpt, last)

    if cfg.log.checkpoint and cfg.log.only_last_checkpoint:
        dir_ckpt = f"{log_dir}/checkpoint/{agent.time_steps_total:0>10}"
        os.makedirs(dir_ckpt, exist_ok=True)
        agent.save(dir_ckpt)

    logger.info(f"Finished @ Time steps {int(cfg.common.max_time_steps):,d}")
    envs.close()
    envs_test.close()
    logger.close()


if __name__ == "__main__":
    run()
