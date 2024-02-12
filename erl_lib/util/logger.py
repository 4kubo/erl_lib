"""Implementations for logging."""
import copy
import logging
import os
import wandb
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from omegaconf import OmegaConf

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter

from erl_lib.base.datatype import Metrics

os.environ["WANDB_SILENT"] = "true"


def flatten_dict(
    dt: dict,
    delimiter: str = "/",
    prevent_delimiter: bool = False,
    flatten_list: bool = False,
):
    """Flatten dict.

    Output and input are of the same dict type.
    Input dict remains the same after the operation.
    """

    def _raise_delimiter_exception():
        raise ValueError(
            f"Found delimiter `{delimiter}` in key when trying to flatten "
            f"array. Please avoid using the delimiter in your specification."
        )

    dt = copy.copy(dt)
    if prevent_delimiter and any(delimiter in key for key in dt):
        # Raise if delimiter is any of the keys
        _raise_delimiter_exception()

    while_check = (dict, list) if flatten_list else dict

    while any(isinstance(v, while_check) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    if prevent_delimiter and delimiter in subkey:
                        # Raise if delimiter is in any of the subkeys
                        _raise_delimiter_exception()

                    add[delimiter.join([key, str(subkey)])] = v
                remove.append(key)
            elif flatten_list and isinstance(value, list):
                for i, v in enumerate(value):
                    if prevent_delimiter and delimiter in subkey:
                        # Raise if delimiter is in any of the subkeys
                        _raise_delimiter_exception()

                    add[delimiter.join([key, str(i)])] = v
                remove.append(key)

        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


class Logger:
    """Class that implements a logger of statistics."""

    def __init__(self, cfg, log_level=logging.INFO, max_bytes=1000000):
        self.log_dir = cfg.log.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Logger
        self.logger = logging.getLogger("erllib")
        self.logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            "%(asctime)s [%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
        )

        # I don't why, but logs are printed out to stdout. even only with file handler
        # sh = logging.StreamHandler()
        # try:
        #     sh.setLevel(log_level)
        # except ValueError:
        #     msg = (
        #         f"got Unknown log level: '{log_level}'."
        #         f" Please chose log_level via hydra out of"
        #         f" ['INFO', 'DEBUG', 'WARN', 'ERROR']"
        #     )
        #     self.logger.error(msg)
        #
        # sh.setFormatter(fmt)
        # self.logger.addHandler(sh)

        fh = RotatingFileHandler(
            os.path.join(self.log_dir, "logs.txt"), maxBytes=max_bytes, mode="w"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        self.infos = {}
        self.info_epoch = {}

        # TBX logger
        self.summary_writer = SummaryWriter(self.log_dir, flush_secs=30)
        # Wandb if possible
        project, entity = (
            cfg.log.wandb.get("project", "none"),
            cfg.log.wandb.get("entity", "none"),
        )
        use_wandb = cfg.log.wandb.use and not any((project is None, entity is None))
        if use_wandb:
            try:
                group_name = cfg.log.wandb.get("group_name", "none")
                if group_name is None:
                    group_name = f"{cfg.env.task_id}"
                tags = cfg.log.wandb.get("tags", "none")
                if tags is None:
                    tags = [str(cfg.env.task_id), str(cfg.common.seed)]
                else:
                    tags = [str(tag) for tag in tags]
                name = ",".join(tags)

                wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    group=group_name,
                    tags=tags,
                    dir=self.log_dir,
                    mode=cfg.log.wandb.get("mode", None),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    settings=wandb.Settings(init_timeout=300),
                )
                self.info(f"Logs will be synced with wandb as {project}.")
                self._wandb = wandb
                self.wandb_metrics = cfg.log.wandb.get("metrics", [])
            except wandb.Error as e:
                self.warning("Failed to init wandb. Logs will be saved locally.")
                self.warning(e)
                self._wandb = None
                self.wandb_metrics = None
        else:
            self._wandb = None
            self.wandb_metrics = None

    def update(self, infos_dict: dict):
        self.info_epoch.update(**infos_dict)

    def append(self, scope: str, index: dict, value_dict: dict):
        """Update the statistics for the current episode."""
        if value_dict == {}:
            return
        if scope not in self.infos:
            value_dict = OrderedDict([(key, v) for key, v in value_dict.items()])
            self.infos[scope] = Metrics(index, value_dict)
        else:
            ordered_value_dict = OrderedDict()
            for key in self.infos[scope].values.index:
                if key not in value_dict:
                    self.warning(f"A metric `{key}` is expected to be updated.")
                    ordered_value_dict[key] = 0.0
                else:
                    ordered_value_dict[key] = value_dict[key]

            self.infos[scope].append(index, ordered_value_dict)

    def end_epoch(self, num_samples_total: int):
        """Finalize collected data and add final fixed values.

        Any kwargs passed to end_episode overwrites tracked data if present.
        This can be used to store fixed values that are tracked per episode
        and do not need to be averaged.
        """
        results = {}
        for key, metrics in self.infos.items():
            values = torch.asarray(metrics.values.values).numpy()
            df_values = np.concatenate(
                [torch.asarray(metrics.index.values).numpy(), values], 1
            )
            index = metrics.index.index + metrics.values.index
            df = pd.DataFrame(df_values, columns=index)

            key_split = key.split("/")
            if 1 < len(key_split):
                dir_name = os.path.join(self.log_dir, "metrics", *key_split[:-1])
                key = key_split[-1]
            else:
                dir_name = os.path.join(self.log_dir, "metrics")

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            file_name = os.path.join(dir_name, f"{key}.csv.gz")
            if not os.path.exists(file_name):
                df.to_csv(file_name, compression="gzip", index=False)
            else:
                df.to_csv(
                    file_name, compression="gzip", mode="a", index=False, header=False
                )

            # Tensorboard and console output
            for i, index in enumerate(metrics.values.index):
                index_result = f"{key}/{index}"
                results[index_result] = values[:, i].mean()

        results.update(**self.info_epoch)
        # Log all metrics to Tensorboard
        self.log_to_tbx(results, num_samples_total)
        # Log specified Wan
        self.log_to_wandb(results, num_samples_total)
        self.infos = {}
        return results

    def log_to_tbx(self, results, num_samples_total):
        tmp = results.copy()

        flat_result = flatten_dict(tmp, delimiter="/")
        valid_result = {}

        for attr, value in flat_result.items():
            # print(f"{attr}: {value:.4f}")
            full_attr = "/".join([attr])
            if isinstance(value, (float, np.float32, np.float64)) and not np.isnan(
                value
            ):
                valid_result[full_attr] = value
                self.summary_writer.add_scalar(
                    full_attr, value, global_step=num_samples_total
                )
            else:
                self.warning(f"{attr}={value} is skipped to log into tbx")
        self.summary_writer.flush()

    def log_to_wandb(self, results, total_step):
        if self._wandb:
            tmp = results.copy()

            flat_result = flatten_dict(tmp, delimiter="/")
            target_metrics = {}
            invalid_metric_ids = []
            for i, metric in enumerate(self.wandb_metrics):
                if metric in flat_result:
                    target_metrics[metric] = flat_result[metric]
                else:
                    invalid_metric_ids.append((i, metric))
            self._wandb.log(target_metrics, step=total_step)
            # Remove the invalid metrics from target metrics
            offset = 0
            for idx, metric in invalid_metric_ids:
                self.warning(f"Metrics {metric} is invalid, so we stop to track it")
                self.wandb_metrics.pop(idx - offset)
                offset += 1

    def save_video(self, frames, total_step):
        if self._wandb:
            frames = np.stack(frames).transpose(0, 3, 1, 2)
            self._wandb.log(
                {"eval_video": self._wandb.Video(frames, fps=self.fps, format="mp4")},
                step=total_step,
            )

    def debug(self, msg, **kwargs):
        self.logger.debug(msg, **kwargs)

    def info(self, msg, **kwargs):
        self.logger.info(msg, **kwargs)

    def warning(self, msg, **kwargs):
        self.logger.warning(msg, **kwargs)

    def close(self):
        if self._wandb:
            self._wandb.finish()

    @property
    def log_level(self):
        return self.logger.level
