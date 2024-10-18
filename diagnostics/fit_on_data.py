import argparse
from pathlib import Path
import json
import numpy as np
import torch
from omegaconf import OmegaConf

from erl_lib.base.datatype import TransitionBatch
from erl_lib.util.env import make_envs
from erl_lib.util.misc import ReplayBuffer, Normalizer
from erl_lib.agent.model_based.modules.gaussian_mlp import PS_MM
from erl_lib.base import (
    OBS,
    ACTION,
    REWARD,
    MASK,
    NEXT_OBS,
    WEIGHT,
)
from diagnostics.utils import train


# Model
weight_decay_ratios = [0.25, 0.5, 0.75, 1.0]
# Training
silent = False
device = "cuda"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_each_scale(
    # task_name="MBWalker2d",
    path_data=".",
    path_out=None,
    file_prefix=None,
    ckpt_path="checkpoint/0000500000",
    seed=0,
    max_epoch=10000,
    num_members=8,
    pfv=False,
    dropout_rate=0.01,
    layer_norm=1.0,
    output_scale=1.0,
    lr=0.001,
    keep_threshold=0.5,
    improvement_threshold=0.1,
    normalized_target=True,
    learned_reward=False,
    train_loss_fn="nll",
    batch_size=500_000,
    mini_batch_size=2048,
):
    path_data = Path(path_data)
    cfg = OmegaConf.load(f"{path_data}/.hydra/config.yaml")
    if cfg.agent.dynamics_model.num_members != num_members:
        raise NotImplementedError(
            "`--num-members` should be equal to `cfg.agent.dynamics_model.num_members`."
        )

    dim_obs, dim_act = make_envs(cfg.env, 1)[1:3]

    dim_input = dim_obs + dim_act
    if learned_reward:
        dim_output = dim_obs + 1
    else:
        dim_output = dim_obs
    cfg.agent.steps_per_iter = 1000
    cfg.agent.dynamics_model.dim_input = dim_input
    cfg.agent.dynamics_model.dim_output = dim_output
    cfg.agent.dynamics_model.prediction_strategy = PS_MM

    # Data
    split_section_dict = {
        OBS: dim_obs,
        ACTION: dim_act,
        REWARD: 1,
        NEXT_OBS: dim_obs,
        MASK: 1,
        WEIGHT: num_members,
    }
    dir_checkpoint = path_data / ckpt_path
    with open(dir_checkpoint / "replay_buffer.json", "r") as f:
        saved_params = json.load(f)
    num_stored = saved_params["num_stored"]
    dim_data = sum(split_section_dict.values())
    data = torch.load(
        dir_checkpoint / "replay_buffer.pt", map_location=torch.device("cpu")
    )[:, :dim_data]
    # Dataset with split into training / validation
    data_train = data[:batch_size]
    rng = np.random.default_rng(seed=seed)
    batch_idx = rng.permutation(batch_size)
    data_train = data_train[batch_idx]
    batch_size_train = int(0.8 * batch_size)
    batch_train = TransitionBatch(
        data_train[:batch_size_train], split_section_dict, device=device
    )
    data_val = TransitionBatch(
        data_train[batch_size_train:], split_section_dict, device=device
    )
    target_obs = batch_train.next_obs - batch_train.obs
    input_sample = batch_train.obs.cpu().numpy()
    if learned_reward:
        output_sample = torch.cat([batch_train.reward, target_obs], 1).cpu().numpy()
    else:
        output_sample = target_obs.cpu().numpy()
    input_normalizer = Normalizer(dim_obs, device, name="input_normalizer")
    output_normalizer = Normalizer(
        dim_output, device, scale=output_scale, name="output_normalizer"
    )
    input_normalizer.update_stats(input_sample)
    output_normalizer.update_stats(output_sample)
    input_normalizer.to()
    output_normalizer.to()

    # Dataset of out-of-distribution for evaluation
    size_ood = min(num_stored - batch_size, 5000)
    data_eval = data[-size_ood:]
    batch_eval = TransitionBatch(data_eval, split_section_dict, device=device)

    result = train(
        batch_train,
        data_val,
        input_normalizer,
        output_normalizer,
        dim_input,
        dim_output,
        None,
        None,
        batch_eval=batch_eval,
        cfg=cfg.agent.dynamics_model,
        dropout_rate=dropout_rate,
        layer_norm=layer_norm,
        seed=seed,
        pfv=pfv,
        # Training
        lr=lr,
        normalized_target=str2bool(normalized_target),
        learned_reward=learned_reward,
        train_loss_fn=train_loss_fn,
        batch_size_train=mini_batch_size,
        keep_threshold=keep_threshold,
        improvement_threshold=improvement_threshold,
        min_epoch=max_epoch,
        max_epoch=max_epoch,
    )
    train_metric_hist = {
        f"train/{key}": torch.stack(val).cpu().numpy() for key, val in result[1].items()
    }
    val_metric_hist = {
        f"val/{key.split('/', 1)[1]}": torch.stack(val).cpu().numpy()
        for key, val in result[2].items()
    }
    eval_metric_hist = {
        key: torch.stack(val).cpu().numpy() for key, val in result[3].items()
    }
    # val_rmse = torch.stack((result[2]["eval/rmse"])).cpu().numpy()
    # val_nll = torch.stack((result[2]["eval/nll"])).cpu().numpy()
    infos = result[4]
    if path_out is not None:
        path_out = Path(path_out)
        if not path_out.exists():
            path_out.mkdir(exist_ok=True, parents=True)
            print(f"Created dir: {path_out}")
        file_name = (
            "train_val_scores.npz" if file_prefix is None else f"{file_prefix}.npz"
        )
        file_name = path_out / file_name
        np.savez(file_name, **train_metric_hist, **val_metric_hist, **eval_metric_hist)
        # Summary of the learning
        infos = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in infos.info.items()
        }
        with open(path_out / "last_infos.json", "w") as f:
            json.dump(infos, f)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--path-data", type=str, required=True, help="Path to dataset of replay buffer."
    )
    arg_parser.add_argument(
        "--path-out", type=str, default=None, help="Path to save results."
    )
    arg_parser.add_argument(
        "--file-prefix", type=str, default=None, help="Prefix of output file name."
    )
    arg_parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoint/0000500000",
        help="Relative path to checkpoint from `--path-data`.",
    )
    arg_parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    arg_parser.add_argument(
        "--num-members", type=int, default=8, help="The number of ensemble members."
    )
    arg_parser.add_argument("--learned-reward", action="store_true")
    arg_parser.add_argument(
        "--pfv",
        type=str,
        help="Use Priors on Function Value, c.f. eq. (10) and (11) from 'Augmenting Neural Networks with Priors on Function Values'",
    )
    arg_parser.add_argument(
        "--dropout-rate", type=float, default=0.01, help="Dropout rate."
    )
    arg_parser.add_argument(
        "--layer-norm",
        type=float,
        default=1.0,
        help="The value of epsilon used for layer normalization."
        "If the value is 0, layer normalization is not applied.",
    )
    arg_parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help="Output scale for denormalization of model's output.",
    )
    arg_parser.add_argument(
        "--max-epoch",
        type=int,
        default=1000,
        help="The maximum number of epoch in model training.",
    )
    arg_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=10240,
        help="The batch size for training.",
    )
    arg_parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=2048,
        help="The batch size for training.",
    )
    arg_parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.5,
        help="The keep threshold used for early stopping in model training.",
    )
    arg_parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.1,
        help="The improvement threshold used for early stopping in model training.",
    )
    arg_parser.add_argument(
        "--normalized-target",
        choices=["True", "False"],
        default="True",
    )
    arg_parser.add_argument(  # with choice selection
        "--train-loss-fn",
        type=str,
        default="gauss_adapt",
        help="The loss function used for training the model.",
        choices=["nll", "gauss_adapt"],
    )

    args = arg_parser.parse_args()

    train_each_scale(**args.__dict__)
