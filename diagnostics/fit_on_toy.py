import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict

from erl_lib.util.misc import ReplayBuffer, Normalizer, TransitionIterator
from erl_lib.agent.model_based.modules.gaussian_mlp import GaussianMLP, PS_MM
from diagnostics.utils import train, legend


# plt.rcParams["axes.linewidth"] = 0.5  # axis line width
# plt.rcParams["axes.grid"] = True  # make grid

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Latin Modern Roman"]

# plt.grid(True)


# Model
weight_decay_ratios = [0.25, 0.5, 1.0]
# Training
silent = False
device = "cuda"


# %% Data generation
def gen_data(
    y_func,
    train_size,
    n,
    y_scale=1.0,
    mu1=-1,
    range1=1,
    obs_noise1=0.1,
    mu2=1,
    range2=1,
    obs_noise2=0.1,
    num_samples=1000,
    lb=-4,
    ub=4,
):
    # Training data
    train_sub_size = train_size // 2
    w = (ub - lb) * 0.5
    c = (ub + lb) * 0.5
    if mu1 < -1 or 1 < mu1:
        raise ValueError("mu1 must be between -1 and 1")
    if mu2 < -1 or 1 < mu2:
        raise ValueError("mu2 must be between -1 and 1")

    x1 = mu1 + (np.random.rand(train_sub_size) - 0.5) * range1
    # x1 = mu1 + np.random.rand(train_sub_size) * range1
    x1 = np.clip(x1, -1, 1)
    x1 = np.tile(x1[:, None], (1, n))
    x2 = mu2 + (np.random.rand(train_sub_size) - 0.5) * range2
    x2 = np.clip(x2, -1, 1)
    x2 = np.tile(x2[:, None], (1, n))
    x_train = np.vstack([x1, x2])
    x_train = x_train * w + c

    y1 = y_func(x1) + obs_noise1 * np.random.randn(train_sub_size, n)
    y2 = y_func(x2) + obs_noise2 * np.random.randn(train_sub_size, n)
    y_train = np.vstack([y1, y2]) * y_scale

    # Test data
    x_test = np.linspace(-1, 1, num_samples)[:, None]
    x_test = np.tile(x_test, (1, n))
    y_test = y_func(x_test) * y_scale
    x_test = x_test * w + c

    return x_train, y_train, x_test, y_test


def gen_simple_sins(train_size, n, base_freq=4.0, scaled=True, **kwargs):
    def func(x):
        arange = np.arange(1, n + 1)[None, :]

        raw_phase = x.copy()
        raw_phase += arange % 2
        phi = raw_phase * np.pi * base_freq
        out = np.sin(phi)
        if scaled:
            scale = (arange - 0.5) ** 2
            out *= scale
        return out

    return gen_data(func, train_size, n=n, **kwargs)


def generate_data(
    dim_input,
    dim_output,
    train_size,
    num_members=8,
    num_samples=1000,
    path_out=None,
    seed: int = 0,
    x_scale=1.0,
    y_scale=1.0,
    obs_noise1=0.05,
    obs_noise2=0.05,
    **kwargs,
):
    np.random.seed(seed)
    lb = 1 * x_scale
    ub = 3 * x_scale

    x_kwargs = dict(
        lb=lb,
        ub=ub,
        mu1=-0.6,
        mu2=0.5,
        range1=0.6,
        range2=0.5,
    )
    kwargs = dict(
        num_samples=num_samples,
        n=dim_input,
        obs_noise1=obs_noise1,
        obs_noise2=obs_noise2,
        **dict(x_kwargs, **kwargs),
    )
    x_train, y_train, x_test, y_test = gen_simple_sins(
        train_size, y_scale=y_scale, **kwargs
    )

    # Push data to replay buffer
    split_section_dict = OrderedDict(
        obs=dim_input,
        action=0,
        reward=0,
        next_obs=dim_output,
        mask=0,
    )

    buffer = ReplayBuffer(
        capacity=train_size,
        device=device,
        valid_ratio=0.2,
        split_validation=True,
        split_section_dict=split_section_dict,
        num_sample_weights=num_members,
    )

    buffer.add_batch([x_train, y_train])

    data_train, data_val = buffer.split_data()

    x_train = data_train.obs.cpu().numpy()
    x_val = data_val.obs.cpu().numpy()

    y_train = data_train.next_obs
    y_val = data_val.next_obs

    y_train = y_train.cpu().numpy()
    y_val = y_val.cpu().numpy()

    # # Plot the data
    # height = 2 * dim_output
    #
    # fig, axes = plt.subplots(
    #     nrows=dim_output,
    #     ncols=1,
    #     sharex=True,
    #     figsize=(4, height),
    #     # dpi=100
    # )
    #
    # if not isinstance(axes, np.ndarray):
    #     axes = [axes]
    #
    # for idx, ax in enumerate(axes):
    #     ax.plot(x_test[:, idx], y_test[:, idx], c="r", label="Eval")
    #     ax.plot(
    #         x_train[:, idx],
    #         y_train[:, idx],
    #         "x",
    #         c="black",
    #         markersize=6,
    #         label="Train",
    #     )
    #     u = y_test.max()
    #     l = y_test.min()
    #     w = (u - l) * 0.5
    #     c = (u + l) * 0.5
    #     ax.set_ylim(ymax=c + w * 2.5, ymin=c - w * 2.5)
    #     ax.grid(True)
    # ax.set_xlabel("Input axis")
    # ax.legend(loc="lower right")
    #
    # if path_out is not None:
    #     fig.savefig(path_out / "data.pdf")
    # plt.show()

    return data_train, data_val, x_train, y_train, x_test, y_test, x_val, y_val


def pred_ensemble(x_test, model, device):
    x_tensor = torch.from_numpy(x_test).to(device, dtype=torch.float32)
    if model.normalize_input:
        x_tensor = model.input_normalizer.normalize(x_tensor)
    model.eval()

    with torch.no_grad():
        mus, y_pred_logstd = model.base_forward(x_tensor)
        mu, scale = model.forward(x_tensor, prediction_strategy=PS_MM)
        if model.normalize_delta:
            scale_mu, scale_std = (
                model.output_normalizer.mean,
                model.output_normalizer.std,
            )
            mu = scale_mu + mu * scale_std
            scale *= scale_std
        elif model.normalized_target:
            assert model.input_normalizer is not None
            scale_mu, scale_std = (
                model.input_normalizer.mean,
                model.input_normalizer.std,
            )
            mu = scale_mu + mu * scale_std
            scale *= scale_std

        mus = mus.cpu().numpy()
        mu = mu.cpu().numpy()
        scale = scale.cpu().numpy()

    return mus, mu, scale


# %%
def plot_ensemble(
    mus,
    mu,
    scale,
    x_train,
    y_train,
    x_test,
    y_test,
    x_val,
    y_val,
    path_out=None,
    file_name=None,
):
    num_members = mus.shape[0]
    dim_output = y_test.shape[1]
    height = 3 * dim_output
    fig, axes = plt.subplots(
        nrows=dim_output, ncols=1, sharex=True, figsize=(6, height), dpi=100
    )
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, ax in enumerate(axes):
        # Each member model's prediction
        for member in range(num_members):
            y_pred_i = mus[member, :, idx]

            ax.plot(x_test, y_pred_i, c="gray", alpha=0.2)

        # Mean prediction
        y_pred_mean_i = mu[:, idx]
        y_pred_std_i = scale[:, idx]
        y_pred_ub = y_pred_mean_i + 2 * y_pred_std_i
        y_pred_lb = y_pred_mean_i - 2 * y_pred_std_i

        ub_g = y_test[:, idx].max()
        lb_g = y_test[:, idx].min()
        c = (ub_g + lb_g) * 0.5
        w = (ub_g - lb_g) * 0.5
        plot_ub = c + w * 2
        plot_lb = c - w * 2
        ax.set_ylim(plot_lb, plot_ub)

        ax.plot(x_test[:, idx], y_pred_mean_i, "b-", markersize=4)
        ax.fill_between(
            x_test[:, idx],
            y_pred_lb,
            y_pred_ub,
            color="b",
            alpha=0.2,
        )
        # Test data
        ax.plot(x_test[:, idx], y_test[:, idx], "r", label="Test")
        # Training data
        ax.plot(
            x_train[:, idx],
            y_train[:, idx],
            "x",
            markersize=8,
            c="black",
            label="Training",
        )
        # Validation data
        ax.plot(
            x_val[:, idx],
            y_val[:, idx],
            "^",
            markersize=8,
            c="green",
            label="Validation",
        )
        ax.set_ylabel(f"Output")
        ax.grid(True)
    ax.set_xlabel("Input")
    legend(fig, adjust=1)

    if path_out is not None:
        if file_name is None:
            fig.savefig(path_out / "prediction.pdf")
        else:
            fig.savefig(path_out / file_name)
    plt.show()


# %%
def train_each_scale(
    path_out=None,
    file_name=None,
    seed=0,
    max_epoch=10000,
    train_size=20,
    num_members=8,
    pfv=False,
    dropout_rate=0.01,
    layer_norm=1.0,
    output_scale=1.0,
    lr=0.001,
    keep_threshold=0.5,
    improvement_threshold=0.1,
    batch_size_train=8,
    noise_std=0.05,
    train_loss_fn="nll",  # nll or gauss_adapt
):
    min_epoch = max_epoch  # x_scales = (0.01, 1.0, 100.0)
    # y_scales = (0.01, 1.0, 100.0)
    x_scales = (1.0,)
    y_scales = (1.0,)

    dim_input = 1
    dim_output = dim_input

    if path_out is not None:
        path_out = Path(path_out)
        if not path_out.exists():
            path_out.mkdir(exist_ok=True, parents=True)
            print(f"Created dir: {path_out}")

    # for seed in range(num_seeds):
    for y_scale in y_scales:
        for x_scale in x_scales:
            y_scale_str = str(y_scale).replace(".", "_")
            x_scale_str = str(x_scale).replace(".", "_")
            if path_out is not None:
                if len(y_scales) == 1 and len(x_scales) == 1:
                    path_out_s = path_out
                else:
                    path_out_s = path_out / f"y{y_scale_str}_x{x_scale_str}"
                    path_out_s.mkdir(exist_ok=True, parents=True)
                    print(f"Created dir: {path_out_s}")
            else:
                path_out_s = None

            # Data
            (
                data_train,
                data_val,
                x_train,
                y_train,
                x_test,
                y_test,
                x_val,
                y_val,
            ) = generate_data(
                dim_input,
                dim_output,
                train_size,
                num_members=num_members,
                path_out=path_out_s,
                seed=seed,
                x_scale=x_scale,
                y_scale=y_scale,
                scaled=False,
                obs_noise1=noise_std,
                obs_noise2=noise_std,
            )
            input_normalizer = Normalizer(dim_input, device, name="input_normalizer")
            output_normalizer = Normalizer(
                dim_output, device, scale=output_scale, name="output_normalizer"
            )
            input_normalizer.update_stats(x_train)
            output_normalizer.update_stats(y_train)
            input_normalizer.to()
            output_normalizer.to()

            # Train with normalization
            model = train(
                # >>>>>>> master
                data_train,
                data_val,
                input_normalizer,
                output_normalizer,
                dim_input,
                dim_output,
                num_members,
                weight_decay_ratios,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
                seed=seed,
                pfv=pfv,
                # Training
                train_loss_fn=train_loss_fn,
                lr=lr,
                batch_size_train=batch_size_train,
                keep_threshold=keep_threshold,
                improvement_threshold=improvement_threshold,
                min_epoch=min_epoch,
                max_epoch=max_epoch,
            )[0]
            mus, mu, scale = pred_ensemble(x_test, model, device)
            plot_ensemble(
                mus,
                mu,
                scale,
                x_train,
                y_train,
                x_test,
                y_test,
                x_val,
                y_val,
                path_out=path_out_s,
                file_name=file_name,
            )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--path-out", type=str, default=None, help="Path to save results."
    )
    arg_parser.add_argument(
        "--file-name", type=str, default=None, help="File name if specified"
    )
    arg_parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    arg_parser.add_argument(
        "--train-size", type=int, default=20, help="The size of data used for training."
    )
    arg_parser.add_argument(
        "--num-members", type=int, default=8, help="The number of ensemble members."
    )
    arg_parser.add_argument(
        "--pfv",
        action="store_true",
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
        default=10000,
        help="The maximum number of epoch in model training.",
    )
    arg_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    arg_parser.add_argument(
        "--batch-size-train", type=int, default=8, help="The batch size for training."
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
        "--noise-std",
        type=float,
        default=0.05,
        help="The standard deviation of noise added to the data.",
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
