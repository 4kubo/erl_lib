import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import OrderedDict

from erl_lib.util.misc import ReplayBuffer, Normalizer, TransitionIterator
from erl_lib.agent.model_based.modules.gaussian_mlp import GaussianMLP, PS_MM
from erl_lib.agent.model_based.model_train.de_trainer import DETrainer


#plt.rcParams["axes.linewidth"] = 0.5  # axis line width
#plt.rcParams["axes.grid"] = True  # make grid

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = ["Latin Modern Roman"]

#plt.grid(True)


# Model
weight_decay_ratios = [0.25, 0.5, 1.0]
# Training
silent = False
device = "cuda"


class FakeLogger:
    log_level = 20

    def update(self, *args, **kwargs):
        pass

    def append(self, scope, index, info):
        self.info = info


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
    mus, mu, scale, x_train, y_train, x_test, y_test, x_val, y_val, path_out=None
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
        fig.savefig(path_out / "prediction.pdf")
    plt.show()


def plot_learning_curve(train_losses, eval_scores, label_prefixs, file_prefix=None):
    fig, ax = plt.subplots(figsize=(4, 3))

    left = 10
    max_loss = -np.inf
    min_loss = np.inf
    for train_loss, eval_score, label_prefix in zip(
        train_losses, eval_scores, label_prefixs
    ):
        num_iter = len(train_loss)
        max_loss = max(
            max(train_loss[left : int(num_iter * 0.1) + left]),
            max(eval_score[left : int(num_iter * 0.1) + left]),
            max_loss,
        )
        min_loss = min(min(train_loss[left:]), min(eval_score[left:]), min_loss)

        ax.plot(train_loss, label=f"{label_prefix} Training loss")
        ax.plot(eval_score, label=f"{label_prefix} Evaluation MSE")

    ax.set_xlim(left=10)
    ax.set_ylim(top=max_loss, bottom=min_loss)

    ax.legend()
    ax.set_ylabel("Training loss / Evaluation MSE")
    ax.set_xlabel("Training epoch")
    ax.grid(True)
    if file_prefix is None:
        fig.show()
    else:
        fig.savefig(f"{file_prefix}_learning_curve.pdf")


def legend(fig, adjust=False):
    options = dict(
        fontsize="medium",
        numpoints=1,
        labelspacing=0,
        columnspacing=1.2,
        handlelength=1.5,
        handletextpad=0.5,
        ncol=4,
        loc="lower center",
    )
    # Find all labels and remove duplicates.
    entries = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries[label] = handle
    leg = fig.legend(entries.values(), entries.keys(), **options)
    leg.get_frame().set_edgecolor("white")
    if adjust is not False:
        pad = adjust if isinstance(adjust, (int, float)) else 0.5
        extent = leg.get_window_extent(fig.canvas.get_renderer())
        extent = extent.transformed(fig.transFigure.inverted())
        yloc, xloc = options["loc"].split()
        y0 = dict(lower=extent.y1, center=0, upper=0)[yloc]
        y1 = dict(lower=1, center=1, upper=extent.y0)[yloc]
        x0 = dict(left=extent.x1, center=0, right=0)[xloc]
        x1 = dict(left=1, center=1, right=extent.x0)[xloc]
        fig.tight_layout(rect=[x0, y0, x1, y1], h_pad=pad, w_pad=pad)


# %%
def train(
    data_train,
    data_val,
    x_test,
    dim_input,
    dim_output,
    num_members,
    dropout_rate,
    layer_norm,
    lr,
    batch_size_train,
    normalize_input=True,
    keep_threshold=0.5,
    improvement_threshold=0.1,
    min_epoch=1000,
    max_epoch=10000,
    seed=0,
    output_scale=1.0,
    train_loss_fn="nll", # nll or gauss_adapt
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_normalizer = Normalizer(dim_input, device, name="input_normalizer")
    output_normalizer = Normalizer(
        dim_output, device, scale=output_scale, name="output_normalizer"
    )

    model = GaussianMLP(
        term_fn=None,
        input_normalizer=input_normalizer,
        output_normalizer=output_normalizer,
        normalize_input=normalize_input,
        batch_size=256,
        dim_input=dim_input,
        dim_output=dim_output,
        device=device,
        num_members=num_members,
        dim_hidden=256,
        drop_rate_base=dropout_rate,
        layer_norm=layer_norm,
        weight_decay_ratios=weight_decay_ratios,
        noise_wd=0.01,
        learned_reward=False,
        delta_prediction=False,
        lb_std=1e-3 / output_scale,
        training_loss_fn=train_loss_fn, # nll or gauss_adapt
    )

    model_trainer = DETrainer(
        model,
        lr=lr,
        keep_threshold=keep_threshold,
        improvement_threshold=improvement_threshold,
        silent=silent,
        logger=FakeLogger(),
    )
    # Preprocess for model training
    target = data_train.next_obs

    input_normalizer.update_stats(data_train.obs.cpu().numpy())
    input_normalizer.to()
    output_normalizer.update_stats(target.cpu().numpy())
    output_normalizer.to()

    iterator_train = TransitionIterator(
        data_train, batch_size_train, shuffle_each_epoch=True, device=device
    )
    iterator_val = TransitionIterator(
        data_val, 256, shuffle_each_epoch=False, device=device
    )

    mus_init, mu_init, scale_init = pred_ensemble(x_test, model, device)
    # Train the model
    train_losses, eval_scores = model_trainer.train(
        iterator_train,
        iterator_val,
        0,
        keep_epochs=min_epoch,
        num_max_epochs=max_epoch,
    )
    # Make prediction
    mus, mu, scale = pred_ensemble(x_test, model, device)
    train_losses = torch.as_tensor(train_losses).cpu().numpy()
    eval_scores = torch.tensor(eval_scores).cpu().numpy()[1:]
    noises = model.print_noise()
    print("Unnormalized noise: ", noises * output_normalizer.std.detach().cpu().numpy())
    
    return model, mus_init, mu_init, scale_init, mus, mu, scale, train_losses, eval_scores


def compute_rmse_loss(model, data, device):
    model.eval()
    with torch.no_grad():
        x = data.obs.cpu().numpy()
        y = data.next_obs.cpu().numpy()

        mus, mu, _ = pred_ensemble(x, model, device)

        loss_mean = ((mu - y) ** 2).sum(axis=1).mean()
        loss_mean = np.sqrt(loss_mean)
        loss_individual = np.sqrt(((mus - y.reshape(1, *y.shape)) ** 2).sum(axis=2).mean())
    
    return loss_mean, loss_individual


def train_each_scale(
    path_out=None,
    seed=0,
    max_epoch=10000,
    train_size=20,
    dropout_rate=0.01,
    layer_norm=1.0,
    output_scale=1.0,
    lr=0.001,
    keep_threshold=0.5,
    improvement_threshold=0.1,
    batch_size_train=8,
    noise_std=0.05,
    train_loss_fn="nll", # nll or gauss_adapt
):
    min_epoch = max_epoch
    num_members = 8
    # x_scales = (0.01, 1.0, 100.0)
    # y_scales = (0.01, 1.0, 100.0)
    x_scales = (1.0,)
    y_scales = (1.0,)

    dim_input = 1
    dim_output = dim_input

    if path_out is not None:
        path_out = Path(path_out)
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

            # Train with normalization
            #result
            
            model, mus_init, mu_init, scale_init, mus, mu, scale, train_losses, eval_scores = train(
                data_train,
                data_val,
                x_test,
                dim_input,
                dim_output,
                num_members,
                seed=seed,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
                output_scale=output_scale,
                # Training
                lr=lr,
                batch_size_train=batch_size_train,
                keep_threshold=keep_threshold,
                improvement_threshold=improvement_threshold,
                min_epoch=min_epoch,
                max_epoch=max_epoch,
                train_loss_fn=train_loss_fn,
            )# (mus_init, mu_init, scale_init), (mus, mu, scale), train_losses, eval_scores

            # Compute RMSE loss
            loss_train_mean, loss_train_individual = compute_rmse_loss(model, data_train, device)
            loss_val_mean, loss_val_individual = compute_rmse_loss(model, data_val, device)
            print(f"Train RMSE loss: {loss_train_mean}")
            print(f"Train individual RMSE loss: {loss_train_individual}")
            print(f"Validation RMSE loss: {loss_val_mean}")
            print(f"Validation individual RMSE loss: {loss_val_individual}")

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
            )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--path-out", type=str, default=None, help="Path to save results."
    )
    arg_parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    arg_parser.add_argument(
        "--train-size", type=int, default=20, help="The size of data used for training."
    )
    arg_parser.add_argument(
        "--num-members", type=int, default=8, help="The number of ensemble members."
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
        help="Output scale for denormalization of model's ouput.",
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
    arg_parser.add_argument( # with choice selection
        "--train-loss-fn",
        type=str,
        default="gauss_adapt",
        help="The loss function used for training the model.",
        choices=["nll", "gauss_adapt"],
    )

    args = arg_parser.parse_args()

    train_each_scale(
        max_epoch=args.max_epoch,
        train_size=args.train_size,
        batch_size_train=args.batch_size_train,
        dropout_rate=args.dropout_rate,
        layer_norm=args.layer_norm,
        output_scale=args.output_scale,
        lr=args.lr,
        keep_threshold=args.keep_threshold,
        improvement_threshold=args.improvement_threshold,
        path_out=args.path_out,
        seed=args.seed,
        noise_std=args.noise_std,
        train_loss_fn=args.train_loss_fn,
    )
