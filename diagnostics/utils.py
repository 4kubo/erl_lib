import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker


import hydra


# from collections import defaultdict
import warnings

# import pathlib
import numpy as np
import torch

# import pandas as pd

from erl_lib.util.misc import ReplayBuffer, Normalizer, TransitionIterator
from erl_lib.agent.model_based.modules.gaussian_mlp import GaussianMLP, PS_MM
from erl_lib.agent.model_based.model_train.de_trainer import DETrainer


#### Plots
def plots(amount, cols=4, size=(2, 2.3), xticks=4, yticks=5, grid=(1, 1), **kwargs):
    rows = int(np.ceil(amount / cols))
    size = (cols * size[0], rows * size[1])
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, **kwargs)
    axes = axes.flatten()
    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(xticks))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(yticks))
        if grid:
            grid = (grid, grid) if not hasattr(grid, "__len__") else grid
            ax.grid(which="both", color="#eeeeee")
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(int(grid[0])))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(int(grid[1])))
            ax.tick_params(which="minor", length=0)
    for ax in axes[amount:]:
        ax.axis("off")
    return fig, axes


def curve(ax, domain, values, low=None, high=None, label=None, order=0, **kwargs):
    finite = np.isfinite(values)
    ax.plot(domain[finite], values[finite], label=label, zorder=1000 - order, **kwargs)
    if low is not None:
        kwargs["lw"] = 0
        kwargs["alpha"] = 0.2
        ax.fill_between(
            domain[finite],
            low[finite],
            high[finite],
            zorder=100 - order,
            **kwargs,
        )


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


def binning(xs, ys, borders, reducer=np.nanmean, fill="nan"):
    assert fill in ("nan", "last", "zeros")
    xs = xs if isinstance(xs, np.ndarray) else np.asarray(xs)
    ys = ys if isinstance(ys, np.ndarray) else np.asarray(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    binned = []
    for start, stop in zip(borders[:-1], borders[1:]):
        left = (xs <= start).sum()
        right = (xs <= stop).sum()
        value = np.nan
        if left < right:
            value = reduce(ys[left:right], reducer)
        if np.isnan(value):
            if fill == "zeros":
                value = 0
            if fill == "last" and binned:
                value = binned[-1]
        binned.append(value)
    return borders[1:], np.array(binned)


def reduce(values, reducer=np.nanmean, *args, **kwargs):
    with warnings.catch_warnings():  # Buckets can be empty.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return reducer(values, *args, **kwargs)


# Model diagnostics
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


def train(
    data_train,
    data_val,
    input_normalizer,
    output_normalizer,
    dim_input,
    dim_output,
    num_members,
    weight_decay_ratios,
    cfg=None,
    dropout_rate=0.01,
    layer_norm=1e-5,
    pfv=False,
    train_loss_fn="nll",
    lr=0.001,
    batch_size_train=2048,
    normalize_input=True,
    keep_threshold=0.5,
    improvement_threshold=0.1,
    min_epoch=1000,
    max_epoch=10000,
    seed=0,
    silent=False,
    device="cuda",
):
    class Logger:
        def update(self, *args, **kwargs):
            pass

        def append(self, scope, index, info):
            self.info = info

    logger = Logger()

    np.random.seed(seed)
    torch.manual_seed(seed)

    if cfg is None:
        model = GaussianMLP(
            term_fn=None,
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
            normalize_input=normalize_input,
            batch_size=0,
            dim_input=dim_input,
            dim_output=dim_output,
            device=device,
            num_members=num_members,
            training_loss_fn=train_loss_fn,
            drop_rate_base=dropout_rate,
            layer_norm=layer_norm,
            dim_hidden=256,
            weight_decay_ratios=weight_decay_ratios,
            noise_wd=0.01,
            learned_reward=False,
            delta_prediction=False,
            prediction_strategy=PS_MM,
            priors_on_function_values=pfv,
        )
    else:
        model = hydra.utils.instantiate(
            cfg,
            batch_size=None,
            term_fn=None,
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
            drop_rate_base=dropout_rate,
            layer_norm=layer_norm,
            priors_on_function_values=pfv,
        )

    model_trainer = DETrainer(
        model,
        lr=lr,
        keep_threshold=keep_threshold,
        improvement_threshold=improvement_threshold,
        silent=silent,
        logger=logger,
        denormalized_mse=True,
    )
    # Preprocess for model training
    iterator_train = TransitionIterator(
        data_train, batch_size_train, shuffle_each_epoch=True, device=device
    )
    iterator_val = TransitionIterator(
        data_val, float("inf"), shuffle_each_epoch=False, device=device
    )

    # Train the model
    train_losses, eval_scores = model_trainer.train(
        iterator_train,
        iterator_val,
        0,
        keep_epochs=min_epoch,
        num_max_epochs=max_epoch,
    )
    return model, train_losses, eval_scores, logger
