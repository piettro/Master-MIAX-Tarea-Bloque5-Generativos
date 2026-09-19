"""Shared plotting functions used across the whole pipeline.

Concentrating recurring figures here keeps a consistent look across the
report and avoids repeating the same matplotlib block in every training or
evaluation script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.config import SETTINGS

FloatArray = npt.NDArray[np.float32]


def apply_style() -> None:
    """Set the visual style shared by every figure in the project."""
    plt.style.use(SETTINGS.matplotlib_style)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def save_figure(fig: matplotlib.figure.Figure, name: str) -> Path:
    """Store a figure under ``results/figures`` and return its path."""
    from src.utils.config import ensure_directories

    ensure_directories()
    path = SETTINGS.figures_dir / f"{name}.png"
    fig.savefig(path)
    return path


def loss_curve(
    history: dict[str, Any],
    title: str,
    file_name: str | None = None,
    keys: tuple[str, ...] = ("loss", "val_loss"),
) -> tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the training-loss evolution, evidence of convergence per epoch."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = {"loss": "training", "val_loss": "validation"}

    for key in keys:
        if key in history:
            ax.plot(history[key], label=labels.get(key, key), linewidth=1.6)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()

    if file_name:
        save_figure(fig, file_name)
    return fig, ax


def adversarial_loss_curve(
    discriminator_loss: list[float],
    generator_loss: list[float],
    title: str,
    file_name: str | None = None,
) -> tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the discriminator/generator dynamic of an adversarial model.

    In an adversarial scheme the loss does not decrease monotonically: what
    is sought is an equilibrium in which neither network dominates the
    other for long.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(discriminator_loss, label="discriminator", linewidth=1.4)
    ax.plot(generator_loss, label="generator", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()

    if file_name:
        save_figure(fig, file_name)
    return fig, ax


def compare_trajectories(
    x_real: FloatArray,
    x_synthetic: FloatArray,
    title: str,
    file_name: str | None = None,
    n: int = 6,
    asset: int = 0,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot real vs. synthetic windows for a single asset, side by side.

    This is the first qualitative check for any generator: the series
    produced should look like a return series with volatility clustering,
    not a smooth trajectory nor one saturated at the range's extremes.
    """
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.2), sharey=True)

    for j in range(n):
        axes[0, j].plot(x_real[j, :, asset], linewidth=0.9, color="#2874a6")
        axes[1, j].plot(x_synthetic[j, :, asset], linewidth=0.9, color="#c0392b")
        for i in range(2):
            axes[i, j].set_xticks([])

    axes[0, 0].set_ylabel("real")
    axes[1, 0].set_ylabel("synthetic")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()

    if file_name:
        save_figure(fig, file_name)
    return fig, axes


def compare_distributions(
    x_real: FloatArray,
    x_synthetic: FloatArray,
    title: str,
    file_name: str | None = None,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Compare marginal distribution, volatility clustering, and correlation.

    A generator can reproduce the histogram of returns and still fail on
    the dependence between assets, which is precisely what determines a
    portfolio's behaviour during a stress episode. That is why the
    correlation structure is contrasted as well.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].hist(
        x_real.ravel(), bins=120, density=True, alpha=0.55, label="real", color="#2874a6"
    )
    axes[0].hist(
        x_synthetic.ravel(),
        bins=120,
        density=True,
        alpha=0.55,
        label="synthetic",
        color="#c0392b",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Marginal distribution")
    axes[0].legend()

    def volatility_by_step(x: FloatArray) -> FloatArray:
        return x.std(axis=(0, 2))

    axes[1].plot(volatility_by_step(x_real), label="real", color="#2874a6")
    axes[1].plot(volatility_by_step(x_synthetic), label="synthetic", color="#c0392b")
    axes[1].set_title("Volatility across the window")
    axes[1].set_xlabel("day within the window")
    axes[1].legend()

    real_corr = np.corrcoef(x_real.reshape(-1, x_real.shape[2]).T)
    synthetic_corr = np.corrcoef(x_synthetic.reshape(-1, x_synthetic.shape[2]).T)
    triangle = np.triu_indices_from(real_corr, k=1)

    axes[2].scatter(
        real_corr[triangle], synthetic_corr[triangle], s=8, alpha=0.6, color="#1e8449"
    )
    limits = [-0.2, 1.0]
    axes[2].plot(limits, limits, color="#4c4c4c", linestyle="--", linewidth=1)
    axes[2].set_xlim(limits)
    axes[2].set_ylim(limits)
    axes[2].set_xlabel("real correlation")
    axes[2].set_ylabel("synthetic correlation")
    axes[2].set_title("Cross-asset correlation")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()

    if file_name:
        save_figure(fig, file_name)
    return fig, axes


def ratio_curve(
    summary: pd.DataFrame,
    metric: str = "pr_auc_mean",
    title: str | None = None,
    file_name: str | None = None,
    std_column: str = "pr_auc_std",
) -> tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot the classifier's metric against the amount of synthetic data.

    This is the central figure of the study: the horizontal axis is the
    amount of synthetic data added, the vertical axis is the classifier's
    quality, with one curve per generative model.
    """
    from src.utils.config import MODEL_COLORS, MODEL_DISPLAY_NAMES

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for name, group in summary.groupby("model"):
        group = group.sort_values("ratio")
        color = MODEL_COLORS.get(name)
        label = MODEL_DISPLAY_NAMES.get(name, name)

        ax.plot(
            group["ratio"], group[metric], marker="o", label=label, color=color,
            linewidth=1.8,
        )

        if std_column in group:
            lower = group[metric] - group[std_column].fillna(0)
            upper = group[metric] + group[std_column].fillna(0)
            ax.fill_between(group["ratio"], lower, upper, alpha=0.15, color=color)

    ax.set_xlabel("synthetic data added (multiples of the real budget)")
    ax.set_ylabel(metric.replace("_mean", "").replace("_", " ").upper())
    ax.set_title(title or "Effect of synthetic data on the test set")
    ax.legend()
    fig.tight_layout()

    if file_name:
        save_figure(fig, file_name)
    return fig, ax


def nearest_neighbor_novelty(
    x_real: FloatArray,
    x_synthetic: FloatArray,
    title: str,
    file_name: str | None = None,
    n_sample: int = 800,
    seed: int = 42,
) -> tuple[matplotlib.figure.Figure, plt.Axes, dict[str, float]]:
    """Compare how much novelty a generator adds versus copying the data.

    For each synthetic sample, the distance to the nearest real window is
    measured. As a reference, the same distance is computed between
    distinct real windows. A generator that merely reproduces the training
    set will show much smaller distances than that reference, a sign of
    memorisation; one that greatly exceeds it is producing noise unrelated
    to the data.
    """
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(seed)
    flat_real = x_real.reshape(len(x_real), -1)
    flat_synthetic = x_synthetic.reshape(len(x_synthetic), -1)

    synthetic_idx = rng.choice(
        len(flat_synthetic), size=min(n_sample, len(flat_synthetic)), replace=False
    )
    real_idx = rng.choice(len(flat_real), size=min(n_sample, len(flat_real)), replace=False)

    synthetic_distances = pairwise_distances(
        flat_synthetic[synthetic_idx], flat_real
    ).min(axis=1)

    # The reference measures the distance between distinct real windows;
    # each window's comparison with itself (which would be zero) is
    # excluded.
    real_distances = pairwise_distances(flat_real[real_idx], flat_real)
    real_distances[np.arange(len(real_idx)), real_idx] = np.inf
    real_distances = real_distances.min(axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(
        real_distances, bins=60, density=True, alpha=0.6, color="#2874a6",
        label="between real windows",
    )
    ax.hist(
        synthetic_distances, bins=60, density=True, alpha=0.6, color="#c0392b",
        label="synthetic to nearest real",
    )
    ax.set_xlabel("Euclidean distance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if file_name:
        save_figure(fig, file_name)

    summary = {
        "mean_synthetic_distance": float(synthetic_distances.mean()),
        "mean_real_distance": float(real_distances.mean()),
        "ratio": float(synthetic_distances.mean() / real_distances.mean()),
    }
    return fig, ax, summary
