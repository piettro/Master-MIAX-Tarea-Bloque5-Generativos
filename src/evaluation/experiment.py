"""Protocol for comparing datasets with different synthetic-data ratios.

Every generator is evaluated by the same function, `ratio_sweep`, which
guarantees that all four generators are compared under exactly the same
conditions: the same real-data budget, the same ratio multipliers, the same
classifier architecture, the same seeds, and the same test set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.models.classifier import evaluate, optimal_threshold, train_classifier
from src.utils.config import SETTINGS

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

RESULT_COLUMNS: list[str] = [
    "model",
    "ratio",
    "seed",
    "n_real",
    "n_synthetic",
    "pr_auc",
    "pr_auc_val",
    "roc_auc",
    "f1",
    "f1_val",
    "precision",
    "recall",
    "threshold",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "epochs",
]


def mix_real_and_synthetic(
    x_real: FloatArray,
    y_real: FloatArray,
    x_synthetic: FloatArray | None,
    y_synthetic: FloatArray | None,
    ratio: float,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Combine the real budget with a proportional amount of synthetic data.

    The number of synthetic samples added is ``ratio`` times the number of
    real ones. All real data is kept in every configuration, so synthetic
    data always acts as an addition, never as a substitute.

    Args:
        x_real: Real windows.
        y_real: Real labels.
        x_synthetic: Synthetic windows, or ``None`` if unavailable.
        y_synthetic: Synthetic labels, or ``None`` if unavailable.
        ratio: Synthetic-to-real multiplier. Values ``<= 0`` return the real
            data unchanged.
        seed: Random seed for sampling and shuffling. Defaults to
            :data:`src.utils.config.SETTINGS.seed`.

    Returns:
        A tuple ``(x_mixed, y_mixed)``, shuffled.
    """
    if ratio <= 0 or x_synthetic is None or len(x_synthetic) == 0:
        return x_real, y_real

    rng = np.random.default_rng(SETTINGS.seed if seed is None else seed)
    n_synthetic = int(round(ratio * len(x_real)))
    with_replacement = n_synthetic > len(x_synthetic)
    idx = rng.choice(len(x_synthetic), size=n_synthetic, replace=with_replacement)

    x_mixed = np.concatenate([x_real, x_synthetic[idx]], axis=0)
    y_mixed = np.concatenate([y_real, y_synthetic[idx]], axis=0)

    order = rng.permutation(len(x_mixed))
    return x_mixed[order], y_mixed[order]


def ratio_sweep(
    model_name: str,
    x_real: FloatArray,
    y_real: FloatArray,
    x_synthetic: FloatArray | None,
    y_synthetic: FloatArray | None,
    x_val: FloatArray,
    y_val: FloatArray,
    x_test: FloatArray,
    y_test: FloatArray,
    ratios: list[float] | None = None,
    n_seeds: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[float, int], dict[str, list[float]]]]:
    """Train and evaluate the classifier for every ratio and seed combination.

    Args:
        model_name: Short identifier of the generator being evaluated (for
            example ``"cgan"``), stored in the ``model`` column.
        x_real: Real training windows (the fixed budget).
        y_real: Real training labels.
        x_synthetic: Synthetic windows produced by the generator, or
            ``None`` for the ``ratio=0`` baseline.
        y_synthetic: Synthetic labels, or ``None``.
        x_val: Validation windows.
        y_val: Validation labels.
        x_test: Test windows.
        y_test: Test labels.
        ratios: Synthetic-to-real ratios to sweep. Defaults to
            :data:`src.utils.config.SETTINGS.synthetic_ratios`.
        n_seeds: Number of seeds per ratio. Defaults to
            :data:`src.utils.config.SETTINGS.n_seeds`.
        verbose: If ``True``, log one line per (ratio, seed) combination.

    Returns:
        A tuple ``(results, histories)``: ``results`` has one row per
        (ratio, seed) combination with the columns in
        :data:`RESULT_COLUMNS`; ``histories`` maps ``(ratio, seed)`` to the
        Keras training history of that run, needed to document convergence.
    """
    ratios = ratios if ratios is not None else list(SETTINGS.synthetic_ratios)
    n_seeds = n_seeds if n_seeds is not None else SETTINGS.n_seeds

    rows: list[dict[str, Any]] = []
    histories: dict[tuple[float, int], dict[str, list[float]]] = {}

    for ratio in ratios:
        for k in range(n_seeds):
            seed = SETTINGS.seed + k

            x_mixed, y_mixed = mix_real_and_synthetic(
                x_real, y_real, x_synthetic, y_synthetic, ratio, seed
            )
            model, history = train_classifier(x_mixed, y_mixed, x_val, y_val, seed=seed)

            threshold = optimal_threshold(model, x_val, y_val)
            metrics = evaluate(model, x_test, y_test, threshold=threshold)

            # The test set covers a single market regime and its metrics
            # are noisy. Validation results are recorded too, so the effect
            # of synthetic data can be told apart from the effect of the
            # regime shift between the two periods.
            val_metrics = evaluate(model, x_val, y_val, threshold=threshold)
            metrics["pr_auc_val"] = val_metrics["pr_auc"]
            metrics["f1_val"] = val_metrics["f1"]

            metrics.update(
                {
                    "model": model_name,
                    "ratio": float(ratio),
                    "seed": seed,
                    "n_real": int(len(x_real)),
                    "n_synthetic": int(len(x_mixed) - len(x_real)),
                    "epochs": len(history.history["loss"]),
                }
            )
            rows.append(metrics)
            histories[(float(ratio), seed)] = history.history

            if verbose:
                logger.info(
                    "%-10s ratio=%-5.2f seed=%d PR-AUC(test)=%.4f "
                    "PR-AUC(val)=%.4f F1=%.4f",
                    model_name,
                    ratio,
                    seed,
                    metrics["pr_auc"],
                    metrics["pr_auc_val"],
                    metrics["f1"],
                )

    return pd.DataFrame(rows)[RESULT_COLUMNS], histories


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by model and ratio, averaging over seeds."""
    return (
        results.groupby(["model", "ratio"])
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            pr_auc_val_mean=("pr_auc_val", "mean"),
            pr_auc_val_std=("pr_auc_val", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            n_synthetic=("n_synthetic", "max"),
        )
        .reset_index()
    )


def save_results(results: pd.DataFrame, model_name: str) -> Path:
    """Persist a generator's ratio-sweep results to ``results/tables``."""
    from src.utils.config import ensure_directories

    ensure_directories()
    path = SETTINGS.tables_dir / f"results_{model_name}.csv"
    results.to_csv(path, index=False)
    return path


def load_results(model_names: list[str]) -> pd.DataFrame:
    """Gather the ratio-sweep results tables of every available generator.

    Args:
        model_names: Model identifiers to look for.

    Returns:
        The concatenation of every table found.

    Raises:
        FileNotFoundError: If none of the requested tables exist.
    """
    chunks = []
    for name in model_names:
        path = SETTINGS.tables_dir / f"results_{name}.csv"
        if path.exists():
            chunks.append(pd.read_csv(path))
        else:
            logger.warning("Results table not found for '%s' at %s.", name, path)
    if not chunks:
        raise FileNotFoundError("No results table is available.")
    return pd.concat(chunks, ignore_index=True)


def save_synthetic_data(
    model_name: str,
    x_synthetic: FloatArray,
    y_synthetic: FloatArray,
    losses: dict[str, list[float]] | None = None,
) -> Path:
    """Write a generator's synthetic dataset to disk in the agreed format.

    This is the single point of contact between a generator and the rest of
    the pipeline. Any generator that follows this contract slots into the
    comparison without touching shared code.

    Args:
        model_name: Short identifier of the generator (for example
            ``"cgan"``).
        x_synthetic: Synthetic windows.
        y_synthetic: Synthetic labels.
        losses: Optional mapping of loss-curve name to per-epoch or
            per-iteration values, stored with a ``loss_`` prefix.

    Returns:
        The path the file was written to.
    """
    from src.utils.config import ensure_directories

    ensure_directories()
    path = SETTINGS.models_dir / f"synthetic_{model_name}.npz"

    arrays: dict[str, Any] = {
        "x_synthetic": np.asarray(x_synthetic, dtype="float32"),
        "y_synthetic": np.asarray(y_synthetic, dtype="float32"),
    }
    if losses is not None:
        for key, values in losses.items():
            arrays[f"loss_{key}"] = np.asarray(values, dtype="float32")

    np.savez_compressed(path, **arrays)
    return path


def load_synthetic_data(model_name: str) -> dict[str, np.ndarray]:
    """Load the synthetic dataset produced by a generator.

    Raises:
        FileNotFoundError: If no synthetic dataset exists for that model.
    """
    path = SETTINGS.models_dir / f"synthetic_{model_name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No synthetic dataset found at {path}. Train and run "
            f"generator '{model_name}' first."
        )
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def threshold_sensitivity(
    returns: "pd.DataFrame",
    positive_rates: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Measure how the crisis definition changes with the labelling threshold.

    NOTE: added -- required by the professor's feedback recorded in
    materials/class_transcription.txt. During the defense, the professor
    asked directly whether the crisis threshold (fixed at the 10th
    percentile of drawdown, ``config.POSITIVE_RATE``) had been explored, and
    pointed out that a less strict cut would produce more crisis windows.
    This function answers that question without touching the reference
    pipeline: it recomputes the label, the resulting drawdown threshold, and
    the number of independent crisis episodes for each candidate positive
    rate, on the same market-return series the main pipeline uses.

    Args:
        returns: Daily log returns of the investment universe, as produced
            by :func:`src.data.dataset.compute_log_returns`.
        positive_rates: Candidate target positive rates. Defaults to
            :data:`src.utils.config.SETTINGS.threshold_sensitivity_rates`.

    Returns:
        A DataFrame with one row per candidate rate: the resulting drawdown
        threshold, the number of positive windows, and the number of
        independent crisis episodes (runs of consecutive positive windows)
        in the training block.
    """
    from src.data.dataset import (
        build_windows,
        calibrate_threshold,
        future_drawdown,
        label_windows,
        market_series,
        temporal_split,
    )

    positive_rates = positive_rates or SETTINGS.threshold_sensitivity_rates

    market_returns = market_series(returns)
    drawdown = future_drawdown(market_returns)
    _, drawdowns, _ = build_windows(returns, drawdown)
    train_idx, _, _ = temporal_split(len(drawdowns))
    train_drawdowns = drawdowns[train_idx]

    rows: list[dict[str, Any]] = []
    for rate in positive_rates:
        threshold = calibrate_threshold(train_drawdowns, positive_rate=rate)
        labels = label_windows(train_drawdowns, threshold)

        # An episode is a run of consecutive positive windows: since windows
        # overlap by construction, counting positive windows overstates how
        # many genuinely distinct crisis events exist.
        transitions = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
        n_episodes = int((transitions == 1).sum())

        rows.append(
            {
                "positive_rate": rate,
                "drawdown_threshold": threshold,
                "n_positive_windows": int(labels.sum()),
                "n_episodes": n_episodes,
            }
        )

    return pd.DataFrame(rows)
