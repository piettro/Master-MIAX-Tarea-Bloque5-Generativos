"""Statistical significance testing and the final comparison table.

Five seeds per configuration and a test set covering a single market regime
mean many apparent differences do not survive scrutiny. This module runs a
paired contrast (paired by seed, against that seed's own ratio-0 reference)
for every generator and ratio, and builds the summary table the final report
relies on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from src.utils.config import SETTINGS

logger = logging.getLogger(__name__)

SIGNIFICANCE_LEVEL: float = 0.05


def paired_significance_test(results: pd.DataFrame) -> pd.DataFrame:
    """Contrast every (model, ratio) pair against its own ratio-0 reference.

    The contrast is a paired t-test, paired by seed: the same seed produces
    one ratio-0 run and one run at each other ratio, so pairing removes the
    seed-to-seed variability that otherwise dominates this problem and
    isolates the effect of the added synthetic data.

    Args:
        results: Concatenated ratio-sweep results for every generator, as
            produced by :func:`src.evaluation.experiment.load_results`.

    Returns:
        A DataFrame with one row per (model, ratio > 0) combination: the
        mean difference and p-value on both test and validation PR-AUC.
    """
    generative_models = [m for m in results["model"].unique() if m != "baseline"]
    rows: list[dict[str, Any]] = []

    for name in generative_models:
        model_results = results[results["model"] == name]
        reference = model_results[model_results["ratio"] == 0].sort_values("seed")

        for ratio in sorted(model_results["ratio"].unique()):
            if ratio == 0:
                continue
            group = model_results[model_results["ratio"] == ratio].sort_values("seed")

            _, p_test = stats.ttest_rel(
                group["pr_auc"].to_numpy(), reference["pr_auc"].to_numpy()
            )
            _, p_val = stats.ttest_rel(
                group["pr_auc_val"].to_numpy(), reference["pr_auc_val"].to_numpy()
            )
            rows.append(
                {
                    "model": name,
                    "ratio": ratio,
                    "delta_test": group["pr_auc"].mean() - reference["pr_auc"].mean(),
                    "p_test": float(p_test),
                    "delta_val": (
                        group["pr_auc_val"].mean() - reference["pr_auc_val"].mean()
                    ),
                    "p_val": float(p_val),
                }
            )

    return pd.DataFrame(rows)


def save_significance_table(contrasts: pd.DataFrame) -> Path:
    """Persist the paired-contrast table to ``results/tables/contrasts.csv``."""
    from src.utils.config import ensure_directories

    ensure_directories()
    path = SETTINGS.tables_dir / "contrasts.csv"
    contrasts.to_csv(path, index=False)
    return path


def summarize_significant_effects(
    contrasts: pd.DataFrame, alpha: float = SIGNIFICANCE_LEVEL
) -> list[str]:
    """Format the statistically significant rows as human-readable lines.

    Args:
        contrasts: Output of :func:`paired_significance_test`.
        alpha: Significance threshold.

    Returns:
        One string per significant effect, for logging or reporting; an
        empty list if none reach significance.
    """
    significant = contrasts[
        (contrasts["p_test"] < alpha) | (contrasts["p_val"] < alpha)
    ]
    lines = []
    for _, row in significant.iterrows():
        which = "test" if row["p_test"] < alpha else "validation"
        delta = row["delta_test"] if row["p_test"] < alpha else row["delta_val"]
        p_value = min(row["p_test"], row["p_val"])
        direction = "improvement" if delta > 0 else "degradation"
        lines.append(
            f"{row['model']:10s} ratio {row['ratio']:<5g} {direction} of "
            f"{delta:+.4f} on {which} (p={p_value:.3f})"
        )
    return lines


def build_comparison_table(
    summary: pd.DataFrame, contrasts: pd.DataFrame, display_names: dict[str, str]
) -> pd.DataFrame:
    """Build the one-row-per-generator table that anchors the final report.

    Args:
        summary: Output of :func:`src.evaluation.experiment.summarize`.
        contrasts: Output of :func:`paired_significance_test`.
        display_names: Mapping from internal model key to a display name,
            typically :data:`src.utils.config.SETTINGS`-adjacent
            ``MODEL_DISPLAY_NAMES``.

    Returns:
        A DataFrame sorted by descending best test PR-AUC, with one row per
        generator: its no-synthetics reference, its best PR-AUC and the
        ratio it was reached at, the resulting variation, the p-value of
        that specific ratio, and the corresponding validation figures.
    """
    generative_models = [m for m in summary["model"].unique() if m != "baseline"]
    rows: list[dict[str, Any]] = []

    for name in generative_models:
        model_summary = summary[summary["model"] == name].sort_values("ratio")
        baseline_row = model_summary[model_summary["ratio"] == 0].iloc[0]
        best_row = model_summary.loc[model_summary["pr_auc_mean"].idxmax()]

        matching_contrast = contrasts[
            (contrasts["model"] == name) & (contrasts["ratio"] == best_row["ratio"])
        ]

        rows.append(
            {
                "model": display_names.get(name, name),
                "pr_auc_test_no_synthetics": baseline_row["pr_auc_mean"],
                "best_pr_auc_test": best_row["pr_auc_mean"],
                "best_ratio": best_row["ratio"],
                "variation": best_row["pr_auc_mean"] - baseline_row["pr_auc_mean"],
                "p_value": (
                    float(matching_contrast["p_test"].iloc[0])
                    if len(matching_contrast)
                    else float("nan")
                ),
                "pr_auc_val_no_synthetics": baseline_row["pr_auc_val_mean"],
                "pr_auc_val_at_best_ratio": best_row["pr_auc_val_mean"],
                "seed_std": best_row["pr_auc_std"],
            }
        )

    return pd.DataFrame(rows).sort_values("best_pr_auc_test", ascending=False)


def save_comparison_table(comparison: pd.DataFrame) -> Path:
    """Persist the final comparison table to ``results/tables``."""
    from src.utils.config import ensure_directories

    ensure_directories()
    path = SETTINGS.tables_dir / "final_comparison.csv"
    comparison.to_csv(path, index=False)
    return path
