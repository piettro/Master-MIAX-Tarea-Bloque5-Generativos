"""Unit tests for src.evaluation.statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.statistics import (
    build_comparison_table,
    paired_significance_test,
    summarize_significant_effects,
)


def _make_results(deltas: dict[float, float], seeds: int = 5) -> pd.DataFrame:
    """Build a fake results table where seed k's PR-AUC at each ratio is a
    fixed baseline plus that ratio's delta, so the true effect is known.
    """
    rows = []
    for seed in range(seeds):
        baseline = 0.20 + 0.01 * seed
        for ratio, delta in deltas.items():
            rows.append(
                {
                    "model": "fake_model",
                    "ratio": ratio,
                    "seed": seed,
                    "pr_auc": baseline + delta,
                    "pr_auc_val": baseline + delta,
                }
            )
    return pd.DataFrame(rows)


def test_paired_significance_detects_a_consistent_effect():
    results = _make_results({0.0: 0.0, 1.0: 0.10})  # a large, consistent lift
    contrasts = paired_significance_test(results)

    row = contrasts[contrasts["ratio"] == 1.0].iloc[0]
    assert row["delta_test"] == pytest.approx(0.10)
    assert row["p_test"] < 0.05


def test_paired_significance_no_effect_when_deltas_are_zero():
    results = _make_results({0.0: 0.0, 0.5: 0.0})
    contrasts = paired_significance_test(results)

    row = contrasts[contrasts["ratio"] == 0.5].iloc[0]
    assert abs(row["delta_test"]) < 1e-9


def test_summarize_significant_effects_reports_only_significant_rows():
    results = pd.concat(
        [
            _make_results({0.0: 0.0, 1.0: 0.10}),
        ]
    )
    contrasts = paired_significance_test(results)
    lines = summarize_significant_effects(contrasts, alpha=0.05)
    assert len(lines) == 1
    assert "improvement" in lines[0]


def test_build_comparison_table_picks_the_best_ratio():
    summary = pd.DataFrame(
        {
            "model": ["fake_model"] * 3,
            "ratio": [0.0, 0.25, 1.0],
            "pr_auc_mean": [0.20, 0.22, 0.35],
            "pr_auc_std": [0.02, 0.02, 0.03],
            "pr_auc_val_mean": [0.30, 0.31, 0.33],
            "pr_auc_val_std": [0.01, 0.01, 0.01],
        }
    )
    contrasts = pd.DataFrame(
        {
            "model": ["fake_model", "fake_model"],
            "ratio": [0.25, 1.0],
            "p_test": [0.5, 0.01],
            "delta_test": [0.02, 0.15],
            "p_val": [0.5, 0.2],
            "delta_val": [0.01, 0.03],
        }
    )

    table = build_comparison_table(summary, contrasts, {"fake_model": "Fake Model"})
    row = table.iloc[0]

    assert row["model"] == "Fake Model"
    assert row["best_ratio"] == 1.0
    assert row["p_value"] == 0.01
