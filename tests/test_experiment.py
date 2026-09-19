"""Unit tests for src.evaluation.experiment."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.experiment import (
    load_synthetic_data,
    mix_real_and_synthetic,
    ratio_sweep,
    save_synthetic_data,
    summarize,
)


def test_mix_with_zero_ratio_returns_real_data_unchanged(small_windows):
    x, y = small_windows
    x_mixed, y_mixed = mix_real_and_synthetic(x, y, None, None, ratio=0.0)
    assert x_mixed is x
    assert y_mixed is y


def test_mix_produces_expected_sample_count(small_windows, rng):
    x_real, y_real = small_windows
    x_synth = rng.normal(size=(500, 60, 6)).astype("float32")
    y_synth = (rng.random(500) < 0.1).astype("float32")

    x_mixed, y_mixed = mix_real_and_synthetic(x_real, y_real, x_synth, y_synth, ratio=2.0, seed=1)

    assert len(x_mixed) == len(x_real) + 2 * len(x_real)
    assert len(y_mixed) == len(x_mixed)


def test_mix_keeps_every_real_sample(small_windows, rng):
    x_real, y_real = small_windows
    x_synth = rng.normal(size=(100, 60, 6)).astype("float32")
    y_synth = (rng.random(100) < 0.1).astype("float32")

    x_mixed, _ = mix_real_and_synthetic(x_real, y_real, x_synth, y_synth, ratio=1.0, seed=1)

    # Every real window must still be findable in the mixed set.
    flat_mixed = {tuple(row.ravel().round(6)) for row in x_mixed}
    for row in x_real:
        assert tuple(row.ravel().round(6)) in flat_mixed


def test_ratio_sweep_produces_one_row_per_ratio_and_seed(small_windows):
    x, y = small_windows
    x_train, x_val, x_test = x[:100], x[100:150], x[150:]
    y_train, y_val, y_test = y[:100], y[100:150], y[150:]

    results, histories = ratio_sweep(
        "test_model",
        x_train,
        y_train,
        None,
        None,
        x_val,
        y_val,
        x_test,
        y_test,
        ratios=[0.0],
        n_seeds=2,
        verbose=False,
    )

    assert len(results) == 2
    assert set(results["model"]) == {"test_model"}
    assert len(histories) == 2


def test_ratio_sweep_ratio_zero_never_adds_synthetic_samples(small_windows, rng):
    x, y = small_windows
    x_train, x_val, x_test = x[:100], x[100:150], x[150:]
    y_train, y_val, y_test = y[:100], y[100:150], y[150:]
    x_synth = rng.normal(size=(200, 60, 6)).astype("float32")
    y_synth = (rng.random(200) < 0.1).astype("float32")

    results, _ = ratio_sweep(
        "test_model",
        x_train,
        y_train,
        x_synth,
        y_synth,
        x_val,
        y_val,
        x_test,
        y_test,
        ratios=[0.0],
        n_seeds=1,
        verbose=False,
    )

    assert (results["n_synthetic"] == 0).all()


def test_summarize_aggregates_over_seeds():
    import pandas as pd

    results = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "ratio": [0.0, 0.0, 1.0, 1.0],
            "pr_auc": [0.2, 0.3, 0.4, 0.5],
            "pr_auc_val": [0.2, 0.3, 0.4, 0.5],
            "f1": [0.1, 0.2, 0.3, 0.4],
            "recall": [0.1] * 4,
            "precision": [0.1] * 4,
            "n_synthetic": [0, 0, 100, 100],
        }
    )
    summary = summarize(results)
    assert len(summary) == 2
    row_ratio_0 = summary[summary["ratio"] == 0.0].iloc[0]
    assert row_ratio_0["pr_auc_mean"] == pytest.approx(0.25)


def test_save_and_load_synthetic_data_round_trip(small_windows, tmp_path, monkeypatch):
    # SETTINGS is an immutable, frozen dataclass instance, and
    # src.evaluation.experiment imported it by name (`from ... import
    # SETTINGS`), so the module-level symbol -- not the frozen instance --
    # is what needs replacing here.
    import dataclasses

    from src.utils.config import SETTINGS

    patched_settings = dataclasses.replace(SETTINGS, models_dir=tmp_path)
    monkeypatch.setattr("src.evaluation.experiment.SETTINGS", patched_settings)

    x, y = small_windows
    path = save_synthetic_data("unit_test_model", x, y, losses={"total": [1.0, 0.5]})
    assert path.exists()

    reloaded = load_synthetic_data("unit_test_model")
    assert np.array_equal(reloaded["x_synthetic"], x)
    assert np.array_equal(reloaded["y_synthetic"], y)
    assert np.allclose(reloaded["loss_total"], [1.0, 0.5])


def test_load_synthetic_data_missing_raises(tmp_path, monkeypatch):
    import dataclasses

    from src.utils.config import SETTINGS

    patched_settings = dataclasses.replace(SETTINGS, models_dir=tmp_path)
    monkeypatch.setattr("src.evaluation.experiment.SETTINGS", patched_settings)

    with pytest.raises(FileNotFoundError):
        load_synthetic_data("does_not_exist")
