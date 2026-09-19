"""Unit tests for src.data.dataset: the part of the pipeline most likely to
silently leak information if a boundary condition is wrong.
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import (
    TanhScaler,
    build_windows,
    calibrate_threshold,
    clean_returns,
    compute_log_returns,
    future_drawdown,
    label_windows,
    market_series,
    sample_real_budget,
    temporal_split,
)


def test_compute_log_returns_shape_and_no_nan(synthetic_price_frame):
    returns = compute_log_returns(synthetic_price_frame)
    assert returns.shape == (len(synthetic_price_frame) - 1, synthetic_price_frame.shape[1])
    assert not returns.isna().any().any()


def test_clean_returns_clips_only_extreme_values(synthetic_price_frame):
    returns = compute_log_returns(synthetic_price_frame)
    returns_with_outlier = returns.copy()
    returns_with_outlier.iloc[10, 0] = 50.0  # an implausible single-day jump

    cleaned, n_clipped = clean_returns(returns_with_outlier, n_sigmas=12.0)

    assert n_clipped == 1
    assert cleaned.iloc[10, 0] < 50.0
    # Every other value must be untouched.
    untouched = returns_with_outlier.drop(returns_with_outlier.index[10])
    assert np.allclose(cleaned.drop(cleaned.index[10]).to_numpy(), untouched.to_numpy())


def test_future_drawdown_is_non_positive(synthetic_price_frame):
    returns = compute_log_returns(synthetic_price_frame)
    drawdown = future_drawdown(market_series(returns), window_y=30)
    valid = drawdown.dropna()
    assert (valid <= 0).all()


def test_build_windows_never_reads_the_future(synthetic_price_frame):
    returns = compute_log_returns(synthetic_price_frame)
    drawdown = future_drawdown(market_series(returns), window_y=30)
    x, drawdowns, dates = build_windows(returns, drawdown, window_x=60, window_y=30)

    # Window i must end exactly at date i and never include a later date.
    assert len(x) == len(drawdowns) == len(dates)
    assert x.shape[1:] == (60, returns.shape[1])
    # No window should extend past len(returns) - window_y.
    assert dates[-1] <= returns.index[-31]


def test_temporal_split_respects_embargo_and_order():
    n = 1000
    train_idx, val_idx, test_idx = temporal_split(
        n, train_fraction=0.7, val_fraction=0.15, embargo=90
    )

    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()
    assert val_idx.min() - train_idx.max() - 1 == 90
    assert test_idx.min() - val_idx.max() - 1 == 90
    # No index should be repeated across splits.
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx.tolist()))


def test_calibrate_threshold_reproduces_target_rate():
    drawdowns = np.linspace(-0.5, 0.0, 1000).astype("float32")
    threshold = calibrate_threshold(drawdowns, positive_rate=0.10)
    labels = label_windows(drawdowns, threshold)
    assert abs(labels.mean() - 0.10) < 0.02


def test_tanh_scaler_round_trip(rng):
    x = rng.normal(0, 5.0, size=(50, 60, 4)).astype("float32")
    scaler = TanhScaler(n_sigmas=4.0)
    scaled = scaler.fit_transform(x)

    assert scaled.min() >= -1.0 and scaled.max() <= 1.0

    restored = scaler.inverse_transform(scaled)
    # Values well within the clipping range should round-trip closely;
    # extreme outliers are expected to be lossy by design (that is the
    # point of clipping).
    within_range = np.abs(scaled) < 0.99
    assert np.allclose(restored[within_range], x[within_range], atol=1e-2)


def test_tanh_scaler_params_round_trip(rng):
    x = rng.normal(0, 2.0, size=(30, 60, 3)).astype("float32")
    scaler = TanhScaler(n_sigmas=3.0).fit(x)
    params = scaler.get_params()

    rebuilt = TanhScaler.from_params(params["mean"], params["scale"], params["n_sigmas"])
    assert np.allclose(rebuilt.transform(x), scaler.transform(x))


def test_sample_real_budget_preserves_positive_rate(small_windows):
    x, y = small_windows
    x_budget, y_budget = sample_real_budget(x, y, n_real=100, seed=1)

    assert len(x_budget) == 100
    assert abs(y_budget.mean() - y.mean()) < 0.05


def test_sample_real_budget_returns_all_data_if_budget_exceeds_pool(small_windows):
    x, y = small_windows
    x_budget, y_budget = sample_real_budget(x, y, n_real=len(x) + 100, seed=1)
    assert len(x_budget) == len(x)
