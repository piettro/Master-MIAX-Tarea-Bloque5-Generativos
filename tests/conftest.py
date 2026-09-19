"""Shared pytest fixtures: small synthetic data, no network or heavy compute.

Every fixture here avoids downloading real price data or training anything
expensive, so the suite runs in seconds and exercises pure logic and cheap
model calls only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# Set before any test module gets a chance to import a src module that
# imports Keras, so the PyTorch backend is always selected regardless of
# test collection order.
os.environ.setdefault("KERAS_BACKEND", "torch")


@pytest.fixture()
def rng() -> np.random.Generator:
    """A deterministic random-number generator for test data."""
    return np.random.default_rng(0)


@pytest.fixture()
def synthetic_price_frame(rng: np.random.Generator) -> pd.DataFrame:
    """A small, fully-populated price DataFrame: 400 days, 5 assets."""
    n_days, n_assets = 400, 5
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    log_returns = rng.normal(0.0003, 0.01, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)


@pytest.fixture()
def small_windows(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A small set of (window, label) pairs shaped like the real pipeline's."""
    n, window_x, n_assets = 200, 60, 6
    x = rng.normal(0, 0.3, (n, window_x, n_assets)).astype("float32")
    y = (rng.random(n) < 0.10).astype("float32")
    x[y == 1] *= 2.0  # give the positive class a distinguishable signature
    return np.clip(x, -1.0, 1.0).astype("float32"), y
