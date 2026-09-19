"""Unit tests for src.evaluation.mutual_information."""

from __future__ import annotations

import numpy as np

from src.evaluation.mutual_information import (
    DESCRIPTOR_NAMES,
    mutual_information_bits,
    mutual_information_table,
    window_descriptors,
)


def test_window_descriptors_shape(small_windows):
    x, _ = small_windows
    descriptors = window_descriptors(x)
    assert descriptors.shape == (len(x), len(DESCRIPTOR_NAMES))


def test_mutual_information_is_higher_for_an_informative_label(rng):
    n = 400
    x = rng.normal(0, 0.2, size=(n, 60, 4)).astype("float32")

    # An uninformative label: independent of the data.
    y_random = (rng.random(n) < 0.5).astype("float32")

    # An informative label: strongly tied to the window's own volatility,
    # one of the six descriptors mutual information is estimated over.
    volatility = x.std(axis=(1, 2))
    y_informative = (volatility > np.median(volatility)).astype("float32")

    mi_random = mutual_information_bits(x, y_random, n_samples=n, seed=1).sum()
    mi_informative = mutual_information_bits(x, y_informative, n_samples=n, seed=1).sum()

    assert mi_informative > mi_random


def test_mutual_information_bits_are_non_negative(small_windows):
    x, y = small_windows
    mi = mutual_information_bits(x, y, n_samples=len(x), seed=1)
    assert (mi >= -1e-9).all()  # allow tiny negative numerical noise near zero


def test_mutual_information_table_shape(small_windows, rng):
    x_real, y_real = small_windows
    x_other = rng.normal(size=x_real.shape).astype("float32")
    y_other = (rng.random(len(x_real)) < 0.1).astype("float32")

    table = mutual_information_table(
        [("Real", x_real, y_real), ("Other", x_other, y_other)],
        n_samples=len(x_real),
        seed=1,
        n_repetitions=2,
    )

    assert list(table.index) == ["Real", "Other"]
    assert set(DESCRIPTOR_NAMES).issubset(table.columns)
    assert "total" in table.columns and "total_std" in table.columns
