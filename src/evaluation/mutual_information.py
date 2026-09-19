"""Mutual information between window descriptors and the crisis label.

Realism metrics answer whether synthetic samples *look like* real ones;
ratio-sweep results answer whether they *help* the classifier. This module
answers the question that explains both: how much information about the
label each dataset actually carries. A generator can reproduce the marginal
distribution of returns and still say nothing about whether a crisis is
coming, and if it does, no amount of it can help the classifier learn.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

#: Aggregated, interpretable descriptors of a window, in the same order the
#: reference linear model uses (see src.models.classifier and the
#: exploratory analysis). Mutual information is estimated on these instead
#: of on the full flattened window, whose 1,380 components would need far
#: more samples than exist for the estimator to be reliable.
DESCRIPTOR_NAMES: list[str] = [
    "volatility",
    "cumulative_return",
    "drawdown_within_window",
    "worst_session",
    "cross_asset_dispersion",
    "mean_absolute_return",
]


def window_descriptors(x: FloatArray) -> FloatArray:
    """Reduce each window to six aggregated, interpretable magnitudes.

    Args:
        x: Windows of shape ``(n_windows, window_x, n_assets)``.

    Returns:
        An array of shape ``(n_windows, 6)``, columns matching
        :data:`DESCRIPTOR_NAMES`.
    """
    portfolio_return = x.mean(axis=2)
    trajectory = np.cumsum(portfolio_return, axis=1)
    running_max = np.maximum.accumulate(trajectory, axis=1)
    return np.column_stack(
        [
            portfolio_return.std(axis=1),
            portfolio_return.sum(axis=1),
            (trajectory - running_max).min(axis=1),
            portfolio_return.min(axis=1),
            x.std(axis=2).mean(axis=1),
            np.abs(portfolio_return).mean(axis=1),
        ]
    )


def mutual_information_bits(
    x: FloatArray, y: FloatArray, n_samples: int, seed: int
) -> FloatArray:
    """Mutual information, in bits, between each descriptor and the label.

    Args:
        x: Windows to estimate from.
        y: Binary labels aligned with ``x``.
        n_samples: If ``x`` has more rows than this, a random subsample of
            this size is used, so that every dataset compared is evaluated
            on the same number of observations (mutual-information
            estimates grow with sample size, which would otherwise favour
            the more numerous synthetic sets).
        seed: Random seed for subsampling and for the estimator itself.

    Returns:
        An array of shape ``(6,)`` with the mutual information of each
        descriptor in :data:`DESCRIPTOR_NAMES`, in bits.
    """
    rng = np.random.default_rng(seed)
    if len(x) > n_samples:
        idx = rng.choice(len(x), size=n_samples, replace=False)
        x, y = x[idx], y[idx]

    mi_nats = mutual_info_classif(
        window_descriptors(x),
        y.astype(int),
        discrete_features=False,
        n_neighbors=3,
        random_state=seed,
    )
    return mi_nats / np.log(2.0)


def mutual_information_table(
    datasets: list[tuple[str, FloatArray, FloatArray]],
    n_samples: int,
    seed: int,
    n_repetitions: int = 5,
) -> pd.DataFrame:
    """Build the mutual-information comparison table across several datasets.

    Args:
        datasets: A list of ``(display_name, x, y)`` tuples, typically the
            real data first, followed by each generator's synthetic set.
        n_samples: Number of observations each dataset is subsampled to.
        seed: Base random seed; repetition ``k`` uses ``seed + k``.
        n_repetitions: Number of repeated estimates averaged per dataset.
            The estimator is noisy enough that a single pass is not
            reliable; both the mean and its standard deviation are kept.

    Returns:
        A DataFrame indexed by dataset name, with one column per descriptor
        plus ``total`` (sum of per-descriptor mutual information) and
        ``total_std`` (its standard deviation across repetitions).
    """
    per_descriptor: dict[str, FloatArray] = {}
    totals: dict[str, tuple[float, float]] = {}

    for name, x, y in datasets:
        repetitions = np.array(
            [
                mutual_information_bits(x, y, n_samples, seed + k)
                for k in range(n_repetitions)
            ]
        )
        per_descriptor[name] = repetitions.mean(axis=0)
        totals[name] = (
            repetitions.sum(axis=1).mean(),
            repetitions.sum(axis=1).std(ddof=1) if n_repetitions > 1 else 0.0,
        )

    table = pd.DataFrame(per_descriptor, index=DESCRIPTOR_NAMES).T
    table["total"] = [totals[name][0] for name in table.index]
    table["total_std"] = [totals[name][1] for name in table.index]
    return table
