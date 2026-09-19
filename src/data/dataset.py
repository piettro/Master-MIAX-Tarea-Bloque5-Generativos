"""Construction of the supervised dataset from historical prices.

The full pipeline is: adjusted close prices -> logarithmic returns -> 60-day
sliding windows (X) -> binary label of a severe 30-day-ahead drawdown (y) ->
chronological split with embargo -> scaling. Every step is a pure function of
its inputs, which is what makes the whole pipeline byte-for-byte
reproducible given a fixed seed and a fixed price cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.config import SETTINGS

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]


# --- Returns and cleaning ----------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices into daily logarithmic returns.

    Args:
        prices: Adjusted close prices indexed by date, one column per asset.

    Returns:
        Daily log returns, with the first row (which has no prior price to
        difference against) dropped.
    """
    return np.log(prices).diff().dropna()


def clean_returns(
    returns: pd.DataFrame, n_sigmas: float = 12.0
) -> tuple[pd.DataFrame, int]:
    """Neutralise implausible single-day jumps caused by data errors.

    A single-day move larger than ``n_sigmas`` standard deviations rarely
    corresponds to a genuine market move and is usually a mis-adjusted split
    or dividend. Such values are clipped to the limit, which preserves the
    sign of the move without letting a single bad record dominate the scale
    of the whole series.

    Args:
        returns: Daily log returns, one column per asset.
        n_sigmas: Number of standard deviations beyond which a return is
            considered implausible and clipped.

    Returns:
        A tuple of ``(cleaned_returns, n_clipped_values)``.
    """
    limits = n_sigmas * returns.std()
    n_clipped = int((returns.abs() > limits).sum().sum())
    cleaned = returns.clip(lower=-limits, upper=limits, axis=1)
    return cleaned, n_clipped


# --- Building X and y ---------------------------------------------------------


def market_series(returns: pd.DataFrame) -> pd.Series:
    """Daily return of the equal-weighted portfolio of the universe."""
    return returns.mean(axis=1)


def future_drawdown(
    market_returns: pd.Series, window_y: int | None = None
) -> pd.Series:
    """Maximum drawdown of the portfolio over the following ``window_y`` days.

    For every date, the forward horizon's cumulative trajectory is
    reconstructed and the maximum decline from its own running peak is
    measured. The result is non-positive and expresses the worst chained
    loss of the period.

    Args:
        market_returns: Daily return of the reference portfolio.
        window_y: Forward horizon, in trading days. Defaults to
            :data:`src.utils.config.SETTINGS.window_y`.

    Returns:
        A series of the same length as ``market_returns``, with the last
        ``window_y`` entries set to ``NaN`` (no complete forward horizon is
        available for them).
    """
    window_y = window_y or SETTINGS.window_y
    values = market_returns.to_numpy()
    n = len(values)
    drawdown = np.full(n, np.nan)

    for i in range(n - window_y):
        segment = np.exp(np.cumsum(values[i : i + window_y]))
        running_peak = np.maximum.accumulate(segment)
        drawdown[i] = float((segment / running_peak - 1.0).min())

    return pd.Series(drawdown, index=market_returns.index, name="future_drawdown")


def build_windows(
    returns: pd.DataFrame,
    drawdown: pd.Series,
    window_x: int | None = None,
    window_y: int | None = None,
) -> tuple[FloatArray, FloatArray, pd.DatetimeIndex]:
    """Generate the input windows and the drawdown associated with each one.

    Window ``i`` covers the ``window_x`` days preceding cutoff date ``i``,
    and its label describes what happens over the following ``window_y``
    days. No future observation ever enters the model's input.

    Args:
        returns: Daily log returns, one column per asset.
        drawdown: Forward drawdown series, as produced by
            :func:`future_drawdown`.
        window_x: Number of past trading days per window. Defaults to
            :data:`src.utils.config.SETTINGS.window_x`.
        window_y: Forward horizon in trading days. Defaults to
            :data:`src.utils.config.SETTINGS.window_y`.

    Returns:
        A tuple ``(X, drawdowns, dates)`` where ``X`` has shape
        ``(n_windows, window_x, n_assets)``, ``drawdowns`` has shape
        ``(n_windows,)``, and ``dates`` holds the cutoff date of each window.
    """
    window_x = window_x or SETTINGS.window_x
    window_y = window_y or SETTINGS.window_y

    matrix = returns.to_numpy()
    windows: list[np.ndarray] = []
    drawdowns: list[float] = []
    dates: list[pd.Timestamp] = []

    for i in range(window_x, len(returns) - window_y):
        windows.append(matrix[i - window_x : i])
        drawdowns.append(drawdown.iloc[i])
        dates.append(returns.index[i])

    x_array = np.array(windows, dtype="float32")
    drawdown_array = np.array(drawdowns, dtype="float32")
    return x_array, drawdown_array, pd.DatetimeIndex(dates)


def calibrate_threshold(
    train_drawdowns: FloatArray, positive_rate: float | None = None
) -> float:
    """Find the drawdown threshold that yields the target positive rate.

    The threshold is fit exclusively on the training block. Calibrating it
    on the full sample would leak information from the test period into the
    very definition of the problem.

    Args:
        train_drawdowns: Forward drawdowns restricted to the training block.
        positive_rate: Target share of positive (crisis) windows. Defaults
            to :data:`src.utils.config.SETTINGS.positive_rate`.

    Returns:
        The drawdown threshold (a negative number, or zero) below which a
        window is labelled as a crisis.
    """
    positive_rate = positive_rate or SETTINGS.positive_rate
    return float(np.quantile(train_drawdowns, positive_rate))


def label_windows(drawdowns: FloatArray, threshold: float) -> FloatArray:
    """Label as positive the windows whose forward drawdown is severe enough."""
    return (drawdowns <= threshold).astype("float32")


# --- Chronological split -------------------------------------------------------


def temporal_split(
    n: int,
    train_fraction: float | None = None,
    val_fraction: float | None = None,
    embargo: int | None = None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return train/validation/test indices as consecutive temporal blocks.

    An ``embargo``-long span is discarded between blocks so that no window
    from one split shares market observations with another. Without that
    gap, the overlap between adjacent windows would leak information from
    test into training and inflate every reported metric.

    Args:
        n: Total number of windows.
        train_fraction: Share of windows assigned to training. Defaults to
            :data:`src.utils.config.SETTINGS.train_fraction`.
        val_fraction: Share of windows assigned to validation. Defaults to
            :data:`src.utils.config.SETTINGS.val_fraction`.
        embargo: Number of windows discarded at each split boundary.
            Defaults to :data:`src.utils.config.SETTINGS.embargo`.

    Returns:
        A tuple ``(train_idx, val_idx, test_idx)`` of integer index arrays.
    """
    train_fraction = (
        train_fraction if train_fraction is not None else SETTINGS.train_fraction
    )
    val_fraction = val_fraction if val_fraction is not None else SETTINGS.val_fraction
    embargo = embargo if embargo is not None else SETTINGS.embargo

    train_cut = int(n * train_fraction)
    val_cut = int(n * (train_fraction + val_fraction))

    train_idx = np.arange(0, train_cut)
    val_idx = np.arange(train_cut + embargo, val_cut)
    test_idx = np.arange(val_cut + embargo, n)

    return train_idx, val_idx, test_idx


# --- Scaling --------------------------------------------------------------------


class TanhScaler:
    """Standardise per asset and compress the result into ``[-1, 1]``.

    Every generator in this project ends with a ``tanh`` activation, whose
    range is exactly that interval. Plain standardisation would leave part
    of the probability mass outside the generator's reach, while a min-max
    scaling would collapse most of the distribution near zero because of the
    fat tails characteristic of financial return series. Clipping at
    ``n_sigmas`` standard deviations solves both problems at the cost of
    saturating a very small fraction of values.
    """

    def __init__(self, n_sigmas: float = 4.0) -> None:
        self.n_sigmas = n_sigmas
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def fit(self, x: FloatArray) -> "TanhScaler":
        """Compute the per-asset mean and scale from ``x``.

        Args:
            x: Array of shape ``(n_windows, window_x, n_assets)``.

        Returns:
            ``self``, to allow chaining with :meth:`transform`.
        """
        self.mean_ = x.mean(axis=(0, 1), keepdims=True)
        self.scale_ = x.std(axis=(0, 1), keepdims=True) * self.n_sigmas
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, x: FloatArray) -> FloatArray:
        """Scale ``x`` using previously fitted statistics and clip to [-1, 1].

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TanhScaler must be fitted before calling transform().")
        return np.clip((x - self.mean_) / self.scale_, -1.0, 1.0).astype("float32")

    def inverse_transform(self, x: FloatArray) -> FloatArray:
        """Undo :meth:`transform`, returning data on the original scale."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "TanhScaler must be fitted before calling inverse_transform()."
            )
        return (x * self.scale_ + self.mean_).astype("float32")

    def fit_transform(self, x: FloatArray) -> FloatArray:
        """Fit on ``x`` and immediately transform it."""
        return self.fit(x).transform(x)

    def get_params(self) -> dict[str, Any]:
        """Return the fitted parameters as a plain dictionary."""
        return {"mean": self.mean_, "scale": self.scale_, "n_sigmas": self.n_sigmas}

    @classmethod
    def from_params(
        cls, mean: FloatArray, scale: FloatArray, n_sigmas: float
    ) -> "TanhScaler":
        """Rebuild a fitted scaler from parameters saved to disk."""
        scaler = cls(n_sigmas=float(n_sigmas))
        scaler.mean_ = mean
        scaler.scale_ = scale
        return scaler


# --- Persistence ----------------------------------------------------------------


def save_dataset(path: Path, **arrays: npt.NDArray[Any]) -> None:
    """Write the processed dataset to a compressed ``.npz`` file.

    Args:
        path: Destination path.
        **arrays: Named arrays to persist (splits, labels, dates, scaler
            parameters).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    """Load the processed dataset produced by the data-preparation stage.

    Args:
        path: Path to the ``.npz`` dataset. Defaults to
            :data:`src.utils.config.SETTINGS.dataset_path`.

    Returns:
        A dictionary with the scaled splits, labels, per-window dates, and a
        fitted :class:`TanhScaler` under the ``"scaler"`` key.

    Raises:
        FileNotFoundError: If no dataset exists at the resolved path.
    """
    path = path or SETTINGS.dataset_path
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset found at {path}. Run the dataset-building stage "
            "first (see main.py build-dataset)."
        )

    with np.load(path, allow_pickle=True) as f:
        data = {key: f[key] for key in f.files}

    # Dates are stored as ISO strings. Storing them as integers would be
    # fragile, because the internal unit of datetime64 dtypes depends on the
    # pandas version, and reading with the wrong unit silently shifts the
    # whole series by decades.
    for key in ("dates_train", "dates_val", "dates_test"):
        if key in data:
            data[key] = pd.DatetimeIndex(data[key].astype(str))

    data["scaler"] = TanhScaler.from_params(
        data["scaler_mean"], data["scaler_scale"], data["scaler_n_sigmas"]
    )
    return data


def sample_real_budget(
    x: FloatArray, y: FloatArray, n_real: int, seed: int | None = None
) -> tuple[FloatArray, FloatArray]:
    """Draw a stratified training subset that preserves the positive rate.

    The real-data budget is the variable that creates the scarcity synthetic
    data is meant to compensate for. Stratified sampling ensures the share
    of crisis episodes does not depend on the luck of the draw.

    Args:
        x: Full pool of windows to sample from.
        y: Labels aligned with ``x``.
        n_real: Number of windows to keep. If greater than or equal to
            ``len(x)``, the inputs are returned unchanged.
        seed: Random seed. Defaults to :data:`src.utils.config.SETTINGS.seed`.

    Returns:
        A tuple ``(x_subset, y_subset)`` of length ``n_real`` (or
        ``len(x)`` if no subsampling was needed).
    """
    seed = SETTINGS.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    if n_real >= len(x):
        return x, y

    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]

    n_positive = int(round(n_real * len(positive_idx) / len(y)))
    n_positive = max(1, min(n_positive, len(positive_idx)))
    n_negative = n_real - n_positive

    selection = np.concatenate(
        [
            rng.choice(positive_idx, size=n_positive, replace=False),
            rng.choice(negative_idx, size=n_negative, replace=False),
        ]
    )
    rng.shuffle(selection)
    return x[selection], y[selection]
