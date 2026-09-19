"""Download and caching of S&P 500 adjusted close prices.

The full download from Yahoo Finance is slow and depends on an external
service, so it is meant to run once; its result is cached to disk and every
other part of the pipeline reads the cache. `load_price_universe` prefers the
small, version-controlled cache and only falls back to the full raw download
if that cache is missing.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from src.utils.config import SETTINGS

logger = logging.getLogger(__name__)


def download_price_universe(
    start_date: str | None = None,
    tickers_url: str | None = None,
    raw_output_path: Path | None = None,
    universe_output_path: Path | None = None,
) -> pd.DataFrame:
    """Download S&P 500 close prices and cache the full-history subset.

    This performs a real network call to Yahoo Finance through ``yfinance``
    and can take several minutes. It is not required for the rest of the
    pipeline to run: :func:`load_price_universe` will use the
    version-controlled cache produced by a previous call to this function.

    Args:
        start_date: First date to request, in ``YYYY-MM-DD`` format.
            Defaults to :data:`src.utils.config.SETTINGS.start_date`.
        tickers_url: URL of a CSV file whose header row lists the starting
            ticker universe. Defaults to
            :data:`src.utils.config.SETTINGS.tickers_url`.
        raw_output_path: Where to write the full, unfiltered price matrix.
        universe_output_path: Where to write the reduced, full-history
            subset that the rest of the pipeline consumes.

    Returns:
        The full-history subset of adjusted close prices, indexed by date.

    Raises:
        ImportError: If ``yfinance`` is not installed.
        RuntimeError: If the download returns no usable data.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download price data. Install it with "
            "'pip install yfinance' or run the pipeline with the bundled "
            "cache instead (no download needed)."
        ) from exc

    start_date = start_date or SETTINGS.start_date
    tickers_url = tickers_url or SETTINGS.tickers_url
    raw_output_path = raw_output_path or SETTINGS.raw_prices_path
    universe_output_path = universe_output_path or SETTINGS.universe_prices_path

    try:
        tickers = list(pd.read_csv(tickers_url))
    except Exception as exc:  # network/parsing failures from an external URL
        raise RuntimeError(
            f"Could not read the ticker universe from {tickers_url}: {exc}"
        ) from exc

    logger.info("Starting universe: %d tickers.", len(tickers))

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        close_prices = yf.download(
            tickers, start=start_date, auto_adjust=True, progress=True
        )["Close"]

    if close_prices.empty:
        raise RuntimeError("The price download returned an empty DataFrame.")

    logger.info("Downloaded price matrix: %s", close_prices.shape)

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    close_prices.to_csv(raw_output_path)
    logger.info("Raw cache written to %s", raw_output_path)

    survivors = close_prices.dropna(axis=1)
    logger.info(
        "Series with full history since %s: %s", start_date, survivors.shape
    )

    universe_output_path.parent.mkdir(parents=True, exist_ok=True)
    survivors.to_csv(universe_output_path)
    size_mb = universe_output_path.stat().st_size / 1e6
    logger.info(
        "Universe cache written to %s (%.1f MB)", universe_output_path, size_mb
    )

    return survivors


def load_price_universe(
    path: Path | None = None, n_tickers: int | None = None
) -> pd.DataFrame:
    """Load the cached full-history price universe.

    Columns with any missing value are dropped, so the resulting universe is
    free of gaps and needs no imputation. That criterion introduces
    survivorship bias, a limitation documented in the exploratory analysis
    rather than corrected here.

    The reduced cache (already filtered to full-history tickers) is
    preferred because it ships with the repository. The full raw download is
    used only as a fallback.

    Args:
        path: Explicit path to a price cache. If omitted, the
            version-controlled universe cache is tried first, then the raw
            download cache.
        n_tickers: Number of assets to keep. Defaults to
            :data:`src.utils.config.SETTINGS.n_tickers`.

    Returns:
        A DataFrame of prices indexed by date, with exactly ``n_tickers``
        columns (or fewer if the cache does not contain that many).

    Raises:
        FileNotFoundError: If no price cache exists at the resolved path.
    """
    if path is None:
        path = (
            SETTINGS.universe_prices_path
            if SETTINGS.universe_prices_path.exists()
            else SETTINGS.raw_prices_path
        )

    if not path.exists():
        raise FileNotFoundError(
            f"No price cache found at {path}. Run "
            "src.data.loader.download_price_universe() first, or place a "
            "compatible CSV at that path."
        )

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.dropna(axis=1)

    n_tickers = n_tickers or SETTINGS.n_tickers
    if prices.shape[1] > n_tickers:
        # NOTE: the previous dropna already keeps only assets with full
        # history, so all remaining columns cover the same period and there
        # is no principled ordering criterion left to break ties with. This
        # keeps the first `n_tickers` columns in the file, which are in
        # alphabetical order: an arbitrary but stable and therefore
        # reproducible tie-break.
        prices = prices.iloc[:, :n_tickers]

    return prices
