"""Central configuration for the synthetic financial data generation project.

Every path, hyperparameter, and constant used across the pipeline is defined
here. Centralising configuration guarantees that every classifier
configuration is trained and evaluated under identical conditions, which is
an explicit requirement of the assignment brief: differences observed
between generators must be attributable to the data they contribute, not to
accidental drift in shared parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths ------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"
TABLES_DIR: Path = PROJECT_ROOT / "results" / "tables"

RAW_PRICES_PATH: Path = RAW_DATA_DIR / "prices_close_sp500.csv"

# Reduced, version-controlled cache with the universe already filtered. It
# lets anyone clone the repository and run the full pipeline without
# depending on a download or on Yahoo Finance being reachable.
UNIVERSE_PRICES_PATH: Path = PROCESSED_DATA_DIR / "prices_universe.csv"
DATASET_PATH: Path = PROCESSED_DATA_DIR / "dataset.npz"

TICKERS_URL: str = (
    "https://raw.githubusercontent.com/alfonso-santos/"
    "microcredencial-carteras-python-2023/main/Tema_5_APT/data/sp500_tickers.csv"
)

# --- Dataset construction ----------------------------------------------------

#: Start date for the price download. The further back this goes, the fewer
#: tickers survive the full-history filter, but the more market cycles the
#: universe covers.
START_DATE: str = "1945-01-01"

#: Number of assets in the final universe. The default reproduces the
#: universe size used in the course material. It can be reduced to make
#: training cheaper without touching the rest of the pipeline.
N_TICKERS: int = 23

#: Trading days of history observed by the model.
WINDOW_X: int = 60

#: Forward horizon over which the drawdown label is measured.
WINDOW_Y: int = 30

#: Target share of the positive class. The drawdown threshold that produces
#: this share is calibrated on the training block only and applied unchanged
#: to validation and test.
POSITIVE_RATE: float = 0.10

# --- Temporal split -----------------------------------------------------------

TRAIN_FRACTION: float = 0.70
VAL_FRACTION: float = 0.15

#: Days discarded between blocks. A window covers 60 days of the past and 30
#: of the future, so a 90-day gap prevents any single market observation
#: from appearing on both sides of a split boundary.
EMBARGO: int = WINDOW_X + WINDOW_Y

# --- Experiment ---------------------------------------------------------------

#: Budget of real samples used to train both the generators and the
#: classifier. Limiting the real budget is what creates the scarcity that
#: synthetic data is meant to compensate for.
N_REAL_SAMPLES: int = 3000

#: Multipliers of synthetic data over the real budget.
#:
#: NOTE: improved over the original submission. The correction lecture
#: (materials/class_transcription.txt) has the professor pointing out, live,
#: that the interesting region is between "no synthetics" and "as many
#: synthetics as real data" (ratios 0-1), and that several other teams found
#: their best result there. The coarse grid {0, 0.25, 0.5, 1, 2, 4} used in
#: the delivered notebooks skips most of that region. The finer points below
#: (0.10, 0.75) fill it in without discarding the original grid, so existing
#: result tables remain comparable at the ratios they already cover.
SYNTHETIC_RATIOS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 4.0)

#: Repetitions with a different seed for every configuration.
N_SEEDS: int = 5

SEED: int = 42

# --- Classifier training -------------------------------------------------------

CLASSIFIER_EPOCHS: int = 80
CLASSIFIER_BATCH_SIZE: int = 64
EARLY_STOPPING_PATIENCE: int = 12

# --- Crisis-threshold sensitivity ---------------------------------------------

# NOTE: added -- required by the professor's feedback recorded in
# materials/class_transcription.txt. During the live defense the professor
# asked directly whether the positive-rate threshold had been explored,
# noting that a less strict cut would yield more crisis windows to train on.
# POSITIVE_RATE above is kept as the reference value used throughout the
# main pipeline, so every reported figure stays reproducible; this list is
# consumed only by the dedicated sensitivity analysis implemented in
# src.evaluation.experiment.threshold_sensitivity.
THRESHOLD_SENSITIVITY_RATES: tuple[float, ...] = (0.05, 0.075, 0.10, 0.15, 0.20)

# --- Aesthetics -----------------------------------------------------------------

MATPLOTLIB_STYLE: str = "ggplot"

MODEL_COLORS: dict[str, str] = {
    "baseline": "#4c4c4c",
    "noise": "#8c8c8c",
    "cgan": "#c0392b",
    "cvae": "#2874a6",
    "diffusion": "#1e8449",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Real data only",
    "noise": "Gaussian noise",
    "cgan": "cGAN",
    "cvae": "CVAE",
    "diffusion": "Diffusion",
}


@dataclass(frozen=True)
class Settings:
    """Immutable, importable view of the module-level constants above.

    NOTE: improved over the original submission -- the delivered project
    kept every constant as a bare module-level name (config.N_REALES, and
    so on). That is simple but exposes a mutable global to every caller.
    Wrapping the same values in a frozen dataclass keeps the ergonomics
    (attribute access, a single import) while turning accidental
    cross-run mutation into a raised FrozenInstanceError instead of a
    silent bug. The module-level constants above are kept alongside this
    class for direct, backward-compatible imports.
    """

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR
    raw_prices_path: Path = RAW_PRICES_PATH
    universe_prices_path: Path = UNIVERSE_PRICES_PATH
    dataset_path: Path = DATASET_PATH
    tickers_url: str = TICKERS_URL

    start_date: str = START_DATE
    n_tickers: int = N_TICKERS
    window_x: int = WINDOW_X
    window_y: int = WINDOW_Y
    positive_rate: float = POSITIVE_RATE

    train_fraction: float = TRAIN_FRACTION
    val_fraction: float = VAL_FRACTION
    embargo: int = EMBARGO

    n_real_samples: int = N_REAL_SAMPLES
    synthetic_ratios: tuple[float, ...] = SYNTHETIC_RATIOS
    n_seeds: int = N_SEEDS
    seed: int = SEED

    classifier_epochs: int = CLASSIFIER_EPOCHS
    classifier_batch_size: int = CLASSIFIER_BATCH_SIZE
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    threshold_sensitivity_rates: tuple[float, ...] = THRESHOLD_SENSITIVITY_RATES

    matplotlib_style: str = MATPLOTLIB_STYLE


SETTINGS = Settings()


def ensure_directories() -> None:
    """Create the output folder structure if it does not exist yet.

    Raises:
        OSError: If a directory cannot be created (for example, due to
            insufficient permissions on the target filesystem).
    """
    for directory in (
        SETTINGS.raw_data_dir,
        SETTINGS.processed_data_dir,
        SETTINGS.models_dir,
        SETTINGS.figures_dir,
        SETTINGS.tables_dir,
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Could not create directory {directory}: {exc}") from exc
