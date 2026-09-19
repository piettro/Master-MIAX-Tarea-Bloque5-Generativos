"""Reference classifier and the single training/evaluation protocol.

The assignment brief requires every configuration compared in the study to
share one architecture, so that differences observed between generators are
attributable to the data they contribute and not to accidental drift in the
model itself. Defining the architecture in one place, and requiring every
caller to go through :func:`train_classifier` and :func:`evaluate`, is what
enforces that requirement in code rather than in convention.
"""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402  (KERAS_BACKEND must be set before importing keras)
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import keras  # noqa: E402
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D  # noqa: E402
from keras.models import Sequential  # noqa: E402

from src.utils.config import SETTINGS  # noqa: E402

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

#: Reference configuration. Every classifier used in the generator
#: comparison shares this architecture, and it is also the starting point
#: that the architecture search below explores around.
REFERENCE_ARCHITECTURE: dict[str, Any] = {
    "filters": (64, 128, 128),
    "kernel_size": 3,
    "dense_units": 100,
    "dropout": 0.3,
}

#: The assignment brief asks, in addition to training and evaluation, for a
#: search over a valid architecture using the real data. The grid sweeps the
#: four decisions with the most influence on network capacity -- depth,
#: convolutional block width, kernel size, and regularisation -- around the
#: reference configuration, which is flagged in the resulting table.
ARCHITECTURE_GRID: list[dict[str, Any]] = [
    {"filters": (32, 64, 64), "kernel_size": 3, "dense_units": 100, "dropout": 0.3},
    {"filters": (64, 128, 128), "kernel_size": 3, "dense_units": 100, "dropout": 0.3},
    {"filters": (64, 128, 256), "kernel_size": 3, "dense_units": 100, "dropout": 0.3},
    {"filters": (64, 128, 128), "kernel_size": 5, "dense_units": 100, "dropout": 0.3},
    {"filters": (64, 128, 128), "kernel_size": 3, "dense_units": 50, "dropout": 0.3},
    {"filters": (64, 128, 128), "kernel_size": 3, "dense_units": 200, "dropout": 0.3},
    {"filters": (64, 128, 128), "kernel_size": 3, "dense_units": 100, "dropout": 0.1},
    {"filters": (64, 128, 128), "kernel_size": 3, "dense_units": 100, "dropout": 0.5},
    {"filters": (32, 64), "kernel_size": 3, "dense_units": 100, "dropout": 0.3},
    {"filters": (64, 128), "kernel_size": 5, "dense_units": 50, "dropout": 0.5},
]


def build_classifier(
    n_steps: int,
    n_assets: int,
    seed: int | None = None,
    filters: tuple[int, ...] | None = None,
    kernel_size: int | None = None,
    dense_units: int | None = None,
    dropout: float | None = None,
) -> keras.Model:
    """Build the one-dimensional convolutional classifier used throughout.

    The structure is a stack of convolution-and-pooling blocks that extract
    local patterns from the temporal window, followed by a dense head. The
    only problem-specific choice is the output layer, a single sigmoid unit,
    since this is binary classification rather than regression.

    Every hyperparameter is optional; omitted, it reproduces the reference
    configuration exactly. The architecture search below sweeps them, but no
    other part of the pipeline does, so the generator comparison always uses
    one fixed architecture.

    Args:
        n_steps: Number of time steps per input window.
        n_assets: Number of assets (channels) per input window.
        seed: If given, seeds every relevant random-number generator before
            building the model, for reproducible weight initialisation.
        filters: Number of filters in each convolutional block.
        kernel_size: Convolution kernel size, shared by every block.
        dense_units: Width of the dense layer before the output.
        dropout: Dropout rate applied after the dense layer.

    Returns:
        A compiled Keras model with binary cross-entropy loss and PR-AUC as
        its tracked metric.
    """
    if seed is not None:
        keras.utils.set_random_seed(seed)

    filters = REFERENCE_ARCHITECTURE["filters"] if filters is None else filters
    kernel_size = (
        REFERENCE_ARCHITECTURE["kernel_size"] if kernel_size is None else kernel_size
    )
    dense_units = (
        REFERENCE_ARCHITECTURE["dense_units"] if dense_units is None else dense_units
    )
    dropout = REFERENCE_ARCHITECTURE["dropout"] if dropout is None else dropout

    layers_stack: list[Any] = [Input(shape=(n_steps, n_assets))]
    for n_filters in filters:
        layers_stack.append(
            Conv1D(filters=n_filters, kernel_size=kernel_size, activation="relu")
        )
        layers_stack.append(MaxPooling1D(pool_size=2))
    layers_stack += [
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout),
        Dense(1, activation="sigmoid"),
    ]

    model = Sequential(layers_stack, name="drawdown_classifier")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def class_weights(y: FloatArray) -> dict[int, float] | None:
    """Compute weights inversely proportional to each class's frequency.

    With a positive rate near ten percent, unweighted cross-entropy tends
    toward the trivial solution of always predicting the majority class.
    The same weighting is applied to every configuration compared, so it
    favours none of them.

    Args:
        y: Binary labels.

    Returns:
        A dictionary mapping ``{0: weight, 1: weight}``, or ``None`` if one
        of the classes is entirely absent from ``y``.
    """
    n_positive = float((y == 1).sum())
    n_negative = float((y == 0).sum())
    if n_positive == 0 or n_negative == 0:
        return None
    total = n_positive + n_negative
    return {0: total / (2.0 * n_negative), 1: total / (2.0 * n_positive)}


def train_classifier(
    x_train: FloatArray,
    y_train: FloatArray,
    x_val: FloatArray,
    y_val: FloatArray,
    seed: int | None = None,
    verbose: int = 0,
    hyperparameters: dict[str, Any] | None = None,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train the classifier with early stopping on validation PR-AUC.

    Early stopping watches the area under the precision-recall curve on
    validation, the appropriate metric when the class of interest is a
    minority, and restores the best epoch's weights so the result does not
    depend on the exact moment training was stopped.

    Args:
        x_train: Training windows.
        y_train: Training labels.
        x_val: Validation windows.
        y_val: Validation labels.
        seed: Random seed for reproducible initialisation.
        verbose: Keras verbosity level passed to ``model.fit``.
        hyperparameters: Optional architecture overrides, as produced by one
            row of :data:`ARCHITECTURE_GRID`.

    Returns:
        A tuple ``(model, history)`` with the trained model and its Keras
        training history.
    """
    model = build_classifier(
        x_train.shape[1],
        x_train.shape[2],
        seed=seed,
        **(hyperparameters or {}),
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=SETTINGS.early_stopping_patience,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=SETTINGS.classifier_epochs,
        batch_size=SETTINGS.classifier_batch_size,
        class_weight=class_weights(y_train),
        callbacks=[early_stopping],
        verbose=verbose,
    )
    return model, history


def optimal_threshold(model: keras.Model, x_val: FloatArray, y_val: FloatArray) -> float:
    """Find the decision threshold that maximises F1 on validation.

    The default cut of 0.5 is inappropriate under class imbalance and
    class-weighted training. The threshold is tuned on validation and then
    applied to test without being revisited.

    Args:
        model: A trained classifier.
        x_val: Validation windows.
        y_val: Validation labels.

    Returns:
        The threshold in ``[0.05, 0.95]`` that maximises F1 on validation.
    """
    val_probabilities = model.predict(x_val, verbose=0).ravel()
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_val, (val_probabilities >= t).astype(int), zero_division=0)
        for t in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def evaluate(
    model: keras.Model, x_test: FloatArray, y_test: FloatArray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute test-set metrics for an already-trained classifier.

    Args:
        model: A trained classifier.
        x_test: Evaluation windows.
        y_test: Evaluation labels.
        threshold: Decision threshold, typically produced by
            :func:`optimal_threshold`.

    Returns:
        A dictionary with PR-AUC, ROC-AUC, F1, precision, recall, the
        threshold used, and the four confusion-matrix counts.
    """
    test_probabilities = model.predict(x_test, verbose=0).ravel()
    y_pred = (test_probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    true_neg, false_pos, false_neg, true_pos = matrix.ravel()

    return {
        "pr_auc": float(average_precision_score(y_test, test_probabilities)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "true_negatives": int(true_neg),
        "false_positives": int(false_pos),
        "false_negatives": int(false_neg),
        "true_positives": int(true_pos),
    }


def search_architecture(
    x_train: FloatArray,
    y_train: FloatArray,
    x_val: FloatArray,
    y_val: FloatArray,
    grid: list[dict[str, Any]] | None = None,
    n_seeds: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep the architecture grid and rank configurations by validation PR-AUC.

    The search uses only the real-data budget and the validation split;
    **the test set never takes part**, so the chosen architecture is not
    contaminated by the period later reported on.

    These figures should be read knowing that the same validation split also
    drives early stopping and the decision threshold, so they are optimistic
    in absolute terms. What the table supports is ranking configurations
    against each other, which is what it is used for.

    Args:
        x_train: Training windows (the real-data budget).
        y_train: Training labels.
        x_val: Validation windows.
        y_val: Validation labels.
        grid: Architectures to try. Defaults to :data:`ARCHITECTURE_GRID`.
        n_seeds: Number of seeds averaged per configuration.
        verbose: If ``True``, log a one-line summary per configuration.

    Returns:
        A DataFrame with one row per configuration, sorted by descending
        mean validation PR-AUC, flagging the reference configuration.
    """
    grid = ARCHITECTURE_GRID if grid is None else grid
    rows: list[dict[str, Any]] = []

    for hyperparameters in grid:
        full_config = dict(REFERENCE_ARCHITECTURE)
        full_config.update(hyperparameters)

        pr_auc_scores, f1_scores, epoch_counts = [], [], []
        for k in range(n_seeds):
            seed = SETTINGS.seed + k
            model, history = train_classifier(
                x_train, y_train, x_val, y_val, seed=seed, hyperparameters=full_config
            )
            metrics = evaluate(
                model, x_val, y_val, threshold=optimal_threshold(model, x_val, y_val)
            )
            pr_auc_scores.append(metrics["pr_auc"])
            f1_scores.append(metrics["f1"])
            epoch_counts.append(len(history.history["loss"]))

        n_params = build_classifier(
            x_train.shape[1], x_train.shape[2], seed=SETTINGS.seed, **full_config
        ).count_params()

        rows.append(
            {
                "filters": "-".join(str(f) for f in full_config["filters"]),
                "kernel_size": full_config["kernel_size"],
                "dense_units": full_config["dense_units"],
                "dropout": full_config["dropout"],
                "parameters": int(n_params),
                "pr_auc_val_mean": float(np.mean(pr_auc_scores)),
                "pr_auc_val_std": (
                    float(np.std(pr_auc_scores, ddof=1)) if n_seeds > 1 else 0.0
                ),
                "f1_val_mean": float(np.mean(f1_scores)),
                "mean_epochs": float(np.mean(epoch_counts)),
                "is_reference": full_config == REFERENCE_ARCHITECTURE,
            }
        )

        if verbose:
            last = rows[-1]
            marker = "  <- reference" if last["is_reference"] else ""
            logger.info(
                "filters=%-12s kernel=%d dense=%-4d dropout=%.1f "
                "params=%9d PR-AUC(val)=%.4f +- %.4f%s",
                last["filters"],
                full_config["kernel_size"],
                full_config["dense_units"],
                full_config["dropout"],
                n_params,
                last["pr_auc_val_mean"],
                last["pr_auc_val_std"],
                marker,
            )

    return (
        pd.DataFrame(rows)
        .sort_values("pr_auc_val_mean", ascending=False)
        .reset_index(drop=True)
    )
