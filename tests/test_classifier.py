"""Unit tests for src.models.classifier."""

from __future__ import annotations

import numpy as np

from src.models.classifier import (
    REFERENCE_ARCHITECTURE,
    build_classifier,
    class_weights,
    evaluate,
    optimal_threshold,
    train_classifier,
)


def test_build_classifier_default_matches_reference_architecture():
    model_default = build_classifier(60, 6, seed=42)
    model_explicit = build_classifier(
        60,
        6,
        seed=42,
        filters=REFERENCE_ARCHITECTURE["filters"],
        kernel_size=REFERENCE_ARCHITECTURE["kernel_size"],
        dense_units=REFERENCE_ARCHITECTURE["dense_units"],
        dropout=REFERENCE_ARCHITECTURE["dropout"],
    )
    assert model_default.count_params() == model_explicit.count_params()


def test_build_classifier_same_seed_gives_identical_weights():
    model_a = build_classifier(60, 6, seed=7)
    model_b = build_classifier(60, 6, seed=7)
    for weight_a, weight_b in zip(model_a.get_weights(), model_b.get_weights()):
        assert np.array_equal(weight_a, weight_b)


def test_build_classifier_respects_hyperparameter_overrides():
    small_model = build_classifier(60, 6, seed=1, filters=(8, 16), dense_units=10)
    large_model = build_classifier(60, 6, seed=1)
    assert small_model.count_params() < large_model.count_params()


def test_class_weights_none_when_a_class_is_absent():
    y_all_negative = np.zeros(50, dtype="float32")
    assert class_weights(y_all_negative) is None


def test_class_weights_favours_minority_class():
    y = np.array([0.0] * 90 + [1.0] * 10, dtype="float32")
    weights = class_weights(y)
    assert weights is not None
    assert weights[1] > weights[0]


def test_train_and_evaluate_end_to_end(small_windows):
    x, y = small_windows
    x_train, x_val = x[:150], x[150:]
    y_train, y_val = y[:150], y[150:]

    model, history = train_classifier(x_train, y_train, x_val, y_val, seed=1)
    assert len(history.history["loss"]) > 0

    threshold = optimal_threshold(model, x_val, y_val)
    assert 0.05 <= threshold <= 0.95

    metrics = evaluate(model, x_val, y_val, threshold=threshold)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    total_predictions = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )
    assert total_predictions == len(y_val)
