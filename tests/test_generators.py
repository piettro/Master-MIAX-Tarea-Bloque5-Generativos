"""Unit tests for the generator interface and the simplest concrete generator.

The GAN, CVAE, and diffusion generators are exercised end-to-end (with
trivially cheap settings) in the smoke tests documented in
docs/architecture.md rather than here, since even a handful of epochs on
each makes the suite noticeably slower; NoiseGenerator has no training loop
and is cheap enough to test thoroughly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.generators import GENERATOR_REGISTRY, create_generator
from src.models.generators.noise import (
    DEFAULT_RELATIVE_SIGMA,
    NoiseGenerator,
    calibrate_noise_intensity,
)


def test_registry_contains_all_four_families():
    assert set(GENERATOR_REGISTRY) == {"noise", "cgan", "cvae", "diffusion"}


def test_create_generator_unknown_name_raises():
    with pytest.raises(ValueError):
        create_generator("not_a_real_generator")


def test_create_generator_returns_expected_type():
    generator = create_generator("noise")
    assert isinstance(generator, NoiseGenerator)


def test_noise_generator_requires_fit_before_generate():
    generator = NoiseGenerator()
    with pytest.raises(RuntimeError):
        generator.generate(10, 0.1)


def test_noise_generator_output_shape_and_range(small_windows):
    x, y = small_windows
    generator = NoiseGenerator(relative_sigma=0.25, seed=1).fit(x, y)

    x_synth, y_synth = generator.generate(50, positive_rate=0.2)

    assert x_synth.shape == (50, x.shape[1], x.shape[2])
    assert x_synth.min() >= -1.0 and x_synth.max() <= 1.0
    assert abs(y_synth.mean() - 0.2) < 0.05


def test_noise_generator_provenance_matches_labels(small_windows):
    x, y = small_windows
    generator = NoiseGenerator(relative_sigma=0.1, seed=1).fit(x, y)

    x_synth, y_synth, origin = generator.generate_with_provenance(60, positive_rate=0.1)

    assert origin.shape == (60,)
    assert np.array_equal(y_synth, y[origin])


def test_noise_generator_distance_to_own_origin_is_small(small_windows):
    x, y = small_windows
    generator = NoiseGenerator(relative_sigma=0.05, seed=1).fit(x, y)
    x_synth, _, origin = generator.generate_with_provenance(30, positive_rate=0.1)

    distance_to_origin = np.linalg.norm(
        (x_synth - x[origin]).reshape(len(x_synth), -1), axis=1
    ).mean()
    distance_to_arbitrary_window = np.linalg.norm(
        (x_synth - x[0]).reshape(len(x_synth), -1), axis=1
    ).mean()

    # With a small relative sigma, a synthetic sample must sit close to the
    # real window it was derived from, and much closer than to an arbitrary
    # one -- the exact bug this test guards against previously went
    # unnoticed in the original notebooks (see docs/architecture.md).
    assert distance_to_origin < distance_to_arbitrary_window / 2


def test_calibrate_noise_intensity_returns_one_row_per_candidate(small_windows):
    x, y = small_windows
    candidates = (0.1, 0.5, 1.0)
    chosen_sigma, rows = calibrate_noise_intensity(
        x, y, positive_rate=0.1, candidate_sigmas=candidates, n_calibration_samples=50
    )

    # The sweep always reports every candidate that was tried...
    assert len(rows) == len(candidates)
    assert {row["relative_sigma"] for row in rows} == set(candidates)
    # ...but the *chosen* sigma is the value validated in the original
    # submission (see the module docstring), not necessarily one of the
    # candidates just swept: the sweep is exposed for audit, not for
    # automatic selection.
    assert chosen_sigma == DEFAULT_RELATIVE_SIGMA
