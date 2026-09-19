"""Gaussian-noise perturbation generator: the mandatory simple baseline.

The assignment brief requires, alongside the three neural generative
families, a fourth, simple model that any more complex generator must
outperform to justify its extra cost. This one perturbs real windows with
Gaussian noise and nothing else, so any statistical property it reproduces
does so trivially, by construction.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from src.models.generators.base import BaseGenerator

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.intp]

#: Relative noise intensity (a multiple of the data's own standard
#: deviation). Chosen in the original notebook by sweeping
#: {0.05, 0.10, 0.25, 0.50, 1.00, 2.00} and picking the value that best
#: balances the marginal statistics against the cross-asset correlation
#: error; see notebooks/original_submission/03_generador_ruido.ipynb.
DEFAULT_RELATIVE_SIGMA: float = 0.25


class NoiseGenerator(BaseGenerator):
    """Adds Gaussian noise to resampled real windows.

    Because every synthetic sample is a real window plus noise, this
    generator cannot invent new market configurations; it can only produce
    local variants of ones already observed. That is precisely what makes
    it a meaningful lower bound: any generator that fails to beat it is not
    using its extra modelling capacity productively.
    """

    name = "noise"

    def __init__(self, relative_sigma: float = DEFAULT_RELATIVE_SIGMA, seed: int = 42) -> None:
        """Initialise the generator.

        Args:
            relative_sigma: Noise standard deviation, expressed as a
                multiple of the real data's own standard deviation.
            seed: Random seed for reproducible sampling.
        """
        self.relative_sigma = relative_sigma
        self.seed = seed
        self._x_real: FloatArray | None = None
        self._y_real: FloatArray | None = None
        self._sigma: float | None = None

    def fit(self, x_real: FloatArray, y_real: FloatArray) -> "NoiseGenerator":
        """Store the real-data pool and compute the absolute noise scale.

        There is no optimisation here: "training" this generator only means
        remembering the pool it will resample from and the data's own
        dispersion, which sets the absolute noise scale.
        """
        self._x_real = x_real
        self._y_real = y_real
        self._sigma = self.relative_sigma * float(x_real.std())
        return self

    def generate(
        self, n_samples: int, positive_rate: float
    ) -> tuple[FloatArray, FloatArray]:
        """Resample real windows with replacement and add Gaussian noise.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._x_real is None or self._y_real is None or self._sigma is None:
            raise RuntimeError("NoiseGenerator must be fitted before generate().")

        x_synth, y_synth, _ = self.generate_with_provenance(n_samples, positive_rate)
        return x_synth, y_synth

    def generate_with_provenance(
        self, n_samples: int, positive_rate: float, seed: int | None = None
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        """Generate samples and also return which real window each came from.

        Knowing the provenance of each synthetic sample lets the novelty
        analysis measure the distance to the *specific* real window a
        sample was derived from, instead of to an arbitrary one -- the
        distinction that matters when calibrating the noise intensity in
        :func:`calibrate_noise_intensity`.

        Args:
            n_samples: Number of synthetic windows to produce.
            positive_rate: Fraction of positive labels among the samples.
            seed: Optional override of the instance seed.

        Returns:
            A tuple ``(x_synthetic, y_synthetic, source_indices)``.
        """
        if self._x_real is None or self._y_real is None or self._sigma is None:
            raise RuntimeError("NoiseGenerator must be fitted before generate().")

        rng = np.random.default_rng(self.seed if seed is None else seed)

        n_positive = int(round(n_samples * positive_rate))
        n_negative = n_samples - n_positive

        positive_idx = np.where(self._y_real == 1)[0]
        negative_idx = np.where(self._y_real == 0)[0]

        selection = np.concatenate(
            [
                rng.choice(positive_idx, size=n_positive, replace=True),
                rng.choice(negative_idx, size=n_negative, replace=True),
            ]
        )
        rng.shuffle(selection)

        noisy = self._x_real[selection] + rng.normal(
            0.0, self._sigma, size=self._x_real[selection].shape
        )
        x_synthetic = np.clip(noisy, -1.0, 1.0).astype("float32")
        y_synthetic = self._y_real[selection].astype("float32")
        return x_synthetic, y_synthetic, selection


def calibrate_noise_intensity(
    x_real: FloatArray,
    y_real: FloatArray,
    positive_rate: float,
    candidate_sigmas: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00),
    n_calibration_samples: int = 2000,
    seed: int = 42,
) -> "tuple[float, list[dict[str, float]]]":
    """Sweep relative noise levels and report their effect on key statistics.

    Args:
        x_real: Real training windows.
        y_real: Real training labels.
        positive_rate: Positive rate to use when sampling calibration
            batches.
        candidate_sigmas: Relative sigma values to try.
        n_calibration_samples: Number of samples drawn per candidate.
        seed: Random seed.

    Returns:
        A tuple ``(chosen_sigma, rows)`` where ``rows`` has one dictionary
        per candidate with its dispersion, kurtosis, cross-asset
        correlation error, and mean distance to each sample's own source
        window. ``chosen_sigma`` is :data:`DEFAULT_RELATIVE_SIGMA`, the
        value validated in the original submission; the sweep is exposed so
        that choice can be audited or revisited.
    """
    from scipy import stats

    n_assets = x_real.shape[2]
    real_corr = np.corrcoef(x_real.reshape(-1, n_assets).T)
    triangle = np.triu_indices_from(real_corr, k=1)

    rows: list[dict[str, float]] = []
    for sigma in candidate_sigmas:
        generator = NoiseGenerator(relative_sigma=sigma, seed=seed).fit(x_real, y_real)
        x_sample, _, origin = generator.generate_with_provenance(
            n_calibration_samples, positive_rate
        )
        synth_corr = np.corrcoef(x_sample.reshape(-1, n_assets).T)

        rows.append(
            {
                "relative_sigma": sigma,
                "std": float(x_sample.std()),
                "kurtosis": float(stats.kurtosis(x_sample.ravel())),
                "correlation_error": float(
                    np.abs(synth_corr[triangle] - real_corr[triangle]).mean()
                ),
                "distance_to_source": float(
                    np.linalg.norm(
                        (x_sample - x_real[origin]).reshape(len(x_sample), -1), axis=1
                    ).mean()
                ),
            }
        )

    return DEFAULT_RELATIVE_SIGMA, rows
