"""Conditional denoising-diffusion generator.

A forward process adds Gaussian noise to a real window across ``T`` steps,
with no parameters of its own, until nothing of the original signal
remains. The reverse process is the only part that is trained: a network
that, given a noisy window and the current step, recovers a clean window.
Generating a new sample means starting from pure noise and applying that
network iteratively.

The network is conditioned on the class label by concatenation and on the
diffusion step through a sinusoidal embedding, since the same noisy window
means something different depending on how much noise it already carries.

One deliberate deviation from the textbook formulation is load-bearing and
documented in detail below: the network predicts the clean window directly,
not the noise that was added, because the standard epsilon-prediction
parameterisation failed outright on this data.
"""

from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402  (KERAS_BACKEND must be set before importing keras)
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402

import keras  # noqa: E402
from keras import Model, layers  # noqa: E402

from src.models.generators.base import BaseGenerator  # noqa: E402
from src.utils.config import SETTINGS  # noqa: E402

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

N_DIFFUSION_STEPS: int = 200
TIME_EMBEDDING_DIM: int = 64
HIDDEN_UNITS: int = 768
N_DENSE_BLOCKS: int = 3
DEFAULT_EPOCHS: int = 1000
DEFAULT_BATCH_SIZE: int = 64
LEARNING_RATE: float = 2e-4
DEFAULT_CALIBRATION_TEMPERATURES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


def _sinusoidal_time_embedding(n_steps: int, dim: int) -> FloatArray:
    """Precompute a fixed sinusoidal embedding, one row per diffusion step."""
    positions = np.arange(n_steps)[:, None]
    frequencies = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))[None, :]
    embedding = np.zeros((n_steps, dim), dtype="float32")
    embedding[:, 0::2] = np.sin(positions * frequencies)
    embedding[:, 1::2] = np.cos(positions * frequencies)
    return embedding


class DiffusionGenerator(BaseGenerator):
    """Class-conditional denoising-diffusion model over flattened windows.

    Why the network predicts the clean window instead of the noise:

    The classical parameterisation trains the network to estimate the noise
    added at each step, which works well when the signal is recoverable
    from the noisy observation. It is not, here: daily log returns have an
    autocorrelation of about 0.019 (see the exploratory analysis), so they
    are close to white noise along the time axis, and the added Gaussian
    noise is not statistically distinguishable from the signal it was added
    to. Because the reverse step divides by the square root of a
    diffusion-schedule coefficient, a mediocre noise estimator makes the
    reverse process diverge outright: in practice, the epsilon-prediction
    training loss plateaued at 0.80 out of a maximum of 1.00 (the loss a
    network that always predicts zero would get, since the data is
    normalised to unit variance), and sampling produced saturated noise.

    Predicting the clean window instead gives a bounded target, and
    clipping that prediction at every step keeps sampling numerically
    stable; the training loss then falls to about 0.34.
    """

    name = "diffusion"

    def __init__(
        self,
        n_steps: int = N_DIFFUSION_STEPS,
        time_embedding_dim: int = TIME_EMBEDDING_DIM,
        hidden_units: int = HIDDEN_UNITS,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
    ) -> None:
        """Initialise the diffusion generator.

        Args:
            n_steps: Number of steps in the forward/reverse process.
            time_embedding_dim: Dimensionality of the sinusoidal step
                embedding.
            hidden_units: Width of the dense blocks in the denoising
                network.
            epochs: Training epochs (each epoch iterates the full real-data
                budget once).
            batch_size: Mini-batch size.
            seed: Random seed. Defaults to
                :data:`src.utils.config.SETTINGS.seed`.
        """
        self.n_steps = n_steps
        self.time_embedding_dim = time_embedding_dim
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = SETTINGS.seed if seed is None else seed

        self.window_x: int | None = None
        self.n_assets: int | None = None
        self.estimator: Model | None = None
        self._scale: float | None = None
        self._clip_limit: float | None = None
        self._temperature: float = 1.0
        self._loss_history: list[float] = []

        betas = np.linspace(1e-4, 0.02, n_steps).astype("float32")
        alphas = (1.0 - betas).astype("float32")
        cumulative_alphas = np.cumprod(alphas).astype("float32")
        self._betas = betas
        self._alphas = alphas
        self._cumulative_alphas = cumulative_alphas
        self._cumulative_alphas_prev = np.concatenate(
            [[1.0], cumulative_alphas[:-1]]
        ).astype("float32")
        self._sqrt_cumulative_alphas = np.sqrt(cumulative_alphas)
        self._sqrt_one_minus_cumulative = np.sqrt(1.0 - cumulative_alphas)
        self._time_embedding = _sinusoidal_time_embedding(n_steps, time_embedding_dim)

    @property
    def loss_history(self) -> dict[str, list[float]]:
        return {"total": self._loss_history}

    @property
    def chosen_temperature(self) -> float:
        """Sampling temperature selected by :func:`calibrate_temperature`."""
        return self._temperature

    def _dimension(self) -> int:
        if self.window_x is None or self.n_assets is None:
            raise RuntimeError("DiffusionGenerator must be fitted before this call.")
        return self.window_x * self.n_assets

    def _build_estimator(self) -> Model:
        dimension = self._dimension()
        x_in = layers.Input(shape=(dimension,), name="noisy_window")
        t_in = layers.Input(shape=(self.time_embedding_dim,), name="step")
        y_in = layers.Input(shape=(1,), name="label")

        hidden = layers.Concatenate()([x_in, t_in, y_in])
        for _ in range(N_DENSE_BLOCKS):
            hidden = layers.Dense(self.hidden_units, activation="silu")(hidden)
        output = layers.Dense(dimension, name="clean_window")(hidden)

        model = Model([x_in, t_in, y_in], output, name="x0_estimator")
        model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
        return model

    def fit(self, x_real: FloatArray, y_real: FloatArray) -> "DiffusionGenerator":
        """Train the denoising network to recover clean windows.

        Each iteration draws a batch of real windows, samples a random step
        per window, adds the corresponding amount of noise, and trains the
        network to recover the original window. The starting loss is
        informative: with data normalised to unit variance, a network that
        always predicted the mean would score close to 1.00.

        Args:
            x_real: Real training windows.
            y_real: Real training labels.

        Returns:
            ``self``.
        """
        self.window_x, self.n_assets = x_real.shape[1], x_real.shape[2]
        dimension = self._dimension()
        x_flat = x_real.reshape(len(x_real), dimension)

        self._scale = float(x_flat.std())
        x0 = (x_flat / self._scale).astype("float32")
        self._clip_limit = float(np.abs(x0).max())

        keras.utils.set_random_seed(self.seed)
        self.estimator = self._build_estimator()
        logger.info(
            "Diffusion scale: %.4f (std after normalising: %.4f). Clip limit: +-%.2f",
            self._scale,
            x0.std(),
            self._clip_limit,
        )

        rng = np.random.default_rng(self.seed)
        self._loss_history = []
        start_time = time.time()

        for epoch in range(self.epochs):
            order = rng.permutation(len(x0))
            epoch_losses = []

            for start in range(0, len(x0), self.batch_size):
                idx = order[start : start + self.batch_size]
                windows = x0[idx]
                labels = y_real[idx].reshape(-1, 1)

                steps = rng.integers(0, self.n_steps, len(idx))
                noise = rng.normal(size=windows.shape).astype("float32")
                noisy = (
                    self._sqrt_cumulative_alphas[steps][:, None] * windows
                    + self._sqrt_one_minus_cumulative[steps][:, None] * noise
                )

                loss = self.estimator.train_on_batch(
                    [noisy, self._time_embedding[steps], labels], windows
                )
                epoch_losses.append(float(loss))

            self._loss_history.append(float(np.mean(epoch_losses)))
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                logger.info(
                    "epoch %4d  error %.4f  (%.1f min)",
                    epoch,
                    self._loss_history[-1],
                    (time.time() - start_time) / 60,
                )

        logger.info(
            "Training finished in %.1f minutes. Error %.4f -> %.4f",
            (time.time() - start_time) / 60,
            self._loss_history[0],
            self._loss_history[-1],
        )
        return self

    def _sample_flat(
        self, n_samples: int, labels: FloatArray, temperature: float, batch: int = 1000
    ) -> FloatArray:
        if self.estimator is None or self._scale is None or self._clip_limit is None:
            raise RuntimeError("DiffusionGenerator must be fitted before sampling.")

        dimension = self._dimension()
        rng = np.random.default_rng(self.seed)
        labels = np.asarray(labels, dtype="float32").reshape(-1, 1)
        chunks: list[np.ndarray] = []

        for start in range(0, n_samples, batch):
            m = min(batch, n_samples - start)
            x = rng.normal(size=(m, dimension)).astype("float32")
            batch_labels = labels[start : start + m]

            with torch.no_grad():
                for t in range(self.n_steps - 1, -1, -1):
                    time_embedding = np.repeat(
                        self._time_embedding[t][None, :], m, axis=0
                    )
                    x0_estimate = keras.ops.convert_to_numpy(
                        self.estimator([x, time_embedding, batch_labels], training=False)
                    )
                    x0_estimate = np.clip(x0_estimate, -self._clip_limit, self._clip_limit)

                    coeff_x0 = (
                        np.sqrt(self._cumulative_alphas_prev[t])
                        * self._betas[t]
                        / (1.0 - self._cumulative_alphas[t])
                    )
                    coeff_xt = (
                        np.sqrt(self._alphas[t])
                        * (1.0 - self._cumulative_alphas_prev[t])
                        / (1.0 - self._cumulative_alphas[t])
                    )
                    x = coeff_x0 * x0_estimate + coeff_xt * x

                    if t > 0 and temperature > 0:
                        variance = (
                            self._betas[t]
                            * (1.0 - self._cumulative_alphas_prev[t])
                            / (1.0 - self._cumulative_alphas[t])
                        )
                        x = x + temperature * np.sqrt(variance) * rng.normal(
                            size=x.shape
                        ).astype("float32")

            chunks.append(x)

        raw = np.concatenate(chunks) * self._scale
        return np.clip(raw, -1.0, 1.0).astype("float32")

    def calibrate_temperature(
        self,
        x_real: FloatArray,
        y_real: FloatArray,
        candidate_temperatures: tuple[float, ...] = DEFAULT_CALIBRATION_TEMPERATURES,
        n_calibration_samples: int = 1000,
    ) -> "tuple[float, list[dict[str, float]]]":
        """Sweep the reverse-process noise temperature and pick the closest fit.

        The variance injected at every reverse step assumes a perfect
        clean-window estimate; the trained network's estimate is
        regularised instead, so the raw process tends to over-disperse the
        samples. This sweep picks the temperature whose sample dispersion
        is closest to the real data's, following the same calibrate-then-fix
        pattern used for the noise generator's sigma.

        Args:
            x_real: Real training windows (used only for the reference
                dispersion, kurtosis, and correlation).
            y_real: Real training labels (used only for the reference
                positive rate).
            candidate_temperatures: Temperatures to try.
            n_calibration_samples: Number of samples drawn per candidate.

        Returns:
            A tuple ``(chosen_temperature, rows)``. Also stores the chosen
            temperature on the instance, so subsequent calls to
            :meth:`generate` use it by default.
        """
        from scipy import stats

        n_assets = x_real.shape[2]
        real_corr = np.corrcoef(x_real.reshape(-1, n_assets).T)
        triangle = np.triu_indices(n_assets, k=1)
        real_std = float(x_real.std())
        positive_rate = float(y_real.mean())

        rng = np.random.default_rng(self.seed)
        calibration_labels = (rng.random(n_calibration_samples) < positive_rate).astype(
            "float32"
        )

        rows: list[dict[str, float]] = []
        for temperature in candidate_temperatures:
            samples = self._sample_flat(
                n_calibration_samples, calibration_labels, temperature
            )
            cube = samples.reshape(-1, self.window_x, self.n_assets)
            corr = np.corrcoef(cube.reshape(-1, n_assets).T)
            rows.append(
                {
                    "temperature": temperature,
                    "std": float(samples.std()),
                    "kurtosis": float(stats.kurtosis(samples.ravel())),
                    "mean_correlation": float(corr[triangle].mean()),
                    "correlation_error": float(
                        np.abs(corr[triangle] - real_corr[triangle]).mean()
                    ),
                    "saturated_pct": 100 * float((np.abs(samples) > 0.99).mean()),
                    "dispersion_error": abs(float(samples.std()) - real_std),
                }
            )

        best_row = min(rows, key=lambda row: row["dispersion_error"])
        self._temperature = float(best_row["temperature"])
        logger.info("Chosen sampling temperature: %.1f", self._temperature)
        return self._temperature, rows

    def generate(
        self, n_samples: int, positive_rate: float
    ) -> tuple[FloatArray, FloatArray]:
        """Run the reverse process from pure noise to produce clean windows.

        Uses :attr:`chosen_temperature` (set by :meth:`calibrate_temperature`,
        or 1.0 by default) as the reverse-process noise scale.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.window_x is None or self.n_assets is None:
            raise RuntimeError("DiffusionGenerator must be fitted before generate().")

        rng = np.random.default_rng(self.seed)
        y_synthetic = (rng.random(n_samples) < positive_rate).astype("float32")
        x_flat = self._sample_flat(n_samples, y_synthetic, self._temperature)
        x_synthetic = x_flat.reshape(n_samples, self.window_x, self.n_assets).astype(
            "float32"
        )
        return x_synthetic, y_synthetic
