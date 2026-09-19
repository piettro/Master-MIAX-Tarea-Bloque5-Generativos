"""Conditional generative adversarial network (cGAN).

Two networks with opposite objectives are trained together: a generator
that produces synthetic windows from noise, and a discriminator that tries
to tell them apart from real ones. The generator never sees the real data
directly; every signal it receives about them arrives through the
discriminator's gradient. The class label is concatenated to both networks'
inputs, which lets a caller request samples of a specific class explicitly
-- the property this whole project needs, since the real scarcity is in
crisis episodes, not in observations overall.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402  (KERAS_BACKEND must be set before importing keras)
import numpy.typing as npt  # noqa: E402

import keras  # noqa: E402
from keras import Model, layers  # noqa: E402
from keras.optimizers import Adam  # noqa: E402

from src.models.generators.base import BaseGenerator  # noqa: E402
from src.utils.config import SETTINGS  # noqa: E402

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

NOISE_DIM: int = 100
GENERATOR_UNITS: tuple[int, ...] = (256, 512, 1024)
DISCRIMINATOR_UNITS: tuple[int, ...] = (512, 256)
LEAKY_RELU_SLOPE: float = 0.2
DISCRIMINATOR_DROPOUT: float = 0.3
LEARNING_RATE: float = 2e-4
ADAM_BETA_1: float = 0.5
DEFAULT_ITERATIONS: int = 12_000
DEFAULT_BATCH_SIZE: int = 64
#: One-sided label smoothing on the real side. It keeps the discriminator
#: from reaching overconfidence, which would otherwise zero out the
#: gradient that reaches the generator.
REAL_LABEL_SMOOTHING: float = 0.9


class ConditionalGAN(BaseGenerator):
    """Class-conditional GAN operating on flattened windows.

    A dense architecture over the flattened window is used, matching the
    usual approach for tabular data of this moderate dimensionality
    (``window_x * n_assets`` components) and keeping training cost within
    what a conventional CPU can handle.
    """

    name = "cgan"

    def __init__(
        self,
        window_x: int | None = None,
        n_assets: int | None = None,
        noise_dim: int = NOISE_DIM,
        iterations: int = DEFAULT_ITERATIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
    ) -> None:
        """Initialise the cGAN.

        Args:
            window_x: Time steps per window. Inferred from the data passed
                to :meth:`fit` if left as ``None``.
            n_assets: Assets per window. Inferred from the data passed to
                :meth:`fit` if left as ``None``.
            noise_dim: Dimensionality of the generator's input noise vector.
            iterations: Number of adversarial training steps.
            batch_size: Mini-batch size for each step.
            seed: Random seed. Defaults to
                :data:`src.utils.config.SETTINGS.seed`.
        """
        self.window_x = window_x
        self.n_assets = n_assets
        self.noise_dim = noise_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.seed = SETTINGS.seed if seed is None else seed

        self.generator: Model | None = None
        self.discriminator: Model | None = None
        self._adversarial_model: Model | None = None
        self._positive_rate: float = SETTINGS.positive_rate

        self.discriminator_loss: list[float] = []
        self.generator_loss: list[float] = []
        self.discriminator_accuracy: list[float] = []

    @property
    def loss_history(self) -> dict[str, list[float]]:
        return {
            "discriminator": self.discriminator_loss,
            "generator": self.generator_loss,
            "discriminator_accuracy": self.discriminator_accuracy,
        }

    def _dimension(self) -> int:
        if self.window_x is None or self.n_assets is None:
            raise RuntimeError("ConditionalGAN must be fitted before this call.")
        return self.window_x * self.n_assets

    def _build_generator(self) -> Model:
        dimension = self._dimension()
        noise = layers.Input(shape=(self.noise_dim,), name="noise")
        label = layers.Input(shape=(1,), name="label")

        hidden = layers.Concatenate()([noise, label])
        for units in GENERATOR_UNITS:
            hidden = layers.Dense(units)(hidden)
            hidden = layers.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)(hidden)
            hidden = layers.BatchNormalization(momentum=0.8)(hidden)

        window = layers.Dense(dimension, activation="tanh", name="window")(hidden)
        return Model([noise, label], window, name="generator")

    def _build_discriminator(self) -> Model:
        dimension = self._dimension()
        window = layers.Input(shape=(dimension,), name="window")
        label = layers.Input(shape=(1,), name="label")

        hidden = layers.Concatenate()([window, label])
        for units in DISCRIMINATOR_UNITS:
            hidden = layers.Dense(units)(hidden)
            hidden = layers.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)(hidden)
            hidden = layers.Dropout(DISCRIMINATOR_DROPOUT)(hidden)

        validity = layers.Dense(1, activation="sigmoid", name="validity")(hidden)
        model = Model([window, label], validity, name="discriminator")
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1),
            metrics=["accuracy"],
        )
        return model

    def _sample_labels(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.choice(
            [0.0, 1.0],
            size=(n, 1),
            p=[1 - self._positive_rate, self._positive_rate],
        ).astype("float32")

    def fit(self, x_real: FloatArray, y_real: FloatArray) -> "ConditionalGAN":
        """Train generator and discriminator with the standard cGAN loop.

        Each iteration has two steps: the discriminator is trained on one
        batch of real windows and one of generated windows; then, with the
        discriminator frozen, the generator is trained through the combined
        model to fool it. Convergence in this scheme is not a descending
        loss -- it is the discriminator's accuracy settling near 0.5, the
        point at which it can no longer tell real from generated windows.

        Args:
            x_real: Real training windows, shape
                ``(n_real, window_x, n_assets)``.
            y_real: Real training labels, shape ``(n_real,)``.

        Returns:
            ``self``.
        """
        self.window_x, self.n_assets = x_real.shape[1], x_real.shape[2]
        dimension = self._dimension()
        x_flat = x_real.reshape(len(x_real), dimension)
        self._positive_rate = float(y_real.mean())

        keras.utils.set_random_seed(self.seed)
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        self.discriminator.trainable = False
        noise_input = layers.Input(shape=(self.noise_dim,))
        label_input = layers.Input(shape=(1,))
        self._adversarial_model = Model(
            [noise_input, label_input],
            self.discriminator([self.generator([noise_input, label_input]), label_input]),
            name="cgan",
        )
        self._adversarial_model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1),
        )

        rng = np.random.default_rng(self.seed)
        self.discriminator_loss, self.generator_loss, self.discriminator_accuracy = (
            [],
            [],
            [],
        )

        for iteration in range(self.iterations):
            indices = rng.integers(0, len(x_flat), self.batch_size)
            real_windows = x_flat[indices]
            real_labels = y_real[indices].reshape(-1, 1)

            noise = rng.normal(0, 1, (self.batch_size, self.noise_dim)).astype("float32")
            fake_labels = self._sample_labels(rng, self.batch_size)
            fake_windows = self.generator.predict([noise, fake_labels], verbose=0)

            self.discriminator.trainable = True
            metrics_real = self.discriminator.train_on_batch(
                [real_windows, real_labels],
                np.full((self.batch_size, 1), REAL_LABEL_SMOOTHING, dtype="float32"),
            )
            metrics_fake = self.discriminator.train_on_batch(
                [fake_windows, fake_labels],
                np.zeros((self.batch_size, 1), dtype="float32"),
            )

            self.discriminator.trainable = False
            noise = rng.normal(0, 1, (self.batch_size, self.noise_dim)).astype("float32")
            fake_labels = self._sample_labels(rng, self.batch_size)
            generator_loss = self._adversarial_model.train_on_batch(
                [noise, fake_labels], np.ones((self.batch_size, 1), dtype="float32")
            )

            self.discriminator_loss.append(
                0.5 * (float(metrics_real[0]) + float(metrics_fake[0]))
            )
            self.discriminator_accuracy.append(
                0.5 * (float(metrics_real[1]) + float(metrics_fake[1]))
            )
            self.generator_loss.append(float(generator_loss))

            if iteration % 1000 == 0:
                logger.info(
                    "iteration %6d  D loss %.4f  D accuracy %.3f  G loss %.4f",
                    iteration,
                    self.discriminator_loss[-1],
                    self.discriminator_accuracy[-1],
                    self.generator_loss[-1],
                )

        logger.info(
            "Training finished. Final discriminator accuracy (last 1000 steps): %.3f",
            float(np.mean(self.discriminator_accuracy[-1000:])),
        )
        return self

    def generate(
        self, n_samples: int, positive_rate: float
    ) -> tuple[FloatArray, FloatArray]:
        """Sample noise, sort labels, and decode through the generator.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.generator is None or self.window_x is None or self.n_assets is None:
            raise RuntimeError("ConditionalGAN must be fitted before generate().")

        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, 1, (n_samples, self.noise_dim)).astype("float32")
        labels = rng.choice(
            [0.0, 1.0], size=(n_samples, 1), p=[1 - positive_rate, positive_rate]
        ).astype("float32")

        x_flat = self.generator.predict([noise, labels], verbose=0)
        x_synthetic = x_flat.reshape(n_samples, self.window_x, self.n_assets).astype(
            "float32"
        )
        y_synthetic = labels.ravel().astype("float32")
        return x_synthetic, y_synthetic
