"""Conditional variational autoencoder (CVAE).

An encoder maps each window, together with its class label, onto a
distribution over a small latent space; a decoder reconstructs the window
from a sample of that distribution and the same label. The training
objective combines a reconstruction term with the Kullback-Leibler
divergence to a standard normal prior, and the reparameterisation trick
makes the sampling step differentiable.

Two regularisation choices below responded to a real problem observed while
fitting this model, and are kept exactly as validated in the original
submission (notebooks/original_submission/05_generador_cvae.ipynb):

* A small latent space and a low KL weight avoid posterior collapse, where
  the decoder starts ignoring the latent code and every sample comes out
  flat.
* A mild dropout (0.1) closes most of the train/validation gap that
  appeared without it (roughly 12.6 down to roughly 9), without triggering
  the same collapse a stronger dropout produced.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402  (KERAS_BACKEND must be set before importing keras)
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402

import keras  # noqa: E402
import keras.ops as ops  # noqa: E402
from keras import Model, layers  # noqa: E402

from src.models.generators.base import BaseGenerator  # noqa: E402
from src.utils.config import SETTINGS  # noqa: E402

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

LATENT_DIM: int = 16
KL_WEIGHT: float = 0.001
DROPOUT: float = 0.1
ENCODER_UNITS: tuple[int, ...] = (256, 128)
DECODER_UNITS: tuple[int, ...] = (128, 256)
DEFAULT_EPOCHS: int = 150
DEFAULT_BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3


class _LatentSampling(layers.Layer):
    """Reparameterisation trick: ``z = mean + exp(0.5 * log_var) * epsilon``.

    Noise is drawn with ``keras.random.normal`` rather than
    ``tf.random.normal`` so the layer works unchanged on the PyTorch Keras
    backend this project runs on.
    """

    def call(self, inputs: tuple[FloatArray, FloatArray]) -> FloatArray:
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class _CVAECore(keras.Model):
    """Keras subclassed model implementing the combined CVAE loss.

    Keras 3's functional API does not support ``add_loss``, so the two loss
    terms (reconstruction and KL) are combined in a custom training step,
    which also lets both be tracked separately to watch for KL collapse.
    """

    def __init__(
        self, encoder: Model, decoder: Model, beta: float = 1.0, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def call(self, inputs: tuple[FloatArray, FloatArray], training: bool = False) -> FloatArray:
        window, label = inputs
        _, _, z = self.encoder([window, label], training=training)
        return self.decoder([z, label], training=training)

    def compute_losses(
        self, window: FloatArray, label: FloatArray, training: bool = False
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        z_mean, z_log_var, z = self.encoder([window, label], training=training)
        reconstruction = self.decoder([z, label], training=training)
        reconstruction_loss = ops.mean(
            ops.sum(ops.square(window - reconstruction), axis=-1)
        )
        kl_loss = -0.5 * ops.mean(
            ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
        )
        return reconstruction_loss + self.beta * kl_loss, reconstruction_loss, kl_loss

    def train_step(
        self, data: tuple[tuple[FloatArray, FloatArray], FloatArray]
    ) -> dict[str, float]:
        (window, label), _ = data
        self.zero_grad()
        total, reconstruction, kl = self.compute_losses(window, label, training=True)
        total.backward()
        trainable = self.trainable_weights
        gradients = [w.value.grad for w in trainable]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable)
        self.loss_tracker.update_state(total)
        self.reconstruction_tracker.update_state(reconstruction)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(
        self, data: tuple[tuple[FloatArray, FloatArray], FloatArray]
    ) -> dict[str, float]:
        (window, label), _ = data
        total, reconstruction, kl = self.compute_losses(window, label, training=False)
        self.loss_tracker.update_state(total)
        self.reconstruction_tracker.update_state(reconstruction)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


class ConditionalVAE(BaseGenerator):
    """Class-conditional VAE for financial return windows."""

    name = "cvae"

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        beta: float = KL_WEIGHT,
        dropout: float = DROPOUT,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = None,
    ) -> None:
        """Initialise the CVAE.

        Args:
            latent_dim: Dimensionality of the latent space.
            beta: Weight of the KL term relative to reconstruction.
            dropout: Dropout rate applied in both encoder and decoder.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            seed: Random seed. Defaults to
                :data:`src.utils.config.SETTINGS.seed`.
        """
        self.latent_dim = latent_dim
        self.beta = beta
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = SETTINGS.seed if seed is None else seed

        self.window_x: int | None = None
        self.n_assets: int | None = None
        self.decoder: Model | None = None
        self._core: _CVAECore | None = None
        self._loss_history: list[float] = []
        self._val_loss_history: list[float] = []
        self._positive_rate: float = SETTINGS.positive_rate

    @property
    def loss_history(self) -> dict[str, list[float]]:
        return {"total": self._loss_history, "validation": self._val_loss_history}

    def _dimension(self) -> int:
        if self.window_x is None or self.n_assets is None:
            raise RuntimeError("ConditionalVAE must be fitted before this call.")
        return self.window_x * self.n_assets

    def _build(self) -> tuple[Model, Model]:
        dimension = self._dimension()

        x_in = layers.Input(shape=(dimension,), name="window")
        y_in = layers.Input(shape=(1,), name="label")
        hidden = layers.Concatenate()([x_in, y_in])
        for units in ENCODER_UNITS:
            hidden = layers.Dense(units, activation="relu")(hidden)
            hidden = layers.Dropout(self.dropout)(hidden)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(hidden)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(hidden)
        z = _LatentSampling(name="z")([z_mean, z_log_var])
        encoder = Model([x_in, y_in], [z_mean, z_log_var, z], name="encoder")

        z_in = layers.Input(shape=(self.latent_dim,), name="z")
        y_dec_in = layers.Input(shape=(1,), name="label_decoder")
        hidden = layers.Concatenate()([z_in, y_dec_in])
        for units in DECODER_UNITS:
            hidden = layers.Dense(units, activation="relu")(hidden)
            hidden = layers.Dropout(self.dropout)(hidden)
        x_out = layers.Dense(dimension, activation="tanh", name="reconstruction")(hidden)
        decoder = Model([z_in, y_dec_in], x_out, name="decoder")

        return encoder, decoder

    def fit(
        self,
        x_real: FloatArray,
        y_real: FloatArray,
        x_val: FloatArray | None = None,
        y_val: FloatArray | None = None,
    ) -> "ConditionalVAE":
        """Train encoder and decoder jointly by minimising the CVAE loss.

        Args:
            x_real: Real training windows.
            y_real: Real training labels.
            x_val: Optional validation windows, used only to monitor
                overfitting; no early stopping or model selection is
                performed on it.
            y_val: Optional validation labels.

        Returns:
            ``self``.
        """
        self.window_x, self.n_assets = x_real.shape[1], x_real.shape[2]
        dimension = self._dimension()
        x_flat = x_real.reshape(len(x_real), dimension)
        self._positive_rate = float(y_real.mean())

        keras.utils.set_random_seed(self.seed)
        encoder, self.decoder = self._build()
        self._core = _CVAECore(encoder, self.decoder, beta=self.beta)
        self._core.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE))

        validation_data = None
        if x_val is not None and y_val is not None:
            x_val_flat = x_val.reshape(len(x_val), dimension)
            validation_data = ([x_val_flat, y_val], x_val_flat)

        history = self._core.fit(
            [x_flat, y_real],
            x_flat,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        self._loss_history = list(history.history["loss"])
        self._val_loss_history = list(history.history.get("val_loss", []))

        z_mean, z_log_var, _ = encoder.predict([x_flat, y_real], verbose=0)
        kl_final = float(
            -0.5 * np.mean(np.sum(1 + z_log_var - z_mean**2 - np.exp(z_log_var), axis=1))
        )
        logger.info(
            "Epochs trained: %d. Loss %.2f -> %.2f. Final KL divergence: %.3f "
            "(should be clearly > 0, or the decoder is ignoring the latent code).",
            len(self._loss_history),
            self._loss_history[0],
            self._loss_history[-1],
            kl_final,
        )
        return self

    def generate(
        self, n_samples: int, positive_rate: float
    ) -> tuple[FloatArray, FloatArray]:
        """Sample the latent prior and decode conditioned on sampled labels.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.decoder is None or self.window_x is None or self.n_assets is None:
            raise RuntimeError("ConditionalVAE must be fitted before generate().")

        rng = np.random.default_rng(self.seed)
        y_synthetic = (rng.random(n_samples) < positive_rate).astype("float32")
        z_samples = rng.normal(size=(n_samples, self.latent_dim)).astype("float32")

        x_flat = self.decoder.predict([z_samples, y_synthetic], verbose=0)
        x_synthetic = x_flat.reshape(n_samples, self.window_x, self.n_assets).astype(
            "float32"
        )
        return x_synthetic, y_synthetic
