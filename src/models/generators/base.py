"""Common interface shared by every synthetic-data generator.

NOTE: improved over the original submission -- the delivered notebooks
implement each generator as free-standing notebook code with no shared
contract beyond a naming convention (save a ``.npz`` with ``X_synth`` and
``y_synth``). Introducing :class:`BaseGenerator` makes that contract
explicit and lets `main.py` treat all four generators polymorphically: the
training script, the ratio sweep, and the comparison report all iterate
over `BaseGenerator` instances instead of branching on the generator's name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


class BaseGenerator(ABC):
    """Abstract interface for a class-conditional synthetic-data generator.

    Every concrete generator is trained exclusively on the real-data budget
    (never on validation or test, and never on the full training history),
    so that the classifier comparison remains valid: a generator that has
    seen data the classifier has not would invalidate the comparison.
    """

    #: Short, lowercase name used as a key in results tables, file names,
    #: and configuration dictionaries (for example, ``"cgan"``).
    name: str = "base"

    @abstractmethod
    def fit(self, x_real: FloatArray, y_real: FloatArray) -> "BaseGenerator":
        """Train the generator on the real-data budget.

        Args:
            x_real: Real training windows, shape
                ``(n_real, window_x, n_assets)``.
            y_real: Real training labels, shape ``(n_real,)``.

        Returns:
            ``self``, to allow ``generator = SomeGenerator().fit(x, y)``.
        """

    @abstractmethod
    def generate(self, n_samples: int, positive_rate: float) -> tuple[FloatArray, FloatArray]:
        """Produce ``n_samples`` synthetic windows conditioned on class.

        Args:
            n_samples: Number of synthetic windows to produce.
            positive_rate: Fraction of the generated samples that should
                carry the positive (crisis) label.

        Returns:
            A tuple ``(x_synthetic, y_synthetic)`` with
            ``x_synthetic.shape == (n_samples, window_x, n_assets)`` and
            values in ``[-1, 1]``.
        """

    @property
    def loss_history(self) -> dict[str, list[float]]:
        """Training curves to persist and plot for convergence evidence.

        Returns:
            A mapping from curve name (for example ``"generator"`` and
            ``"discriminator"`` for an adversarial model, or ``"total"`` for
            a single-loss model) to the list of per-epoch or per-iteration
            values. The base implementation returns an empty mapping;
            concrete generators override this once training has completed.
        """
        return {}
