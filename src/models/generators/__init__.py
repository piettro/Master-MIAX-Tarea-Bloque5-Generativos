"""Factory for the four synthetic-data generators compared in this project."""

from __future__ import annotations

from src.models.generators.base import BaseGenerator
from src.models.generators.diffusion import DiffusionGenerator
from src.models.generators.gan import ConditionalGAN
from src.models.generators.noise import NoiseGenerator
from src.models.generators.vae import ConditionalVAE

GENERATOR_REGISTRY: dict[str, type[BaseGenerator]] = {
    "noise": NoiseGenerator,
    "cgan": ConditionalGAN,
    "cvae": ConditionalVAE,
    "diffusion": DiffusionGenerator,
}


def create_generator(name: str, **kwargs: object) -> BaseGenerator:
    """Instantiate a generator by its registry name.

    Args:
        name: One of "noise", "cgan", "cvae", "diffusion".
        **kwargs: Forwarded to the generator's constructor.

    Returns:
        An unfitted BaseGenerator instance.

    Raises:
        ValueError: If name is not in GENERATOR_REGISTRY.
    """
    try:
        generator_cls = GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATOR_REGISTRY))
        raise ValueError(
            f"Unknown generator '{name}'. Available generators: {available}."
        ) from exc
    return generator_cls(**kwargs)


__all__ = [
    "BaseGenerator",
    "ConditionalGAN",
    "ConditionalVAE",
    "DiffusionGenerator",
    "GENERATOR_REGISTRY",
    "NoiseGenerator",
    "create_generator",
]
