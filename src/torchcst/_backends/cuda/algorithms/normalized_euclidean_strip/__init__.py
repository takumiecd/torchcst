"""Normalized Strip registration metadata; GPU code remains lazily imported."""

from torchcst._backends.cuda.registry import Registry

from .full.algorithm import NormalizedFullAlgorithm
from .window.algorithm import NormalizedWindowAlgorithm

__all__ = [
    "REGISTRY",
    "NormalizedFullAlgorithm",
    "NormalizedWindowAlgorithm",
    "make_registry",
]


def make_registry():
    registry = Registry()
    registry.register(NormalizedFullAlgorithm())
    registry.register(NormalizedWindowAlgorithm())
    return registry


REGISTRY = make_registry()
