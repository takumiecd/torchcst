"""Normalized Strip registration metadata; GPU code remains lazily imported."""

from torchcst._backends.cuda.registry import Registry
from torchcst._backends.cuda.dispatch.select import FixedSelector

from .full.algorithm import NormalizedFullAlgorithm
from .window.algorithm import NormalizedWindowAlgorithm
from .plans import FULL

__all__ = [
    "REGISTRY",
    "NormalizedFullAlgorithm",
    "NormalizedWindowAlgorithm",
    "make_registry",
    "BOOTSTRAP_SELECTOR",
]


def make_registry():
    registry = Registry()
    registry.register(NormalizedFullAlgorithm())
    registry.register(NormalizedWindowAlgorithm())
    return registry


REGISTRY = make_registry()

# Explicit bootstrap configuration until an approved measured policy is loaded.
# This is not a leaderboard winner or an Algorithm-specific dispatch branch.
BOOTSTRAP_SELECTOR = FixedSelector(
    FULL, registry=REGISTRY, revision="normalized-bootstrap-v1"
)
