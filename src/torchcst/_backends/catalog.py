"""One lazy backend-independent catalog; declaration imports load no evaluators."""

from functools import cache

from .registry import Registry


def make_registry():
    from .cuda.algorithms.normalized_euclidean_strip import (
        NormalizedFullAlgorithm,
        NormalizedWindowAlgorithm,
    )
    from .cuda.algorithms.strip_torus.fused.algorithm import FusedStripTorusAlgorithm
    from .torch.algorithms.factored import FactoredAlgorithm
    from .torch.algorithms.materialized import MaterializedAlgorithm
    from .torch.algorithms.normalized_radial import NormalizedRadialAlgorithm
    from .torch.algorithms.tiled import TiledAlgorithm

    registry = Registry()
    for algorithm in (
        MaterializedAlgorithm(),
        FactoredAlgorithm(),
        TiledAlgorithm(),
        NormalizedRadialAlgorithm(),
        NormalizedFullAlgorithm(),
        NormalizedWindowAlgorithm(),
        FusedStripTorusAlgorithm(),
    ):
        registry.register(algorithm)
    return registry


@cache
def get_registry():
    return make_registry()


def __getattr__(name):
    if name == "REGISTRY":
        return get_registry()
    raise AttributeError(name)
