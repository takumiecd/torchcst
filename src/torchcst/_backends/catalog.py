"""One lazy backend-independent catalog; declaration imports load no evaluators."""

from functools import cache

from .registry import Registry


def make_registry():
    from .cuda.algorithms.linear.normalized_euclidean_strip import (
        NormalizedFullAlgorithm,
        NormalizedWindowAlgorithm,
    )
    from .cuda.algorithms.linear.strip_torus.fused.algorithm import (
        FusedStripTorusAlgorithm,
    )
    from .cuda.algorithms.polar_update import FusedPolarUpdateAlgorithm
    from .torch.algorithms.atom_update import AtomUpdateAlgorithm
    from .torch.algorithms.linear.factored import FactoredAlgorithm
    from .torch.algorithms.linear.materialized import MaterializedAlgorithm
    from .torch.algorithms.linear.normalized_radial import NormalizedRadialAlgorithm
    from .torch.algorithms.linear.tiled import TiledAlgorithm
    from .torch.algorithms.polar_update import PolarUpdateAlgorithm

    registry = Registry()
    for algorithm in (
        MaterializedAlgorithm(),
        FactoredAlgorithm(),
        TiledAlgorithm(),
        NormalizedRadialAlgorithm(),
        NormalizedFullAlgorithm(),
        NormalizedWindowAlgorithm(),
        FusedStripTorusAlgorithm(),
        AtomUpdateAlgorithm(),
        PolarUpdateAlgorithm(),
        FusedPolarUpdateAlgorithm(),
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
