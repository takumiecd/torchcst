"""One lazy backend-independent catalog; declaration imports load no evaluators."""

from functools import cache

from .registry import Registry


def make_registry():
    from .cuda.algorithms.conv2d.channel_algorithm import ChannelFourierAlgorithm
    from .cuda.algorithms.linear.normalized_euclidean_strip import (
        NormalizedFullAlgorithm,
        NormalizedWindowAlgorithm,
    )
    from .cuda.algorithms.linear.radius_cell_mass.algorithm import (
        RadiusCellMassAlgorithm,
    )
    from .cuda.algorithms.linear.radius_cell_mass.rational_wide import (
        RationalWideAlgorithm,
        RationalWideAxisGradientAlgorithm,
        RationalWideScaleWeightAlgorithm,
    )
    from .cuda.algorithms.polar_update import FusedPolarUpdateAlgorithm
    from .cuda.algorithms.polar_update.algorithm import (
        FlatTorusPolarUpdateAlgorithm,
        RadiusFlatTorusPolarUpdateAlgorithm,
    )
    from .torch.algorithms.atom_update import AtomUpdateAlgorithm
    from .torch.algorithms.conv2d import (
        TorchChannelConv2dAlgorithm,
        TorchConv2dAlgorithm,
    )
    from .torch.algorithms.linear.factored import FactoredAlgorithm
    from .torch.algorithms.linear.materialized import MaterializedAlgorithm
    from .torch.algorithms.linear.normalized_radial import NormalizedRadialAlgorithm
    from .torch.algorithms.polar_update import PolarUpdateAlgorithm

    registry = Registry()
    for algorithm in (
        ChannelFourierAlgorithm(),
        TorchChannelConv2dAlgorithm(),
        TorchConv2dAlgorithm(),
        RadiusCellMassAlgorithm(),
        RationalWideAlgorithm(),
        RationalWideAxisGradientAlgorithm(),
        RationalWideScaleWeightAlgorithm(),
        MaterializedAlgorithm(),
        FactoredAlgorithm(),
        NormalizedRadialAlgorithm(),
        NormalizedFullAlgorithm(),
        NormalizedWindowAlgorithm(),
        AtomUpdateAlgorithm(),
        PolarUpdateAlgorithm(),
        FusedPolarUpdateAlgorithm(),
        FlatTorusPolarUpdateAlgorithm(),
        RadiusFlatTorusPolarUpdateAlgorithm(),
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
