"""Stage 4: pure, engine-free init/birth-placement functions.

See :mod:`torchcst.init.init_placement` for the full contract
(``docs/absorb-and-gram-design.md`` section 6).
"""

from .init_placement import (
    GreedySelection,
    coverage_lattice,
    logdet_greedy,
    to_synapse_births,
    variance_amplitudes,
)

__all__ = [
    "GreedySelection",
    "coverage_lattice",
    "logdet_greedy",
    "to_synapse_births",
    "variance_amplitudes",
]
