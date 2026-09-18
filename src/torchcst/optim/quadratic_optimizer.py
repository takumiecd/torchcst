"""CST SGD, Momentum, RMSProp, and Adam on the local Taylor model.

This family uses the shared N/D coordinator.  The Quadratic solvers add
``-η N(d)/D(d)`` to the current displacement instead of replacing ``d``
with that vector.
"""

from __future__ import annotations

from .config import QuadraticOptimizerConfig
from .nd_optimizer import _NDModelOptimizer


class _QuadraticModelOptimizer(_NDModelOptimizer):
    """Gradient family: ``d ← Π(d - η N(d)/D(d))``."""

    config_type = QuadraticOptimizerConfig


class CSTQuadraticSGD(_QuadraticModelOptimizer):
    """Quadratic SGD: current ``N`` divided by one."""


class CSTQuadraticMomentum(_QuadraticModelOptimizer):
    """Quadratic Momentum: EMA ``N`` divided by one."""

    _use_numerator_moment = True


class CSTQuadraticRMSProp(_QuadraticModelOptimizer):
    """Quadratic RMSProp: current ``N`` and EMA ``D``."""

    _use_denominator_moment = True


class CSTQuadraticAdam(_QuadraticModelOptimizer):
    """Quadratic Adam: EMA ``N`` and EMA ``D``."""

    _use_numerator_moment = True
    _use_denominator_moment = True
