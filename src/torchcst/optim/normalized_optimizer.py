"""Composable CST optimizers built from independent N, D, and solver parts."""

from __future__ import annotations

from .config import NormalizedOptimizerConfig
from .nd_optimizer import _NDModelOptimizer


class _NormalizedModelOptimizer(_NDModelOptimizer):
    """Implicit family: ``d ← Π(-η N(d)/D(d))``."""

    config_type = NormalizedOptimizerConfig


class CSTNormalizedSGD(_NormalizedModelOptimizer):
    """CST SGD: current affine numerator divided by one."""


class CSTNormalizedMomentum(_NormalizedModelOptimizer):
    """CST Momentum: EMA affine numerator divided by one."""

    _use_numerator_moment = True


class CSTNormalizedRMSProp(_NormalizedModelOptimizer):
    """CST RMSProp: current affine numerator and EMA denominator."""

    _use_denominator_moment = True


class CSTNormalizedAdam(_NormalizedModelOptimizer):
    """CST Adam: EMA affine numerator and EMA denominator."""

    _use_numerator_moment = True
    _use_denominator_moment = True


CSTSGD = CSTNormalizedSGD
CSTMomentum = CSTNormalizedMomentum
CSTRMSProp = CSTNormalizedRMSProp
CSTImplicitAdam = CSTNormalizedAdam
