"""Composable CST optimizers built from independent N, D, and solver parts."""

from __future__ import annotations

import torch

from torchcst._derivatives.frame import AutogradFrameGeometry
from torchcst.nn import CSTLinear

from .atom_grad import LinearJGHAtomGrad
from .config import NormalizedOptimizerConfig
from .moments import (
    DenominatorMoment,
    NormalizedMomentSystem,
    NumeratorMoment,
    UnitDenominator,
)
from .optimizer import _ModelOptimizer
from .solvers import NormalizedSolveResult, NormalizedUpdateProblem


class _NormalizedModelOptimizer(_ModelOptimizer):
    """Common coordinator for the composable normalized optimizer family."""

    config_type = NormalizedOptimizerConfig
    _use_numerator_moment = False
    _use_denominator_moment = False

    def _make_geometry(self, site: CSTLinear) -> AutogradFrameGeometry:
        # The normalized N/D system works directly in fixed atom coordinates,
        # but the shared model coordinator still uses a geometry as its
        # validated point/context boundary.  Cache this lightweight reference
        # geometry so it is not rebuilt for every step.
        if not hasattr(self, "_normalized_geometries"):
            self._normalized_geometries = {}
        if site not in self._normalized_geometries:
            self._normalized_geometries[site] = site.cst_frame_geometry()
        return self._normalized_geometries[site]

    def _make_atom_grad(self, site: CSTLinear, moments) -> LinearJGHAtomGrad:
        c = self.cst_config
        return LinearJGHAtomGrad(mode=c.atom_grad_mode, factored=c.factored)

    def _make_moments(self) -> NormalizedMomentSystem:
        c = self.cst_config
        numerator = NumeratorMoment(c.betas[0] if self._use_numerator_moment else 0.0)
        denominator = (
            DenominatorMoment(c.betas[1], eps=c.eps)
            if self._use_denominator_moment
            else UnitDenominator()
        )
        return NormalizedMomentSystem(
            numerator=numerator,
            denominator=denominator,
        )

    def _solve(self, context, expanded):
        del context
        c = self.cst_config
        problem = NormalizedUpdateProblem(
            expanded.numerator,
            expanded.denominator,
            learning_rate=c.lr,
        )
        return c.solver.solve(problem, trust_radius=c.trust_radius)

    def _validate_solve(self, solve, context) -> None:
        if not isinstance(solve, NormalizedSolveResult):
            raise TypeError("normalized solver must return a NormalizedSolveResult")
        displacement = solve.displacement
        point = context.current_point
        if displacement.shape != point.shape:
            raise ValueError("CST displacement has the wrong shape")
        if displacement.device != point.device or displacement.dtype != point.dtype:
            raise ValueError("CST displacement must match its CST point")
        if not bool(torch.isfinite(displacement).all()):
            raise FloatingPointError("CST displacement must be finite")
        norm = torch.linalg.vector_norm(displacement)
        if not bool(norm <= self.cst_config.trust_radius * (1.0 + 10.0 * torch.finfo(displacement.dtype).eps)):
            raise ValueError("CST displacement exceeds the trust radius")

    def _moment_contract(self):
        c = self.cst_config
        solver = c.solver
        return {
            "betas": c.betas,
            "eps": c.eps,
            "numerator": "ema" if self._use_numerator_moment else "current",
            "denominator": "ema" if self._use_denominator_moment else "unit",
            "solver": {
                "type": type(solver).__name__,
                "max_iter": getattr(solver, "max_iter", None),
                "tolerance": getattr(solver, "tolerance", None),
                "damping": getattr(solver, "damping", None),
            },
        }


class CSTSGD(_NormalizedModelOptimizer):
    """CST SGD: current affine numerator divided by one."""


class CSTMomentum(_NormalizedModelOptimizer):
    """CST Momentum: EMA affine numerator divided by one."""

    _use_numerator_moment = True


class CSTRMSProp(_NormalizedModelOptimizer):
    """CST RMSProp: current affine numerator and EMA denominator."""

    _use_denominator_moment = True


class CSTNormalizedAdam(_NormalizedModelOptimizer):
    """CST Adam: EMA affine numerator and EMA denominator."""

    _use_numerator_moment = True
    _use_denominator_moment = True


# A descriptive alias for callers who want to emphasize that this is the new
# composable family while the historical CSTAdam API remains available.
CSTImplicitAdam = CSTNormalizedAdam

