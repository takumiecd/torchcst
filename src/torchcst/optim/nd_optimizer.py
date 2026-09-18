"""Shared coordinator for composable N/D CST optimizers.

Normalized and Quadratic wrappers select moments and a solver.  This
coordinator owns sites, expands ``N`` and ``D``, and commits the accepted
displacement.  It does not choose whether the solver replaces ``d`` with
``-η N/D`` or adds that vector to ``d``.
"""

from __future__ import annotations

import torch

from torchcst._derivatives.frame import AutogradFrameGeometry
from torchcst.nn import CSTLinear

from .atom_grad import LinearJGAtomGrad, LinearJGHAtomGrad
from .moments import (
    DenominatorMoment,
    NormalizedMomentSystem,
    NumeratorMoment,
    UnitDenominator,
)
from .optimizer import _ModelOptimizer
from .solvers import NormalizedSolveResult, NormalizedUpdateProblem


class _NDModelOptimizer(_ModelOptimizer):
    """Common coordinator for the composable N/D optimizer families."""

    _use_numerator_moment = False
    _use_denominator_moment = False

    def _make_geometry(self, site: CSTLinear) -> AutogradFrameGeometry:
        # The N/D system works directly in fixed atom coordinates, but the
        # shared model coordinator still uses a geometry as its validated
        # point/context boundary.  Cache this lightweight reference
        # geometry so it is not rebuilt for every step.
        if not hasattr(self, "_nd_geometries"):
            self._nd_geometries = {}
        if site not in self._nd_geometries:
            self._nd_geometries[site] = site.cst_frame_geometry()
        return self._nd_geometries[site]

    def _make_atom_grad(
        self, site: CSTLinear, moments
    ) -> LinearJGAtomGrad | LinearJGHAtomGrad:
        c = self.cst_config
        atom_grad_type = (
            LinearJGHAtomGrad if self._uses_curvature else LinearJGAtomGrad
        )
        return atom_grad_type(mode=c.atom_grad_mode, factored=c.factored)

    def _make_moments(self) -> NormalizedMomentSystem:
        c = self.cst_config
        include_curvature = self._uses_curvature
        numerator = NumeratorMoment(
            c.betas[0] if self._use_numerator_moment else 0.0,
            include_curvature=include_curvature,
        )
        denominator = (
            DenominatorMoment(
                c.betas[1],
                eps=c.eps,
                include_curvature=include_curvature,
            )
            if self._use_denominator_moment
            else UnitDenominator()
        )
        return NormalizedMomentSystem(
            numerator=numerator,
            denominator=denominator,
        )

    @property
    def _uses_curvature(self) -> bool:
        """Whether the configured solver can evaluate a curvature correction."""

        max_iter = getattr(self.cst_config.solver, "max_iter", None)
        # Injected solvers without a max_iter contract retain the historical
        # JGH path; only the explicit max_iter=1 mode selects first order.
        return max_iter is None or max_iter >= 2

    def _solve(self, context, expanded):
        del context
        c = self.cst_config
        problem = NormalizedUpdateProblem(
            expanded.numerator,
            expanded.denominator,
            learning_rate=c.lr,
        )
        if c.initial_zero_step and expanded.previous_step == 0:
            displacement = c.solver.project_displacement(
                problem.scaled_step_at_zero(),
                trust_radius=c.trust_radius,
            )
            return NormalizedSolveResult(
                displacement=displacement.detach(),
                residual_norm=torch.zeros(
                    (), device=problem.device, dtype=problem.dtype
                ),
                iterations=0,
                converged=True,
                on_boundary=c.solver.displacement_is_on_boundary(
                    displacement,
                    trust_radius=c.trust_radius,
                ),
                solver_mode="zero_point",
            )
        return c.solver.solve(problem, trust_radius=c.trust_radius)

    def _validate_solve(self, solve, context) -> None:
        if not isinstance(solve, NormalizedSolveResult):
            raise TypeError("N/D solver must return a NormalizedSolveResult")
        displacement = solve.displacement
        point = context.current_point
        if displacement.shape != point.shape:
            raise ValueError("CST displacement has the wrong shape")
        if displacement.device != point.device or displacement.dtype != point.dtype:
            raise ValueError("CST displacement must match its CST point")
        if not bool(torch.isfinite(displacement).all()):
            raise FloatingPointError("CST displacement must be finite")
        if not self.cst_config.solver.displacement_is_valid(
            displacement,
            trust_radius=self.cst_config.trust_radius,
        ):
            raise ValueError("CST displacement exceeds the solver constraint")

    def _moment_contract(self):
        c = self.cst_config
        solver = c.solver
        return {
            "betas": c.betas,
            "eps": c.eps,
            "numerator": "ema" if self._use_numerator_moment else "current",
            "denominator": "ema" if self._use_denominator_moment else "unit",
            "initial_zero_step": c.initial_zero_step,
            "solver": {
                "type": type(solver).__name__,
                "max_iter": getattr(solver, "max_iter", None),
                "tolerance": getattr(solver, "tolerance", None),
                "damping": getattr(solver, "damping", None),
            },
        }
