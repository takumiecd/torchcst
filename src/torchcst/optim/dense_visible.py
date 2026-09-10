"""Visible-space Adam moments with atom-local CST updates."""

from torchcst._derivatives.local_tangent import AtomLocalTangentGeometry

from ._regularized_update import solve
from .config import DenseVisibleAdamConfig
from .first_order import _TangentModelOptimizer
from .moments import MomentSystem
from .moments.dense_visible import (
    DenseVisibleFirstMoment,
    DenseVisibleSecondMoment,
)


class CSTDenseVisibleAdam(_TangentModelOptimizer):
    """Keep dense visible ``m`` and ``v`` and solve atom-local CST blocks."""

    config_type = DenseVisibleAdamConfig

    def _make_geometry(self, site):
        c = self.cst_config
        if not hasattr(self, "_tangent_operators"):
            self._tangent_operators = {}
        if site not in self._tangent_operators:
            self._tangent_operators[site] = site.cst_derivatives().tangent_ops(
                backend=c.tangent_backend if c.factored_geometry else "reference",
                atom_tile=c.tangent_atom_tile,
                execution="triton"
                if c.device_execution and site.atoms.p.is_cuda
                else "eager",
            )
        ops = self._tangent_operators[site]
        return AtomLocalTangentGeometry(
            ops.derivatives,
            factored=c.factored_geometry,
            ops=ops,
            recompression_rtol=c.solve_rtol,
        )

    def _make_moments(self):
        c = self.cst_config
        return MomentSystem(
            first=DenseVisibleFirstMoment(c.betas[0]),
            second=DenseVisibleSecondMoment(
                c.betas[1],
                eps=c.eps,
                row_chunk=c.row_chunk_size,
                atom_chunk=c.tangent_atom_tile,
            ),
        )

    def _solve(self, context, expanded):
        c = self.cst_config
        return solve(
            expanded.second.metric.blocks,
            expanded.first.corrected.constant,
            c.lr,
            c.update_damping,
            rtol=c.solve_rtol,
        )

    def _validate_solve(self, result, context):
        self._validate_displacement(result, context)

    def _moment_contract(self):
        c = self.cst_config
        return {
            "betas": c.betas,
            "eps": c.eps,
            "update_damping": c.update_damping,
            "second_moment": "dense_visible_diagonal",
        }
