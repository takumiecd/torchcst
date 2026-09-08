"""Primary tangent-only CST optimizer."""

from .config import FirstOrderAdamConfig
from .moments import MomentSystem, SeparableDiagonalSecondMoment
from .optimizer import _ModelOptimizer


class CSTAdam(_ModelOptimizer):
    """First-order CST Adam with current-tangent moment compression/transport.

    Keyword options construct FirstOrderAdamConfig; alternatively pass cst=.
    Experimental atom_block/atom_diag RMS differ from diagonal Adam.
    """

    config_type = FirstOrderAdamConfig

    def _make_geometry(self, site):
        from torchcst._derivatives.tangent import TangentGeometry

        return TangentGeometry(
            site.cst_derivatives(), factored=self.cst_config.factored_geometry
        )

    def _make_moments(self):
        from .moments.atom_rms import AtomRMS
        from .moments.tangent import TangentFirstMoment

        c = self.cst_config
        second = (
            SeparableDiagonalSecondMoment(c.betas[1], eps=c.eps)
            if c.second_moment == "separable"
            else AtomRMS(
                c.betas[1],
                eps=c.eps,
                diagonal=c.second_moment == "atom_diag",
                rtol=c.tangent_rtol,
            )
        )
        return MomentSystem(
            first=TangentFirstMoment(c.betas[0], damping=c.first_moment_damping),
            second=second,
        )

    def _solve(self, context, expanded):
        from .quadratic import TangentProblem, solve_tangent

        problem = TangentProblem(context, expanded, learning_rate=self.cst_config.lr)
        return solve_tangent(problem, radius=self.cst_config.trust_radius)
