"""Optional second-order CST optimizer."""

from torchcst._derivatives.factored_frame import FactoredFrameGeometry

from .config import SecondOrderAdamConfig
from .moments import (
    AcceptedFrameFirstMoment,
    MomentSystem,
    SeparableDiagonalSecondMoment,
)
from .optimizer import _ModelOptimizer
from .problem import QuarticProblem


class CSTSecondOrderAdam(_ModelOptimizer):
    """Optional full second-order weight map with accepted Taylor-frame moments."""

    config_type = SecondOrderAdamConfig

    def _make_geometry(self, site):
        c = self.cst_config
        geometry = (
            FactoredFrameGeometry(site.cst_derivatives())
            if c.factored_geometry
            else site.cst_frame_geometry()
        )
        geometry.device_solver = c.gram_solver
        geometry.pcg_options = self._pcg_options(c)
        return geometry

    def _make_moments(self):
        c = self.cst_config
        return MomentSystem(
            first=AcceptedFrameFirstMoment(c.betas[0], damping=c.first_moment_damping),
            second=SeparableDiagonalSecondMoment(c.betas[1], eps=c.eps),
        )

    def _solve(self, context, expanded):
        c = self.cst_config
        problem = QuarticProblem(
            context, expanded, learning_rate=c.lr, evaluation=c.quartic_evaluation
        )
        return c.quartic.solve(problem, trust_radius=c.trust_radius)
