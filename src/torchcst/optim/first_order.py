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

        c = self.cst_config
        if not hasattr(self, "_tangent_operators"):
            self._tangent_operators = {}
        if site not in self._tangent_operators:
            self._tangent_operators[site] = site.cst_derivatives().tangent_ops(
                backend=c.tangent_backend if c.factored_geometry else "reference",
                atom_tile=c.tangent_atom_tile,
            )
        ops = self._tangent_operators[site]
        return TangentGeometry(
            ops.derivatives,
            factored=c.factored_geometry,
            ops=ops,
            recompression=c.recompression,
            recompression_max_iter=c.recompression_max_iter,
            recompression_rtol=c.recompression_rtol,
        )

    def _check_tangent_configuration(self):
        for site, ops in self._tangent_operators.items():
            if ops._modules != (site.kernel, site.input_chart, site.output_chart):
                raise ValueError(
                    "fixed kernel/chart configuration changed; rebuild optimizer"
                )
            ops.check_configuration()

    def _begin_capture(self):
        self._check_tangent_configuration()
        super()._begin_capture()

    def _build_cst_proposals(self):
        self._check_tangent_configuration()
        return super()._build_cst_proposals()

    def _tangent_descriptors(self):
        return {
            site.name: self._tangent_operators[site.module].descriptor
            for site in self._sites
        }

    def state_dict(self):
        self._check_tangent_configuration()
        state = super().state_dict()
        state["tangent_descriptors"] = self._tangent_descriptors()
        return state

    def load_state_dict(self, state_dict):
        self._check_tangent_configuration()
        if (
            isinstance(state_dict, dict)
            and state_dict.get("algorithm") == type(self).__name__
            and state_dict.get("tangent_descriptors") != self._tangent_descriptors()
        ):
            raise ValueError(
                "checkpoint tangent descriptors do not match fixed kernel/chart configuration"
            )
        super().load_state_dict(state_dict)

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
