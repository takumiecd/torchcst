"""Primary tangent-only CST optimizer."""

from .config import FirstOrderAdamConfig
from .moments import MomentSystem, SeparableDiagonalSecondMoment
from .optimizer import _ModelOptimizer


class _TangentModelOptimizer(_ModelOptimizer):
    """Shared fixed-tangent configuration and checkpoint validation."""

    def _make_geometry(self, site):
        from torchcst._derivatives.tangent import TangentGeometry

        c = self.cst_config
        if not hasattr(self, "_tangent_operators"):
            self._tangent_operators = {}
        if site not in self._tangent_operators:
            self._tangent_operators[site] = site.cst_derivatives().tangent_ops(
                backend=c.tangent_backend if c.factored_geometry else "reference",
                atom_tile=c.tangent_atom_tile,
                gram_action=c.recompression_action,
                execution="triton"
                if c.device_execution and site.atoms.p.is_cuda
                else "eager",
            )
        ops = self._tangent_operators[site]
        if c.update_solver in ("pcg", "krylov") and ops.backend == "reference":
            raise ValueError("Matrix-free update requires a factorized tangent backend")
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


class CSTAdam(_TangentModelOptimizer):
    """First-order CST Adam with tangent moment transport and a trust region."""

    config_type = FirstOrderAdamConfig

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

        if self.cst_config.update_solver == "krylov":
            from ._trust_krylov import solve
            from .quadratic import MatrixFreeTangentProblem

            c = self.cst_config
            return solve(
                MatrixFreeTangentProblem(context, expanded, learning_rate=c.lr),
                radius=c.trust_radius,
                basis_size=c.update_basis_size,
                basis_memory_mb=c.update_basis_memory_mb,
                check_interval=c.update_check_interval,
                rtol=c.update_rtol,
            )
        if self.cst_config.update_approximation != "full":
            from ._local_tangent import solve
            from .quadratic import MatrixFreeTangentProblem

            c = self.cst_config
            return solve(
                MatrixFreeTangentProblem(context, expanded, learning_rate=c.lr),
                approximation=c.update_approximation,
                radius=c.trust_radius,
                rtol=c.update_rtol,
            )
        if self.cst_config.update_solver == "pcg":
            from ._trust_pcg import solve
            from .quadratic import MatrixFreeTangentProblem

            c = self.cst_config
            problem = MatrixFreeTangentProblem(context, expanded, learning_rate=c.lr)
            return solve(
                problem,
                radius=c.trust_radius,
                max_iter=c.update_max_iter,
                shift_steps=c.update_shift_steps,
                rtol=c.update_rtol,
            )
        problem = TangentProblem(context, expanded, learning_rate=self.cst_config.lr)
        return solve_tangent(problem, radius=self.cst_config.trust_radius)
