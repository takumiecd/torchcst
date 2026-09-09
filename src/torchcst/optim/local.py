"""Public atom-local optimizer using the common mixed-model coordinator."""

import torch

from torchcst._derivatives.local_tangent import AtomLocalTangentGeometry

from ._regularized_update import solve
from .config import LocalAdamConfig
from .first_order import _TangentModelOptimizer
from .moments import MomentSystem
from .moments.atom_rms import AtomRMSState
from .moments.tangent import TangentFirstMoment
from .moments.transported_rms import CholeskyParameterRMS, TransportedParameterRMS


class CSTLocalAdam(_TangentModelOptimizer):
    """Atom-local transported first/second moments without a trust region.

    Pass a model, with dense=AdamWConfig(...) when it has ordinary trainable
    parameters. Cross-atom history and covariance are deliberately omitted.
    """

    config_type = LocalAdamConfig

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
        second = (
            CholeskyParameterRMS(c.betas[1], eps=c.eps, damping=c.first_moment_damping)
            if c.whitening == "cholesky"
            else TransportedParameterRMS(c.betas[1], eps=c.eps, rtol=c.tangent_rtol)
        )
        return MomentSystem(
            first=TangentFirstMoment(c.betas[0], damping=c.first_moment_damping),
            second=second,
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
        contract = {
            "betas": c.betas,
            "eps": c.eps,
            "first_moment_damping": c.first_moment_damping,
            "update_damping": c.update_damping,
            "tangent_rtol": c.tangent_rtol,
            "second_moment": "transported_parameter_outer_product",
        }

        # Keep existing eigen checkpoints compatible. A changed coordinate
        # convention must never silently reinterpret the stored C and basis.
        if c.whitening != "eigen":
            contract["whitening"] = c.whitening
        return contract

    @classmethod
    def _map_state(cls, value, *, clone=False, reference=None):
        if isinstance(value, AtomRMSState) and reference is not None:
            # Basis precision is part of the transport state, independently of
            # the parameter dtype. Preserve it across checkpoint restoration.
            return AtomRMSState(
                cls._map_state(value.point, clone=clone, reference=reference),
                value.basis.detach()
                .to(device=reference.device, dtype=torch.float64)
                .clone(),
                cls._map_state(value.value, clone=clone, reference=reference),
                value.beta_power,
            )
        return super()._map_state(value, clone=clone, reference=reference)
