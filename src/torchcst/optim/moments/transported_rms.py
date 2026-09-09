"""Atom-local transported outer products of the accumulated parameter gradient."""

import math

import torch

from torchcst.optim.atom_grad import AtomGradRequest
from torchcst.optim.moments.atom_rms import (
    AtomBlockMetric,
    AtomRMS,
    AtomRMSState,
    symmetric_root,
)
from torchcst.optim.moments.base import ExpandedSecondMoment


class TransportedParameterRMS(AtomRMS):
    def initialize(self, context):
        point = context.current_point
        k, q = point.shape
        return AtomRMSState(
            point.clone(),
            point.new_zeros(k, q, q, dtype=torch.float64),
            point.new_zeros(k, q, q),
            1.0,
        )

    @property
    def observation_request(self):
        return AtomGradRequest(jg=True)

    def whitening_basis(self, gram):
        values, vectors = torch.linalg.eigh(gram)
        active = values > self.rtol * values.clamp_min(0).amax(-1, keepdim=True)
        scale = torch.where(
            active, values.clamp_min(torch.finfo(values.dtype).tiny).rsqrt(), 0
        )
        return vectors * scale.unsqueeze(-2)

    def expand(self, state, observation, context, *, next_step):
        if not isinstance(state, AtomRMSState):
            raise TypeError("expected an atom RMS state")
        if not math.isclose(state.beta_power, self.beta ** (next_step - 1)):
            raise ValueError("atom RMS step or beta mismatch")
        observation.require(self.observation_request)
        point, geometry = context.current_point, context.geometry
        gram = geometry.cross(point, point, local=True).double()
        gram = (gram + gram.transpose(-1, -2)) * 0.5
        basis = self.whitening_basis(gram)
        if next_step == 1:
            transported = torch.zeros_like(gram)
        else:
            cross = geometry.cross(point, state.point, local=True).double()
            transfer = basis.transpose(-1, -2) @ cross @ state.basis.double()
            transported = transfer @ state.value.double() @ transfer.transpose(-1, -2)
        h = (basis.transpose(-1, -2) @ observation.jg.double().unsqueeze(-1)).squeeze(
            -1
        )
        value = self.beta * transported + (1 - self.beta) * h.unsqueeze(
            -1
        ) * h.unsqueeze(-2)
        value = (value + value.transpose(-1, -2)) * 0.5
        power = self.beta**next_step
        root = symmetric_root(value / (1 - power)) + self.eps * torch.eye(
            point.shape[-1], device=point.device, dtype=value.dtype
        )
        coordinates = basis.transpose(-1, -2) @ gram
        blocks = coordinates.transpose(-1, -2) @ root @ coordinates
        # C is FP32 for FP32 parameters. Keep small basis/metric calculations FP64.
        pending = AtomRMSState(
            point.clone(), basis.detach(), value.to(point).detach(), power
        )
        return ExpandedSecondMoment(
            AtomBlockMetric(blocks.detach(), geometry.visible_shape), pending
        )


class CholeskyParameterRMS(TransportedParameterRMS):
    """Regularized basis without spectral direction selection.

    B=L^{-T}, L L^T=R+damping I. Q=JB is contractive rather than
    orthonormal when damping is positive. C's PSD square root is unchanged.
    """

    def __init__(self, beta, *, eps, damping):
        super().__init__(beta, eps=eps)
        self.damping = damping

    def whitening_basis(self, gram):
        from torchcst._runtime.validation import require

        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        factor, info = torch.linalg.cholesky_ex(
            gram + self.damping * eye, check_errors=False
        )
        require((info == 0).all(), "Cholesky whitening failed", FloatingPointError)
        basis = torch.linalg.solve_triangular(
            factor.transpose(-1, -2), eye.expand_as(gram), upper=True
        )
        require(
            torch.isfinite(basis).all(),
            "non-finite whitening basis",
            FloatingPointError,
        )
        return basis
