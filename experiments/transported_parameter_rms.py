"""Atom-local transported outer products of the accumulated parameter gradient."""

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
    @property
    def observation_request(self):
        return AtomGradRequest(jg=True)

    def expand(self, state, observation, context, *, next_step):
        observation.require(self.observation_request)
        point, geometry = context.current_point, context.geometry
        gram = geometry.cross(point, point, local=True).double()
        gram = (gram + gram.transpose(-1, -2)) * 0.5
        values, vectors = torch.linalg.eigh(gram)
        active = values > self.rtol * values.clamp_min(0).amax(-1, keepdim=True)
        scale = torch.where(
            active, values.clamp_min(torch.finfo(values.dtype).tiny).rsqrt(), 0
        )
        basis = vectors * scale.unsqueeze(-2)
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
