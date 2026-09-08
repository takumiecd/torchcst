"""Experimental RMS in independently transported atom tangent spaces.

This is not diagonal Adam. Cross-atom metric terms are omitted. The diagonal
variant discards within-atom off-diagonals after each transport/observation.
"""

import math
from dataclasses import dataclass

import torch

from ..atom_grad import AtomGradRequest
from .base import ExpandedSecondMoment, SecondMomentComponent


def symmetric_root(matrix):
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.transpose(-1, -2)))
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(
        -1, -2
    )


@dataclass(frozen=True)
class AtomRMSState:
    point: torch.Tensor
    basis: torch.Tensor
    value: torch.Tensor
    beta_power: float


@dataclass(frozen=True)
class AtomBlockMetric:
    blocks: torch.Tensor
    visible_shape: tuple[int, ...]


class AtomRMS(SecondMomentComponent):
    def __init__(self, beta, *, eps, diagonal=False, rtol=1e-6):
        self.beta = beta
        self.eps = eps
        self.diagonal = diagonal
        self.rtol = rtol

    @property
    def observation_request(self):
        return AtomGradRequest(atom_square=True)

    def initialize(self, context):
        point = context.current_point
        k, q = point.shape
        shape = (k, q) if self.diagonal else (k, q, q)
        return AtomRMSState(
            point.clone(), point.new_zeros(k, q, q), point.new_zeros(shape), 1.0
        )

    def expand(self, state, observation, context, *, next_step):
        if not isinstance(state, AtomRMSState):
            raise TypeError("expected an atom RMS state")
        if not math.isclose(state.beta_power, self.beta ** (next_step - 1)):
            raise ValueError("atom RMS step or beta mismatch")
        observation.require(self.observation_request)
        point, geometry = context.current_point, context.geometry
        gram = geometry.cross(point, point, local=True).double()
        gram = 0.5 * (gram + gram.transpose(-1, -2))
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
            old = (
                torch.diag_embed(state.value.double())
                if self.diagonal
                else state.value.double()
            )
            transported = transfer @ old @ transfer.transpose(-1, -2)
        incoming = basis.transpose(-1, -2) @ observation.atom_square.double() @ basis
        value = self.beta * transported + (1 - self.beta) * incoming
        value = 0.5 * (value + value.transpose(-1, -2))
        if self.diagonal:
            value = value.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        power = self.beta**next_step
        corrected = value / (1 - power)
        root = (
            torch.diag_embed(corrected.sqrt())
            if self.diagonal
            else symmetric_root(corrected)
        )
        root = root + self.eps * torch.eye(
            point.shape[-1], device=point.device, dtype=root.dtype
        )
        coordinates = basis.transpose(-1, -2) @ gram
        blocks = coordinates.transpose(-1, -2) @ root @ coordinates
        pending = AtomRMSState(
            point.clone(), basis.to(point).detach(), value.to(point).detach(), power
        )
        return ExpandedSecondMoment(
            AtomBlockMetric(blocks.to(point).detach(), geometry.visible_shape), pending
        )

    def compress(self, expanded, accepted_displacement, context):
        return expanded.pending_state
