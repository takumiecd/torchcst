"""Low-rank certificate accumulated cheaply at backward finalization."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from cstf.compute import Observation
from cstf.storage import SynapseStore


class CertificateSnapshot(NamedTuple):
    """SVD reading plus the two registered scalar audit values."""

    U_r: Tensor
    S: Tensor
    V_r: Tensor
    sigma2_over_sigma1: float
    participation_ratio: float


class CertificateSubspace:
    """Accumulate ``G += sum weight * g_out.T @ x``; factor only on snapshot."""

    name = "certificate_subspace"

    def __init__(self, store: SynapseStore | None = None, rank: int = 1) -> None:
        if isinstance(store, int) and not isinstance(store, bool):
            if rank != 1:
                raise ValueError("rank was specified twice")
            rank = store
            store = None
        if store is not None and not isinstance(store, SynapseStore):
            raise TypeError("store must be a SynapseStore or None")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an int")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.store = store
        self.rank = rank
        self.G: Tensor | None = None
        if store is not None:
            self.G = store.w.detach().new_zeros((store.d_out, store.d_in))

    @property
    def has_signal(self) -> bool:
        return self.G is not None and bool(torch.count_nonzero(self.G))

    def accumulate(self, x: Tensor, g_out: Tensor, weight: float = 1.0) -> None:
        """Add one weighted outer-product sum without computing an SVD."""
        if not isinstance(x, Tensor) or not isinstance(g_out, Tensor):
            raise TypeError("x and g_out must be Tensors")
        if x.ndim == 0 or g_out.ndim == 0:
            raise ValueError("x and g_out must have feature dimensions")
        x_flat = x.detach().reshape(-1, x.shape[-1])
        g_flat = g_out.detach().reshape(-1, g_out.shape[-1])
        if x_flat.shape[0] != g_flat.shape[0]:
            raise ValueError("captured x and g_out batch dimensions do not align")
        contribution = float(weight) * (g_flat.transpose(0, 1) @ x_flat)
        if self.G is None:
            self.G = contribution.detach().clone()
        else:
            if self.G.shape != contribution.shape:
                raise ValueError("certificate observation dimensions changed")
            self.G = self.G.to(contribution) + contribution
        self.G = self.G.detach()

    def update(self, observations: tuple[Observation, ...]) -> None:
        """Accumulate a finalized update's observations."""
        for observation in observations:
            self.accumulate(
                observation.x, observation.g_out, observation.micro_weight
            )

    def snapshot(self) -> CertificateSnapshot:
        """Compute the only SVD in the lifecycle, immediately before an event."""
        if self.G is None:
            raise RuntimeError("certificate dimensions are not initialized")
        matrix = self.G.detach()
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        count = min(self.rank, singular.numel())
        u_r = u[:, :count].clone()
        v_r = vh[:count].clone()
        sigma1 = float(singular[0]) if singular.numel() else 0.0
        sigma2 = float(singular[1]) if singular.numel() > 1 else 0.0
        ratio = sigma2 / sigma1 if sigma1 > 0 else 0.0
        if u_r.shape[1]:
            leading = u_r[:, 0]
            denominator = float(leading.square().sum())
            participation = (
                float(leading.abs().sum().square()) / denominator
                if denominator > 0
                else 0.0
            )
        else:
            participation = 0.0
        return CertificateSnapshot(
            u_r,
            singular[:count].clone(),
            v_r,
            ratio,
            participation,
        )

    def reset(self) -> None:
        """Begin the next observation window without changing shape/device."""
        if self.G is not None:
            self.G.zero_()
