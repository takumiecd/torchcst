"""Low-rank certificate accumulated cheaply at backward finalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import Tensor

from torchcst._validation import require_int
from torchcst.compute import Observation, flatten_capture_pair
from torchcst.storage import SynapseStore

from .base import WeightedMeasurement, weighted_sum


class CertificateSnapshot(NamedTuple):
    """SVD reading plus the two registered scalar audit values."""

    U_r: Tensor
    S: Tensor
    V_r: Tensor
    sigma2_over_sigma1: float
    participation_ratio: float


def _participation_ratio(u_r: Tensor) -> float:
    """``(sum|u|)^2 / sum u^2`` of the leading left singular vector."""
    if not u_r.shape[1]:
        return 0.0
    leading = u_r[:, 0]
    denominator = float(leading.square().sum())
    if denominator <= 0:
        return 0.0
    return float(leading.abs().sum().square()) / denominator


class CertificateSubspace:
    """Accumulate ``G += sum weight * g_out.T @ x``; factor only on snapshot.

    Four accumulation entry points feed the same ``G``, one per caller shape:
    :meth:`accumulate` takes raw boundary tensors, :meth:`update` a finalized
    update's :class:`~torchcst.compute.Observation` tuple,
    :meth:`update_reduced` an already-reduced ``g_out.T @ x`` matrix, and
    :meth:`finalize_update` is the engine's instrument boundary (weighted
    measurements).
    """

    name = "certificate_subspace"

    def __init__(self, store: SynapseStore | None = None, rank: int = 1) -> None:
        if store is not None and not isinstance(store, SynapseStore):
            raise TypeError("store must be a SynapseStore or None")
        require_int(rank, "rank", minimum=1)
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
        x_flat, g_flat = flatten_capture_pair(x, g_out, x.shape[-1], g_out.shape[-1])
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
            self.accumulate(observation.x, observation.g_out, observation.micro_weight)

    def update_reduced(self, contribution: Tensor) -> None:
        """Accumulate one already-reduced ``g_out.T @ x`` contribution."""
        if not isinstance(contribution, Tensor):
            raise TypeError("contribution must be a Tensor")
        if contribution.ndim != 2:
            raise ValueError("reduced certificate contribution must be rank 2")
        value = contribution.detach()
        if self.G is None:
            self.G = value.clone()
        else:
            if self.G.shape != value.shape:
                raise ValueError("certificate contribution dimensions changed")
            self.G = self.G.to(value) + value
        self.G = self.G.detach()

    def prepare(self, view: Any, module: Any) -> None:
        """Certificate dimensions are stable; no per-update preparation is needed."""
        del view, module

    def _measure(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        del module
        x_flat, g_flat = flatten_capture_pair(x, g_out, x.shape[-1], g_out.shape[-1])
        return {"matrix": g_flat.transpose(0, 1) @ x_flat}

    def reduce_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        """Reduce one signed certificate contribution inside backward."""
        return self._measure(module, x, g_out)

    def measure_after_backward(
        self, module: Any, x: Tensor, g_out: Tensor
    ) -> dict[str, Tensor]:
        """Measure one signed certificate contribution after backward."""
        return self._measure(module, x, g_out)

    def finalize_update(
        self,
        measurements: tuple[WeightedMeasurement, ...],
        view: Any,
    ) -> None:
        """Accumulate the weighted signed certificate at the update boundary."""
        del view
        contribution = weighted_sum(measurements, "matrix")
        if contribution is not None:
            self.update_reduced(contribution)

    def snapshot(self) -> CertificateSnapshot:
        """Compute the only SVD in the lifecycle, immediately before an event."""
        if self.G is None:
            raise RuntimeError("certificate dimensions are not initialized")
        u, singular, vh = torch.linalg.svd(self.G.detach(), full_matrices=False)
        count = min(self.rank, singular.numel())
        u_r = u[:, :count].clone()
        v_r = vh[:count].clone()
        sigma1 = float(singular[0]) if singular.numel() else 0.0
        sigma2 = float(singular[1]) if singular.numel() > 1 else 0.0
        ratio = sigma2 / sigma1 if sigma1 > 0 else 0.0
        return CertificateSnapshot(
            u_r,
            singular[:count].clone(),
            v_r,
            ratio,
            _participation_ratio(u_r),
        )

    def reset(self) -> None:
        """Begin the next observation window without changing shape/device."""
        if self.G is not None:
            self.G.zero_()

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "torchcst-certificate-subspace-v1",
            "G": None if self.G is None else self.G.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-certificate-subspace-v1"
        ):
            raise ValueError("unsupported CertificateSubspace state schema")
        value = state.get("G")
        if value is not None and not isinstance(value, Tensor):
            raise TypeError("CertificateSubspace G must be a Tensor or None")
        self.G = None if value is None else value.detach().clone()
