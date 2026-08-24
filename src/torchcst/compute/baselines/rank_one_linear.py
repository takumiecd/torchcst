"""Factorized compute path for Sphere×Sphere rank-one atoms."""

from __future__ import annotations

from torch import Tensor

from torchcst.storage import SynapseStore, SynapseView

from ..capture import flatten_capture_pair
from ._control_base import _ControlLinear


class RankOneLinear(_ControlLinear):
    """Linear map ``sum_k c_k u_k v_k^T`` without materializing dense ``W``.

    A control family, not CST: a chart-free rank-one control (free ``u``,
    ``v``; no factor, no bandwidth) that sits outside the theory's
    representation class and upper-bounds what coordinates alone can buy.
    """

    def __init__(
        self,
        store: SynapseStore,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__(store, in_features, out_features)
        if store.d_in != in_features or store.d_out != out_features:
            raise ValueError("rank-one coordinate widths must equal feature widths")
        if store.spec.factor_in != "dot" or store.spec.factor_out != "dot":
            raise ValueError("RankOneLinear requires dot factors")

    def _freeze_view(self, view: SynapseView) -> SynapseView:
        # Cache only detached structure; fresh differentiable gathers are
        # made from store Parameters on every forward.
        return SynapseView(
            site=view.site,
            version=view.version,
            ids=view.ids,
            s=view.s.detach(),
            t=view.t.detach(),
            w=view.w.detach(),
            mass=view.mass.detach(),
        )

    def _live_factors(self) -> tuple[Tensor, Tensor, Tensor]:
        slots = self._cached_slots.to(device=self.store.w.device)
        return (
            self.store.s.index_select(0, slots),
            self.store.t.index_select(0, slots),
            self.store.w.index_select(0, slots),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal in_features")
        view = self._view()
        source, target, weights = self._live_factors()
        source = source.to(device=x.device)
        target = target.to(device=x.device)
        weights = weights.to(device=x.device)
        output = ((x @ source.transpose(0, 1)) * weights) @ target
        return self._capture_output(output, x, view.version)

    def atom_grads(self, x: Tensor, g_out: Tensor) -> Tensor:
        """Return signed ``dL/dw`` contributions in packed live-ID order."""
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal in_features")
        if g_out.ndim == 0 or g_out.shape[-1] != self.out_features:
            raise ValueError("g_out's final dimension must equal out_features")
        x_flat, g_flat = flatten_capture_pair(
            x, g_out, self.in_features, self.out_features
        )
        self._view()
        source, target, _ = self._live_factors()
        source = source.detach().to(x_flat)
        target = target.detach().to(g_flat)
        return (
            (x_flat @ source.transpose(0, 1)) * (g_flat @ target.transpose(0, 1))
        ).sum(0)

    def dense_weight(self) -> Tensor:
        """Materialize ``[out_features, in_features]`` for tests/debugging."""
        self._view()
        source, target, weights = self._live_factors()
        return (target.transpose(0, 1) * weights) @ source
