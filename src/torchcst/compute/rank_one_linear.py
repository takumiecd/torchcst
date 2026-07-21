"""Factorized compute path for Sphere×Sphere rank-one atoms."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst.storage import SynapseStore, SynapseView

from .capture import BackwardContext


class RankOneLinear(nn.Module):
    """Linear map ``sum_k c_k u_k v_k^T`` without materializing dense ``W``."""

    def __init__(
        self,
        store: SynapseStore,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if not isinstance(store, SynapseStore):
            raise TypeError("store must be a SynapseStore")
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if store.d_in != in_features or store.d_out != out_features:
            raise ValueError("rank-one coordinate widths must equal feature widths")
        if store.spec.kernel_in != "dot" or store.spec.kernel_out != "dot":
            raise ValueError("RankOneLinear requires dot kernels")
        self.store = store
        self.in_features = in_features
        self.out_features = out_features
        self.capture_site = store.site
        self._cached_version = -1
        self._cached_view: SynapseView | None = None
        self._cached_slots = torch.zeros(0, dtype=torch.int64)
        self._backward_context: BackwardContext | None = None

    def _view(self) -> SynapseView:
        if self._cached_version != self.store.version:
            view = self.store.view()
            # Cache only detached structure; fresh differentiable gathers are
            # made from store Parameters on every forward.
            self._cached_view = SynapseView(
                site=view.site,
                version=view.version,
                ids=view.ids,
                s=view.s.detach(),
                t=view.t.detach(),
                w=view.w.detach(),
                mass=view.mass.detach(),
            )
            self._cached_slots = self.store._slots.slots_of(view.ids)
            self._cached_version = view.version
        assert self._cached_view is not None
        return self._cached_view

    def set_backward_context(self, context: BackwardContext | None) -> None:
        if context is not None and not isinstance(context, BackwardContext):
            raise TypeError("context must be a BackwardContext or None")
        self._backward_context = context

    @property
    def capture_enabled(self) -> bool:
        return self._backward_context is not None

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

        context = self._backward_context
        if context is not None and torch.is_grad_enabled() and output.requires_grad:
            input_fact = x.detach()
            site = self.capture_site
            version = view.version

            def queue(grad_output: Tensor) -> None:
                context.queue(site, input_fact, grad_output.detach(), version)

            output.register_hook(queue)
        return output

    def atom_grads(self, x: Tensor, g_out: Tensor) -> Tensor:
        """Return signed ``dL/dw`` contributions in packed live-ID order."""
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal in_features")
        if g_out.ndim == 0 or g_out.shape[-1] != self.out_features:
            raise ValueError("g_out's final dimension must equal out_features")
        x_flat = x.detach().reshape(-1, self.in_features)
        g_flat = g_out.detach().reshape(-1, self.out_features)
        if x_flat.shape[0] != g_flat.shape[0]:
            raise ValueError("captured x and g_out batch dimensions do not align")
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
