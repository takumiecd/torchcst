"""Shared scaffold for the control-family linears (entry, rank-one)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst._validation import require_int
from torchcst.storage import SynapseStore, SynapseView

from ..capture import BackwardContext, register_capture_hook


class _ControlLinear(nn.Module):
    """Store wiring, the version-keyed structural cache, and capture plumbing.

    Concrete controls implement :meth:`_freeze_view` (what to cache per
    structural version) plus their own forward/atom_grads/dense_weight math.
    """

    def __init__(
        self,
        store: SynapseStore,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if not isinstance(store, SynapseStore):
            raise TypeError("store must be a SynapseStore")
        require_int(in_features, "in_features", minimum=1)
        require_int(out_features, "out_features", minimum=1)
        # Keep the store registered as a child so model.parameters() includes w.
        self.store = store
        self.in_features = in_features
        self.out_features = out_features
        self.capture_site = store.site
        self._cached_version = -1
        self._cached_view: SynapseView | None = None
        self._cached_slots = torch.zeros(0, dtype=torch.int64)
        self._backward_context: BackwardContext | None = None

    def _freeze_view(self, view: SynapseView) -> SynapseView:
        """Return the structural snapshot this control caches per version."""
        raise NotImplementedError

    def _view(self) -> SynapseView:
        if self._cached_version != self.store.version:
            view = self.store.view()
            self._cached_view = self._freeze_view(view)
            self._cached_slots = self.store._slots.slots_of(view.ids)
            self._cached_version = view.version
        assert self._cached_view is not None
        return self._cached_view

    def _live_weights(self) -> Tensor:
        slots = self._cached_slots.to(device=self.store.w.device)
        return self.store.w.index_select(0, slots)

    def set_backward_context(self, context: BackwardContext | None) -> None:
        """Enable capture for one engine-owned update, or disable it."""
        if context is not None and not isinstance(context, BackwardContext):
            raise TypeError("context must be a BackwardContext or None")
        self._backward_context = context

    @property
    def capture_enabled(self) -> bool:
        return self._backward_context is not None

    def _capture_output(self, output: Tensor, x: Tensor, version: int) -> Tensor:
        """Arm the capture hook when an engine update is in flight."""
        if self._backward_context is not None:
            register_capture_hook(
                output, self._backward_context, self.capture_site, x, version
            )
        return output
