"""Dedicated gather/scatter compute path for the entry family."""

from __future__ import annotations

from torch import Tensor

from torchcst.storage import SynapseStore, SynapseView

from ..capture import flatten_capture_pair
from ._control_base import _ControlLinear


class EntryLinear(_ControlLinear):
    """Sparse linear map backed by entry-family ``(source, target, weight)`` rows.

    A control family, not CST: one atom is one sparse matrix entry -- the
    DST/SET/RigL baseline, and the sigma->0 limit of the triangular kernel
    family.

    The module is a read-only compute edge: it neither creates operations nor
    mutates the store, and ``forward`` materializes no dense weight matrix.
    """

    def __init__(
        self,
        store: SynapseStore,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__(store, in_features, out_features)
        if store.d_in != 1 or store.d_out != 1:
            raise ValueError("EntryLinear requires scalar entry coordinates")
        if store.spec.kernel_in != "delta" or store.spec.kernel_out != "delta":
            raise ValueError("EntryLinear requires delta kernels")

    def _freeze_view(self, view: SynapseView) -> SynapseView:
        source = view.s[:, 0]
        target = view.t[:, 0]
        if bool(((source < 0) | (source >= self.in_features)).any()):
            raise ValueError("entry source coordinate exceeds in_features")
        if bool(((target < 0) | (target >= self.out_features)).any()):
            raise ValueError("entry target coordinate exceeds out_features")
        # The cached read is structural.  Do not retain the index-select
        # autograd graph carried by SynapseView.w across training updates.
        return SynapseView(
            site=view.site,
            version=view.version,
            ids=view.ids,
            s=view.s,
            t=view.t,
            w=view.w.detach(),
            mass=view.mass.detach(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal in_features")
        view = self._view()
        source = view.s[:, 0].to(device=x.device)
        target = view.t[:, 0].to(device=x.device)
        weights = self._live_weights().to(device=x.device)
        contributions = x.index_select(-1, source) * weights
        output = contributions.new_zeros((*x.shape[:-1], self.out_features))
        output = output.index_add(-1, target, contributions)
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
        view = self._view()
        source = view.s[:, 0].to(device=x_flat.device)
        target = view.t[:, 0].to(device=g_flat.device)
        return (x_flat.index_select(1, source) * g_flat.index_select(1, target)).sum(
            dim=0
        )

    def dense_weight(self) -> Tensor:
        """Materialize ``[out_features, in_features]`` solely for tests/debugging."""
        view = self._view()
        source = view.s[:, 0].to(device=self.store.w.device)
        target = view.t[:, 0].to(device=self.store.w.device)
        flat_index = target * self.in_features + source
        dense = self._live_weights().new_zeros(self.out_features * self.in_features)
        return dense.index_add(0, flat_index, self._live_weights()).reshape(
            self.out_features, self.in_features
        )
