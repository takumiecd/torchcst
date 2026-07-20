"""Dedicated gather/scatter compute path for the entry family."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst.storage import NeuronStore, SynapseStore, SynapseView

from .capture import BackwardContext


class EntryLinear(nn.Module):
    """Sparse linear map backed by entry-family ``(source, target, weight)`` rows.

    The module is a read-only compute edge: it neither creates operations nor
    mutates the store.  No dense weight matrix is materialized by ``forward``.
    """

    def __init__(
        self,
        store: SynapseStore,
        in_features: int,
        out_features: int,
        in_neurons: NeuronStore | None = None,
        out_neurons: NeuronStore | None = None,
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
        if store.d_in != 1 or store.d_out != 1:
            raise ValueError("EntryLinear requires scalar entry coordinates")
        if store.spec.kernel_in != "delta" or store.spec.kernel_out != "delta":
            raise ValueError("EntryLinear requires delta kernels")
        self._validate_neurons(in_neurons, in_features, "in_neurons")
        self._validate_neurons(out_neurons, out_features, "out_neurons")

        # Keep the store registered as a child so model.parameters() includes w.
        self.store = store
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
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
            source = view.s[:, 0]
            target = view.t[:, 0]
            if bool(((source < 0) | (source >= self.in_features)).any()):
                raise ValueError("entry source coordinate exceeds in_features")
            if bool(((target < 0) | (target >= self.out_features)).any()):
                raise ValueError("entry target coordinate exceeds out_features")
            # The cached read is structural.  Do not retain the index-select
            # autograd graph carried by SynapseView.w across training updates.
            self._cached_view = SynapseView(
                site=view.site,
                version=view.version,
                ids=view.ids,
                s=view.s,
                t=view.t,
                w=view.w.detach(),
                mass=view.mass.detach(),
            )
            self._cached_slots = self.store._slots.slots_of(view.ids)
            self._cached_version = view.version
        assert self._cached_view is not None
        return self._cached_view

    def set_backward_context(self, context: BackwardContext | None) -> None:
        """Enable capture for one engine-owned update, or disable it."""
        if context is not None and not isinstance(context, BackwardContext):
            raise TypeError("context must be a BackwardContext or None")
        self._backward_context = context

    @property
    def capture_enabled(self) -> bool:
        return self._backward_context is not None

    def _live_weights(self) -> Tensor:
        slots = self._cached_slots.to(device=self.store.w.device)
        return self.store.w.index_select(0, slots)

    @staticmethod
    def _validate_neurons(
        store: NeuronStore | None, width: int, name: str
    ) -> None:
        if store is not None and not isinstance(store, NeuronStore):
            raise TypeError(f"{name} must be a NeuronStore or None")
        if store is not None and store.n_max != width:
            raise ValueError(f"{name}.n_max must equal its feature width")

    @staticmethod
    def _gate(
        store: NeuronStore | None, reference: Tensor
    ) -> Tensor | None:
        if store is None:
            return None
        return store.gate_vector().to(device=reference.device, dtype=reference.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal in_features")
        view = self._view()
        in_gate = self._gate(self.in_neurons, x)
        gated_x = x if in_gate is None else x * in_gate
        source = view.s[:, 0].to(device=x.device)
        target = view.t[:, 0].to(device=x.device)
        weights = self._live_weights().to(device=x.device)
        contributions = gated_x.index_select(-1, source) * weights
        output = contributions.new_zeros((*x.shape[:-1], self.out_features))
        output = output.index_add(-1, target, contributions)
        out_gate = self._gate(self.out_neurons, output)
        if out_gate is not None:
            output = output * out_gate

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
        view = self._view()
        in_gate = self._gate(self.in_neurons, x_flat)
        out_gate = self._gate(self.out_neurons, g_flat)
        if in_gate is not None:
            x_flat = x_flat * in_gate
        if out_gate is not None:
            g_flat = g_flat * out_gate
        source = view.s[:, 0].to(device=x_flat.device)
        target = view.t[:, 0].to(device=g_flat.device)
        return (
            x_flat.index_select(1, source) * g_flat.index_select(1, target)
        ).sum(dim=0)

    def dense_weight(self) -> Tensor:
        """Materialize ``[out_features, in_features]`` solely for tests/debugging."""
        view = self._view()
        source = view.s[:, 0].to(device=self.store.w.device)
        target = view.t[:, 0].to(device=self.store.w.device)
        flat_index = target * self.in_features + source
        dense = self._live_weights().new_zeros(self.out_features * self.in_features)
        result = dense.index_add(0, flat_index, self._live_weights()).reshape(
            self.out_features, self.in_features
        )
        if self.in_neurons is not None:
            result = result * self.in_neurons.gate_vector().to(result)
        if self.out_neurons is not None:
            result = result * self.out_neurons.gate_vector().to(result)[:, None]
        return result
