"""General CST linear map over continuous coordinates."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from torchcst.representation import ContinuousKernel
from torchcst.storage import NeuronStore, SynapseStore

from .backends import (
    Factored,
    Materialized,
    NativeTruncated,
    crossover_materializes,
    validate_backend,
)
from .backends.materialize import LeanLinearMaterialize, linear_weight
from .backends.native import NativeTruncatedFunction, neighbor_tables
from .capture import register_capture_hook
from .cst_map import _ContinuousCSTMap


class CSTLinear(_ContinuousCSTMap):
    """Apply a continuous CST measure to feature rows.

    Neuron coordinates are fixed floating buffers. Synapse source and target
    coordinates, atom weights, and global kernel bandwidths remain learnable.

    A CST layer is a composition, not a primitive: neurons come first
    (:meth:`torchcst.storage.NeuronStore.propose` for hidden populations, or
    data-supplied coordinates for pinned ones), synapses derive from the
    populations they connect (:meth:`torchcst.storage.SynapseStore.between`),
    and this module merely applies the composed site.

    *How* the measure is applied is the ``backend`` (see
    :mod:`torchcst.compute.backends`): :class:`Factored` below the FLOP
    crossover, :class:`Materialized` above it (the default ``"auto"``
    switches per forward on the live atom count), or
    :class:`NativeTruncated` for the no-W local-kernel form.  The module
    owns state -- stores, views, capture, mass -- and dispatches tensors
    into the chosen backend's pure functions.
    """

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel: ContinuousKernel,
        kernel_out: ContinuousKernel | None = None,
        *,
        track_mass: bool = True,
        backend="auto",
        gauge=None,
    ) -> None:
        super().__init__(
            in_neurons, out_neurons, synapses, kernel, kernel_out,
            track_mass=track_mass, gauge=gauge,
        )
        self.backend = validate_backend(
            backend,
            kernel_in=self.kernel_in,
            kernel_out=self.kernel_out,
            track_mass=track_mass,
            gauge=self.gauge,
        )

    def _resolved_backend(self, count: int):
        """The backend this forward actually runs, given the live count."""
        if self.backend == "auto":
            if crossover_materializes(
                count, self.in_features, self.out_features
            ):
                return Materialized()
            return Factored()
        if isinstance(self.backend, NativeTruncated) and count == 0:
            return Factored()  # an empty site has no neighbor tables
        return self.backend

    def dense_weight(self) -> Tensor:
        """Materialize ``K_out diag(w) K_in.T`` -- the Materialized path's W."""
        self._view()
        source, target, weights = self._live_factors()
        config = (
            self.backend
            if isinstance(self.backend, Materialized)
            else Materialized()
        )
        if config.lean:
            return LeanLinearMaterialize.apply(
                source, target, weights,
                self.in_neurons.mu.to(source),
                self.out_neurons.mu.to(target),
                self.kernel_in.sigma.to(source),
                self.kernel_out.sigma.to(target),
                config.compute_dtype,
            )
        k_in, k_out = self._kernel_matrices(source, target)
        self._refresh_mass_scale(k_in, k_out)
        return linear_weight(k_in, k_out, weights, config.compute_dtype)

    def _forward_native(self, x: Tensor, config: NativeTruncated) -> Tensor:
        source, target, weights = self._live_factors()
        source = source.to(device=x.device)
        target = target.to(device=x.device)
        weights = weights.to(device=x.device)
        mu_in = self.in_neurons.mu.to(source)
        mu_out = self.out_neurons.mu.to(target)
        sigma_in = self.kernel_in.sigma.to(source)
        sigma_out = self.kernel_out.sigma.to(target)
        idx_in, pad_in = neighbor_tables(
            source, mu_in, sigma_in, config.radius
        )
        idx_out, pad_out = neighbor_tables(
            target, mu_out, sigma_out, config.radius
        )
        rows = x.reshape(-1, self.in_features)
        output = NativeTruncatedFunction.apply(
            rows, source, target, weights, mu_in, mu_out, sigma_in,
            sigma_out, idx_in, pad_in, idx_out, pad_out,
        )
        return output.reshape(*x.shape[:-1], self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal the input neuron width")
        view = self._view()
        backend = self._resolved_backend(int(self._cached_slots.shape[0]))
        if isinstance(backend, Factored):
            return self._forward_rows(x)
        if isinstance(backend, NativeTruncated):
            output = self._forward_native(x, backend)
        else:
            weight = self.dense_weight().to(dtype=x.dtype, device=x.device)
            output = F.linear(x, weight)
        if self._backward_context is not None:
            register_capture_hook(
                output, self._backward_context, self.capture_site, x,
                view.version,
            )
        return output
