"""General CST linear map over continuous coordinates."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.representation import ContinuousKernel, L2NormalizedColumns
from torchcst.storage import NeuronStore, SynapseStore

from .backends import (
    Factored,
    Materialized,
    NativeTruncated,
    crossover_materializes,
    validate_backend,
)
from .backends.materialize import (
    LeanL2LinearMaterialize,
    LeanLinearMaterialize,
    linear_weight,
)
from .backends.native import NativeTruncatedFunction, neighbor_tables
from .capture import register_capture_hook
from .cst_map import _ContinuousCSTMap


# A no-grad call retains no kernel graph, so one full materialisation avoids
# chunk launch overhead when its two column matrices remain modest.  The cap is
# in scalar elements (128 MiB at fp32), independent of the parameter dtype.
_NO_GRAD_FULL_COLUMN_LIMIT = 32 * 1024 * 1024


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
        self._eval_weight_key: tuple[object, ...] | None = None
        self._eval_weight_cache: Tensor | None = None

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self._eval_weight_key = None
            self._eval_weight_cache = None
        return result

    def _eval_dense_weight(self) -> Tensor:
        """Cache graph-free W only while the module remains in eval mode."""
        tensors = (
            self.synapses.s,
            self.synapses.t,
            self.synapses.w,
            self.in_neurons.mu,
            self.out_neurons.mu,
            self.kernel_in.sigma,
            self.kernel_out.sigma,
        )
        key = (
            self.synapses.version,
            type(self.gauge),
            self.backend,
            *((id(value), value._version) for value in tensors),
        )
        if self._eval_weight_key != key or self._eval_weight_cache is None:
            self._eval_weight_cache = self.dense_weight()
            self._eval_weight_key = key
        return self._eval_weight_cache

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
        full_column_elements = source.shape[0] * (
            self.in_neurons.mu.shape[0] + self.out_neurons.mu.shape[0]
        )
        if (
            config.lean
            and not torch.is_grad_enabled()
            and full_column_elements <= _NO_GRAD_FULL_COLUMN_LIMIT
        ):
            k_in, k_out = self._kernel_matrices(source, target)
            self._refresh_mass_scale(k_in, k_out)
            return linear_weight(k_in, k_out, weights, config.compute_dtype)
        if config.lean:
            mu_in = self.in_neurons.mu.to(source)
            mu_out = self.out_neurons.mu.to(target)
            sigma_in, _ = self.kernel_in._prepare(mu_in, source, None)
            sigma_out, _ = self.kernel_out._prepare(mu_out, target, None)
            materialize = (
                LeanL2LinearMaterialize
                if isinstance(self.gauge, L2NormalizedColumns)
                else LeanLinearMaterialize
            )
            args = (
                source, target, weights, mu_in, mu_out, sigma_in, sigma_out,
                config.compute_dtype,
            )
            if materialize is LeanL2LinearMaterialize:
                return materialize.apply(*args, config.compile_l2)
            return materialize.apply(*args)
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
            weight = (
                self._eval_dense_weight()
                if not self.training and not torch.is_grad_enabled()
                else self.dense_weight()
            ).to(dtype=x.dtype, device=x.device)
            output = F.linear(x, weight)
        if self._backward_context is not None:
            register_capture_hook(
                output, self._backward_context, self.capture_site, x,
                view.version,
            )
        return output
