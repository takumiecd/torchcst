"""Shared mechanics for continuous CST compute modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst.representation import Box, ContinuousKernel
from torchcst.storage import NeuronStore, SynapseStore, SynapseView

from .capture import BackwardContext, flatten_capture_pair, register_capture_hook


class _ContinuousCSTMap(nn.Module):
    """Own a continuous CST measure and its endpoint charts.

    This internal base contains representation, mass, and capture mechanics
    shared by linear and convolutional compute modules.  It deliberately has
    no public forward contract: each concrete module defines its own input
    geometry.

    The kernel objects and the store's spec must name the same family, so a
    store calibrated for one profile cannot be driven by another: mass and
    rent constants are family-specific.
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
    ) -> None:
        super().__init__()
        if not isinstance(track_mass, bool):
            raise TypeError("track_mass must be a bool")
        if not isinstance(in_neurons, NeuronStore) or not isinstance(
            out_neurons, NeuronStore
        ):
            raise TypeError("in_neurons and out_neurons must be NeuronStores")
        if not isinstance(synapses, SynapseStore):
            raise TypeError("synapses must be a SynapseStore")
        if not isinstance(kernel, ContinuousKernel):
            raise TypeError("kernel must be a ContinuousKernel")
        if kernel_out is not None and not isinstance(kernel_out, ContinuousKernel):
            raise TypeError("kernel_out must be a ContinuousKernel or None")
        outgoing = kernel if kernel_out is None else kernel_out
        if (
            synapses.spec.kernel_in != kernel.family
            or synapses.spec.kernel_out != outgoing.family
        ):
            raise ValueError(
                "continuous CST maps require a spec naming the kernel families"
            )
        if not isinstance(synapses.spec.domain_in, Box) or not isinstance(
            synapses.spec.domain_out, Box
        ):
            raise ValueError("continuous CST maps require Box coordinate domains")
        self._validate_neurons(in_neurons, synapses.d_in, "in_neurons")
        self._validate_neurons(out_neurons, synapses.d_out, "out_neurons")

        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.synapses = synapses
        self.kernel_in = kernel
        self.kernel_out = kernel if kernel_out is None else kernel_out
        self.in_features = in_neurons.n_max
        self.out_features = out_neurons.n_max
        self.capture_site = synapses.site
        # Mass tracking feeds diagnostics and mass-based retention policies
        # only; the represented forward/backward math never reads it. A
        # caller that uses neither may pass track_mass=False to skip
        # _refresh_mass_scale's per-call bookkeeping (including the
        # host-syncing checks in SynapseStore.set_mass_scale).
        self._track_mass = track_mass
        # This map is purely synaptic: it never applies a neuron gate. A
        # neuron's gate belongs to the neuron and must be applied exactly
        # once, by whichever :class:`~torchcst.compute.CSTBoundary` owns that
        # boundary -- never by the map that produces or consumes it.
        self._cached_version = -1
        self._cached_view: SynapseView | None = None
        self._cached_slots = torch.zeros(0, dtype=torch.int64)
        self._backward_context: BackwardContext | None = None
        self._mass_signature: tuple[int, ...] | None = None
        self._mass_sigmas: tuple[Tensor, ...] = ()
        # Non-registered back-reference to the CSTBoundary that owns this
        # map's output boundary, if any -- set by CSTBoundary.__init__. Not
        # an nn.Module attribute: registering it would duplicate parameter
        # ownership (the boundary already registers its own submodules) and
        # create a module cycle (the boundary also holds this map).
        self.__dict__["_out_boundary"] = None

    @property
    def out_boundary(self):
        """The CSTBoundary owning this map's output boundary, if one exists."""
        return self.__dict__.get("_out_boundary")

    @property
    def store(self) -> SynapseStore:
        """Compatibility name shared by all CST compute modules."""
        return self.synapses

    @staticmethod
    def _validate_neurons(store: NeuronStore, dim: int, name: str) -> None:
        if store.mu.ndim != 2:
            raise ValueError(f"{name}.mu must be rank 2 for continuous coordinates")
        if store.mu.shape[1] != dim:
            raise ValueError(f"{name}.mu coordinate dimension does not match synapses")
        if not store.mu.is_floating_point():
            raise TypeError(f"{name}.mu must have a floating dtype")

    def _view(self) -> SynapseView:
        if self._cached_version != self.synapses.version:
            view = self.synapses.view()
            self._cached_view = SynapseView(
                site=view.site,
                version=view.version,
                ids=view.ids,
                s=view.s.detach(),
                t=view.t.detach(),
                w=view.w.detach(),
                mass=view.mass.detach(),
                lineages=view.lineages,
                bounds_in=view.bounds_in,
                bounds_out=view.bounds_out,
                domain_in=view.domain_in,
                domain_out=view.domain_out,
            )
            # Slots are cached device-resident (slots_of returns CPU
            # bookkeeping) so _live_factors' per-forward `.to(device=...)`
            # is a true no-op until the next structural event. That keeps
            # the hot path free of host-to-device copies -- which is also
            # what makes it CUDA-graph-capturable.
            self._cached_slots = self.synapses._slots.slots_of(view.ids).to(
                device=self.synapses.w.device
            )
            self._cached_version = view.version
        assert self._cached_view is not None
        return self._cached_view

    def _live_factors(self) -> tuple[Tensor, Tensor, Tensor]:
        slots = self._cached_slots.to(device=self.synapses.w.device)
        return (
            self.synapses.s.index_select(0, slots),
            self.synapses.t.index_select(0, slots),
            self.synapses.w.index_select(0, slots),
        )

    def set_backward_context(self, context: BackwardContext | None) -> None:
        if context is not None and not isinstance(context, BackwardContext):
            raise TypeError("context must be a BackwardContext or None")
        self._backward_context = context

    @property
    def capture_enabled(self) -> bool:
        return self._backward_context is not None

    def _kernel_matrices(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        in_mu = self.in_neurons.mu.to(device=source.device, dtype=source.dtype)
        out_mu = self.out_neurons.mu.to(device=target.device, dtype=target.dtype)
        return self.kernel_in(in_mu, source), self.kernel_out(out_mu, target)

    def _current_mass_signature(self) -> tuple[int, ...]:
        return (
            self.synapses.version,
            self.synapses.s._version,
            self.synapses.t._version,
            self.in_neurons.mu._version,
            self.out_neurons.mu._version,
            self.in_neurons.version,
            self.out_neurons.version,
            self.in_neurons.gate._version,
            self.out_neurons.gate._version,
            self.kernel_in.sigma._version,
            self.kernel_out.sigma._version,
        )

    def _refresh_mass_scale(self, k_in: Tensor, k_out: Tensor) -> None:
        if not self._track_mass:
            return
        kernels = tuple(dict.fromkeys((self.kernel_in, self.kernel_out)))
        sigmas = tuple(kernel.sigma.detach().clone() for kernel in kernels)
        signature = self._current_mass_signature()
        unchanged = (
            self._mass_signature == signature
            and len(sigmas) == len(self._mass_sigmas)
            and all(
                torch.equal(now, old.to(now))
                for now, old in zip(sigmas, self._mass_sigmas)
            )
        )
        if unchanged:
            return
        in_gate = self.in_neurons.gate_vector().detach().to(k_in)
        # Mass reads both endpoint gates even though this map never applies
        # them itself: mass measures an atom's effective magnitude in the
        # composed network, where the owning boundaries apply their gates.
        out_gate = self.out_neurons.gate_vector().detach().to(k_out)
        scale = torch.linalg.vector_norm(
            in_gate[:, None] * k_in.detach(), dim=0
        ) * torch.linalg.vector_norm(out_gate[:, None] * k_out.detach(), dim=0)
        self.synapses.set_mass_scale(scale, version=self.synapses.version)
        self._mass_signature = signature
        self._mass_sigmas = sigmas

    def _forward_rows(self, x: Tensor) -> Tensor:
        """Apply the represented map to rows and queue capture if requested."""
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal the input neuron width")
        view = self._view()
        source, target, weights = self._live_factors()
        source = source.to(device=x.device)
        target = target.to(device=x.device)
        weights = weights.to(device=x.device)
        k_in, k_out = self._kernel_matrices(source, target)
        self._refresh_mass_scale(k_in, k_out)

        output = ((x @ k_in) * weights) @ k_out.transpose(0, 1)

        if self._backward_context is not None:
            register_capture_hook(
                output, self._backward_context, self.capture_site, x, view.version
            )
        return output

    def atom_grads(self, x: Tensor, g_out: Tensor) -> Tensor:
        """Return the signed update contribution for every live atom weight."""
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal the input neuron width")
        if g_out.ndim == 0 or g_out.shape[-1] != self.out_features:
            raise ValueError(
                "g_out's final dimension must equal the output neuron width"
            )
        x_flat, g_flat = flatten_capture_pair(
            x, g_out, self.in_features, self.out_features
        )
        self._view()
        source, target, _ = self._live_factors()
        source = source.detach().to(device=x_flat.device, dtype=x_flat.dtype)
        target = target.detach().to(device=g_flat.device, dtype=g_flat.dtype)
        with torch.no_grad():
            k_in, k_out = self._kernel_matrices(source, target)
            return ((x_flat @ k_in) * (g_flat @ k_out)).sum(dim=0)

    def candidate_weight_grads(
        self,
        x: Tensor,
        g_out: Tensor,
        source: Tensor,
        target: Tensor,
        *,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Score zero-weight continuous candidates without mutating the store."""
        if source.ndim != 2 or source.shape[1] != self.synapses.d_in:
            raise ValueError("source candidates have the wrong coordinate shape")
        if target.ndim != 2 or target.shape[1] != self.synapses.d_out:
            raise ValueError("target candidates have the wrong coordinate shape")
        if source.shape[0] != target.shape[0]:
            raise ValueError("source and target candidate counts must match")
        count = source.shape[0]
        if chunk_size is None:
            chunk_size = max(count, 1)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an int or None")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        x_flat, g_flat = flatten_capture_pair(
            x, g_out, self.in_features, self.out_features
        )
        if count == 0:
            return self.synapses.w.detach().new_zeros(0).to(x_flat)
        source = source.detach().to(device=x_flat.device, dtype=x_flat.dtype)
        target = target.detach().to(device=g_flat.device, dtype=g_flat.dtype)
        values: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                k_in, k_out = self._kernel_matrices(
                    source[start:stop], target[start:stop]
                )
                values.append(
                    ((x_flat @ k_in) * (g_flat @ k_out)).sum(dim=0)
                )
        return torch.cat(values)

    def pre_gate_rows(self, x: Tensor) -> Tensor:
        """Return the map applied to rows, with no capture.

        The value a :class:`~torchcst.compute.CSTBoundary`'s gate multiplies.
        A boundary uses this to reconstruct its activation from a captured
        input, which is what keeps the dormant gate field exact.
        """
        x_flat = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        if x_flat.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal the input neuron width")
        self._view()
        source, target, weights = self._live_factors()
        source = source.detach().to(x_flat)
        target = target.detach().to(x_flat)
        weights = weights.detach().to(x_flat)
        with torch.no_grad():
            k_in, k_out = self._kernel_matrices(source, target)
            return ((x_flat @ k_in) * weights) @ k_out.transpose(0, 1)

    def input_row_energy(self) -> Tensor:
        """Return ``sum_o M[j, o]^2`` for the represented map ``M``.

        How strongly each input row is answered downstream.  A neuron gate at
        this map's *input* boundary scales a feature that this map then carries
        into output space, so its solve curvature is the activation energy
        times this response.  Computed from the Gram of the outgoing kernel, so
        the ``[in_features, out_features]`` matrix is never materialized.

        This map never applies its own outgoing gate, so the energy never
        scales ``k_out`` by it -- it is conservative (an overestimate) at a
        hidden boundary, where the true downstream response is damped by the
        owning :class:`~torchcst.compute.CSTBoundary`'s gate.
        """
        self._view()
        source, target, weights = self._live_factors()
        with torch.no_grad():
            k_in, k_out = self._kernel_matrices(
                source.detach(), target.detach()
            )
            weighted_in = k_in * weights.detach().to(k_in)
            gram_out = k_out.transpose(0, 1) @ k_out
            return (weighted_in @ gram_out).mul(weighted_in).sum(dim=1)

    def kernel_columns(self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate read-only kernel columns for arbitrary source/target rows.

        Returns ``(k_in, k_out)`` with shapes ``[in_features, N]`` and
        ``[out_features, N]``.  No gradient is tracked and no store state is
        read or mutated; this is the capability instruments use to evaluate
        candidate/live kernel directions (``KernelPort``) instead of reaching
        into kernel/neuron internals directly.
        """
        if source.ndim != 2 or source.shape[1] != self.synapses.d_in:
            raise ValueError("source coordinates have the wrong shape")
        if target.ndim != 2 or target.shape[1] != self.synapses.d_out:
            raise ValueError("target coordinates have the wrong shape")
        if source.shape[0] != target.shape[0]:
            raise ValueError("source and target coordinate counts must match")
        with torch.no_grad():
            return self._kernel_matrices(source.detach(), target.detach())

    def dense_weight(self) -> Tensor:
        """Materialize ``K_out diag(w) K_in.T`` for diagnostics or fast paths."""
        self._view()
        source, target, weights = self._live_factors()
        k_in, k_out = self._kernel_matrices(source, target)
        self._refresh_mass_scale(k_in, k_out)
        return (k_out * weights) @ k_in.transpose(0, 1)
