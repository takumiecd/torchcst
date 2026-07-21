"""General Gaussian CST linear map over continuous coordinates."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from torchcst.representation import Box, GaussianKernel
from torchcst.storage import NeuronStore, SynapseStore, SynapseView

from .capture import BackwardContext


class CSTLinear(nn.Module):
    """Compose two Gaussian kernel matrices without materializing dense ``W``.

    Neuron coordinates are fixed floating buffers in step 9.  Synapse source
    and target coordinates, atom weights, and global kernel bandwidths remain
    learnable; learnable neuron ``mu`` is intentionally deferred.
    """

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel: GaussianKernel,
        kernel_out: GaussianKernel | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(in_neurons, NeuronStore) or not isinstance(
            out_neurons, NeuronStore
        ):
            raise TypeError("in_neurons and out_neurons must be NeuronStores")
        if not isinstance(synapses, SynapseStore):
            raise TypeError("synapses must be a SynapseStore")
        if not isinstance(kernel, GaussianKernel):
            raise TypeError("kernel must be a GaussianKernel")
        if kernel_out is not None and not isinstance(kernel_out, GaussianKernel):
            raise TypeError("kernel_out must be a GaussianKernel or None")
        if (
            synapses.spec.kernel_in != "gaussian"
            or synapses.spec.kernel_out != "gaussian"
        ):
            raise ValueError("CSTLinear requires a continuous Gaussian spec")
        if not isinstance(synapses.spec.domain_in, Box) or not isinstance(
            synapses.spec.domain_out, Box
        ):
            raise ValueError("CSTLinear requires Box coordinate domains")
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
        self._cached_version = -1
        self._cached_view: SynapseView | None = None
        self._cached_slots = torch.zeros(0, dtype=torch.int64)
        self._backward_context: BackwardContext | None = None
        self._mass_signature: tuple[int, ...] | None = None
        self._mass_sigmas: tuple[Tensor, ...] = ()

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
            self._cached_slots = self.synapses._slots.slots_of(view.ids)
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
        out_gate = self.out_neurons.gate_vector().detach().to(k_out)
        scale = torch.linalg.vector_norm(
            in_gate[:, None] * k_in.detach(), dim=0
        ) * torch.linalg.vector_norm(out_gate[:, None] * k_out.detach(), dim=0)
        self.synapses.set_mass_scale(scale, version=self.synapses.version)
        self._mass_signature = signature
        self._mass_sigmas = sigmas

    def forward(self, x: Tensor) -> Tensor:
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

        in_gate = self.in_neurons.gate_vector().to(device=x.device, dtype=x.dtype)
        gated_x = x * in_gate
        output = ((gated_x @ k_in) * weights) @ k_out.transpose(0, 1)
        out_gate = self.out_neurons.gate_vector().to(
            device=output.device, dtype=output.dtype
        )
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
        """Return the signed update contribution for every live atom weight."""
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal the input neuron width")
        if g_out.ndim == 0 or g_out.shape[-1] != self.out_features:
            raise ValueError(
                "g_out's final dimension must equal the output neuron width"
            )
        x_flat = x.detach().reshape(-1, self.in_features)
        g_flat = g_out.detach().reshape(-1, self.out_features)
        if x_flat.shape[0] != g_flat.shape[0]:
            raise ValueError("captured x and g_out batch dimensions do not align")
        self._view()
        source, target, _ = self._live_factors()
        source = source.detach().to(device=x_flat.device, dtype=x_flat.dtype)
        target = target.detach().to(device=g_flat.device, dtype=g_flat.dtype)
        with torch.no_grad():
            k_in, k_out = self._kernel_matrices(source, target)
            in_gate = self.in_neurons.gate_vector().to(x_flat)
            out_gate = self.out_neurons.gate_vector().to(g_flat)
            return (((x_flat * in_gate) @ k_in) * ((g_flat * out_gate) @ k_out)).sum(
                dim=0
            )

    def dense_weight(self) -> Tensor:
        """Materialize ``K_out diag(w) K_in.T`` solely for tests/debugging."""
        self._view()
        source, target, weights = self._live_factors()
        k_in, k_out = self._kernel_matrices(source, target)
        self._refresh_mass_scale(k_in, k_out)
        return (k_out * weights) @ k_in.transpose(0, 1)
