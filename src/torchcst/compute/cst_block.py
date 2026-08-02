"""A CST layer that owns its neurons' gate, applied exactly once.

``CSTLinear`` applies both endpoint gates itself.  That is fine for a single
map, but wiring two of them together gates the shared hidden store twice --
once as layer 1's output, once as layer 2's input -- so a neuron enters the
composed function as ``gamma^2``.  A dormant ``gamma=0`` then has identically
zero first derivative and can never be woken: the growth field is not merely
hard to measure, it is exactly zero.

This block is the supported composition.  It wraps a gate-free map and applies
the *producing* store's gate once, after the activation:

    h = gamma * activation(synapse_map(x))

``gamma`` multiplies a factor that does not contain it, exactly like a synapse
amplitude multiplies its kernel outer product, so the dormant field is exact
and nonzero and the same scalar solve applies.

An optional ``normalize`` module centers the pre-activation before the
activation is applied:

    h = gamma * activation(normalize(synapse_map(x)))

A CST layer has no bias and no normalization of its own, so nothing keeps the
sum of its atoms centered; a narrow coordinate domain then saturates the
activation and freezes training at chance once the layer is stacked. Passing
the closure ``lambda p: activation(norm(p))`` as ``activation`` "works" but
leaves ``norm``'s parameters owned by nobody -- forget to add them to the
optimizer and they silently never train. ``normalize`` is a real submodule
attribute instead, so it is picked up by ``block.parameters()`` for free.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor, nn

from torchcst.storage import NeuronStore, SynapseStore

from .capture import BackwardContext
from .cst_map import _ContinuousCSTMap

__all__ = ["CSTBlock"]


class CSTBlock(nn.Module):
    """Gate-free CST map + activation + one application of the output gate.

    The wrapped map must be constructed with ``gate_input=False`` and
    ``gate_output=False``: this block, not the map, owns the boundary gate.
    An incoming boundary is gated by whoever produced it (an upstream block,
    or an explicit gate on the network's raw input).
    """

    def __init__(
        self,
        linear: _ContinuousCSTMap,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(linear, _ContinuousCSTMap):
            raise TypeError("linear must be a continuous CST compute module")
        if linear._gate_input or linear._gate_output:
            raise ValueError(
                "CSTBlock owns the boundary gate: build the map with "
                "gate_input=False, gate_output=False"
            )
        if activation is not None and not callable(activation):
            raise TypeError("activation must be callable or None")
        if normalize is not None and not isinstance(normalize, nn.Module):
            raise TypeError(
                "normalize must be an nn.Module or None -- a plain callable "
                "would not register its parameters on the block"
            )
        self.linear = linear
        self.activation = activation
        # An nn.Module assigned as an attribute is auto-registered as a
        # submodule, so its parameters (e.g. a LayerNorm's weight/bias) show
        # up in block.parameters() without the caller wiring them in by hand.
        self.normalize = normalize
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.capture_site = linear.capture_site
        # One entry per forward, in forward order -- the same order the engine
        # queues its own (x, g_out) facts, which is what lets a gate instrument
        # pair the two per microbatch under gradient accumulation.
        self._gate_grads: list[Tensor] = []

    @property
    def store(self) -> SynapseStore:
        return self.linear.store

    @property
    def out_neurons(self) -> NeuronStore:
        return self.linear.out_neurons

    @property
    def in_neurons(self) -> NeuronStore:
        return self.linear.in_neurons

    def set_backward_context(self, context: BackwardContext | None) -> None:
        self.linear.set_backward_context(context)

    @property
    def capture_enabled(self) -> bool:
        return self.linear.capture_enabled

    def __getattr__(self, name: str) -> Any:
        """Delegate the map's remaining compute capabilities (kernel ports…)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        linear = (self.__dict__.get("_modules") or {}).get("linear")
        if linear is None:
            raise AttributeError(name)
        return getattr(linear, name)

    # -- forward ----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        pre = self.linear(x)
        activated = self._normalize_and_activate(pre)
        gate = self.out_neurons.gate_vector().to(
            device=activated.device, dtype=activated.dtype
        )
        gated = activated * gate
        if torch.is_grad_enabled() and gated.requires_grad:
            queue = self._gate_grads

            def collect(grad: Tensor) -> None:
                queue.append(grad.detach())

            gated.register_hook(collect)
        return gated

    # -- gate field -------------------------------------------------------

    def reset_gate_capture(self) -> None:
        """Drop any gradients queued but not consumed by an instrument."""
        self._gate_grads.clear()

    def take_gate_grad(self) -> Tensor | None:
        """Pop the oldest queued post-gate gradient, or ``None`` if empty."""
        if not self._gate_grads:
            return None
        return self._gate_grads.pop(0)

    def gate_tangent(
        self,
        x: Tensor,
        g_gated: Tensor,
        *,
        row_energy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the gate field and the activation energy for every chart row.

        ``x`` is the block's captured input and ``g_gated`` the gradient of
        its *returned* tensor.  The activation is recomputed from ``x`` through
        the gate-free map, so neither factor involves ``gamma``: the first
        value is exactly ``dL/dgamma``, and it stays exact -- and nonzero --
        for a dormant row.

        The second value is the solve curvature ``sum(activation^2)``, scaled
        by ``row_energy`` when given.  At a *terminal* boundary the activation
        already is the model's output direction (the FC-2 topology) and no
        scale is needed.  At a hidden boundary the feature of ``gamma_j`` in
        output space is the activation times the consumer's response to row
        ``j``, so pass the consumer's
        :meth:`~torchcst.compute.CSTLinear.input_row_energy`; without it the
        step is conservative rather than wrong.

        The remaining constant is the objective's own curvature, which a
        loss-blind policy may not read.  The tree's realized-profit trial (or a
        damped acceptance ladder) is what absorbs it.
        """
        activated = self.activated_rows(x)
        g_flat = g_gated if g_gated.ndim == 2 else g_gated.reshape(
            -1, g_gated.shape[-1]
        )
        if g_flat.shape != activated.shape:
            raise ValueError("gate gradient does not match the activation shape")
        g_flat = g_flat.to(activated)
        curvature = activated.square().sum(dim=0)
        if row_energy is not None:
            if row_energy.shape != curvature.shape:
                raise ValueError("row_energy must have one entry per chart row")
            curvature = curvature * row_energy.to(curvature)
        return ((g_flat * activated).sum(dim=0), curvature)

    def activated_rows(self, x: Tensor) -> Tensor:
        """Return ``activation(normalize(synapse_map(x)))`` with no gate, no capture."""
        with torch.no_grad():
            pre = self.linear.pre_gate_rows(x)
            return self._normalize_and_activate(pre)

    def _normalize_and_activate(self, pre: Tensor) -> Tensor:
        """Apply the shared pre-activation pipeline: normalize, then activate.

        :meth:`forward` and :meth:`activated_rows` must feed the activation
        the exact same tensor -- the dormant-gate field is reconstructed from
        the latter and compared against gradients captured from the former --
        so both call this one place rather than each inlining the two steps.
        """
        normalized = pre if self.normalize is None else self.normalize(pre)
        return normalized if self.activation is None else self.activation(normalized)
