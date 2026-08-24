"""A neuron boundary that owns its gate, applied exactly once.

A raw :class:`~torchcst.compute.CSTLinear` map is purely synaptic and never
applies a neuron gate. Wiring two maps together over a shared hidden store by
gating each map's own endpoints would gate that store twice -- once as layer
1's output, once as layer 2's input -- so a neuron enters the composed
function as ``gamma^2``. A dormant ``gamma=0`` then has identically zero first
derivative and can never be woken: the growth field is not merely hard to
measure, it is exactly zero.

``CSTBoundary`` is the supported composition. It takes the *map's output* and
applies ``normalize -> activation -> gate``, exactly once:

    y = gamma * activation(normalize(pre))

``gamma`` multiplies a factor that does not contain it, exactly like a synapse
amplitude multiplies its factor outer product, so the dormant field is exact
and nonzero and the same scalar solve applies. Composition is then explicit:

    y = boundary(map(x))

and a stack is ``b2(m2(b1(m1(x))))``. A terminal boundary -- the topology a
single, non-composed ``CSTLinear`` used to be -- is simply
``CSTBoundary(map)`` with no activation: ``forward = pre * gate``. The
network's raw input is never gated by anything unless a caller multiplies it
by ``store.gate_vector()`` themselves; nothing upstream of the first map owns
that boundary.

An optional ``normalize`` module centers the pre-activation before the
activation is applied:

    y = gamma * activation(normalize(pre))

A CST layer has no bias and no normalization of its own, so nothing keeps the
sum of its atoms centered; a narrow coordinate domain then saturates the
activation and freezes training at chance once boundaries are stacked. Passing
the closure ``lambda p: activation(norm(p))`` as ``activation`` "works" but
leaves ``norm``'s parameters owned by nobody -- forget to add them to the
optimizer and they silently never train. ``normalize`` is a real submodule
attribute instead, so it is picked up by ``boundary.parameters()`` for free.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

from torchcst.storage import NeuronStore

from .cst_map import _ContinuousCSTMap

__all__ = ["CSTBoundary"]


class CSTBoundary(nn.Module):
    """Apply ``normalize -> activation -> gate`` to a map's output, once.

    Takes the map's output (``pre``), not the map's input: it does not call
    the map itself, so composition is explicit at the call site
    (``boundary(map(x))``).
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
        if linear.out_boundary is not None:
            raise ValueError(
                "linear already has a CSTBoundary owning its output boundary"
            )
        if activation is not None and not callable(activation):
            raise TypeError("activation must be callable or None")
        if normalize is not None and not isinstance(normalize, nn.Module):
            raise TypeError(
                "normalize must be an nn.Module or None -- a plain callable "
                "would not register its parameters on the boundary"
            )
        # Stored non-registered: linear is the producer, not a submodule of
        # this boundary. Registering it here would duplicate parameter
        # ownership (linear.parameters() already covers them) and, since
        # linear.out_boundary now points back at self, create a module cycle.
        self.__dict__["_linear"] = linear
        linear.__dict__["_out_boundary"] = self
        self.activation = activation
        # An nn.Module assigned as an attribute is auto-registered as a
        # submodule, so its parameters (e.g. a LayerNorm's weight/bias) show
        # up in boundary.parameters() without the caller wiring them in by
        # hand.
        self.normalize = normalize
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        # One entry per forward, in forward order -- the same order the engine
        # queues its own (x, g_out) facts, which is what lets a gate instrument
        # pair the two per microbatch under gradient accumulation.
        self._gate_grads: list[Tensor] = []

    @property
    def out_neurons(self) -> NeuronStore:
        return self._linear.out_neurons

    # -- forward ----------------------------------------------------------

    def forward(self, pre: Tensor) -> Tensor:
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

        ``x`` is the captured input to the producing map and ``g_gated`` the
        gradient of this boundary's *returned* tensor. The activation is
        recomputed from ``x`` through the gate-free map, so neither factor
        involves ``gamma``: the first value is exactly ``dL/dgamma``, and it
        stays exact -- and nonzero -- for a dormant row.

        The second value is the solve curvature ``sum(activation^2)``, scaled
        by ``row_energy`` when given. At a *terminal* boundary the activation
        already is the model's output direction and no scale is needed. At a
        hidden boundary the feature of ``gamma_j`` in output space is the
        activation times the consumer's response to row ``j``, so pass the
        consumer's
        :meth:`~torchcst.compute.CSTLinear.input_row_energy`; without it the
        step is conservative rather than wrong.

        The remaining constant is the objective's own curvature, which a
        loss-blind policy may not read. The tree's realized-profit trial (or a
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
        """Return ``activation(normalize(map(x)))`` with no gate, no capture."""
        with torch.no_grad():
            pre = self._linear.pre_gate_rows(x)
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
