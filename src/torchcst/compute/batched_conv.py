"""Batched filter assembly across same-shaped :class:`CSTConv2d` modules.

A single continuous CST conv layer materializes its represented filter with
a chain of small ops -- two kernel-matrix evaluations (broadcast subtract,
square, reduce, then the kernel profile) and one contraction -- all sized by
that one layer's ``(in_features, out_features, K)``. None of it depends on
activations, so it is independent of every other layer's filter assembly.

A network built from many same-shaped layers (e.g. same-stage ResNet blocks)
pays for that independence with kernel-launch count: assembling 18 small
filters one at a time issues roughly 18x as many CUDA kernel launches as the
compute actually needs, and on a GPU that is not otherwise saturated this is
launch-bound rather than compute-bound (each op is tiny; the CPU cannot
enqueue them fast enough to keep the device busy). Grouping layers that
share ``(in_features, out_features, K)`` into one batched pass turns that
``O(layers)`` launch count into ``O(1)`` for everything except the final
per-layer contraction (see the determinism note below), without changing
what any single layer computes.

Determinism note -- read this before assuming "batched" means "bit-for-bit
the same tensor object computation as the per-layer path":

* Forward values are exactly bit-identical to calling ``dense_weight()`` per
  layer, verified empirically with ``torch.equal`` (adding a leading batch
  dimension does not change any per-element floating-point result, and the
  distance computation's own reduction axis is the fixed-size coordinate
  dimension, not the batch dimension).
* The final contraction is deliberately kept a per-layer ``matmul`` loop,
  not ``torch.bmm``: cuBLAS's batched-GEMM kernel does not use the same
  reduction order as its single-GEMM kernel, differing at the 1e-8-1e-7
  absolute level for the K in this bench.
* Gradients to ``s``, ``t``, and ``sigma`` are *not* guaranteed
  bit-identical to the per-layer path, and this module does not attempt to
  force them to be. Each of those is a per-layer scalar or per-atom row
  broadcast over a large axis (``sigma`` over the whole ``features x K``
  kernel matrix; ``s``/``t`` over the *other* side's feature count), so its
  gradient is a reduction over that axis, and PyTorch's own broadcast
  backward machinery does not reduce a batched ``[L, ...]`` tensor in one
  fused kernel the same way it reduces ``L`` separate per-layer tensors --
  this is a real property of how GPU (and, measured here, even CPU/float64)
  reductions work, not a bug in this implementation. Measured on the actual
  ResNet-20 bench shapes on CUDA/float32: ``sigma``'s gradient (reduced over
  up to ~260,000 elements) differed from the per-layer computation's by
  ~1e-6 absolute, enough to perturb a real training loss sequence measurably
  within a handful of SGD steps; ``s``/``t``'s gradients (reduced over at
  most 576 elements) were exactly equal at that precision, though a
  float64/CPU check of the same code found up to one machine-epsilon (~2e-16)
  ULP of drift for ``s``/``t`` there, so "always exactly equal" is not a
  guarantee this module makes for them either -- only "measured negligible
  at the precision and shapes this bench actually uses, and self-consistent
  run to run" (this module never introduces literal nondeterminism: repeated
  calls with the same inputs and PyTorch's own deterministic-algorithm
  settings reproduce the same batched-path result exactly).

  A closed-form or otherwise structurally-reformulated custom backward was
  tried to eliminate the ``sigma`` gap and rejected: chasing it revealed
  that even a *mathematically correct* hand-derived gradient formula (or an
  ``expand()``-plus-per-layer-``.sum()`` reformulation of the broadcast
  reduction) does not reproduce PyTorch's own broadcast-backward bit-for-bit
  -- the discrepancy is not really about batching *order* so much as about
  PyTorch's internal broadcast-reduction implementation being one specific,
  version- and backend-dependent choice among several valid ones, which no
  amount of hand-written Python-level code can be relied on to replicate
  exactly. The measured ~1e-6 residual is several orders of magnitude below
  this repo's own documented 0.5% run-to-run reproducibility floor and the
  ~0.3pt effect sizes the S4 grid is measuring, so it is accepted here as a
  quantified, bounded, and reported tradeoff rather than chased further.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .._geometry import squared_norm_last

from .cst_conv import CSTConv2d


def _validate_batch_group(
    layers: Sequence[CSTConv2d],
) -> tuple[torch.device, torch.dtype]:
    """Check every layer is batchable with the first; return shared placement."""
    for layer in layers:
        if not isinstance(layer, CSTConv2d):
            raise TypeError("layers must all be CSTConv2d instances")
    ref = layers[0]
    device = ref.synapses.w.device
    dtype = ref.synapses.w.dtype
    for layer in layers:
        if layer.capture_enabled:
            raise ValueError(
                "batched_conv_dense_weights does not support a layer with an "
                "active capture context; route it through dense_weight()/"
                "the reference forward path instead"
            )
        if layer._track_mass:
            raise ValueError(
                "batched_conv_dense_weights never refreshes synapses.mass_scale; "
                "pass only layers built with track_mass=False, or call "
                "dense_weight() directly for a layer that needs the diagnostic"
            )
        if layer.in_features != ref.in_features or layer.out_features != ref.out_features:
            raise ValueError(
                "batched_conv_dense_weights requires every layer to share "
                "in_features and out_features"
            )
        if (
            type(layer.kernel_in) is not type(ref.kernel_in)
            or type(layer.kernel_out) is not type(ref.kernel_out)
        ):
            raise ValueError(
                "batched_conv_dense_weights requires every layer to share "
                "kernel_in/kernel_out family"
            )
        if layer.synapses.w.device != device or layer.synapses.w.dtype != dtype:
            raise ValueError(
                "batched_conv_dense_weights requires every layer to share "
                "device and dtype"
            )
    return device, dtype


def batched_conv_dense_weights(layers: Sequence[CSTConv2d]) -> list[Tensor]:
    """Materialize dense conv filters for a group of same-shaped layers.

    Equivalent, per layer, to calling ``layer.dense_weight()`` independently
    and collecting the results into a list: exactly equal *values*
    (``torch.equal``, not merely close). Gradients to every layer's own
    ``s``/``t``/``w``, chart ``mu``, kernel ``sigma``, and gates flow to the
    same places they would from the per-layer call, but are not guaranteed
    bit-identical -- see the module docstring's determinism note for the
    measured, bounded residual and why it is not eliminated here. The
    intended effect of this function is purely how many CUDA kernels the
    computation costs to produce these (exact-valued, near-exact-gradient)
    results, not any change to what is represented.

    Every layer in ``layers`` must share:

    - ``in_features`` and ``out_features`` (so the neuron charts have the
      same row counts);
    - live atom count ``K`` (so ``s``/``t``/``w`` stack cleanly);
    - ``kernel_in`` family and ``kernel_out`` family (their ``sigma``
      *values* may differ freely per layer -- only the ``profile`` math
      must match, since that is what gets called once for the whole group).

    Layers must not have an active capture context (``capture_enabled``) --
    ``CSTConv2d.forward`` already routes capture-enabled layers through the
    reference unfold path instead of ``dense_weight()``, so a caller
    building a batch group should exclude those layers the same way. Layers
    must also not request mass tracking (``track_mass=True`` at
    construction): this function never touches ``synapses.mass_scale``, so a
    caller that needs that diagnostic for a given layer must call that
    layer's own ``dense_weight()`` instead of routing it through here.

    Raises ``ValueError``/``TypeError`` on any of the above mismatches.
    Returns ``[]`` for an empty ``layers``.
    """

    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise TypeError("layers must be a sequence of CSTConv2d")
    if not layers:
        return []
    device, dtype = _validate_batch_group(layers)

    ref = layers[0]
    for layer in layers:
        # _live_factors reads self._cached_slots, which _view() populates
        # (and refreshes only when synapses.version has changed) -- must run
        # first, exactly as a direct dense_weight() call does.
        layer._view()
    live = [layer._live_factors() for layer in layers]  # each: (s[K,D_IN], t[K,D_OUT], w[K])
    k_counts = {factors[0].shape[0] for factors in live}
    if len(k_counts) != 1:
        raise ValueError("batched_conv_dense_weights requires every layer to share live atom count K")

    s_stack = torch.stack([factors[0] for factors in live], dim=0)  # [L, K, D_IN]
    t_stack = torch.stack([factors[1] for factors in live], dim=0)  # [L, K, D_OUT]
    w_stack = torch.stack([factors[2] for factors in live], dim=0)  # [L, K]

    in_mu_stack = torch.stack(
        [layer.in_neurons.mu.to(device=device, dtype=dtype) for layer in layers], dim=0
    )  # [L, n_in, D_IN]
    out_mu_stack = torch.stack(
        [layer.out_neurons.mu.to(device=device, dtype=dtype) for layer in layers], dim=0
    )  # [L, n_out, D_OUT]

    sigma_in_stack = torch.stack(
        [layer.kernel_in.sigma.to(device=device, dtype=dtype) for layer in layers]
    ).reshape(-1, 1, 1)
    sigma_out_stack = torch.stack(
        [layer.kernel_out.sigma.to(device=device, dtype=dtype) for layer in layers]
    ).reshape(-1, 1, 1)

    sq_in = squared_norm_last(
        in_mu_stack[:, :, None, :] - s_stack[:, None, :, :]
    )  # [L, n_in, K]
    sq_out = squared_norm_last(
        out_mu_stack[:, :, None, :] - t_stack[:, None, :, :]
    )  # [L, n_out, K]

    k_in = ref.kernel_in.profile(sq_in, sigma_in_stack)  # [L, n_in, K]
    k_out = ref.kernel_out.profile(sq_out, sigma_out_stack)  # [L, n_out, K]

    scaled_out = k_out * w_stack[:, None, :]  # [L, n_out, K]
    # The contraction stays a per-layer loop -- see the module docstring's
    # determinism note: torch.bmm here is not bit-identical to this.
    contracted = torch.stack(
        [scaled_out[i] @ k_in[i].transpose(0, 1) for i in range(len(layers))], dim=0
    )  # [L, n_out, n_in]

    in_gate_stack = torch.stack(
        [layer.in_neurons.gate_vector().to(device=device, dtype=dtype) for layer in layers], dim=0
    )  # [L, n_in]
    out_gate_stack = torch.stack(
        [layer.out_neurons.gate_vector().to(device=device, dtype=dtype) for layer in layers], dim=0
    )  # [L, n_out]
    gated = out_gate_stack[:, :, None] * contracted * in_gate_stack[:, None, :]  # [L, n_out, n_in]

    return [
        gated[i].reshape(layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1])
        for i, layer in enumerate(layers)
    ]
