"""Experimental raw-Triweight contraction without weight tiles.

Strip+Torus distances give phi[a,n,k] = e[a,k]^3 (t[a,k]-s[a,n])_+^3.
Sorting t and taking four prefix moments of X replaces the N*K evaluation
per atom with K prefix scans and N lookups. See notes/atom-prefix-linear.ja.md.
This is an autograd oracle/prototype, not a registered production backend.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst import DirectAmpWidth, Triweight
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan


@dataclass
class PrefixPlan:
    """Fresh differentiable atom data; never cache across parameter updates."""

    amplitude: Tensor
    threshold: Tensor
    scale: Tensor
    row_distance: Tensor
    order: Tensor
    counts: Tensor


def prepare_prefix(site, p: Tensor) -> PrefixPlan:
    if (
        type(site.kernel) is not DirectAmpWidth
        or type(site.kernel.profile) is not Triweight
    ):
        raise ValueError(
            "atom prefix prototype requires DirectAmpWidth and raw Triweight"
        )
    plan = execution_plan(site)  # validates Strip+Torus and unnormalized profile
    if p.dtype not in (torch.float32, torch.float64):
        raise ValueError("atom prefix prototype requires float32 or float64")
    if p.ndim != 2 or p.shape[1] != plan.parameter_dim:
        raise ValueError(f"p must have shape [atoms, {plan.parameter_dim}]")
    # execution_plan validates the fixed kernel configuration. As in the
    # existing backend, avoid repeating host-synchronizing checks in capture.
    encoded, amplitude, precision = site.kernel._tile_parameters(p)
    centers = site.chart.geometry.decode_centers(encoded)
    radius = centers[:, :2].norm(dim=-1)
    direction = centers[:, :2] / radius[:, None]
    rho = plan.section[:, 0]
    delta = (rho[None, :] - radius[:, None]).square()
    delta = delta + (plan.section[None, :, 1:] - centers[:, None, 2:]).square().sum(-1)
    # Squared direction differences avoid cancellation between large radii.
    s = (
        radius[:, None]
        * 0.5
        * (plan.circle[None, :, :] - direction[:, None, :]).square().sum(-1)
    )
    threshold = (precision.reciprocal()[:, None] - delta) / (2 * rho[None, :])
    threshold, order = threshold.sort(dim=-1, descending=True)
    scale = (2 * rho[None, :] * precision[:, None]).gather(1, order).pow(3)
    # Membership changes need no derivative: value and first derivative vanish
    # at t=s for Triweight. Do not detach thresholds used in the moments.
    counts = torch.searchsorted(
        -threshold.detach().contiguous(), -s.detach().contiguous(), right=False
    )
    return PrefixPlan(amplitude, threshold, scale, s, order, counts)


def contract_prefix(
    inputs: Tensor, plan: PrefixPlan, *, atom_chunk: int = 16
) -> Tensor:
    if atom_chunk <= 0:
        raise ValueError("atom_chunk must be positive")
    a, k = plan.threshold.shape
    n = plan.row_distance.shape[1]
    if inputs.shape[-1] != k:
        raise ValueError("input feature count does not match chart")
    if inputs.dtype != plan.threshold.dtype or inputs.device != plan.threshold.device:
        raise ValueError("inputs and atoms must have the same dtype and device")
    x = inputs.reshape(-1, k)
    # Preserve zero gradients for empty atom tables and empty batches.
    result = (x.sum(-1, keepdim=True) * 0).expand(-1, n) + plan.amplitude.sum() * 0
    for start in range(0, a, atom_chunk):
        stop = min(start + atom_chunk, a)
        t = plan.threshold[start:stop, None, :]
        s = plan.row_distance[start:stop, None, :]
        weighted = x[:, plan.order[start:stop]].permute(1, 0, 2)
        weighted = weighted * plan.scale[start:stop, None, :]
        counts = plan.counts[start:stop, None, :].expand(-1, x.shape[0], -1)
        moments = []
        for degree in range(4):
            prefix = (weighted * t.pow(degree)).cumsum(-1)
            prefix = torch.nn.functional.pad(prefix, (1, 0))
            moments.append(prefix.gather(-1, counts))
        # Horner evaluation; the algebra is exact over reals, not bitwise FP32.
        value = ((-moments[0] * s + 3 * moments[1]) * s - 3 * moments[2]) * s + moments[
            3
        ]
        result = result + (value * plan.amplitude[start:stop, None, None]).sum(0)
    return result.reshape(*inputs.shape[:-1], n)


def atom_prefix_linear(
    site, inputs: Tensor, p: Tensor, *, atom_chunk: int = 16
) -> Tensor:
    return contract_prefix(inputs, prepare_prefix(site, p), atom_chunk=atom_chunk)
