"""Graph-safe specialization of the existing Euclidean polar update."""

import math

import torch

from torchcst._backends.torch.parameterizations import polar_amp_width as polar


def graph_update(value, p, displacement, *, step_size):
    """Production polar update algebra, specialized to fixed Euclidean centers.

    Avoids public geometry's host finite-value checks during CUDA Graph capture.
    Tests compare against execution.apply_parameter_update; no width SGD.
    """
    from torchcst._backends.torch.updates.polar_amp_width import _project_polar

    polar = _project_polar(p[:, :2])
    q = polar.square().sum(-1, keepdim=True)
    raw = displacement[:, :2]
    tangent = raw - ((raw * polar).sum(-1, keepdim=True) / q) * polar
    chord = polar + tangent
    direction = chord / chord.square().sum(-1, keepdim=True).sqrt()
    energy = tangent.square().sum(-1, keepdim=True)
    if value.setting("activity_mode") == "time_energy":
        energy = energy / step_size
    amp, _ = polar_module_amplitude(value, direction)
    dormant = 1 / (1 + (amp / value.scalar("w_c").to(p)).square())
    task_q = (
        q
        + value.scalar("activity_gain").to(p) * energy
        + 3
        * value.scalar("dormant_expansion_rate").to(p)
        * step_size
        * dormant[:, None]
    ).clamp(1.0, 4.0)
    decay = torch.exp(-4 * value.scalar("radial_regularization").to(p) * step_size)
    regularized = 1 / (1 - (task_q - 1) / task_q * decay)
    # Preserve the same operation order as production, including radius rescaling.
    updated = direction * task_q.sqrt() * (regularized / task_q).sqrt()
    return torch.cat((updated, p[:, 2:] + displacement[:, 2:]), 1)


polar_module_amplitude = polar._amplitude_and_alpha


def fused_update_(value, previous, proposal, *, step_size):
    """Apply the same update in place; AdamW owns moments and step counting.

    Research FP32 CUDA route only. Scalars remain live device tensors and no
    intermediate polar tensors are allocated. ``previous`` must not alias the
    proposal. The caller owns the old-parameter snapshot.
    """
    if (
        previous.shape != proposal.shape
        or proposal.ndim != 2
        or proposal.shape[1] != 4
        or previous.dtype != torch.float32
        or proposal.dtype != torch.float32
        or not proposal.is_cuda
        or previous.device != proposal.device
        or not previous.is_contiguous()
        or not proposal.is_contiguous()
        or previous.data_ptr() == proposal.data_ptr()
        or not math.isfinite(step_size)
        or step_size <= 0
    ):
        raise ValueError(
            "fused polar update requires distinct contiguous CUDA FP32 [A,4] tensors and positive step"
        )
    mode = value.setting("activity_mode")
    if mode not in ("finite_chord", "time_energy"):
        raise ValueError("unsupported polar activity mode")
    scalars = [
        value.scalar(name)
        for name in (
            "amplitude_max",
            "w_c",
            "activity_gain",
            "dormant_expansion_rate",
            "radial_regularization",
        )
    ]
    if any(
        s.device != proposal.device or s.dtype != proposal.dtype or s.numel() != 1
        for s in scalars
    ):
        raise ValueError("polar scalars must be live CUDA FP32 scalar tensors")
    from .update_kernels import apply_polar

    if len(proposal):
        apply_polar[((len(proposal) + 127) // 128,)](
            previous,
            proposal,
            *scalars,
            len(proposal),
            step_size,
            mode == "time_energy",
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        torch.autograd.graph.increment_version(proposal)
    return proposal
