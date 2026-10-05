"""Graph-safe in-place Euclidean Polar update after the optimizer proposal."""

import math

import torch


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
    from .kernels import apply_polar

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
