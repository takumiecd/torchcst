"""In-place Sphere coordinate law; AdamW owns moments and step counting."""

import math

import torch


@torch.no_grad()
def fused_update_(value, charts, previous, proposal, *, step_size):
    if (
        previous.shape != proposal.shape
        or proposal.ndim != 2
        or proposal.shape[1] != 6
        or proposal.dtype != torch.float32
        or previous.dtype != proposal.dtype
        or not proposal.is_cuda
        or previous.device != proposal.device
        or not previous.is_contiguous()
        or not proposal.is_contiguous()
        or torch._C._is_alias_of(previous, proposal)
        or type(step_size) not in (int, float)
        or not math.isfinite(step_size)
        or step_size <= 0
    ):
        raise ValueError(
            "requires distinct contiguous CUDA FP32 [A,6] snapshots and positive step"
        )
    mode = value.setting("activity_mode")
    if mode not in ("finite_chord", "time_energy"):
        raise ValueError("unsupported Polar activity mode")
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
    geometry_scalars = [
        s
        for chart in charts
        for s in (chart.geometry.radius, chart.geometry.chart_margin)
    ]
    if len(charts) != 2 or any(
        s.device != proposal.device or s.dtype != proposal.dtype or s.numel() != 1
        for s in scalars + geometry_scalars
    ):
        raise ValueError("requires live CUDA FP32 Sphere and Polar scalars")
    from .kernels import apply_sphere_polar

    if len(proposal):
        apply_sphere_polar[((len(proposal) + 127) // 128,)](
            previous,
            proposal,
            *scalars,
            *geometry_scalars,
            len(proposal),
            step_size,
            mode == "time_energy",
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
        torch.autograd.graph.increment_version(proposal)
    return proposal
