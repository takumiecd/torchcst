"""Graph-safe in-place Euclidean Polar update after the optimizer proposal."""

import math

import torch

from torchcst._builtin_callbacks import seal_functions as _seal_functions


def fused_update_(
    value,
    previous,
    proposal,
    *,
    step_size,
    periods=None,
    amplitude_weight_decay=0.0,
    capture_launch=None,
):
    """Apply the same update in place; AdamW owns moments and step counting.

    Research FP32 CUDA route only. Scalars remain live device tensors and no
    intermediate polar tensors are allocated. ``previous`` must not alias the
    proposal. The caller owns the old-parameter snapshot.
    """
    if (
        type(amplitude_weight_decay) not in (int, float)
        or not math.isfinite(amplitude_weight_decay)
        or amplitude_weight_decay < 0
        or step_size * amplitude_weight_decay > 1
    ):
        raise ValueError("invalid amplitude_weight_decay or step_size * decay > 1")
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
        or (previous.numel() != 0 and previous.data_ptr() == proposal.data_ptr())
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
    if periods is not None:
        if len(periods) != 2 or any(
            p.device != proposal.device or p.dtype != proposal.dtype or p.numel() != 1
            for p in periods
        ):
            raise ValueError("periods must be two live CUDA FP32 scalar tensors")
        # Match the public geometry's finite-point/displacement rejection.
        # Reduce both checks to a four-byte status, without per-atom scratch.
        # The fixed positive periods are validated by the binding declaration.
        from torchcst._backends.torch import finite

        if finite.checks_enabled() and len(proposal):
            from .kernels import check_periodic_centers

            invalid = torch.zeros((), dtype=torch.int32, device=proposal.device)
            check_periodic_centers[((len(proposal) + 127) // 128,)](
                previous,
                proposal,
                invalid,
                len(proposal),
                128,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if bool(invalid):
                raise ValueError(
                    "points and displacement must contain only finite values"
                )
    from .kernels import apply_polar

    if len(proposal):
        compiled = apply_polar[((len(proposal) + 127) // 128,)](
            previous,
            proposal,
            *scalars,
            len(proposal),
            step_size,
            mode == "time_energy",
            128,
            AMPLITUDE_DECAY=step_size * amplitude_weight_decay,
            HAS_AMPLITUDE_DECAY=amplitude_weight_decay != 0,
            **(
                {
                    "INPUT_PERIOD": periods[0],
                    "OUTPUT_PERIOD": periods[1],
                    "PERIODIC": True,
                }
                if periods is not None
                else {}
            ),
            num_warps=4,
            enable_fp_fusion=False,
        )
        if capture_launch is not None:
            # Record only compiled code and scalar specialization, never inputs.
            capture_launch(
                compiled,
                apply_polar,
                len(proposal),
                step_size,
                mode == "time_energy",
                step_size * amplitude_weight_decay,
            )
        torch.autograd.graph.increment_version(proposal)
    return proposal


# Optional prepared updates may specialize only these original definitions.
_seal_functions(globals())
