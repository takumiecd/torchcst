"""Forward-only bounded-memory mapped CST materialization and GEMM."""

import math

import torch

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_materialize_parallel import materialize_listed_parallel
from prototypes.bounded_gemm import bounded_gemm
from prototypes.bounded_gemm_fp16x3 import bounded_gemm_fp16x3
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


def weight_fp_fusion_enabled(device):
    """Use the fast W kernel on validated Ampere GPUs; preserve Ada accuracy."""
    return torch.cuda.get_device_capability(device) < (8, 9)


def streamed_forward(
    layer,
    x,
    *,
    prepared=None,
    weight_chunk_rows=1024,
    materialize_tile=(64, 32),
    cache_weight_rows=0,
    cache_weight_dtype=torch.float32,
    return_cache=False,
    gemm_mode="ieee",
    materialize_mode="default",
    listed_data=None,
    listed_unroll=1,
):
    """Generate a row window of logical W and multiply it before reusing the buffer."""
    if layer.tile_shape != (64, 64) or layer.shape[0] % 64:
        raise ValueError("streamed prototype currently requires 64x64 full row tiles")
    if (
        type(weight_chunk_rows) is not int
        or weight_chunk_rows < 64
        or weight_chunk_rows % 64
    ):
        raise ValueError("weight_chunk_rows must be a positive multiple of 64")
    if materialize_tile not in ((16, 32), (32, 32), (64, 32), (32, 64), (64, 64)):
        raise ValueError("unsupported materialization tile")
    if gemm_mode not in ("ieee", "tf32x3", "fp16x3"):
        raise ValueError("unknown forward gemm_mode")
    if listed_unroll not in (1, 2, 4, 8):
        raise ValueError("listed_unroll must be 1, 2, 4, or 8")
    if materialize_mode not in (
        "default",
        "listed",
        "listed_parallel",
        "listed_csr",
        "listed_bounded",
    ):
        raise ValueError("unknown forward materialize_mode")
    if (
        materialize_mode
        in ("listed", "listed_parallel", "listed_csr", "listed_bounded")
        and listed_data is None
    ):
        raise ValueError("listed forward requires candidate lists")
    if (
        type(cache_weight_rows) is not int
        or cache_weight_rows < 0
        or cache_weight_rows > layer.shape[0]
        or cache_weight_rows % weight_chunk_rows
    ):
        raise ValueError(
            "cache_weight_rows must contain complete windows at most full W"
        )
    if cache_weight_dtype not in (torch.float32, torch.float16):
        raise ValueError("cache_weight_dtype must be float32 or float16")
    if torch.is_grad_enabled() and (
        x.requires_grad or layer.strip.atoms.p.requires_grad
    ):
        raise NotImplementedError("mapped streamed prototype supports forward only")
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("mapped streamed prototype requires NVIDIA CUDA float32")
    if x.device != layer.strip.atoms.p.device or layer.strip.atoms.p.dtype != x.dtype:
        raise ValueError("input and model must share device and dtype")
    p, circle, section, offsets = (
        prepared
        if prepared is not None
        else prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    )
    if p.shape[1] != 6:
        raise ValueError("streamed factored prototype requires four-dimensional sites")
    flat = x.reshape(-1, layer.shape[1]).contiguous()
    y = flat.new_empty((flat.shape[0], layer.shape[0]))
    cached = torch.empty(
        (cache_weight_rows, layer.shape[1]),
        device=flat.device,
        dtype=cache_weight_dtype,
    )
    if not flat.shape[0]:
        output = y.reshape(*x.shape[:-1], layer.shape[0])
        return (output, cached) if return_cache else output
    chunk = min(weight_chunk_rows, layer.shape[0])
    fp_fusion = weight_fp_fusion_enabled(x.device)
    bn, bk = materialize_tile
    w = flat.new_empty((chunk, layer.shape[1]))
    for start in range(0, layer.shape[0], chunk):
        rows = min(chunk, layer.shape[0] - start)
        target = (
            cached[start : start + rows]
            if cache_weight_dtype == torch.float32 and start < cache_weight_rows
            else w[:rows]
        )
        if materialize_mode in (
            "listed",
            "listed_parallel",
            "listed_csr",
            "listed_bounded",
        ):
            csr = materialize_mode == "listed_csr"
            if csr:
                atom_lists, list_counts, list_bases, _ = listed_data
                max_candidates = 0
            else:
                atom_lists, list_counts, max_candidates = listed_data
                list_bases = None
            kernel = (
                materialize_listed_parallel
                if materialize_mode == "listed_parallel"
                else materialize_listed
            )
            kernel[(rows // 64 * layer.column_groups * 4,)](
                p,
                circle,
                section,
                atom_lists,
                list_counts,
                offsets,
                target,
                K=layer.shape[1],
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=max_candidates,
                CSR=csr,
                BOUNDED=materialize_mode == "listed_bounded",
                Bases=list_bases,
                **(
                    {"LOOP_UNROLL": listed_unroll}
                    if materialize_mode != "listed_parallel"
                    else {}
                ),
                STATION_START=start // 64 * layer.column_groups,
                ROW_START=start,
                num_warps=4 if materialize_mode == "listed_parallel" else 1,
                **({"BA": 2} if materialize_mode == "listed_parallel" else {}),
                enable_fp_fusion=fp_fusion,
            )
        else:
            materialize_logical[(math.ceil(rows / bn), layer.column_groups, 64 // bk)](
                p,
                circle,
                section,
                offsets,
                target,
                N=layer.shape[0],
                K=layer.shape[1],
                S=64,
                T=64,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                D=4,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                BN=bn,
                BK=bk,
                BA=1,
                FACTORED=True,
                ROW_GROUP_START=start // 64,
                LOCAL_W=True,
                num_warps=4,
                enable_fp_fusion=fp_fusion,
            )
        if cache_weight_dtype == torch.float16 and start < cache_weight_rows:
            cached[start : start + rows].copy_(target)
        output_window = y[:, start : start + rows]
        if gemm_mode == "tf32x3":
            bounded_gemm(flat, target.T, output_window)
        elif gemm_mode == "fp16x3":
            bounded_gemm_fp16x3(flat, target.T, output_window)
        else:
            torch.mm(flat, target.T, out=output_window)
    output = y.reshape(*x.shape[:-1], layer.shape[0])
    return (output, cached) if return_cache else output
