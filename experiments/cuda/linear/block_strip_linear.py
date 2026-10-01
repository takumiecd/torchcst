"""Experimental rectangular weight blocks mapped onto a one-dimensional Strip.

Logical block (r, c) is station r * column_groups + c, including the torus
seam. It is adjacency on this Strip, not all neighbors in the matrix grid,
that defines I/B support. Torch supports autograd; Triton is forward-only.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    GridPattern,
    LinePattern,
    StripChart,
    TorusGeometry,
    TriweightSpec,
    presets,
)
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare
from torchcst._backends.torch.kernels import direct_amp_width as _direct
from torchcst._backends.torch.kernels.execution import KernelOptions
from torchcst._backends.torch.operators.strip_torus.preparation import PROFILE_KINDS
from torchcst._backends.torch.parameterizations import direct_amp_width as _coordinates
from torchcst.kernels import ProfileBinding


class BlockStripLinear(nn.Module):
    def __init__(
        self,
        shape,
        tile_shape,
        atoms,
        *,
        device="cpu",
        dtype=torch.float32,
        tile_pitch=4.1,
        sigma=0.5,
        sigma_min=0.2,
    ):
        super().__init__()
        if (
            len(shape) != 2
            or len(tile_shape) != 2
            or any(type(v) is not int or v < 1 for v in (*shape, *tile_shape))
        ):
            raise ValueError("shape and tile_shape must contain two positive integers")
        self.shape, self.tile_shape = tuple(shape), tuple(tile_shape)
        self.row_groups = math.ceil(shape[0] / tile_shape[0])
        self.column_groups = math.ceil(shape[1] / tile_shape[1])
        s, t = tile_shape
        g = self.row_groups * self.column_groups
        grid_width = math.gcd(t, 16)
        chart = StripChart(
            shape=(g * s, t),
            tile_shape=(s, t),
            axis=0,
            tile_pitch=tile_pitch,
            axes=(
                LinePattern(g * s, spacing=min(0.1, 1.5 / max(s - 1, 1))),
                GridPattern((t // grid_width, grid_width), spacing=0.05),
            ),
            geometry=TorusGeometry(
                3,
                major_radius=max(2, g) * tile_pitch / (2 * math.pi),
                minor_radius=0.4,
                representation="intrinsic",
            ),
        )
        kernel = presets.direct_activity(
            amplitude_max=1,
            input_bounds=BandwidthBounds(
                minimum=sigma_min, birth=sigma, maximum=0.8, upper_floor=sigma_min
            ),
            w_c=0.05,
            profile=ProfileBinding(profile=TriweightSpec()),
        )
        self.strip = CSTLinear(
            chart=chart,
            atoms=atoms,
            kernel=kernel,
            kernel_options=KernelOptions(checkpoint_blocks=False),
            backend="tiled",
            device=device,
            dtype=dtype,
        )

    def unroll_weight(self, virtual):
        s, t = self.tile_shape
        return (
            virtual.reshape(self.row_groups, self.column_groups, s, t)
            .permute(0, 2, 1, 3)
            .reshape(self.row_groups * s, self.column_groups * t)[
                : self.shape[0], : self.shape[1]
            ]
        )

    def dense_weight(self):
        """Independent canonical dense oracle; only for small cases."""
        return self.unroll_weight(self.strip.dense_weight())

    def logical_to_virtual(self, indices):
        n, k = indices // self.shape[1], indices % self.shape[1]
        s, t = self.tile_shape
        station = (n // s) * self.column_groups + k // t
        return (station * s + n % s) * t + k % t

    def reference(self, x):
        """Bounded Torch weight tiles, all atoms, ordinary autograd."""
        flat = x.reshape(-1, self.shape[1])
        s, t = self.tile_shape
        layer = self.strip
        p = layer.atoms.p
        center, amp, precision = _coordinates.tile_parameters(
            layer.kernel, layer.chart, p
        )
        outputs = []
        for r in range(self.row_groups):
            width = min(s, self.shape[0] - r * s)
            acc = flat.sum(-1, keepdim=True).expand(-1, width) * 0 + p.sum() * 0
            for c in range(self.column_groups):
                g = r * self.column_groups + c
                w = _direct._single_block(
                    layer.kernel,
                    layer.chart,
                    center,
                    amp,
                    precision,
                    slice(g * s * t, (g + 1) * s * t),
                ).reshape(s, t)
                columns = min(t, self.shape[1] - c * t)
                acc = acc + F.linear(
                    flat[:, c * t : c * t + columns], w[:width, :columns]
                )
            outputs.append(acc)
        return torch.cat(outputs, -1).reshape(*x.shape[:-1], self.shape[0])

    def forward(
        self,
        x,
        *,
        backend="torch",
        batch_tile=None,
        prepared=None,
        output_tile=None,
        num_warps=4,
        column_tile=None,
        support_cull="none",
        fused_config=None,
        weight_chunk_rows=1024,
        materialize_tile=None,
    ):
        if fused_config is not None and backend != "triton_fused":
            raise ValueError("fused_config requires triton_fused")
        if fused_config is not None and batch_tile is not None:
            raise ValueError("specify fused_config or batch_tile")
        if x.shape[-1] != self.shape[1]:
            raise ValueError("input feature count differs from mapped shape")
        if backend == "torch":
            return self.reference(x)
        if backend == "triton_streamed":
            from experiments.cuda.linear.block_streamed_forward import streamed_forward

            return streamed_forward(
                self,
                x,
                prepared=prepared,
                weight_chunk_rows=weight_chunk_rows,
                materialize_tile=(64, 32)
                if materialize_tile is None
                else materialize_tile,
            )
        if weight_chunk_rows != 1024 or materialize_tile is not None:
            raise ValueError(
                "weight_chunk_rows and materialize_tile apply only to triton_streamed"
            )
        if backend not in (
            "triton_direct",
            "triton_reuse",
            "triton_shared",
            "triton_atom_dot",
            "triton_fused",
        ):
            raise ValueError("unknown prototype backend")
        if backend != "triton_atom_dot" and (
            column_tile is not None or support_cull != "none"
        ):
            raise ValueError(
                "column_tile and support_cull apply only to triton_atom_dot"
            )
        if column_tile not in (None, 16, 32, 64, 128) or support_cull not in (
            "none",
            "bounds",
            "exact",
        ):
            raise ValueError("invalid column execution options")
        if output_tile is None:
            output_tile = 16 if backend == "triton_atom_dot" else 4
        if torch.is_grad_enabled() and (
            x.requires_grad or self.strip.atoms.p.requires_grad
        ):
            raise NotImplementedError("mapped Triton prototype supports forward only")
        if x.device.type != "cuda" or x.dtype != torch.float32:
            raise ValueError("mapped Triton requires NVIDIA CUDA float32")
        if x.device != self.strip.atoms.p.device or self.strip.atoms.p.dtype != x.dtype:
            raise ValueError("input and model must share device and dtype")
        from experiments.cuda.linear.block_strip_kernels import (
            block_direct,
            block_direct_reuse,
        )

        p, circle, section, offsets = prepared or prepare(
            self.strip, self.strip.atoms.p, support_layout=True
        )
        flat = x.reshape(-1, self.shape[1]).contiguous()
        y = flat.new_empty((flat.shape[0], self.shape[0]))
        bm = batch_tile or (16 if backend in ("triton_direct", "triton_shared") else 64)
        allowed = (4, 16, 64) if backend == "triton_direct" else (16, 64, 128)
        if backend in ("triton_shared", "triton_atom_dot"):
            allowed = (16, 32, 64)
            valid_outputs = (16, 32) if backend == "triton_atom_dot" else (1, 2, 4, 8)
            if output_tile not in valid_outputs or num_warps not in (4, 8):
                raise ValueError("unsupported shared execution shape")
        elif num_warps != 4 or output_tile != 4:
            raise ValueError("output_tile and num_warps apply only to triton_shared")
        if bm not in allowed:
            raise ValueError("unsupported batch tile")
        s, t = self.tile_shape
        opts = {
            "M": flat.shape[0],
            "N": self.shape[0],
            "K": self.shape[1],
            "S": s,
            "T": t,
            "CG": self.column_groups,
            "G": self.strip.chart.tile_count,
            "D": p.shape[1] - 2,
            "BM": bm,
            "PROFILE": PROFILE_KINDS[self.strip.kernel.profiles[0].binding.profile.id],
        }
        if flat.shape[0]:
            if backend in ("triton_shared", "triton_atom_dot"):
                from experiments.cuda.linear.block_shared_kernel import (
                    block_direct_shared,
                )
                from experiments.cuda.linear.block_support import section_bounds

                bk = column_tile or max(
                    16 if backend == "triton_atom_dot" else 1,
                    min(128, 1 << (t - 1).bit_length()),
                )
                bounds = (
                    section_bounds(section, bk)
                    if support_cull != "none"
                    else torch.stack((section.amin(0), section.amax(0)))
                )
                block_direct_shared[
                    (
                        math.ceil(flat.shape[0] / bm),
                        self.row_groups * math.ceil(s / output_tile),
                    )
                ](
                    flat,
                    p,
                    circle,
                    section,
                    offsets,
                    bounds,
                    y,
                    **opts,
                    BN=output_tile,
                    USE_DOT=backend == "triton_atom_dot",
                    BK=bk,
                    CULL_CHUNKS=support_cull != "none",
                    CHECK_ZERO=support_cull == "exact",
                    num_warps=num_warps,
                    enable_fp_fusion=False,
                )
            elif backend in ("triton_direct", "triton_reuse"):
                bounds = torch.stack((section.amin(0), section.amax(0)))
                kernel = (
                    block_direct_reuse if backend == "triton_reuse" else block_direct
                )
                kernel[(math.ceil(flat.shape[0] / bm), self.shape[0])](
                    flat,
                    p,
                    circle,
                    section,
                    offsets,
                    bounds,
                    y,
                    **opts,
                    BK=min(128, 1 << (t - 1).bit_length())
                    if backend == "triton_reuse"
                    else 128,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                from experiments.cuda.linear.block_fused_config import (
                    default_fused_config,
                    launch_fused,
                )

                config = fused_config or default_fused_config(
                    self.tile_shape,
                    flat.shape[0],
                    batch_tile,
                    atom_density=self.strip.atoms.p.shape[0]
                    / (self.shape[0] * self.shape[1]),
                    gpu_name=torch.cuda.get_device_name(flat.device),
                    logical_shape=self.shape,
                )
                launch_fused(self, flat, (p, circle, section, offsets), y, config)
        return y.reshape(*x.shape[:-1], self.shape[0])
