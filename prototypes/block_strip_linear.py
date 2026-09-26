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
    CSTLinear,
    DirectAmpWidth,
    GridPattern,
    LinePattern,
    StripChart,
    TorusGeometry,
    Triweight,
)
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


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
        kernel = DirectAmpWidth(
            amplitude_max=1,
            sigma_min=sigma_min,
            sigma_birth=sigma,
            sigma_max=0.8,
            w_c=0.05,
            profile=Triweight(sigma_min, normalize_columns=False),
            checkpoint_blocks=False,
        )
        self.strip = CSTLinear(
            chart=chart,
            atoms=atoms,
            kernel=kernel,
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
        center, amp, precision = layer.kernel.tile_parameters(layer.chart, p)
        outputs = []
        for r in range(self.row_groups):
            width = min(s, self.shape[0] - r * s)
            acc = flat.sum(-1, keepdim=True).expand(-1, width) * 0 + p.sum() * 0
            for c in range(self.column_groups):
                g = r * self.column_groups + c
                w = layer.kernel._single_block(
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

    def forward(self, x, *, backend="torch", batch_tile=None, prepared=None):
        if x.shape[-1] != self.shape[1]:
            raise ValueError("input feature count differs from mapped shape")
        if backend == "torch":
            return self.reference(x)
        if backend not in ("triton_direct", "triton_fused"):
            raise ValueError("unknown prototype backend")
        if torch.is_grad_enabled() and (
            x.requires_grad or self.strip.atoms.p.requires_grad
        ):
            raise NotImplementedError("mapped Triton prototype supports forward only")
        if x.device.type != "cuda" or x.dtype != torch.float32:
            raise ValueError("mapped Triton requires NVIDIA CUDA float32")
        if x.device != self.strip.atoms.p.device or self.strip.atoms.p.dtype != x.dtype:
            raise ValueError("input and model must share device and dtype")
        from prototypes.block_strip_kernels import block_direct, block_fused

        p, circle, section, offsets = prepared or prepare(
            self.strip, self.strip.atoms.p, support_layout=True
        )
        flat = x.reshape(-1, self.shape[1]).contiguous()
        y = flat.new_empty((flat.shape[0], self.shape[0]))
        bm = batch_tile or (16 if backend == "triton_direct" else 64)
        if bm not in ((4, 16, 64) if backend == "triton_direct" else (16, 64)):
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
            "PROFILE": PROFILE_KINDS[type(self.strip.kernel.profile)],
        }
        if flat.shape[0]:
            if backend == "triton_direct":
                bounds = torch.stack((section.amin(0), section.amax(0)))
                block_direct[(math.ceil(flat.shape[0] / bm), self.shape[0])](
                    flat,
                    p,
                    circle,
                    section,
                    offsets,
                    bounds,
                    y,
                    **opts,
                    BK=128,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                block_fused[
                    (math.ceil(flat.shape[0] / bm), self.row_groups * math.ceil(s / 16))
                ](
                    flat,
                    p,
                    circle,
                    section,
                    offsets,
                    y,
                    **opts,
                    BN=16,
                    BK=16,
                    BA=8,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return y.reshape(*x.shape[:-1], self.shape[0])
