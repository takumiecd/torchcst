"""Globally L2-normalized, compact-support Euclidean Strip linear maps."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from torchcst.geometry import EuclideanGeometry, GridPattern, LinePattern, StripChart

_SIGMA_LOW = 0.03
_SIGMA_HIGH = 3.25
_FLOOR = 1e-6


class NormalizedStripLinear(nn.Module):
    """Sum globally normalized Triweight atoms on a regular 3D Strip.

    ``p`` has columns ``[amplitude, log_sigma, c0, c1, c2]``. The module owns
    a cloned parameter and snapshots the fixed chart metadata. Each atom is
    normalized over the entire finite operator, including in ``memory='window'``.
    Widths are ``exp(clamp(log_sigma, log(.03), log(3.25)))`` and the norm floor
    is ``1e-6``. Signed amplitudes and ordinary gradients of all five parameters
    are supported. No bias is added.

    CPU float32/float64 uses a support-local PyTorch reference. CUDA requires
    float32, spacing ``(1, .5, .5)``, disabled TF32 and nondeterministic atomic
    accumulation. The opt-in window implementation uses at most 512 weight rows
    and falls back to the full implementation when its metadata guards do not
    apply. CUDA backends support first derivatives only.
    """

    def __init__(self, chart: StripChart, p: Tensor, *, memory: str = "full"):
        super().__init__()
        if memory not in ("full", "window"):
            raise ValueError("memory must be 'full' or 'window'")
        if not isinstance(chart, StripChart):
            raise TypeError("chart must be a StripChart")
        if any(
            tensor.requires_grad for tensor in (*chart.parameters(), *chart.buffers())
        ):
            raise ValueError(
                "NormalizedStripLinear requires a fixed, nontrainable chart"
            )
        if (
            type(chart.geometry) is not EuclideanGeometry
            or chart.geometry.embedding_dim != 3
            or len(chart.shape) != 2
            or chart.axis != 0
            or len(chart.axes) != 2
            or type(chart.axes[0]) is not LinePattern
            or type(chart.axes[1]) is not GridPattern
            or chart.axes[1].dim != 2
        ):
            raise ValueError("requires a regular 3D Euclidean Strip chart")
        line, grid = chart.axes
        spacing = (float(line.spacing[0]), *map(float, grid.spacing))
        origin = (float(line.start[0]), *map(float, grid.start))
        if not all(math.isfinite(v) for v in (*origin, *spacing)) or min(spacing) <= 0:
            raise ValueError("chart origin must be finite and spacing positive")
        if not math.isclose(
            float(chart.tile_pitch), chart.tile_shape[0] * spacing[0], rel_tol=1e-6
        ):
            raise ValueError("Strip rows must be contiguous")
        if not isinstance(p, Tensor) or p.ndim != 2 or p.shape[1] != 5:
            raise ValueError("p must have shape [atoms, 5]")
        if p.dtype not in (torch.float32, torch.float64):
            raise TypeError("p must be float32 or float64")
        if p.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU and CUDA devices are supported")
        if not bool(torch.isfinite(p).all()):
            raise ValueError("p must be finite")
        self.p = nn.Parameter(p.detach().clone().contiguous())
        self.memory = memory
        self._sizes = (chart.shape[0], *grid.shape)
        if math.prod(grid.shape) != chart.shape[1]:
            raise ValueError("chart column shape differs from its GridPattern")
        self.register_buffer("origin", p.new_tensor(origin))
        self.register_buffer("spacing", p.new_tensor(spacing))
        self.register_buffer("sizes", torch.tensor(self._sizes, device=p.device))
        # This model contract is fixed, not a tunable bandwidth buffer.
        self.register_buffer(
            "sigma_bounds",
            torch.tensor(
                [_SIGMA_LOW, _SIGMA_HIGH], dtype=torch.float64, device=p.device
            ),
        )
        self._refresh_metadata()

    @property
    def in_features(self) -> int:
        return self._sizes[1] * self._sizes[2]

    @property
    def out_features(self) -> int:
        return self._sizes[0]

    def _refresh_metadata(self):
        if (
            self.origin.shape != (3,)
            or self.spacing.shape != (3,)
            or self.sizes.shape != (3,)
        ):
            raise RuntimeError("invalid chart metadata shape")
        origin = tuple(float(v) for v in self.origin.detach().cpu())
        spacing = tuple(float(v) for v in self.spacing.detach().cpu())
        sizes = tuple(int(v) for v in self.sizes.detach().cpu())
        # Compare in the stored dtype, including after .float()/.double().
        expected = torch.tensor(
            [_SIGMA_LOW, _SIGMA_HIGH],
            dtype=torch.float64,
            device=self.sigma_bounds.device,
        )
        if self.sigma_bounds.dtype != torch.float64 or not torch.equal(
            self.sigma_bounds, expected
        ):
            raise RuntimeError(
                "checkpoint sigma bounds differ from the fixed normalized model"
            )
        if sizes != self._sizes:
            raise RuntimeError("checkpoint chart sizes differ")
        if not all(math.isfinite(v) for v in (*origin, *spacing)) or min(spacing) <= 0:
            raise RuntimeError("invalid chart metadata")
        self._plan = SimpleNamespace(origin=origin, spacing=spacing, sizes=sizes)
        from ._backends.cuda.schema import OperatorSpec

        self._cuda_operator = OperatorSpec(sizes=sizes, origin=origin, spacing=spacing)
        self._metadata_versions = tuple(
            (id(t), t._version)
            for t in (self.origin, self.spacing, self.sizes, self.sigma_bounds)
        )

    def _apply(self, fn, recurse=True):
        expected = torch.tensor(
            [_SIGMA_LOW, _SIGMA_HIGH],
            dtype=torch.float64,
            device=self.sigma_bounds.device,
        )
        if self.sigma_bounds.dtype != torch.float64 or not torch.equal(
            self.sigma_bounds, expected
        ):
            raise RuntimeError(
                "checkpoint sigma bounds differ from the fixed normalized model"
            )
        result = super()._apply(fn, recurse=recurse)
        # Module dtype conversions must never quantize the fixed contract.
        self.sigma_bounds = expected.to(device=self.p.device)
        self._refresh_metadata()
        return result

    def get_extra_state(self):
        return {
            "format_version": 1,
            "memory": self.memory,
            "sizes": self._sizes,
            "sigma_bounds": (_SIGMA_LOW, _SIGMA_HIGH),
            "norm_floor": _FLOOR,
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state != {**self.get_extra_state(), "memory": state.get("memory")}
            or state.get("memory") not in ("full", "window")
        ):
            raise RuntimeError("normalized Strip checkpoint contract differs")
        self.memory = state["memory"]
        self._refresh_metadata()

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor) or x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("x must have last dimension equal to in_features")
        if x.dtype != self.p.dtype or x.device != self.p.device:
            raise ValueError("x and p must have the same device and dtype")
        if self.p.dtype not in (torch.float32, torch.float64):
            raise TypeError("p must be float32 or float64")
        if self.memory not in ("full", "window"):
            raise ValueError("memory must be 'full' or 'window'")
        versions = tuple(
            (id(t), t._version)
            for t in (self.origin, self.spacing, self.sizes, self.sigma_bounds)
        )
        if versions != self._metadata_versions:
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "chart metadata must be refreshed outside CUDA graph capture"
                )
            self._refresh_metadata()
        flat = x.reshape(-1, self.in_features).contiguous()
        if len(self.p) == 0 or flat.shape[0] == 0:
            result = (
                flat.new_zeros((flat.shape[0], self.out_features))
                + flat.reshape(-1)[:1].mul(0).sum()
                + self.p.reshape(-1)[:1].mul(0).sum()
            )
        elif x.device.type == "cpu":
            result = _reference(flat, self.p, self._plan)
        elif x.device.type == "cuda":
            if x.dtype != torch.float32:
                raise TypeError("normalized Strip CUDA requires float32")
            if torch.are_deterministic_algorithms_enabled():
                raise RuntimeError(
                    "normalized Strip CUDA requires nondeterministic atomic accumulation"
                )
            if torch.is_autocast_enabled():
                raise RuntimeError("normalized Strip CUDA does not support autocast")
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError("normalized Strip CUDA requires TF32 to be disabled")
            from ._backends.cuda.algorithms.normalized_strip import REGISTRY
            from ._backends.cuda.context import context_from_tensors
            from ._backends.cuda.dispatch.select import select_normalized

            context = context_from_tensors(self._cuda_operator, flat, self.p)
            decision = select_normalized(context, memory=self.memory)
            result = REGISTRY.execute(
                decision.plan,
                context,
                x=flat,
                parameters=self.p,
                operator=self._cuda_operator,
            )
        else:
            raise ValueError("only CPU and CUDA devices are supported")
        return result.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, atoms={len(self.p)}, memory={self.memory!r}"


def _reference(x, p, plan):
    """Bounded temporary site chunks; never construct an operator-sized W."""
    y = x.new_zeros((len(x), plan.sizes[0])) + p.reshape(-1)[:1].mul(0).sum()
    eps = torch.finfo(p.dtype).eps
    for atom in p:
        center = atom[2:]
        axes = []
        for c, o, s, n in zip(
            center.detach().tolist(), plan.origin, plan.spacing, plan.sizes
        ):
            margin = 32 * eps * (abs(c) + abs(o) + n * s + _SIGMA_HIGH + 1)
            lo = max(0, min(n, math.ceil((c - _SIGMA_HIGH - margin - o) / s)))
            hi = max(lo, min(n, math.floor((c + _SIGMA_HIGH + margin - o) / s) + 1))
            axes.append((lo, hi))
        lengths = tuple(hi - lo for lo, hi in axes)
        total = math.prod(lengths)
        if not total:
            continue
        precision = torch.exp(
            -2 * atom[1].clamp(math.log(_SIGMA_LOW), math.log(_SIGMA_HIGH))
        )

        def chunk(
            start,
            total=total,
            lengths=lengths,
            axes=axes,
            center=center,
            precision=precision,
        ):
            index = torch.arange(start, min(start + 4096, total), device=p.device)
            r2 = index % lengths[2] + axes[2][0]
            r1 = index // lengths[2] % lengths[1] + axes[1][0]
            r0 = index // (lengths[1] * lengths[2]) + axes[0][0]
            d = [
                o + r.to(p.dtype) * s - c
                for o, r, s, c in zip(plan.origin, (r0, r1, r2), plan.spacing, center)
            ]
            q = ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) * precision
            t = (1 - q).clamp_min(0)
            return r0, r1 * plan.sizes[2] + r2, t * t * t

        norm2 = atom.new_zeros(())
        for start in range(0, total, 4096):
            k = chunk(start)[2]
            norm2 = norm2 + (k * k).sum()
        # Safe zero-support derivative, retaining the inclusive floor tie.
        norm = norm2.clamp_min(1e-30).sqrt().clamp_min(_FLOOR)
        for start in range(0, total, 4096):
            rows, cols, k = chunk(start)
            values = x[:, cols] * (atom[0] * k / norm)
            y = y.index_add(1, rows, values)
    return y + x.reshape(-1)[:1].mul(0).sum()
