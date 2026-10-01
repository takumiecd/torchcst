"""Globally L2-normalized, compact-support Euclidean Strip linear maps."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from torchcst.geometry import EuclideanGeometry, GridPattern, LinePattern, StripChart
from torchcst.geometry.spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
)
from torchcst.kernels.presets import NORMALIZED_RADIAL_TRIWEIGHT
from torchcst.operators.spec import OperatorSpec, SingleChartSpec

_SIGMA_LOW = 0.03
_SIGMA_HIGH = 3.25
_FLOOR = 1e-6


def normalized_strip_declaration(*, sizes, origin, spacing):
    """Canonical contiguous sites; CUDA storage tiling is not mathematical state."""
    chart = ChartSpec(
        kind="product",
        geometry=EuclideanGeometrySpec(intrinsic_dim=3),
        shape=(sizes[0], math.prod(sizes[1:])),
        axes=(
            LinePatternSpec(
                shape=(sizes[0],), start=(origin[0],), spacing=(spacing[0],)
            ),
            GridPatternSpec(shape=sizes[1:], start=origin[1:], spacing=spacing[1:]),
        ),
    )
    return OperatorSpec(
        layout=SingleChartSpec(chart=chart), kernel=NORMALIZED_RADIAL_TRIWEIGHT
    )


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
        self._operator_spec = normalized_strip_declaration(
            sizes=sizes, origin=origin, spacing=spacing
        )
        self._metadata_versions = tuple(
            (id(t), t._version)
            for t in (self.origin, self.spacing, self.sizes, self.sigma_bounds)
        )

    def declaration(self):
        """Snapshot the common single-chart contract at a configuration boundary."""
        self._refresh_metadata()
        return self._operator_spec

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
            from torchcst._backends.torch.operators.normalized_radial import apply

            result = apply(flat, self.p, self._plan)
        elif x.device.type == "cuda":
            from torchcst._backends.cuda.algorithms.normalized_euclidean_strip import (
                REGISTRY,
            )
            from torchcst._backends.cuda.context import context_from_tensors
            from torchcst._backends.cuda.dispatch.select import select_normalized

            context = context_from_tensors(self._operator_spec, flat, self.p)
            decision = select_normalized(context, memory=self.memory)
            result = REGISTRY.execute(
                decision.plan,
                context,
                x=flat,
                parameters=self.p,
                operator=self._operator_spec,
            )
        else:
            raise ValueError("only CPU and CUDA devices are supported")
        return result.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, atoms={len(self.p)}, memory={self.memory!r}"
