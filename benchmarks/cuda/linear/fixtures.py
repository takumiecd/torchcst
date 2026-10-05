"""Pure regular-site fixtures shared by registry tests and benchmarks."""

import math

from torchcst.charts import ProductChartSpec
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels.presets import NORMALIZED_RADIAL_TRIWEIGHT
from torchcst.operators.spec import OperatorSpec, SingleChartSpec
from torchcst.patterns import GridPatternSpec, LinePatternSpec


def operator_spec(*, sizes, origin, spacing):
    """Canonical contiguous sites; CUDA storage tiling is not mathematical state."""
    chart = ProductChartSpec(
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


def strip_torus_model(
    atoms, *, rows=64, columns=128, station_rows=16, device="cuda", backend="triton"
):
    from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, presets
    from torchcst import chart_presets as layouts
    from torchcst import geometry_presets as spaces
    from torchcst import pattern_presets as patterns
    from torchcst._backends.torch.kernels.execution import KernelOptions
    from torchcst.kernels import ProfileBinding

    stations = math.ceil(rows / station_rows)
    chart = layouts.strip(
        shape=(rows, columns),
        tile_shape=(station_rows, columns),
        axes=(
            patterns.line(rows, spacing=min(0.1, 1.5 / max(station_rows - 1, 1))),
            patterns.grid((columns // 16, 16), spacing=0.05),
        ),
        axis=0,
        tile_pitch=4.1,
        geometry=spaces.torus(
            3,
            major_radius=max(2, stations) * 4.1 / (2 * math.pi),
            minor_radius=0.4,
            representation="intrinsic",
        ),
    )
    kernel = presets.direct_activity(
        amplitude_max=1.0,
        input_bounds=BandwidthBounds(
            minimum=0.2, birth=0.5, maximum=0.8, upper_floor=0.2
        ),
        w_c=0.05,
        profile=ProfileBinding(profile=TriweightSpec()),
    )
    return CSTLinear(
        chart=chart,
        atoms=atoms,
        kernel=kernel,
        kernel_options=KernelOptions(checkpoint_blocks=False),
        backend=backend,
        device=device,
    )


def normalized_chart(
    sizes=(64, 4, 4), origin=(0.0, 0.0, 0.0), *, dtype=None, device="cpu"
):
    """One continuous Strip, using public declarations and no test helpers."""
    import torch

    from torchcst import chart_presets as layouts
    from torchcst import compile_chart
    from torchcst import pattern_presets as patterns

    n, h, j = sizes
    spec = layouts.strip(
        shape=(n, h * j),
        tile_shape=(n, h * j),
        axis=0,
        tile_pitch=float(n),
        axes=(
            patterns.line(n, low=origin[0], high=origin[0] + n - 1),
            patterns.grid(
                (h, j),
                low=origin[1:],
                high=(origin[1] + (h - 1) * 0.5, origin[2] + (j - 1) * 0.5),
            ),
        ),
    )
    return compile_chart(spec, device=device, dtype=dtype or torch.float64)


def local_product_state(*, minimum=1.0, maximum=16.0, birth=3.0, w_c=1e6):
    """Production polar contract; one shared set of bounds, normalized Triweight."""

    from torchcst.kernels import BandwidthBounds, TriweightSpec, presets
    from torchcst.kernels.state import KernelState

    return KernelState(
        presets.polar_activity(
            amplitude_max=1.0,
            input_bounds=BandwidthBounds(
                minimum=minimum, maximum=maximum, birth=birth, upper_floor=birth
            ),
            w_c=w_c,
            profile=presets.profile(TriweightSpec()),
            radial_regularization=0.1,
            activity_gain=1.0,
            activity_mode="finite_chord",
            dormant_expansion_rate=0.02,
        )
    )


def local_product_reference(x, p, value, charts, domain):
    """Production full-chart factors, sliced AFTER discrete L2 normalization."""

    from torchcst._backends.torch.kernels import execution

    v, u = execution.factors(value, *charts, p)
    j = slice(domain.input_start, domain.input_start + domain.input_count)
    i = slice(domain.output_start, domain.output_start + domain.output_count)
    return (x @ v[j]) @ u[i].T


def local_product_dense_factors(p, value, domain):
    """Graph-safe Torch baseline, verified against production full-chart factors."""

    import torch

    from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
        decode,
    )

    q = decode(value, p)
    j = torch.arange(domain.input_size, device=p.device, dtype=p.dtype)
    i = torch.arange(domain.output_size, device=p.device, dtype=p.dtype)
    v = (
        (
            1
            - (
                domain.input_origin + j[:, None] * domain.spacing - q[None, :, 2]
            ).square()
            * q[None, :, 1]
        )
        .clamp_min(0)
        .pow(3)
    )
    u = (
        (
            1
            - (
                domain.output_origin + i[:, None] * domain.spacing - q[None, :, 3]
            ).square()
            * q[None, :, 1]
        )
        .clamp_min(0)
        .pow(3)
    )
    v = v / torch.linalg.vector_norm(v, dim=0).clamp_min(1e-6)[None]
    u = u / torch.linalg.vector_norm(u, dim=0).clamp_min(1e-6)[None]
    v = v[domain.input_start : domain.input_start + domain.input_count]
    u = (
        u[domain.output_start : domain.output_start + domain.output_count]
        * q[None, :, 0]
    )
    return v, u
