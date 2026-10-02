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
