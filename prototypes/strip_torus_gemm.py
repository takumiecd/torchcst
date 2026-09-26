"""Small cost probe for the PyTorch Strip + Torus tiled backend."""

from __future__ import annotations

import math
from time import perf_counter

import torch
from torch import Tensor

from torchcst import (
    CSTLinear,
    DirectAmpWidth,
    GridPattern,
    LinePattern,
    StripChart,
    TorusGeometry,
    Triweight,
)
from torchcst.nn._layout import AtomLayout, initial_layout
from torchcst.nn._strip_torus import (
    owners_from_support,
    support_mask,
    validate_tiled,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_forward(
    model: CSTLinear, inputs: Tensor, p: Tensor, layout: AtomLayout
) -> tuple[Tensor, float, float]:
    """Mirror tiled_linear only to isolate tile evaluation and matmul time."""

    chart = model.chart
    packed = layout.pack(p)
    flat_inputs = inputs.reshape(-1, model.in_features)
    outputs = []
    weight_ms = 0.0
    matmul_ms = 0.0
    columns = torch.arange(model.in_features, device=p.device)
    for station in range(chart.tile_count):
        neighbors = dict.fromkeys(
            ((station - 1) % chart.tile_count, station, (station + 1) % chart.tile_count)
        )
        candidates = torch.cat(
            [packed[layout.offsets[g] : layout.offsets[g + 1]] for g in neighbors]
        )
        row_start = station * chart.tile_shape[0]
        row_stop = min(row_start + chart.tile_shape[0], model.out_features)
        rows = torch.arange(row_start, row_stop, device=p.device)
        if candidates.shape[0]:
            _synchronize(inputs.device)
            before_weight = perf_counter()
            weight = model.kernel.weight_tile(chart, candidates, rows, columns)
            _synchronize(inputs.device)
            before_matmul = perf_counter()
            result = flat_inputs @ weight.T
            _synchronize(inputs.device)
            weight_ms += (before_matmul - before_weight) * 1000
            matmul_ms += (perf_counter() - before_matmul) * 1000
        else:
            result = flat_inputs.new_zeros((flat_inputs.shape[0], rows.numel()))
        outputs.append(result)
    result = torch.cat(outputs, dim=-1).reshape(*inputs.shape[:-1], model.out_features)
    return result, weight_ms, matmul_ms


@torch.no_grad()
def probe(model: CSTLinear, inputs: Tensor) -> dict[str, object]:
    """Measure routing, packing, tile evaluation, and matmul separately."""

    chart = validate_tiled(model.chart, model.kernel)
    _synchronize(inputs.device)
    started = perf_counter()
    touched = support_mask(chart, model.kernel, model.atoms.p)
    _synchronize(inputs.device)
    routed = perf_counter()
    owners = owners_from_support(chart, model.atoms.p, touched)
    layout = initial_layout(owners, chart.tile_count)
    layout.pack(model.atoms.p)
    _synchronize(inputs.device)
    packed_at = perf_counter()
    actual, weight_ms, matmul_ms = _timed_forward(model, inputs, model.atoms.p, layout)
    _synchronize(inputs.device)
    tiled_at = perf_counter()
    expected = model.kernel.weight(chart, model.atoms.p)
    expected = torch.nn.functional.linear(inputs, expected)
    _synchronize(inputs.device)
    dense_at = perf_counter()
    return {
        "support_ms": (routed - started) * 1000,
        "packing_ms": (packed_at - routed) * 1000,
        "tiled_forward_ms": (tiled_at - packed_at) * 1000,
        "weight_eval_ms": weight_ms,
        "matmul_ms": matmul_ms,
        "dense_forward_ms": (dense_at - tiled_at) * 1000,
        "max_abs_error": (actual - expected).abs().max().item(),
        "atoms_per_station": (layout.offsets[1:] - layout.offsets[:-1]).tolist(),
        "touched_stations_per_atom": torch.bincount(
            touched.sum(0), minlength=chart.tile_count + 1
        ).tolist(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)
    chart = StripChart(
        shape=(64, 128),
        tile_shape=(16, 128),
        axes=(LinePattern(64, spacing=0.1), GridPattern((8, 16), spacing=0.05)),
        axis=0,
        tile_pitch=4.1,
        geometry=TorusGeometry(
            3,
            major_radius=16.4 / (2 * math.pi),
            minor_radius=0.4,
            max_arc_step=2.0,
            representation="intrinsic",
        ),
    )
    kernel = DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=0.2,
        sigma_birth=0.5,
        sigma_max=0.8,
        w_c=0.05,
        profile=Triweight(0.2, normalize_columns=False),
        checkpoint_blocks=False,
    )
    layer = CSTLinear(chart=chart, atoms=256, kernel=kernel, backend="tiled")
    x = torch.randn(32, layer.in_features)
    print(probe(layer, x))
