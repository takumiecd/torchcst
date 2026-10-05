"""Local routing must preserve exhaustive ownership, including circular ties."""

import math
from dataclasses import replace

import pytest
import torch

from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.operators.strip_torus.tiled import CircleRouting

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def routing(stations, *, start=0.0, spacing=0.1, extra=0.0):
    rows = stations * 8 - 3
    chart = _construction.strip(
        shape=(rows, 1),
        tile_shape=(8, 1),
        axis=0,
        tile_pitch=4.1,
        axes=(
            _construction.line_pattern(
                rows, low=start, high=start + (rows - 1) * spacing
            ),
            _construction.line_pattern(1, spacing=1.0),
        ),
        geometry=_construction.torus(
            2,
            major_radius=(stations * 4.1 + extra) / (2 * math.pi),
            minor_radius=0.4,
            representation="intrinsic",
        ),
    ).cuda()
    return CircleRouting.from_chart(chart)


def decoded_arcs(plan, arcs):
    angle = arcs / plan.major_radius
    return torch.stack((angle.cos(), angle.sin()), 1)


@pytest.mark.parametrize(
    "stations,start,spacing,extra",
    [
        (3, 0.0, 0.1, 0.0),
        (4, -37.3, 0.0, 50.0),
        (31, 107.1, 0.58, 1000.0),
        (1025, -81.7, 0.1, 0.0),
        (4096, 19.2, 0.1, 1000.0),
        (16384, 0.0, 0.1, 0.0),
    ],
)
def test_local_candidates_match_exhaustive_all_station_boundaries(
    stations, start, spacing, extra
):
    from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.preparation import (
        route_and_layout,
    )

    plan = routing(stations, start=start, spacing=spacing, extra=extra)
    assert plan.local_candidates
    torch.manual_seed(213)
    ends = plan.starts + plan.spans
    following = torch.cat((plan.starts[1:], plan.starts[:1] + plan.period))
    midpoints = (ends + following) / 2
    sites = torch.cat((plan.starts, ends, midpoints))
    points = torch.cat(
        (
            sites,
            torch.nextafter(sites, torch.full_like(sites, float("inf"))),
            torch.nextafter(sites, torch.full_like(sites, -float("inf"))),
            (torch.rand(2048, device="cuda") * 20 - 10) * plan.period,
        )
    )
    decoded = decoded_arcs(plan, points)
    # Noncontiguous centers and both signed zeros at the angular cut.
    decoded = decoded.T.contiguous().T
    decoded[:4] = decoded.new_tensor([[1, 0], [1, -0.0], [-1, 0], [-1, -0.0]])
    actual = route_and_layout(plan, decoded)
    exhaustive = route_and_layout(replace(plan, local_candidates=False), decoded)
    for got, wanted in zip(actual, exhaustive):
        torch.testing.assert_close(got, wanted, atol=0, rtol=0)
    # Independent Torch oracle on bounded samples, covering seam and random arcs.
    selection = torch.linspace(0, len(decoded) - 1, 257, device="cuda").long()
    torch.testing.assert_close(
        actual[0][selection], plan.owners(decoded[selection]), atol=0, rtol=0
    )


def test_extreme_coordinate_scale_uses_exhaustive_fallback():
    from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.preparation import (
        route_and_layout,
    )

    plan = routing(31, start=1e6)
    assert not plan.local_candidates
    decoded = decoded_arcs(plan, torch.linspace(-100, 100, 257, device="cuda"))
    torch.testing.assert_close(
        route_and_layout(plan, decoded)[0], plan.owners(decoded), atol=0, rtol=0
    )


def test_captured_routing_handles_unrestricted_atom_jumps():
    from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.preparation import (
        route_and_layout,
    )

    plan = routing(4096)
    decoded = decoded_arcs(plan, torch.zeros(257, device="cuda"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            route_and_layout(plan, decoded)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = route_and_layout(plan, decoded)
    for scale in (-100.0, 0.37, 250.0):
        arcs = torch.linspace(-1, 1, 257, device="cuda") * plan.period * scale
        decoded.copy_(decoded_arcs(plan, arcs))
        graph.replay()
        expected = route_and_layout(replace(plan, local_candidates=False), decoded)
        for got, wanted in zip(actual, expected):
            torch.testing.assert_close(got, wanted, atol=0, rtol=0)
