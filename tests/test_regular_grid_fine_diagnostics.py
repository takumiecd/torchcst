"""Probes retain arithmetic and isolate layout, materialization and tracing."""

import json
from pathlib import Path

import pytest
import test_regular_grid_h_cuda as base
import torch

from benchmarks.cuda.linear.fine_diagnostics import catalog, census, error


def test_fine_catalog_has_distinct_controls():
    names = catalog()
    assert len(names) == len(set(names)) == 19
    assert {
        "runtime",
        "mirror",
        "sorted-P",
        "sorted-partials",
        "batch-contiguous",
        "cached-profiles",
        "cache-pipeline",
        "G-only",
        "eviction-pressure",
    } <= set(names)


def test_fine_census_accounts_for_masked_and_group_padding():
    p = torch.zeros(13, 9)
    p[10] = torch.arange(1, 10)
    p[12] = 2
    snap = {
        "packed": p,
        "order": torch.arange(9),
        "sizes": (32, 65, 33, 65.0, 33.0, 0.0, 0.0),
    }
    result = census(snap)
    assert result["H"]["valid_site_lanes_per_batch_row"] == 45
    assert result["H"]["scheduled_site_lanes_per_batch_row"] == 192
    assert result["G"]["valid_site_lanes_per_batch_row"] == 18
    assert result["G"]["requested_to_group_union_ratio"] == 4.5


def test_fine_error_keeps_existing_tolerance():
    with pytest.raises(AssertionError):
        error(torch.tensor([1.001]), torch.tensor([1.0]))


@base.GPU
@pytest.mark.parametrize("case", catalog())
@pytest.mark.parametrize("sigma", [3.0, 8.0])
def test_fine_probe_layout_cache_reduction_and_tail(tmp_path, case, sigma):
    from dataclasses import asdict

    from benchmarks.cuda.linear.fine_diagnostics import case_worker
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        prepare_routing,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
        InputBatchProducerHRecipe,
    )

    layer = base.model(
        base.periodic.parameters(17), "input_site32", sigma=sigma, device="cuda"
    )
    recipe = InputBatchProducerHRecipe()
    x = torch.randn(37, 130, device="cuda")[:, ::2]
    dy = torch.randn(33, 37, device="cuda").T
    sizes = (37, 65, 33, 65.0, 33.0, 0.0, 0.0)
    p = _prepare(layer.atoms.p, layer.kernel, sizes, recipe)
    order = prepare_routing(p, sizes, recipe, output=False)[0]
    path = tmp_path / "snapshot.pt"
    torch.save(
        {
            "x": x.cpu(),
            "dy": dy.cpu(),
            "packed": p.cpu(),
            "order": order.cpu(),
            "sizes": sizes,
            "recipe": asdict(recipe),
        },
        path,
    )
    result = case_worker(path, case)
    assert result["status"] == "PASS"
    assert result["memory"]["validation_GPU_buffers_released_before_capture"]
    assert result["schedule"]["chunk_starts"] == [0, 16, 32]
    assert len(result["timing"]["samples_ms"]) == (
        21 if case in ("runtime-one", "eviction-pressure") else 9
    )
    assert all(
        r["max_abs"] <= 4e-4 and r["relative_l2"] <= 4e-4
        for r in result["correctness"].values()
    )


@base.GPU
@pytest.mark.parametrize("dense", [False, True])
def test_fine_trace_copy_does_not_mutate_primary(tmp_path, dense):
    from test_regular_grid_batch_producer_cuda import candidate_selector

    from benchmarks.cuda.linear.fine_diagnostics import trace_copy
    from benchmarks.cuda.linear.periodic_comparison import TrainingStep

    if dense:
        layer = torch.nn.Linear(65, 33, bias=False, device="cuda")
    else:
        layer = base.model(base.periodic.parameters(17), "input_site32", device="cuda")
        layer.selector = candidate_selector(16)[0]
    x = torch.randn(37, 65, device="cuda", requires_grad=True)
    step = TrainingStep(layer, x, torch.randn(37, 33, device="cuda"), dense=dense)
    source = next(layer.parameters()).detach().clone()
    result = trace_copy(step, tmp_path)
    assert result["primary_parameters_unchanged"] and torch.equal(
        source, next(layer.parameters())
    )
    trace = json.loads(Path(result["trace_path"]).read_text())
    assert any(e.get("cat") == "kernel" for e in trace["traceEvents"])
    assert result["kernel_launches_per_replay"] > 0
