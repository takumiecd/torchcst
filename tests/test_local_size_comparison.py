"""Matched sizes and captured updates for the local product size campaign."""

import pytest
import torch
from test_local_product_research import (
    test_graph_training_updates_width_and_matches_public_optimizer as check_training,
)

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run


@pytest.mark.parametrize("size", [32, 64, 128])
@pytest.mark.parametrize("stage", ["early", "middle", "late", "narrow-only"])
def test_size_comparison_snapshot(size, stage):
    run = load_run(
        f"benchmarks/cuda/linear/cases/local-size-{size}-{stage}.json",
        "benchmarks/cuda/linear/plans-local-persistent.json",
    )
    assert decode_snapshot(run.snapshot()) == run
    assert run.case.rows == 32
    assert run.case.atoms == int(0.05 * size * size)
    assert run.dense
    assert run.case.widths.rho == (0.25, 0.75, 2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size", [32, 64, 128])
def test_size_comparison_public_optimizer(size):
    check_training("hybrid-persistent-mid4", size_override=size)
