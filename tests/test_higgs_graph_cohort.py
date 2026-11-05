"""HIGGS density permissions are separate from the frozen MNIST cohort."""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from test_graph_capture_qualification import cpu_cuda_api  # noqa: F401
from torch import nn

from benchmarks.training.cell_mass_mnist import make_radius_cst
from benchmarks.training.flat_step_graph import FlatPreparedStep
from benchmarks.training.full_step_graph import PreparedStep
from benchmarks.training.graph_higgs_cohort import (
    HIGGS_ATOM_DIVISORS,
    HIGGS_GRAPH_COHORT,
    higgs_graph_signature,
)
from benchmarks.training.graph_stock_qualification import QualificationError
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor


def model(kind="dense", divisor=20):
    width = 1024
    middle = (
        make_radius_cst(
            width, width * width // divisor, radius_execution="global-response4"
        )
        if kind == "cst"
        else nn.Linear(width, width, bias=False)
    )
    result = nn.Sequential(
        nn.Linear(28, width), nn.ReLU(), middle, nn.ReLU(), nn.Linear(width, 2)
    )
    result._cst_research_cohort = HIGGS_GRAPH_COHORT
    return result


def inputs():
    return torch.ones(8, 28), torch.zeros(8, dtype=torch.int64)


@pytest.mark.parametrize("divisor", HIGGS_ATOM_DIVISORS)
def test_exact_higgs_atom_counts_and_trainable_end_layers(divisor):
    network = model("cst", divisor)
    assert higgs_graph_signature(network, *inputs(), "cst") == (
        HIGGS_GRAPH_COHORT,
        1024,
        1024 * 1024 // divisor,
        "cst",
    )
    assert all(parameter.requires_grad for parameter in network.parameters())


@pytest.mark.parametrize(
    "change", ["tag", "input", "output", "middle", "frozen_input", "frozen_output"]
)
def test_higgs_shape_tag_and_trainability_are_admission_conditions(change):
    network = model()
    if change == "tag":
        network._cst_research_cohort = "higgs_unqualified"
    elif change == "input":
        network[0] = nn.Linear(29, 1024)
    elif change == "output":
        network[4] = nn.Linear(1024, 10)
    elif change == "middle":
        network[2] = nn.Linear(1024, 512, bias=False)
    elif change == "frozen_input":
        network[0].weight.requires_grad_(False)
    else:
        network[4].weight.requires_grad_(False)
    with pytest.raises(QualificationError):
        higgs_graph_signature(network, *inputs(), "dense")


def test_undeclared_atom_density_and_malformed_data_rejected():
    network = model("cst", 200)
    # A shape change must never silently select the nearest admitted density.
    network[2].atoms.p = nn.Parameter(network[2].atoms.p[:-1].detach())
    with pytest.raises(QualificationError, match="atom count"):
        higgs_graph_signature(network, *inputs(), "cst")
    network = model()
    with pytest.raises(QualificationError, match="FP32 rows/INT64 labels"):
        higgs_graph_signature(network, torch.ones(8, 28), torch.zeros(8), "dense")


def test_missing_higgs_declaration_cannot_select_the_legacy_five_percent_path():
    network = model()
    del network._cst_research_cohort
    with pytest.raises(QualificationError, match="explicit research cohort tag"):
        higgs_graph_signature(network, *inputs(), "dense")


@pytest.mark.parametrize("prepared", [PreparedStep, FlatPreparedStep])
@pytest.mark.usefixtures("cpu_cuda_api")
@pytest.mark.parametrize(
    "change",
    [
        "tag",
        "tag_deleted",
        "tag_unknown",
        "input_features",
        "output_features",
        "frozen_input",
        "frozen_output",
    ],
)
def test_prepared_replay_guard_observes_higgs_declaration_changes(prepared, change):
    # CPU CUDA-control-flow mock proves admission/invalidation, not GPU replay.
    network = model()
    optimizer = torch.optim.AdamW(network.parameters(), fused=True)
    tx, ty = inputs()
    program = prepared(network, optimizer, tx, ty, "dense")
    if (
        prepared is FlatPreparedStep
        and program.qualification_error
        == "qualified native storage checker is unavailable"
    ):
        pytest.skip("flat metadata checker is unavailable on this host")
    assert not program.invalidated, program.qualification_error
    assert program.eligible()
    if change == "tag":
        network._cst_research_cohort = None
    elif change == "tag_deleted":
        del network._cst_research_cohort
    elif change == "tag_unknown":
        network._cst_research_cohort = "unknown_higgs_cohort"
    elif change == "input_features":
        network[0].in_features = 29
    elif change == "output_features":
        network[4].out_features = 3
    elif change == "frozen_input":
        network[0].weight.requires_grad_(False)
    else:
        network[4].weight.requires_grad_(False)
    assert not program.eligible()


def test_native_finite_density_predicate_preserves_old_five_percent_rule():
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = Path(executor.__file__).parent
    source = """
#include "limits.cuh"
using namespace radius_cell_mass;
static_assert(owner_window_cohort_atoms(16384) == 13421772);
static_assert(higgs_owner_window_cohort(1024, 5242));
static_assert(higgs_owner_window_cohort(1024, 10485));
static_assert(higgs_owner_window_cohort(1024, 20971));
static_assert(higgs_owner_window_cohort(1024, 52428));
static_assert(higgs_owner_window_cohort(1024, 104857));
static_assert(higgs_owner_window_cohort(2048, 20971));
static_assert(higgs_owner_window_cohort(4096, 1677721));
static_assert(higgs_owner_window_cohort(8192, 6710886));
static_assert(!higgs_owner_window_cohort(1024, 5243));
static_assert(!higgs_owner_window_cohort(512, 13107));
static_assert(!higgs_owner_window_cohort(16384, 13421772));
static_assert(!higgs_owner_window_cohort(32768, 53687091));
"""
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I",
            str(directory),
            "-",
        ],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("divisor", HIGGS_ATOM_DIVISORS)
def test_private_density_graph_replay_tracks_current_atoms_and_all_gradients(divisor):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    torch.manual_seed(413)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        middle = make_radius_cst(
            1024, 1024 * 1024 // divisor, radius_execution="global-response4"
        ).cuda()
        layout = hierarchy_from_operator(middle.declaration())
        configs = layout.input.native_config(), layout.output.native_config()
        p = middle.atoms.p
        x = torch.randn(4, 1024, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        extension = executor._extension()
        old = extension.research_owner_window_fp32_geometry_sort32_flat_c
        entry = extension.research_higgs_owner_window_fp32_geometry_sort32_flat_c

        def evaluate(function, graph_permission):
            y = function(x, p, *configs, graph_permission)
            dx, dp = torch.autograd.grad(y, (x, p), dy)
            return y, dx, dp

        if divisor != 20:
            with pytest.raises(
                RuntimeError, match="fixed power-of-two/atom-density/B4"
            ):
                evaluate(old, True)
        for _ in range(3):
            evaluate(entry, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = evaluate(entry, True)
        with torch.no_grad():
            p[:, 2:].add_(0.375)
            x.add_(0.125)
        expected = evaluate(old, False)
        graph.replay()
        stream.synchronize()
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
        with pytest.raises(RuntimeError, match="fixed power-of-two/atom-density/B4"):
            entry(x, p[:-1].contiguous(), *configs, True)
