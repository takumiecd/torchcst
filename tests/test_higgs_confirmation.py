"""Low-density permissions and dev/confirmation separation contracts."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.training.graph_higgs_cohort import (
    HIGGS_CONFIRMATION_COHORT,
    HIGGS_CONFIRMATION_DIVISORS,
    HIGGS_CONFIRMATION_WIDTHS,
    HIGGS_GRAPH_COHORT,
    higgs_graph_signature,
)
from benchmarks.training.graph_stock_qualification import QualificationError
from benchmarks.training.higgs_confirmation import (
    CONFIRMATION_ROWS,
    DEV_ROWS,
    atom_count,
    choose_best,
    development_rows,
    evaluate,
    load_partition,
    make_model,
    validate_checkpoint,
    validation_slice,
)
from benchmarks.training.higgs_density import make_model as exploratory_model
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor


@pytest.mark.parametrize("width", HIGGS_CONFIRMATION_WIDTHS)
@pytest.mark.parametrize("divisor", HIGGS_CONFIRMATION_DIVISORS)
def test_exact_low_density_cohort_and_shared_ordinary_layers(width, divisor):
    cst, dense = (make_model(width, divisor, 29, kind) for kind in ("cst", "dense"))
    x, y = torch.ones(8, 28), torch.zeros(8, dtype=torch.int64)
    assert higgs_graph_signature(cst, x, y, "cst") == (
        HIGGS_CONFIRMATION_COHORT,
        width,
        width * width // divisor,
        "cst",
    )
    assert higgs_graph_signature(dense, x, y, "dense") == (
        HIGGS_CONFIRMATION_COHORT,
        width,
        None,
        "dense",
    )
    assert cst[2].atoms.p.shape == (atom_count(width, divisor), 4)
    assert all(p.requires_grad for p in cst.parameters())
    for index in (0, 4):
        for name, tensor in cst[index].state_dict().items():
            assert torch.equal(tensor, dense[index].state_dict()[name])


def test_legacy_higgs_cannot_admit_low_density_by_retagging():
    network = make_model(1024, 1000, 29, "cst")
    x, y = torch.ones(8, 28), torch.zeros(8, dtype=torch.int64)
    network._cst_research_cohort = HIGGS_GRAPH_COHORT
    with pytest.raises(QualificationError, match="atom count"):
        higgs_graph_signature(network, x, y, "cst")
    network = exploratory_model(4096, 200, 29, "dense")
    network._cst_research_cohort = HIGGS_CONFIRMATION_COHORT
    with pytest.raises(QualificationError, match="explicit width"):
        higgs_graph_signature(network, x, y, "dense")


@pytest.mark.parametrize(
    "width,divisor", [(4096, 100), (1024, 20), (1024, 250), (1024.0, 100), (1024, True)]
)
def test_confirmation_does_not_silently_expand_density_or_width(width, divisor):
    with pytest.raises(ValueError):
        atom_count(width, divisor)


def test_dense_initialization_is_density_independent_and_rng_preserved():
    torch.manual_seed(73)
    before = torch.random.get_rng_state().clone()
    expected = make_model(1024, 1000, 71, "dense")[2].weight.detach()
    for divisor in HIGGS_CONFIRMATION_DIVISORS:
        assert torch.equal(expected, make_model(1024, divisor, 71, "dense")[2].weight)
    assert torch.equal(before, torch.random.get_rng_state())


def test_selection_uses_only_development_loss_and_earliest_ties():
    metrics = {"loss": 0.5, "accuracy": 0.9, "roc_auc": 0.95}
    assert choose_best(None, metrics, 1)
    previous = {"epoch": 1, "development": metrics}
    assert not choose_best(previous, metrics | {"accuracy": 1.0}, 2)
    assert not choose_best(previous, metrics | {"roc_auc": 1.0}, 2)
    assert choose_best(previous, metrics | {"loss": 0.49, "accuracy": 0.1}, 2)
    with pytest.raises(ValueError):
        choose_best(previous, metrics | {"loss": float("nan")}, 2)


def test_reduced_development_is_available_only_for_short_qualification():
    assert (
        development_rows(
            SimpleNamespace(development_rows=32, steps_per_epoch=16, epochs=2)
        )
        == 32
    )
    assert (
        development_rows(
            SimpleNamespace(development_rows=50_000, steps_per_epoch=0, epochs=10)
        )
        == 50_000
    )
    for steps, epochs in ((0, 2), (5000, 2), (16, 10)):
        with pytest.raises(ValueError, match="short qualification"):
            development_rows(
                SimpleNamespace(
                    development_rows=32, steps_per_epoch=steps, epochs=epochs
                )
            )


def test_validation_partitions_are_disjoint_and_confirmation_cannot_change_dev(
    tmp_path,
):
    rows = DEV_ROWS + CONFIRMATION_ROWS
    x = np.zeros((rows, 28), dtype=np.float32)
    y = np.arange(rows, dtype=np.int64) % 2
    x[DEV_ROWS:] = 99
    path = tmp_path / "screen.npz"
    np.savez_compressed(path, x_val=x, y_val=y)
    dx, dy = load_partition(path, "development")
    cx, cy = load_partition(path, "confirmation")
    assert not dx.any() and bool((cx == 99).all())
    assert len(dx) == len(cx) == 50_000
    np.testing.assert_array_equal(dy.numpy(), y[:DEV_ROWS])
    np.testing.assert_array_equal(cy.numpy(), y[DEV_ROWS:])
    x[DEV_ROWS:] = -7
    y[DEV_ROWS:] = 1 - y[DEV_ROWS:]
    second = tmp_path / "second.npz"
    np.savez_compressed(second, x_val=x, y_val=y)
    ax, ay = load_partition(second, "development")
    assert torch.equal(dx, ax) and torch.equal(dy, ay)
    assert (
        validation_slice("development").stop == validation_slice("confirmation").start
    )
    with pytest.raises(ValueError):
        validation_slice("test")


def test_checkpoint_identity_prevents_cross_seed_or_data_mislabelling():
    args = SimpleNamespace(kind="cst", width=1024, divisor=1000, seed=29)
    configuration = {
        "cohort": HIGGS_CONFIRMATION_COHORT,
        **vars(args),
        "dataset_sha256": "a" * 64,
    }
    validate_checkpoint(configuration, args, "a" * 64)
    for key, value in (
        ("seed", 43),
        ("width", 2048),
        ("divisor", 500),
        ("kind", "dense"),
    ):
        with pytest.raises(ValueError, match="identity"):
            validate_checkpoint(configuration | {key: value}, args, "a" * 64)
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(configuration, args, "b" * 64)


def test_evaluation_preserves_example_order_and_module_mode():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    x = torch.tensor([[2.0, 1.0], [1.0, 2.0], [4.0, 1.0], [1.0, 4.0]])
    y = torch.tensor([0, 1, 0, 1])
    metrics, logits, labels = evaluate(model, x, y)
    np.testing.assert_array_equal(logits, x.numpy())
    np.testing.assert_array_equal(labels, y.numpy())
    assert metrics["accuracy"] == metrics["roc_auc"] == 1.0
    assert model.training


def test_native_confirmation_predicate_is_separate_from_existing_cohorts():
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = """
#include "limits.cuh"
using namespace radius_cell_mass;
static_assert(higgs_confirmation_owner_window_cohort(1024,1048));
static_assert(higgs_confirmation_owner_window_cohort(1024,2097));
static_assert(higgs_confirmation_owner_window_cohort(1024,5242));
static_assert(higgs_confirmation_owner_window_cohort(1024,10485));
static_assert(higgs_confirmation_owner_window_cohort(2048,4194));
static_assert(higgs_confirmation_owner_window_cohort(2048,8388));
static_assert(higgs_confirmation_owner_window_cohort(2048,20971));
static_assert(higgs_confirmation_owner_window_cohort(2048,41943));
static_assert(!higgs_confirmation_owner_window_cohort(1024,52428));
static_assert(!higgs_confirmation_owner_window_cohort(4096,16777));
static_assert(!higgs_confirmation_owner_window_cohort(1024,1049));
static_assert(!higgs_owner_window_cohort(1024,1048));
static_assert(owner_window_cohort_atoms(1024)==52428);
"""
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I",
            str(Path(executor.__file__).parent),
            "-",
        ],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", HIGGS_CONFIRMATION_WIDTHS)
@pytest.mark.parametrize("divisor", HIGGS_CONFIRMATION_DIVISORS)
def test_confirmation_graph_tracks_inputs_atoms_and_all_gradients(width, divisor):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    torch.manual_seed(571)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        middle = make_model(width, divisor, 29, "cst")[2].cuda()
        layout = hierarchy_from_operator(middle.declaration())
        configs = layout.input.native_config(), layout.output.native_config()
        extension = executor._extension()
        entry = extension.research_higgs_confirmation_owner_window_fp32_geometry_sort32_flat_c
        ordinary = extension.research_owner_window_fp32_geometry_sort32_flat_c
        x = torch.randn(4, width, device="cuda", requires_grad=True)
        p, dy = middle.atoms.p, torch.randn_like(x)

        def call(function, permission):
            value = function(x, p, *configs, permission)
            return value, *torch.autograd.grad(value, (x, p), dy)

        for _ in range(3):
            call(entry, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = call(entry, True)
        with torch.no_grad():
            x.add_(0.125)
            p[:, 2:].add_(0.375)
        expected = call(ordinary, False)
        graph.replay()
        stream.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        with pytest.raises(RuntimeError, match="fixed power-of-two/atom-density/B4"):
            entry(x, p[:-1].contiguous(), *configs, True)
