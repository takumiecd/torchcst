"""CPU qualification of the existing runner's public radius fixture and oracle."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from benchmarks.cuda.linear.radius_cell_mass import (
    _chunked_loss_backward,
    chunked_reference,
    distribution,
    fixture,
    oracle,
    stratified_subset,
)

ROOT = Path(__file__).resolve().parents[1]


def case():
    return load_run(
        ROOT / "benchmarks/cuda/linear/cases/radius-cell-mass-128-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )


def test_radius_catalog_and_public_initializer():
    run = case()
    model, x, target, _ = fixture(run.case)
    assert model.atoms.p.shape == (819, 4)
    assert x.shape == target.shape == (4, 128)
    assert not run.case.optimizer.capturable
    report = distribution(model)
    assert len(report) == 2
    assert len(report[0]["radii"]) == 819
    assert sum(report[0]["level_counts"]) == 819
    assert sum(report[0]["bank_counts"]) == 819


def test_radius_quadrature_oracle_all_gradients():
    small = replace(case().case, atoms=8)
    model, x, _, dy = fixture(small)
    model = model.double()
    x = x.double().requires_grad_()
    actual = model(x)
    ga = torch.autograd.grad(actual, (x, model.atoms.p), dy.double())
    expected = oracle(model, x)
    gb = torch.autograd.grad(expected, (x, model.atoms.p), dy.double())
    for a, b in zip((actual, *ga), (expected, *gb)):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-8)


def test_radius_protocol_reconstructs_revision7_contract():
    from benchmarks.cuda.linear.protocol import measurement_operator
    from torchcst.kernels.parameterizations import PolarAmpRadiusSpec

    run = case()
    expected, *_ = fixture(run.case)
    actual = measurement_operator(asdict(run.case))
    assert actual == expected.execution_declaration()
    assert actual.kernel.revision == 7
    assert isinstance(actual.kernel.parameterization, PolarAmpRadiusSpec)


def test_radius_submission_adapter_rejects_research_only():
    from benchmarks.database.adapters.linear import project

    artifact = {
        "schema_version": 1,
        "status": "PASS",
        "records": [],
        "run": case().snapshot(),
    }
    with pytest.raises(ValueError, match="polar_radius_cell_mass is research-only"):
        project(artifact)


@pytest.mark.parametrize("gpu", [False, True])
def test_training_fixture_decays_amplitude_instead_of_coordinates(gpu):
    from benchmarks.cuda.linear.radius_cell_mass import optimizer
    from torchcst import CSTAdamW, CSTOptimizer

    run = case()
    model, *_ = fixture(run.case)
    opt = optimizer(model, run.case, gpu=gpu)
    assert type(opt) is (CSTAdamW if gpu else CSTOptimizer)
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert (
        opt.param_groups[0]["amplitude_weight_decay"] == run.case.optimizer.weight_decay
    )


def test_large_radius_case_explicit_general_plan_and_schema():
    from benchmarks.cuda.linear.protocol import measurement_operator

    run = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-general-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert (run.case.size, run.case.atoms, run.case.rows) == (1024, 52428, 4)
    assert run.baseline == "radius-hybrid-serial"
    assert len(run.plans) == 1
    recipe = run.plans[0].plan.recipe
    assert recipe.moment_method == "merge" and recipe.save_response
    assert recipe.input_direct_levels == recipe.output_direct_levels == 2
    assert recipe.schedule == "sequential"
    assert not recipe.native_autograd and not recipe.native_entry
    assert decode_snapshot(run.snapshot()) == run
    assert measurement_operator(asdict(run.case)).kernel.revision == 7
    with pytest.raises(ValueError, match="size"):
        replace(run.case, size=2048)
    with pytest.raises(ValueError, match="65536"):
        replace(run.case, atoms=65537)
    # The measured N128 native fixture and its defaults remain separate.
    native = load_run(
        ROOT / "benchmarks/cuda/linear/cases/radius-cell-mass-128-native-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert (native.case.size, native.case.atoms, native.case.rows) == (128, 819, 4)
    assert native.plans[0].plan.recipe.native_entry


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_atoms", [1, 4, 64])
def test_chunked_public_reference_matches_complete_current_model(dtype, chunk_atoms):
    model, x, _, dy = fixture(replace(case().case, atoms=11, rows=2))
    model = model.to(dtype=dtype)
    x, dy = x.to(dtype=dtype).requires_grad_(), dy.to(dtype=dtype)
    # Exercise live P and fixed configuration, rather than reconstructing seed41.
    with torch.no_grad():
        model.atoms.p[:, 2:].add_(17.25)
        model.kernel.scalar("lower_kappa").mul_(1.2)
    expected = model(x)
    reference = torch.autograd.grad(expected, (x, model.atoms.p), dy)
    actual = chunked_reference(model, x, dy, chunk_atoms=chunk_atoms)
    for a, b in zip(actual, (expected, *reference)):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
        assert not a.requires_grad
    torch.testing.assert_close(
        chunked_reference(model, x, chunk_atoms=chunk_atoms),
        expected,
        atol=3e-6,
        rtol=3e-6,
    )
    assert model.atoms.p.grad is None and x.grad is None


def test_chunked_reference_has_bounded_graphs_and_releases_temporary_sites(monkeypatch):
    import gc
    import weakref

    from benchmarks.cuda.linear import radius_cell_mass as benchmark

    model, x, _, dy = fixture(replace(case().case, atoms=11, rows=2))
    original = benchmark._reference_model
    parameters, outputs, sites = [], [], []

    def capture(*args):
        part = original(*args)
        assert part.atoms.count <= 3
        parameters.append(weakref.ref(part.atoms.p))
        sites.append(weakref.ref(part))

        def before(module, inputs):
            assert all(ref() is None for ref in outputs)

        part.register_forward_pre_hook(before)
        part.register_forward_hook(
            lambda module, inputs, y: outputs.append(weakref.ref(y))
        )
        return part

    monkeypatch.setattr(benchmark, "_reference_model", capture)
    chunked_reference(model, x, dy, chunk_atoms=3)
    assert len(sites) == 2 and len(outputs) == 4
    gc.collect()
    assert all(ref() is None for ref in (*sites, *parameters, *outputs))
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            chunked_reference(model, x, chunk_atoms=invalid)


def test_chunked_loss_backward_matches_full_model_and_cpu_optimizer():
    import copy

    from benchmarks.cuda.linear.radius_cell_mass import optimizer

    settings = replace(case().case, atoms=11, rows=2)
    model, x, target, _ = fixture(settings)
    other = copy.deepcopy(model)
    x = x.requires_grad_()
    tx = x.detach().clone().requires_grad_()
    (model(x) - target).square().mean().backward()
    _chunked_loss_backward(other, tx, target, chunk_atoms=4)
    for actual, expected in (
        (other.atoms.p.grad, model.atoms.p.grad),
        (tx.grad, x.grad),
    ):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    oa, ob = (
        optimizer(model, settings, gpu=False),
        optimizer(other, settings, gpu=False),
    )
    oa.step()
    ob.step()
    torch.testing.assert_close(other.atoms.p, model.atoms.p, atol=3e-6, rtol=3e-6)
    for name in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            ob.state[other.atoms.p][name],
            oa.state[model.atoms.p][name],
            atol=3e-6,
            rtol=3e-6,
        )


def test_quadrature_subset_uses_current_radius_strata():
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    model, *_ = fixture(replace(case().case, atoms=19))
    subset, indices = stratified_subset(model)
    _, ri, ro = decoded_values(model.kernel, model.atoms.p.detach())
    order = torch.maximum(ri, ro).argsort(stable=True)
    assert len(indices) == 8 and len(set(indices)) == 8
    assert indices[0] == int(order[0]) and indices[-1] == int(order[-1])
    torch.testing.assert_close(subset.atoms.p, model.atoms.p[indices])
    assert subset.execution_declaration() == model.execution_declaration()


def test_n1024_chunked_fourier_vjp_matches_unchunked_small_atoms():
    model, x, _, dy = fixture(replace(case().case, size=1024, atoms=13))
    model = model.double()
    x = x.double().requires_grad_()
    expected = model(x)
    gradients = torch.autograd.grad(expected, (x, model.atoms.p), dy.double())
    actual = chunked_reference(model, x, dy.double(), chunk_atoms=4)
    for a, b in zip(actual, (expected, *gradients)):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-11)


@pytest.mark.parametrize("direction", [float("inf"), -float("inf")])
@pytest.mark.parametrize("coordinate", [711.198486328125, 1024.0])
def test_large_center_accepts_one_adjacent_fp32_value_not_two(direction, coordinate):
    from benchmarks.cuda.linear.radius_cell_mass import parameter_quantization_gate

    expected = torch.tensor([[0.25, 1.0, coordinate, 0.0]], dtype=torch.float32)
    actual = expected.clone()
    toward = torch.tensor(direction)
    actual[0, 2] = torch.nextafter(expected[0, 2], toward)
    report = parameter_quantization_gate(actual, expected, large=True)
    assert report["status"] == "PASS"
    assert report["worst_abs_delta_in_directed_ulp"] == 1
    assert (
        parameter_quantization_gate(actual, expected, large=False)["status"] == "FAIL"
    )
    actual[0, 2] = torch.nextafter(actual[0, 2], toward)
    assert parameter_quantization_gate(actual, expected, large=True)["status"] == "FAIL"


def test_parameter_gate_keeps_absolute_floor_and_amplitude_columns():
    from benchmarks.cuda.linear.radius_cell_mass import parameter_quantization_gate

    expected = torch.zeros(2, 4)
    actual = expected.clone()
    actual[0, 2] = 1.9e-5
    assert parameter_quantization_gate(actual, expected, large=True)["status"] == "PASS"
    actual[0, 2] = 2.1e-5
    assert parameter_quantization_gate(actual, expected, large=True)["status"] == "FAIL"
    # A passing large-center error cannot hide a smaller failing amplitude error.
    expected[0, 2] = 711.198486328125
    actual = expected.clone()
    actual[0, 2] = torch.nextafter(expected[0, 2], torch.tensor(float("inf")))
    actual[1, 0] = 3e-5
    report = parameter_quantization_gate(actual, expected, large=True)
    assert report["status"] == "FAIL" and report["failed_elements"] == 1
    assert report["worst_abs_index"] == [0, 2]
    for column in (0, 1):
        expected = torch.zeros(1, 4)
        expected[0, column] = 711.198486328125
        actual = expected.clone()
        actual[0, column] = torch.nextafter(
            expected[0, column], torch.tensor(float("inf"))
        )
        assert (
            parameter_quantization_gate(actual, expected, large=True)["status"]
            == "FAIL"
        )


def test_gpu_compatibility_factory_uses_same_explicit_polar_plan_without_gpu_work():
    from benchmarks.cuda.linear.radius_cell_mass import optimizer
    from torchcst import CSTAdamW, CSTOptimizer
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )

    settings = case().case
    a, *_ = fixture(settings)
    b, *_ = fixture(settings)
    actual = optimizer(a, settings)
    compatibility = optimizer(b, settings, compatibility=True)
    assert type(actual) is CSTAdamW and type(compatibility) is CSTOptimizer
    assert type(compatibility.base_optimizer) is torch.optim.AdamW
    assert compatibility.param_groups[0]["fused"] is True
    assert compatibility._update_dispatcher.selector._plan == RADIUS_FLAT_TORUS_FUSED
    assert actual._update_dispatcher.selector._plan == RADIUS_FLAT_TORUS_FUSED
    for name in ("lr", "weight_decay", "amplitude_weight_decay", "fused", "capturable"):
        assert actual.param_groups[0][name] == compatibility.param_groups[0][name]


def test_optimizer_bitgate_copies_inputs_and_rejects_one_bit_change():
    from benchmarks.cuda.linear.radius_cell_mass import (
        _copy_optimizer_inputs,
        optimizer,
        optimizer_bitgate,
    )

    settings = replace(case().case, atoms=8)
    a, *_ = fixture(settings)
    b, *_ = fixture(settings)
    oa, ob = optimizer(a, settings, gpu=False), optimizer(b, settings, gpu=False)
    pa, pb = a.atoms.p, b.atoms.p
    pa.grad = torch.randn_like(pa)
    # Seed real non-empty Adam state, then compare the exact snapshot copy.
    oa.step()
    _copy_optimizer_inputs(oa, pa, ob, pb)
    assert all(optimizer_bitgate(oa, pa, ob, pb).values())
    assert ob.state[pb]["exp_avg"].data_ptr() != oa.state[pa]["exp_avg"].data_ptr()
    with torch.no_grad():
        ob.state[pb]["exp_avg"][0, 0] = torch.nextafter(
            ob.state[pb]["exp_avg"][0, 0], torch.tensor(float("inf"))
        )
    with pytest.raises(AssertionError, match="GPU_optimizer_wrapper_bitwise"):
        optimizer_bitgate(oa, pa, ob, pb)
    _copy_optimizer_inputs(oa, pa, ob, pb)
    with torch.no_grad():
        pa[0, 0] = 0.0
        pb[0, 0] = -0.0
    assert torch.equal(pa, pb)
    with pytest.raises(AssertionError, match="GPU_optimizer_wrapper_bitwise"):
        optimizer_bitgate(oa, pa, ob, pb)


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("side", ["actual", "expected", "both"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_parameter_quantization_gate_rejects_nonfinite_values(large, side, value):
    from benchmarks.cuda.linear.radius_cell_mass import parameter_quantization_gate

    expected = torch.zeros(1, 4)
    actual = expected.clone()
    if side in ("actual", "both"):
        actual[0, 2] = value
    if side in ("expected", "both"):
        expected[0, 2] = value
    report = parameter_quantization_gate(actual, expected, large=large)
    assert report["status"] == "FAIL"
    assert report["failed_elements"] == 1
