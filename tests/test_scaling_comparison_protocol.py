"""Common inputs, exact oracles, frozen metadata and declared state trajectories."""

import copy
import hashlib
import struct
import subprocess

import pytest
import torch

from benchmarks.cuda.linear import scaling_comparison as protocol
from benchmarks.cuda.linear.manifest import REGISTRY


@pytest.mark.parametrize("sigma", [3, 8])
def test_geometry_initialization_does_not_change_shared_task(sigma):
    a, *ax = protocol.fixture("sphere", 1024, sigma, batch=2, atoms=7)
    b, *bx = protocol.fixture("torus", 1024, sigma, batch=2, atoms=7)
    assert a.atoms.p.shape == (7, 6)
    assert b.atoms.p.shape == (7, 5)
    for left, right in zip(ax, bx, strict=True):
        assert torch.equal(left, right)
    # Global RNG changes cannot change the declared task or dense fixture.
    torch.manual_seed(99)
    assert torch.equal(
        protocol.dense_fixture(32).weight, protocol.dense_fixture(32).weight
    )
    for actual, expected in zip(ax, protocol.shared_inputs(1024, batch=2), strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("geometry", ["sphere", "torus"])
@pytest.mark.parametrize("sigma", [3, 8])
def test_conditional_baseline_is_registered_and_roundtrips(geometry, sigma):
    plan = protocol.baseline_plan(geometry, sigma)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    if geometry == "sphere":
        expected = (
            "research_cuda_sphere_polar_fused_weight"
            if sigma == 3
            else "research_cuda_sphere_polar_blocked"
        )
        assert plan.algorithm_id == expected


def test_oracle_checks_entire_gradient_and_rejects_changed_operator(monkeypatch):
    model, x, target, dy = protocol.fixture("sphere", 32, 3, batch=2, atoms=7)
    model.selector = None
    report = protocol.oracle_check(model, x, dy, target, "sphere")
    assert report["status"] == "PASS"
    assert set(report["errors"]) == {"Y", "dX", "all_dP"}
    original = model.forward
    monkeypatch.setattr(model, "forward", lambda value: original(value) + 0.01)
    with pytest.raises(AssertionError):
        protocol.oracle_check(model, x, dy, target, "sphere")


def test_archive_without_git_records_hashes_and_separate_hint(monkeypatch):
    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(protocol.subprocess, "check_output", no_git)
    monkeypatch.setenv("CST_FROZEN_SOURCE_COMMIT", "a" * 40)
    metadata = protocol.source_metadata()
    assert metadata["source_commit"] is None
    assert metadata["source_commit_hint"] == "a" * 40
    assert (
        "src/torchcst/_backends/cuda/algorithms/linear/torus_profile_product/_shared/profiles.py"
        in metadata["source_hashes"]
    )
    assert "benchmarks/cuda/linear/sphere_baseline.py" in metadata["source_hashes"]
    assert (
        "benchmarks/cuda/linear/torus_profile_product.py" in metadata["source_hashes"]
    )


def test_zero_amplitude_support_is_not_reported_empty():
    model, *_ = protocol.fixture("sphere", 32, 3, batch=2, atoms=7)
    with torch.no_grad():
        model.atoms.p[:, 0] = 0
        model.atoms.p[:, 2:] = 0
    summary = protocol.support_summary(
        model, "sphere", protocol.baseline_plan("sphere", 3)
    )
    assert all(side["max_count"] > 0 for side in summary["sides"])
    assert all(side["empty_atoms"] == 0 for side in summary["sides"])
    assert len(protocol.fixture_metadata(model)["declaration_sha256"]) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("geometry", ["sphere", "torus"])
@pytest.mark.parametrize("sigma", [3, 8])
def test_same_cotangent_public_parameters_moments_and_exact_clock(geometry, sigma):
    proof = protocol.state_trajectory(geometry, sigma)
    assert proof["status"] == "PASS"
    assert proof["same_cotangent"] and proof["exact_step_counter"]


@pytest.mark.parametrize("geometry", ["sphere", "torus"])
def test_actual_geometry_tensors_distinguish_same_declaration(geometry, monkeypatch):
    model, *_ = protocol.fixture(geometry, 1024, 3, atoms=7)
    declaration = model.execution_declaration()
    # A previously frozen declaration (or metadata-only declaration in a
    # future adapter) cannot substitute for hashing actual runtime tensors.
    monkeypatch.setattr(model, "execution_declaration", lambda: declaration)
    before = protocol.fixture_metadata(model)
    coordinate_name = (
        "input_chart.coordinates" if geometry == "sphere" else "chart.axes.0.start"
    )
    with torch.no_grad():
        dict(model.named_buffers())[coordinate_name].flatten()[0] += 0.125
    after = protocol.fixture_metadata(model)
    assert before["declaration_sha256"] == after["declaration_sha256"]
    assert before["tensor_hashes"] != after["tensor_hashes"]
    key = "buffer:" + coordinate_name
    assert before["tensor_hashes"][key] != after["tensor_hashes"][key]
    expected = {"buffer:" + name for name, _ in model.named_buffers()}
    assert set(before["tensor_hashes"]) == expected
    assert list(before["tensor_hashes"]) == sorted(before["tensor_hashes"])
    # Atom parameters use a separate initialization hash; every other parameter
    # (e.g. a declared live scalar) belongs to the actual fixture fingerprint.
    model.register_parameter("fixture_probe", torch.nn.Parameter(torch.tensor(2.0)))
    assert (
        "parameter:fixture_probe" in protocol.fixture_metadata(model)["tensor_hashes"]
    )


def test_oracle_rejects_nan_operator(monkeypatch):
    model, x, target, dy = protocol.fixture("sphere", 32, 3, batch=2, atoms=7)
    original = model.forward
    monkeypatch.setattr(model, "forward", lambda value: original(value) * float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite oracle Y"):
        protocol.oracle_check(model, x, dy, target, "sphere")


@pytest.mark.parametrize("side", ["actual", "reference"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_state_error_rejects_nonfinite_parameters_or_moments(side, value):
    a, b = torch.ones(2), torch.ones(2)
    (a if side == "actual" else b)[0] = value
    with pytest.raises(FloatingPointError, match="nonfinite"):
        protocol.finite_error("exp_avg", a, b)


def test_finite_tensor_with_nonfinite_error_metric_is_rejected(monkeypatch):
    monkeypatch.setattr(protocol.sphere, "error", lambda *_: {"max_abs": float("nan")})
    with pytest.raises(FloatingPointError, match="error metrics"):
        protocol.finite_error("parameters", torch.ones(2), torch.ones(2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_measurement_cannot_be_reported_as_a_win(value):
    with pytest.raises(AssertionError, match="finite positive"):
        protocol.validate_measurement(
            {"median_ms": 1.0, "samples_ms": [value] * 21},
            {"allocated_bytes": 1, "reserved_bytes": 2},
        )
    with pytest.raises(AssertionError, match="peak"):
        protocol.validate_measurement(
            None, {"allocated_bytes": value, "reserved_bytes": 2}
        )


def test_tensor_hash_does_not_require_numpy_and_preserves_exact_bytes(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("NumPy is not available")

    monkeypatch.setattr(torch.Tensor, "numpy", unavailable)
    fixtures = [
        (torch.tensor(1.25, dtype=torch.float64), struct.pack("<d", 1.25)),
        (torch.tensor([9.0, 1.25, -2.5, 8.0])[1:3], struct.pack("<ff", 1.25, -2.5)),
        (torch.tensor([[1, -2]], dtype=torch.int16), struct.pack("<hh", 1, -2)),
        (torch.empty(0), b""),
    ]
    for tensor, data in fixtures:
        assert protocol.sphere.tensor_hash(tensor) == hashlib.sha256(data).hexdigest()
    model, *_ = protocol.fixture("sphere", 32, 3, atoms=7)
    metadata = protocol.fixture_metadata(model)
    key = "buffer:input_chart.coordinates"
    assert metadata["tensor_specs"][key] == {"dtype": "torch.float32", "shape": [32, 3]}
    assert list(metadata["tensor_specs"]) == sorted(metadata["tensor_specs"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
def test_graph_capture_executes_exactly_twenty_four_updates():
    clock = torch.zeros((), device="cuda", dtype=torch.int64)

    def update():
        clock.add_(1)
        return clock

    graph, result = protocol.capture(update)
    assert int(clock) == 3  # Two eager warmups and the explicitly run capture.
    for _ in range(21):
        graph.replay()
    torch.cuda.synchronize()
    assert int(result) == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
def test_diagnostic_capture_and_five_replays_execute_six_updates():
    clock = torch.zeros((), device="cuda", dtype=torch.int64)

    class Diagnostic:
        update = None

        def __call__(self, events, updates):
            events[0].record()
            clock.add_(1)
            for event in events[1:]:
                event.record()

    protocol.phase_diagnostics(Diagnostic())
    torch.cuda.synchronize()
    assert int(clock) == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("geometry", ["sphere", "torus"])
def test_copied_cst_phase_refreshes_metadata_without_extra_update(geometry):
    model, x, target, _ = protocol.fixture(
        geometry, 64 if geometry == "sphere" else 1024, 3, batch=4, atoms=17
    )
    model = model.cuda()
    x, target = x.cuda().requires_grad_(), target.cuda()
    protocol.bind_plan(model, protocol.baseline_plan(geometry, 3))
    step = protocol.TrainingStep(model, x, target, geometry, "research_graph")
    graph, _ = protocol.capture(step)
    for _ in range(21):
        graph.replay()
    torch.cuda.synchronize()
    assert int(step.opt.state[model.atoms.p]["step"]) == 24
    primary_parameters = model.atoms.p.detach().clone()
    diagnostic_model = copy.deepcopy(model)
    assert "_execution_declaration" not in diagnostic_model.__dict__
    diagnostic = protocol.TrainingStep(
        diagnostic_model,
        x.detach().clone().requires_grad_(),
        target.clone(),
        geometry,
        "research_graph",
    )
    diagnostic.opt.load_state_dict(copy.deepcopy(step.opt.state_dict()))
    report = protocol.phase_diagnostics(diagnostic)
    assert report["samples"] == 5
    assert int(diagnostic.opt.state[diagnostic_model.atoms.p]["step"]) == 30
    assert int(step.opt.state[model.atoms.p]["step"]) == 24
    assert torch.equal(primary_parameters, model.atoms.p)
