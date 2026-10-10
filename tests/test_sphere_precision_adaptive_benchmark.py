"""Precision-corrected cases retain the Sphere model and explicitly change the baseline."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from tools.kernel_dev import __main__ as dev

LINEAR = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
CATALOG = LINEAR / "plans-sphere-precision-adaptive.json"
CASES = [(1024, "1_25", 1.25), (1024, "3", 3), (2048, "1_25", 1.25), (2048, "3", 3)]
CONTROL = "precision-corrected-recompute"
CANDIDATE = "precision-corrected-adaptive"


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_precision_adaptive_cases_preserve_task_controls_and_declared_width(
    size, tag, sigma
):
    run = load_run(
        LINEAR / f"cases/sphere-precision-adaptive-{size}-sigma{tag}.json", CATALOG
    )
    previous = load_run(
        LINEAR / f"cases/sphere-support-adaptive-{size}-sigma{tag}.json",
        LINEAR / "plans-sphere-support-adaptive.json",
    )
    assert [entry.id for entry in run.plans] == [
        "precision-corrected-compact",
        CONTROL,
        CANDIDATE,
    ]
    assert run.baseline == "precision-corrected-compact" and run.dense
    for name, old_name in zip(
        ["precision-corrected-compact", CONTROL, CANDIDATE],
        ["compact-weight", "recompute-g4-merged", "tiny16-adaptive"],
        strict=True,
    ):
        new_plan, old_plan = run.entry(name).plan, previous.entry(old_name).plan
        assert new_plan.algorithm_id != old_plan.algorithm_id
        recipe = asdict(new_plan.recipe)
        assert type(recipe.pop("precision_norm_threshold")) is float
        assert new_plan.recipe.precision_norm_threshold == 0.001
        assert recipe == asdict(old_plan.recipe)
    fields, original = asdict(run.case), asdict(previous.case)
    fields.pop("id")
    original.pop("id")
    assert fields == original
    assert run.case.profile == f"rho{sigma:g}"
    assert (run.case.size, run.case.rows, run.case.atoms, run.case.seed) == (
        size,
        32,
        int(0.05 * size * size),
        41,
    )
    plan = run.entry(CANDIDATE).plan
    assert (
        plan.algorithm_id == "research_cuda_sphere_polar_precision_adaptive"
        and plan.algorithm_revision == "v1"
    )
    assert asdict(plan.recipe) == {
        "support_capacity": 64,
        "support_tile": 16,
        "atom_group": 4,
        "index_bits": 16,
        "merge_output_vjp": True,
        "bins_per_axis": 8,
        "query_rows": 4,
        "candidate_capacity": 128,
        "tiny_capacity": 16,
        "precision_norm_threshold": 0.001,
    }
    assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_precision_adaptive_preparation_preserves_complete_declared_cohort(
    tmp_path, size, tag, sigma
):
    case = LINEAR / f"cases/sphere-precision-adaptive-{size}-sigma{tag}.json"
    output = tmp_path / "prepared"
    original = load_run(case, CATALOG)
    dev.prepare(CATALOG, case, [CONTROL, CANDIDATE], output)
    report = dev.check(output / "plans.json", [output / "case.json"])
    prepared = load_run(output / "case.json", output / "plans.json")
    assert report["status"] == "PASS" and prepared.case == original.case
    assert prepared.case.profile == f"rho{sigma:g}"
    assert prepared.baseline == "precision-corrected-compact" and prepared.dense
    assert [entry.id for entry in prepared.plans] == [
        entry.id for entry in original.plans
    ]
    assert prepared.input_hashes == report["cases"][0]["input_hashes"]


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_precision_adaptive_declarations_remain_lazy(size, tag, sigma):
    case = LINEAR / f"cases/sphere-precision-adaptive-{size}-sigma{tag}.json"
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(case)!r}, '--plans', {str(CATALOG)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(name.endswith(('.precision_executor', '.precision_kernels', '.precision_prepare', '.adaptive_executor', '.adaptive_kernels', '.direct_executor',
                             '.direct_kernels', '.recompute_prepare_kernels')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )


@pytest.mark.parametrize("profile", ["rho1", "rho1_25", "rho1.5", "rho2"])
def test_extra_sharp_declaration_does_not_accept_arbitrary_profiles(profile):
    run = load_run(
        LINEAR / "cases/sphere-precision-adaptive-1024-sigma1_25.json", CATALOG
    )
    with pytest.raises(ValueError):
        replace(run.case, profile=profile)


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_sharp_wrapper_calls_unchanged_worker_with_explicit_plan(
    tmp_path, monkeypatch, size, tag, sigma
):
    from benchmarks.cuda.linear import precision_adaptive_comparison as wrapper

    run = load_run(
        LINEAR / f"cases/sphere-precision-adaptive-{size}-sigma{tag}.json", CATALOG
    )
    plan_file, output = tmp_path / "plan.json", tmp_path / "result.json"
    plan_file.write_text(json.dumps(run.snapshot()["plans"][0]["plan"]))
    calls = []

    def worker(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "PASS", "size": size, "sigma_initial": sigma}

    monkeypatch.setattr(wrapper, "worker", worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--size",
            str(size),
            "--sigma",
            str(sigma),
            "--mode",
            "research_graph",
            "--linear-plan",
            str(plan_file),
            "--output",
            str(output),
            "--verify-only",
        ],
    )
    wrapper.main()
    assert calls == [
        (
            ("sphere", size, sigma, "research_graph"),
            {
                "plan": run.entry("precision-corrected-compact").plan,
                "verify_only": True,
                "phases": False,
            },
        )
    ]
    result = json.loads(output.read_text())
    assert result["study_condition"] == (
        "sharp diagnostic" if sigma == 1.25 else "ordinary sigma3 main"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--size",
            str(size),
            "--sigma",
            str(sigma),
            "--mode",
            "research_graph",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit):
        wrapper.main()


def test_precision_wrapper_rejects_legacy_plan_before_running_worker(
    tmp_path, monkeypatch
):
    from benchmarks.cuda.linear import precision_adaptive_comparison as wrapper

    old = load_run(
        LINEAR / "cases/sphere-support-adaptive-1024-sigma1_25.json",
        LINEAR / "plans-sphere-support-adaptive.json",
    )
    plan_file = tmp_path / "legacy.json"
    plan_file.write_text(wrapper.REGISTRY.dumps_plan(old.entry("compact-weight").plan))

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy route must not execute as corrected precision")

    monkeypatch.setattr(wrapper, "worker", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precision-wrapper",
            "--size",
            "1024",
            "--sigma",
            "1.25",
            "--mode",
            "research_graph",
            "--linear-plan",
            str(plan_file),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(SystemExit):
        wrapper.main()

def test_route_diagnostics_are_candidate_only_and_after_primary(tmp_path, monkeypatch):
    from benchmarks.cuda.linear import precision_adaptive_comparison as wrapper

    run = load_run(
        LINEAR / "cases/sphere-precision-adaptive-1024-sigma1_25.json", CATALOG
    )
    plan_file, output = tmp_path / "plan.json", tmp_path / "result.json"
    plan_file.write_text(wrapper.REGISTRY.dumps_plan(run.entry(CANDIDATE).plan))
    calls = []
    monkeypatch.setattr(
        wrapper,
        "worker",
        lambda *a, **kw: calls.append("primary") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        wrapper,
        "route_diagnostics",
        lambda *a: calls.append("diagnostics") or {"scope": "independent"},
    )
    args = [
        "wrapper",
        "--size",
        "1024",
        "--sigma",
        "1.25",
        "--mode",
        "research_graph",
        "--linear-plan",
        str(plan_file),
        "--output",
        str(output),
        "--route-diagnostics",
    ]
    monkeypatch.setattr(sys, "argv", args)
    wrapper.main()
    assert calls == ["primary", "diagnostics"]
    assert json.loads(output.read_text())["route_diagnostics"] == {
        "scope": "independent"
    }
    monkeypatch.setattr(sys, "argv", [*args, "--verify-only"])
    with pytest.raises(SystemExit):
        wrapper.main()
    assert calls == ["primary", "diagnostics"]
    plan_file.write_text(wrapper.REGISTRY.dumps_plan(run.entry("precision-corrected-compact").plan))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        wrapper.main()
