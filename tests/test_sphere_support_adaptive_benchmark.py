"""Sharp and ordinary adaptive support cases retain the same Sphere contract."""

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
CATALOG = LINEAR / "plans-sphere-support-adaptive.json"
CASES = [(1024, "1_25", 1.25), (1024, "3", 3), (2048, "1_25", 1.25), (2048, "3", 3)]
CONTROL = "recompute-g4-merged"
CANDIDATE = "tiny16-adaptive"


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_adaptive_cases_preserve_task_controls_and_declared_width(size, tag, sigma):
    run = load_run(
        LINEAR / f"cases/sphere-support-adaptive-{size}-sigma{tag}.json", CATALOG
    )
    previous = load_run(
        LINEAR / f"cases/sphere-site-gather-{size}-sigma3.json",
        LINEAR / "plans-sphere-site-gather.json",
    )
    assert [entry.id for entry in run.plans] == ["compact-weight", CONTROL, CANDIDATE]
    assert run.baseline == "compact-weight" and run.dense
    for name in ("compact-weight", CONTROL):
        assert run.entry(name).plan == previous.entry(name).plan
    fields, original = asdict(run.case), asdict(previous.case)
    fields.pop("id")
    original.pop("id")
    fields.pop("profile")
    original.pop("profile")
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
        plan.algorithm_id == "research_cuda_sphere_polar_support_adaptive"
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
    }
    assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_adaptive_preparation_preserves_complete_declared_cohort(
    tmp_path, size, tag, sigma
):
    case = LINEAR / f"cases/sphere-support-adaptive-{size}-sigma{tag}.json"
    output = tmp_path / "prepared"
    original = load_run(case, CATALOG)
    dev.prepare(CATALOG, case, [CONTROL, CANDIDATE], output)
    report = dev.check(output / "plans.json", [output / "case.json"])
    prepared = load_run(output / "case.json", output / "plans.json")
    assert report["status"] == "PASS" and prepared.case == original.case
    assert prepared.case.profile == f"rho{sigma:g}"
    assert prepared.baseline == "compact-weight" and prepared.dense
    assert [entry.id for entry in prepared.plans] == [
        entry.id for entry in original.plans
    ]
    assert prepared.input_hashes == report["cases"][0]["input_hashes"]


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_adaptive_declarations_remain_lazy(size, tag, sigma):
    case = LINEAR / f"cases/sphere-support-adaptive-{size}-sigma{tag}.json"
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(case)!r}, '--plans', {str(CATALOG)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(name.endswith(('.adaptive_executor', '.adaptive_kernels', '.direct_executor',
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
        LINEAR / "cases/sphere-support-adaptive-1024-sigma1_25.json", CATALOG
    )
    with pytest.raises(ValueError):
        replace(run.case, profile=profile)


@pytest.mark.parametrize("size,tag,sigma", CASES)
def test_sharp_wrapper_calls_unchanged_worker_with_explicit_plan(
    tmp_path, monkeypatch, size, tag, sigma
):
    from benchmarks.cuda.linear import support_adaptive_comparison as wrapper

    run = load_run(
        LINEAR / f"cases/sphere-support-adaptive-{size}-sigma{tag}.json", CATALOG
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
                "plan": run.entry("compact-weight").plan,
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


def test_route_diagnostics_are_candidate_only_and_after_primary(tmp_path, monkeypatch):
    from benchmarks.cuda.linear import support_adaptive_comparison as wrapper

    run = load_run(
        LINEAR / "cases/sphere-support-adaptive-1024-sigma1_25.json", CATALOG
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
    plan_file.write_text(wrapper.REGISTRY.dumps_plan(run.entry("compact-weight").plan))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        wrapper.main()
