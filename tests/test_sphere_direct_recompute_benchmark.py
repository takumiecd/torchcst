"""Recomputed-profile comparisons preserve Sphere fixtures and stored controls."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from tools.kernel_dev import __main__ as dev

LINEAR = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
CATALOG = LINEAR / "plans-sphere-direct-recompute.json"
STORED = (
    "direct-g1-separate",
    "direct-g4-separate",
    "direct-g1-merged",
    "direct-g4-merged",
)
RECOMPUTED = tuple(name.replace("direct-", "recompute-", 1) for name in STORED)


@pytest.mark.parametrize("size,atoms", [(1024, 52428), (2048, 209715)])
def test_recomputed_comparison_preserves_fixtures_and_all_controls(size, atoms):
    run = load_run(
        LINEAR / f"cases/sphere-direct-recompute-{size}-sigma3.json", CATALOG
    )
    previous = load_run(
        LINEAR / f"cases/sphere-direct-{size}-sigma3.json",
        LINEAR / "plans-sphere-direct.json",
    )
    assert [entry.id for entry in run.plans] == [
        "compact-weight",
        "old-direct",
        *STORED,
        *RECOMPUTED,
    ]
    assert len(run.plans) == 10 and run.baseline == "compact-weight" and run.dense
    for entry in previous.plans:
        assert run.entry(entry.id).plan == entry.plan
    fields, original = asdict(run.case), asdict(previous.case)
    fields.pop("id")
    original.pop("id")
    assert fields == original
    assert (run.case.size, run.case.atoms, run.case.profile) == (size, atoms, "rho3")
    for stored, recomputed in zip(STORED, RECOMPUTED):
        plan = run.entry(recomputed).plan
        assert plan.algorithm_id == "research_cuda_sphere_polar_direct_recompute"
        assert plan.algorithm_revision == "v1"
        assert asdict(plan.recipe) == asdict(run.entry(stored).plan.recipe)
        assert asdict(plan.recipe) == {
            "support_capacity": 64,
            "support_tile": 16,
            "atom_group": 4 if "g4" in recomputed else 1,
            "index_bits": 16,
            "merge_output_vjp": recomputed.endswith("merged"),
        }
    assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run


@pytest.mark.parametrize("size", [1024, 2048])
def test_recomputed_contributor_preparation_preserves_all_comparisons(tmp_path, size):
    case = LINEAR / f"cases/sphere-direct-recompute-{size}-sigma3.json"
    output = tmp_path / "prepared"
    original = load_run(case, CATALOG)
    # Explicitly request the mechanism control as well as all eight ablations.
    dev.prepare(CATALOG, case, ["old-direct", *STORED, *RECOMPUTED], output)
    report = dev.check(output / "plans.json", [output / "case.json"])
    prepared = load_run(output / "case.json", output / "plans.json")
    assert report["status"] == "PASS"
    assert prepared.case == original.case
    assert prepared.baseline == "compact-weight" and prepared.dense
    assert [entry.id for entry in prepared.plans] == [
        entry.id for entry in original.plans
    ]
    for entry in prepared.plans:
        assert entry.plan == original.entry(entry.id).plan


@pytest.mark.parametrize("size", [1024, 2048])
def test_recomputed_declarations_do_not_import_gpu_execution(size):
    case = LINEAR / f"cases/sphere-direct-recompute-{size}-sigma3.json"
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(case)!r}, '--plans', {str(CATALOG)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(name.endswith(('.direct_executor', '.direct_kernels', '.recompute_prepare',
                             '.recompute_prepare_kernels')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
