"""Direct-contraction comparisons preserve the accepted Sphere task and controls."""

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
CATALOG = LINEAR / "plans-sphere-direct.json"
CANDIDATES = (
    "direct-g1-separate",
    "direct-g4-separate",
    "direct-g1-merged",
    "direct-g4-merged",
)


@pytest.mark.parametrize("size,atoms", [(1024, 52428), (2048, 209715)])
def test_direct_comparison_preserves_task_and_accepted_controls(size, atoms):
    run = load_run(LINEAR / f"cases/sphere-direct-{size}-sigma3.json", CATALOG)
    previous = load_run(
        LINEAR / f"cases/sphere-grouped-compact-{size}-sigma3.json",
        LINEAR / "plans-sphere-grouped-compact.json",
    )
    assert [entry.id for entry in run.plans] == [
        "compact-weight",
        "old-direct",
        *CANDIDATES,
    ]
    assert run.baseline == "compact-weight" and run.dense
    assert run.entry("compact-weight").plan == previous.entry("grouped-compact").plan
    old = run.entry("old-direct").plan
    assert old.algorithm_id == "research_cuda_sphere_polar_support"
    assert old.algorithm_revision == "v1"
    assert asdict(old.recipe) == {"support_capacity": 64}
    fields, original = asdict(run.case), asdict(previous.case)
    fields.pop("id")
    original.pop("id")
    assert fields == original
    assert (run.case.size, run.case.atoms, run.case.profile) == (size, atoms, "rho3")
    for name in CANDIDATES:
        plan = run.entry(name).plan
        assert plan.algorithm_id == "research_cuda_sphere_polar_direct"
        assert plan.algorithm_revision == "v1"
        assert asdict(plan.recipe) == {
            "support_capacity": 64,
            "support_tile": 16,
            "atom_group": 4 if "g4" in name else 1,
            "index_bits": 16,
            "merge_output_vjp": name.endswith("merged"),
        }
    assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run


@pytest.mark.parametrize("size", [1024, 2048])
def test_direct_contributor_preparation_preserves_all_ablations(tmp_path, size):
    case = LINEAR / f"cases/sphere-direct-{size}-sigma3.json"
    output = tmp_path / "prepared"
    original = load_run(case, CATALOG)
    dev.prepare(CATALOG, case, list(CANDIDATES), output)
    report = dev.check(output / "plans.json", [output / "case.json"])
    prepared = load_run(output / "case.json", output / "plans.json")
    assert report["status"] == "PASS"
    assert prepared.case == original.case
    assert prepared.baseline == "compact-weight" and prepared.dense
    assert [entry.id for entry in prepared.plans] == ["compact-weight", *CANDIDATES]
    for entry in prepared.plans:
        assert entry.plan == original.entry(entry.id).plan


@pytest.mark.parametrize("size", [1024, 2048])
def test_direct_declarations_do_not_import_gpu_execution(size):
    case = LINEAR / f"cases/sphere-direct-{size}-sigma3.json"
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(case)!r}, '--plans', {str(CATALOG)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(name.endswith(('.direct_executor', '.direct_kernels')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
