"""Site owners compare the identical Sphere task against compact and scatter."""

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
CATALOG = LINEAR / "plans-sphere-site-gather.json"
CANDIDATES = ("site-gather1", "site-gather4")
CONTROL = "recompute-g4-merged"


@pytest.mark.parametrize("size,atoms", [(1024, 52428), (2048, 209715)])
def test_site_gather_comparison_preserves_fixtures_and_controls(size, atoms):
    run = load_run(LINEAR / f"cases/sphere-site-gather-{size}-sigma3.json", CATALOG)
    previous = load_run(
        LINEAR / f"cases/sphere-direct-recompute-{size}-sigma3.json",
        LINEAR / "plans-sphere-direct-recompute.json",
    )
    assert [entry.id for entry in run.plans] == ["compact-weight", CONTROL, *CANDIDATES]
    assert len(run.plans) == 4 and run.baseline == "compact-weight" and run.dense
    for name in ("compact-weight", CONTROL):
        assert run.entry(name).plan == previous.entry(name).plan
    fields, original = asdict(run.case), asdict(previous.case)
    fields.pop("id")
    original.pop("id")
    assert fields == original
    assert (run.case.size, run.case.atoms, run.case.profile) == (size, atoms, "rho3")
    for group, name in zip((1, 4), CANDIDATES):
        plan = run.entry(name).plan
        assert plan.algorithm_id == "research_cuda_sphere_polar_site_gather"
        assert plan.algorithm_revision == "v1"
        assert asdict(plan.recipe) == {
            **asdict(run.entry(CONTROL).plan.recipe),
            "site_group": group,
        }
        assert asdict(plan.recipe) == {
            "support_capacity": 64,
            "support_tile": 16,
            "atom_group": 4,
            "index_bits": 16,
            "merge_output_vjp": True,
            "site_group": group,
        }
    assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run


@pytest.mark.parametrize("size", [1024, 2048])
def test_site_gather_contributor_preparation_preserves_fixed_cohort(tmp_path, size):
    case = LINEAR / f"cases/sphere-site-gather-{size}-sigma3.json"
    output = tmp_path / "prepared"
    original = load_run(case, CATALOG)
    dev.prepare(CATALOG, case, [CONTROL, *CANDIDATES], output)
    report = dev.check(output / "plans.json", [output / "case.json"])
    prepared = load_run(output / "case.json", output / "plans.json")
    assert report["status"] == "PASS" and prepared.case == original.case
    assert prepared.baseline == "compact-weight" and prepared.dense
    assert [entry.id for entry in prepared.plans] == [
        entry.id for entry in original.plans
    ]
    for entry in prepared.plans:
        assert entry.plan == original.entry(entry.id).plan


@pytest.mark.parametrize("size", [1024, 2048])
def test_site_gather_declarations_do_not_import_gpu_execution(size):
    case = LINEAR / f"cases/sphere-site-gather-{size}-sigma3.json"
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run', '--case', {str(case)!r}, '--plans', {str(CATALOG)!r}, '--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(name.endswith(('.site_gather_executor', '.site_gather_kernels',
                             '.direct_executor', '.direct_kernels', '.recompute_prepare',
                             '.recompute_prepare_kernels')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
