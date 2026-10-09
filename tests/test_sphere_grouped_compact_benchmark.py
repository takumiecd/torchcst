"""Compact-only comparison declarations preserve the measured task."""

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
CATALOG = LINEAR / "plans-sphere-grouped-compact.json"
BASELINE = "sphere-grouped-weight-64-g8-t16"
CANDIDATE = "grouped-compact"


@pytest.mark.parametrize("size,atoms", [(1024, 52428), (2048, 209715)])
def test_compact_preserves_measured_baseline_and_task(size, atoms):
    case = LINEAR / f"cases/sphere-grouped-compact-{size}-sigma3.json"
    run = load_run(case, CATALOG)
    previous = load_run(
        LINEAR / f"cases/sphere-polar-{size}-sigma3-grouped.json",
        LINEAR / "plans-sphere-polar-grouped.json",
    )
    assert [entry.id for entry in run.plans] == [BASELINE, CANDIDATE]
    assert run.baseline == BASELINE and run.dense
    assert run.entry(BASELINE).plan == previous.entry(BASELINE).plan
    fields, oldfields = asdict(run.case), asdict(previous.case)
    fields.pop("id")
    oldfields.pop("id")
    assert fields == oldfields
    assert (run.case.size, run.case.atoms, run.case.profile) == (size, atoms, "rho3")
    assert asdict(run.entry(CANDIDATE).plan.recipe) == {
        "support_capacity": 64,
        "patch_tile": 16,
        "atom_group": 8,
        "index_bits": 16,
    }
    assert (
        run.entry(CANDIDATE).plan.algorithm_id
        == "research_cuda_sphere_polar_grouped_compact"
    )
    assert decode_snapshot(json.loads(json.dumps(run.snapshot()))) == run


@pytest.mark.parametrize("size", [1024, 2048])
def test_compact_contributor_prepare_and_check(tmp_path, size):
    case = LINEAR / f"cases/sphere-grouped-compact-{size}-sigma3.json"
    output = tmp_path / "prepared"
    source = load_run(case, CATALOG)
    dev.prepare(CATALOG, case, [CANDIDATE], output)
    report = dev.check(output / "plans.json", [output / "case.json"])
    prepared = load_run(output / "case.json", output / "plans.json")
    assert report["status"] == "PASS" and prepared.case == source.case
    assert prepared.baseline == BASELINE and prepared.dense
    assert [entry.id for entry in prepared.plans] == [BASELINE, CANDIDATE]


@pytest.mark.parametrize("size", [1024, 2048])
def test_compact_validation_does_not_import_gpu_execution(size):
    case = LINEAR / f"cases/sphere-grouped-compact-{size}-sigma3.json"
    code = f"""
import sys
from benchmarks.cuda.linear.run import main
sys.argv = ['run','--case',{str(case)!r},'--plans',{str(CATALOG)!r},'--validate-only']
main()
assert 'triton' not in sys.modules
assert not any(name.startswith('torchcst._backends.cuda.algorithms.linear.') and name.rsplit('.',1)[-1] in ('executor','kernels','grouped_executor','grouped_kernels') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
    )
