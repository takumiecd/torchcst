"""CPU-only declaration, independent oracle, and legacy runner compatibility."""

import copy
import json
import subprocess
import sys

import pytest
import torch

from benchmarks.cuda.linear import fourier_profile_product as ff
from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog
from benchmarks.cuda.linear.periodic_comparison import (
    FOURIER_CATALOG,
    bind_fourier_plan,
    case_definition,
    fourier_oracle_check,
    load_case,
    state_gate,
    validate_fourier_record,
)


@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_native_full_site_oracle_and_torch_shared(family, modes, floor):
    case = ff.case_definition(size=300, atoms=64, batch=8, family=family, modes=modes)
    layer = ff.fixture(case, dtype=torch.float64, floor=floor, atoms=7)
    generator = torch.Generator().manual_seed(716)
    x = torch.randn(2, 300, dtype=torch.float64, generator=generator)
    dy = torch.randn(2, 300, dtype=torch.float64, generator=generator)
    for kind in ("torch-factored", "torch-shared"):
        replica = copy.deepcopy(layer)
        bind_fourier_plan(replica, kind)
        result = fourier_oracle_check(replica, x, dy)
        assert result["status"] == "PASS"
        assert max(v["max_abs"] for v in result["errors"].values()) < 1e-9
    assert layer.kernel.amplitude_max.dtype == torch.float64


def test_fourier_catalog_registered_without_cuda_import():
    entries = decode_catalog(json.loads(FOURIER_CATALOG.read_text()))
    assert {entry.id for entry in entries} == {"fourier-shared", "fourier-direct"}
    for entry in entries:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.cuda.linear.manifest; assert 'triton' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_frozen_case_mutation_rejected_and_legacy_unchanged(tmp_path):
    case = ff.case_definition()
    assert ff.load_case(case) == case
    altered = copy.deepcopy(case)
    altered["normalization"]["floor"] = 1e-5
    with pytest.raises(ValueError, match="frozen"):
        ff.load_case(altered)
    for size in (1024, 2048):
        for rho in (3, 8):
            path = tmp_path / "legacy.json"
            path.write_text(json.dumps(case_definition(size, rho)))
            assert load_case(path) == case_definition(size, rho)


def test_failures_cannot_publish_selection_timing():
    record = {
        "kind": "fourier-shared",
        "status": "FAIL",
        "failed_stage": "oracle",
        "error": "fixed gate failed",
        "timing": None,
    }
    validate_fourier_record(record)
    record["timing"] = {"median_ms": 1.0}
    with pytest.raises(ValueError, match="without timings"):
        validate_fourier_record(record)


def test_same_cotangent_public_optimizer_cpu():
    result = state_gate(
        device="cpu",
        steps=3,
        fourier_case=ff.case_definition(size=300, atoms=64, batch=8),
    )
    assert result["status"] == "PASS" and result["final_clock"] == 4
