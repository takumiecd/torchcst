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
    fourier_primary,
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


@pytest.mark.parametrize(
    "field,value", [("size", 300.0), ("atoms", 64.0), ("batch", 8.0), ("modes", True)]
)
def test_fourier_case_rejects_noninteger_dimensions(field, value):
    case = ff.case_definition()
    case[field] = value
    with pytest.raises(ValueError, match="integers"):
        ff.load_case(case)


@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
def test_density_case_exact_tiles_and_immutable_geometry(family, modes):
    from torchcst._backends.cuda.algorithms.linear.fourier_tiles.algorithm import (
        FourierTilesAlgorithm,
    )
    from torchcst._backends.cuda.algorithms.linear.fourier_tiles.recipe import (
        FourierTilesRecipe,
    )
    from torchcst._backends.schema import DeviceInfo
    from torchcst.kernels.fourier_layout import fourier_tile_layout
    from torchcst.operators.context import LinearContext

    case = ff.density_case_definition(family=family, modes=modes)
    assert ff.load_case(case) == case
    assert case["atoms"] == int(0.05 * 1024 * 1024) == 52428
    assert case["periods"] == [102.4, 102.4]
    assert case["tile_width"] == 12.8 and case["local_period"] == 38.4
    assert fourier_primary(case) == ("fourier-shared", "fourier-direct", "dense")
    model = ff.fixture(case, atoms=17)
    assert model.atoms.p.dtype == torch.float32
    assert model.chart.geometry.periods.dtype == torch.float64
    assert model.chart.origin.dtype == torch.float64
    assert model.chart.spacing.tolist() == [0.1, 0.1]
    assert model.chart.geometry.periods.tolist() == case["periods"]
    declaration = model.execution_declaration()
    assert fourier_tile_layout(declaration.charts[0], declaration.kernel) == (
        (8, 128, 12.8),
        (8, 128, 12.8),
    )
    context = LinearContext(
        operator=declaration,
        input_shape=(32, 1024),
        input_strides=(1024, 1),
        dtype=torch.float32,
        atom_count=52428,
        device=DeviceInfo("cuda", 0),
        parameter_dim=4,
    )
    assert FourierTilesAlgorithm().supports(context, FourierTilesRecipe()).supported
    # Device-only movement must preserve source FP64 geometry declarations.
    model.to("cpu")
    assert model.chart.geometry.periods.dtype == torch.float64


@pytest.mark.parametrize(
    "field,value",
    [
        ("atoms", 52429),
        ("size", 1024.0),
        ("atom_density", 0.1),
        ("tile_width", 10.0),
        ("local_period", 30.0),
        ("geometry_dtype", "float32"),
        ("required_workers", ["fourier-shared", "dense"]),
    ],
)
def test_density_case_protocol_mutations_rejected(field, value):
    case = ff.density_case_definition()
    case[field] = value
    with pytest.raises(ValueError, match="frozen"):
        ff.load_case(case)


def test_density_source_cases_predeclared():
    from pathlib import Path

    directory = Path(__file__).parents[1] / "benchmarks/cuda/linear/cases"
    for family in ("chebyshev", "kaiser"):
        for modes in (32, 64):
            path = directory / f"fourier-density-{family}-1024-a52428-b32-m{modes}.json"
            assert json.loads(path.read_text()) == ff.density_case_definition(
                family=family, modes=modes
            )


@pytest.mark.parametrize("family", ["chebyshev", "kaiser"])
@pytest.mark.parametrize("modes", [32, 64])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_density_cpu_native_oracle_dispatch_and_normalization(family, modes, floor):
    case = ff.density_case_definition(family=family, modes=modes)
    layer = ff.fixture(case, dtype=torch.float64, floor=floor, atoms=17)
    generator = torch.Generator().manual_seed(1538)
    x = torch.randn(2, 1024, dtype=torch.float64, generator=generator)
    dy = torch.randn(2, 1024, dtype=torch.float64, generator=generator)
    for kind in ("torch-factored", "torch-shared"):
        replica = copy.deepcopy(layer)
        bind_fourier_plan(replica, kind)
        result = fourier_oracle_check(replica, x, dy, case=case)
        assert result["status"] == "PASS"
        assert "full-site norm derivative" in result["scope"]
        assert max(value["max_abs"] for value in result["errors"].values()) < 1e-9
