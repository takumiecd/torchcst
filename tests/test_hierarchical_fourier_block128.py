"""CUDA C++ block128: metadata, host/device scalar math and end-to-end VJP."""

import ctypes
import hashlib
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torchcst import FourierPeakSpec
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.algorithm import (
    Block128HierarchicalFourierAlgorithm,
    FourierContext,
)
from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.contract import (
    Block128FourierRecipe,
    FourierLinearSpec,
    torch_reference,
)


def test_block128_metadata_and_scope():
    from torchcst._backends.registry import Registry
    from torchcst._backends.schema import ExecutionPlan

    alg, recipe = Block128HierarchicalFourierAlgorithm(), Block128FourierRecipe()
    registry = Registry()
    registry.register(alg)
    plan = ExecutionPlan(alg.id, alg.revision, recipe)
    assert registry.loads_plan(registry.dumps_plan(plan)) == plan
    context = FourierContext(FourierLinearSpec(128), "cuda", torch.float32, 4, 128)
    assert alg.supports(context, recipe).supported
    for change in (
        {"atoms": 127},
        {"batch": 0},
        {"batch": 65},
        {"spec": FourierLinearSpec(256)},
        {"dtype": torch.float64},
        {"device_type": "cpu"},
        {"spec": FourierLinearSpec(128, 4)},
    ):
        assert not alg.supports(replace(context, **change), recipe).supported
    for change in (
        {"batch_size": True},
        {"batch_size": 8},
        {"output_tile_sites": 16},
        {"hg_execution": "split_groups"},
        {"h_retention": "save"},
        {"scheduling": "dynamic_jobs"},
    ):
        with pytest.raises(ValueError):
            replace(recipe, **change)


def test_block128_catalog_and_case_snapshot():
    from benchmarks.cuda.linear.manifest import decode_snapshot, load_run

    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    run = load_run(
        root / "cases/block128-fourier-m8-mixed-128-a128.json",
        root / "plans-hierarchical-fourier-block128.json",
    )
    restored = decode_snapshot(run.snapshot())
    assert restored.snapshot() == run.snapshot()
    assert run.entry("cuda-block128").plan.recipe == Block128FourierRecipe()


def test_benchmark_metadata_hashes_native_sources(monkeypatch):
    from benchmarks.automation.ingest import expected_worker_source
    from benchmarks.cuda.linear.manifest import load_run
    from benchmarks.cuda.linear.run import _metadata

    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    run = load_run(
        root / "cases/block128-fourier-m8-mixed-128-a128.json",
        root / "plans-hierarchical-fourier-block128.json",
    )
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="metadata-only",
            major=8,
            minor=9,
            multi_processor_count=1,
            total_memory=1,
        ),
    )
    args = SimpleNamespace(
        device=0,
        source_commit="unrecorded",
        snapshot_sha256="0" * 64,
        worker="correctness",
        plan_id="cuda-block128",
        polar_update=None,
    )
    metadata = _metadata(args, run)
    selected = expected_worker_source(
        {"request": {"source_files": metadata["source_hashes"]}}
    )
    for name in ("block128_math.h", "block128_bindings.cpp", "block128_kernels.cu"):
        entries = [
            (path, digest)
            for path, digest in metadata["source_hashes"].items()
            if path.endswith("/" + name)
        ]
        assert len(entries) == 1
        path, digest = entries[0]
        assert digest == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert selected[path] == digest


@pytest.fixture(scope="module")
def host_math(tmp_path_factory):
    """Compile the very same scalar header without CUDA to test its mathematics."""
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier import contract

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    directory = tmp_path_factory.mktemp("block128_host")
    source = directory / "check.cpp"
    source.write_text(r"""
#include "block128_math.h"
extern "C" void evaluate(double width,double center,const double* fit,double* out) {
    for(int i=0;i<128*3;++i) out[i]=0.;
    for(int c=0;c<4;++c) {
        auto comp=cst128::component(width,c);
        double co[9],ci[9]; cst128::coefficients(fit,width,comp,co,ci);
        for(int i=0;i<128;++i) {
            double f[3]; cst128::series(i,center,comp,co,ci,f);
            for(int k=0;k<3;++k) out[i*3+k]+=f[k];
        }
    }
}
extern "C" void fourier(double width,double center,const double* fit,int c,int site,double* out) {
    auto comp=cst128::component(width,c);
    out[0]=out[1]=out[2]=0.;
    if(comp.bank<0) return;
    for(int r=0;r<17;++r) {
        int n=r<=8?r:r-8;
        double co,ci,beta[3]; cst128::coefficient(fit,width,comp,n,co,ci);
        cst128::beta(comp,center,r,co,ci,beta);
        for(int k=0;k<3;++k) out[k]+=beta[k]*cst128::basis(comp.bank,r,site);
    }
}
""")
    library = directory / "math.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(Path(contract.__file__).parent),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    dll = ctypes.CDLL(str(library))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    dll.evaluate.argtypes = [ctypes.c_double, ctypes.c_double, vector, vector]
    dll.fourier.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        vector,
        ctypes.c_int,
        ctypes.c_int,
        vector,
    ]
    dll.evaluate.restype = dll.fourier.restype = None
    return dll


def test_host_scalar_profiles_and_all_derivative_fields(host_math):
    from benchmarks.research.hierarchical_fourier_cpu import knots
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        coefficient_table,
    )
    from torchcst._backends.torch.kernels.hierarchical_fourier import (
        values_and_derivatives,
    )

    spec = FourierLinearSpec(128)
    fit = coefficient_table(8)[0].numpy()
    widths = [0.1, 0.105, 0.2, 0.333, 1.0, 2.4, 5.0, 9.99, 10.0]
    for boundary, _ in knots(spec.period)[::3]:
        widths.extend([max(0.1, boundary - 1e-8), boundary, min(10.0, boundary + 1e-8)])
    sites = torch.arange(128, dtype=torch.float64) * 0.1
    for j, width in enumerate(widths):
        center = [-0.09, 0.137, 6.381, 12.79, 212.8][j % 5]
        result = np.zeros((128, 3), dtype=np.float64)
        host_math.evaluate(width, center, fit, result)
        ref = values_and_derivatives(
            FourierPeakSpec(modes=8),
            spec.period,
            sites,
            torch.tensor(center, dtype=torch.float64),
            torch.tensor(width, dtype=torch.float64),
            support="centered",
        )
        for field in range(3):
            torch.testing.assert_close(
                torch.from_numpy(result[:, field]), ref[field], atol=3e-9, rtol=3e-8
            )


def test_host_bank_beta_derivatives_match_shifted_series(host_math):
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.executor import (
        coefficient_table,
    )
    from torchcst._backends.torch.kernels.hierarchical_fourier import components
    from torchcst._backends.torch.profiles.fourier_peak import coefficients

    spec = FourierLinearSpec(128)
    fit = coefficient_table(8)[0].numpy()
    for width in (1.2, 2.4, 5.0, 9.0):
        L, _, w, wi, _, levels = components(
            spec.period, torch.tensor(width, dtype=torch.float64)
        )
        b, bq = coefficients(FourierPeakSpec(modes=8), width / L)
        center, site = 6.137, 61
        mode = torch.arange(9, dtype=torch.float64)
        for c in range(4):
            if not (w[c] > 0 and 4 <= levels[c] <= 6):
                continue
            result = np.zeros(3)
            host_math.fourier(width, center, fit, c, site, result)
            phase = (site * 0.1 - center) * torch.pi * mode / L[c]
            expected = torch.stack(
                (
                    (b[c] * w[c] * phase.cos()).sum(),
                    (b[c] * w[c] * phase.sin() * torch.pi * mode / L[c]).sum(),
                    ((bq[c] / L[c] * w[c] + b[c] * wi[c]) * phase.cos()).sum(),
                )
            )
            torch.testing.assert_close(
                torch.from_numpy(result), expected, atol=3e-9, rtol=3e-8
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA and nvcc"
)
@pytest.mark.parametrize("batch", [1, 4, 5])
def test_cuda_block128_y_dx_all_atom_gradients_and_retained_backward(batch):
    from benchmarks.cuda.linear.hierarchical_fourier import initialize
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.block128_executor import (
        apply,
    )

    spec, recipe = FourierLinearSpec(128), Block128FourierRecipe()
    p0 = initialize(SimpleNamespace(size=128, atoms=128, seed=41, profile="m8-mixed"))
    gen = torch.Generator().manual_seed(118)
    x0 = torch.randn(batch, 128, generator=gen)
    dy = torch.randn(batch, 128, generator=gen)
    xp = x0.double().requires_grad_()
    pp = p0.double().requires_grad_()
    yr = torch_reference(xp, pp, spec)
    xr, pr = torch.autograd.grad(yr, (xp, pp), dy.double())
    x = x0.cuda().requires_grad_()
    p = p0.cuda().requires_grad_()
    y = apply(x, p, spec, recipe)
    gx, gp = torch.autograd.grad(y, (x, p), dy.cuda(), retain_graph=True)
    repeated = torch.autograd.grad(y, (x, p), dy.cuda())
    for actual, expected in ((y, yr), (gx, xr), (gp, pr)):
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=4e-4, rtol=4e-4
        )
    torch.testing.assert_close(gx, repeated[0])
    torch.testing.assert_close(gp, repeated[1])
    assert gp[0, 0].abs() > 0  # zero amplitude must retain the amplitude gradient


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA and nvcc"
)
@pytest.mark.parametrize("which", ["x", "p"])
def test_cuda_block128_original_input_version_check(which):
    from benchmarks.cuda.linear.hierarchical_fourier import initialize
    from torchcst._backends.cuda.algorithms.linear.hierarchical_fourier.block128_executor import (
        apply,
    )

    p = (
        initialize(SimpleNamespace(size=128, atoms=128, seed=41, profile="m8-mixed"))
        .cuda()
        .requires_grad_()
    )
    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    y = apply(x, p, FourierLinearSpec(128), Block128FourierRecipe())
    with torch.no_grad():
        (x if which == "x" else p).add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()
