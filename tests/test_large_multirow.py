"""Narrow large-width admission and bounded setup, not a speed assertion."""

import copy
from dataclasses import replace

import pytest
import torch
from test_ffn_rectangular_2048 import context
from test_variance_initialization import layer

from benchmarks.cuda.linear.manifest import REGISTRY
from torchcst import CSTLinear, FixedSelector, init, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.large_multirow import (
    LARGE_GEOMETRY,
    LARGE_LEVELS,
    LARGE_LEVELS_STREAM64,
    LARGE_PACKED,
    LARGE_RECOMPUTE,
    LARGE_ROW8,
    LARGE_STREAM64,
    RadiusLargeGeometryAlgorithm,
    RadiusLargeLevelsAlgorithm,
    RadiusLargeLevelsStream64Algorithm,
    RadiusLargePackedAlgorithm,
    RadiusLargeRecomputeAlgorithm,
    RadiusLargeRow8Algorithm,
    RadiusLargeStream64Algorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_ACTIVE_GRAD4_NATIVE,
)
from torchcst._backends.torch.kernels.radius_cell_mass import atom_energies, factors

CASES = [
    (RadiusLargeRow8Algorithm, LARGE_ROW8),
    (RadiusLargePackedAlgorithm, LARGE_PACKED),
    (RadiusLargeGeometryAlgorithm, LARGE_GEOMETRY),
    (RadiusLargeLevelsAlgorithm, LARGE_LEVELS),
    (RadiusLargeLevelsStream64Algorithm, LARGE_LEVELS_STREAM64),
    (RadiusLargeStream64Algorithm, LARGE_STREAM64),
    (RadiusLargeRecomputeAlgorithm, LARGE_RECOMPUTE),
]


@pytest.mark.parametrize("n", [2048, 8192, 16384])
@pytest.mark.parametrize("cls,plan", CASES)
def test_scope_is_explicit_and_old_cohort_unchanged(n, cls, plan):
    c = context(n, n, count=8 * n)
    assert cls().supports(c, plan.recipe).supported
    assert (
        not RadiusCellMassAlgorithm()
        .supports(c, GLOBAL_ACTIVE_GRAD4_NATIVE.recipe)
        .supported
    )
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    with pytest.raises((KeyError, ValueError)):
        make_algorithm_registry().validate_plan(plan)
    for change in [
        {"atom_count": 8 * n + 1},
        {"input_shape": (257, n)},
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
    ]:
        assert not cls().supports(replace(c, **change), plan.recipe).supported


@pytest.mark.parametrize("n", [8, 2048])
@pytest.mark.parametrize("q", [0.0, 1.0])
def test_bounded_energy_matches_full_public_factors_including_wrapped_images(n, q):
    m = layer(n, n, atoms=31)
    with torch.no_grad():
        amp = torch.linspace(-0.2, 0.2, 31)
        rho = (1 + 3 * q) ** 0.5
        m.atoms.p[:, 0] = amp * rho
        m.atoms.p[:, 1] = (1 - amp.square()).sqrt() * rho
        m.atoms.p[:, 2] = torch.linspace(-0.5, 2 * n + 0.5, 31)
        m.atoms.p[:, 3] = torch.linspace(n + 0.5, -0.5, 31)
    p = m.atoms.p.detach()
    x, y = factors(m.kernel, *m.cst_charts(), p)
    expected = x.square().sum(0) * y.square().sum(0)
    actual = atom_energies(m.kernel, *m.cst_charts(), p)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)


def test_large_public_initializer_does_not_construct_full_factors(monkeypatch):
    from torchcst._backends.torch import initialization

    def reject(*args, **kwargs):
        raise AssertionError("large initializer used full factors")

    monkeypatch.setattr(initialization.execution, "factors", reject)
    m = layer(2048, 2048, atoms=32)
    report = init.variance_scaling_(
        m, 1e-6, q=0, generator=torch.Generator().manual_seed(31)
    )
    assert report.relative_error <= 0.005
    x, y = factors(m.kernel, *m.cst_charts(), m.atoms.p.detach())
    assert float((x.square().sum(0) * y.square().sum(0)).sum()) == pytest.approx(
        report.expected_energy, rel=1e-12
    )


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")


@GPU
@pytest.mark.parametrize("n", [2048, 8192, 16384])
@pytest.mark.parametrize("rows", [1, 9, 65, 256])
def test_cuda_large_vjps_match_row8_and_high_keys(n, rows):
    # Small A is an indexing/geometry gate; full A=8N is independently checked by runner.
    from torchcst import ChebyshevSpec, RadiusBounds, chart_presets, kernels

    torch.manual_seed(39)
    chart = chart_presets.regular_grid(n, geometry="flat_torus")
    base = CSTLinear(
        chart,
        chart,
        atoms=32,
        dtype=torch.float32,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
            coordinates=kernels.polar(
                amplitude_max=1,
                w_c=0.3,
                input_bounds=RadiusBounds(
                    minimum=0.5, birth=0.5, maximum=50, upper_floor=1
                ),
                radius_envelope="rational_zero_opening",
                zero_opening_scale=0.001,
            ),
        ),
    )
    with torch.no_grad():
        base.atoms.p[0, :2] = torch.tensor([0.0, 1.0])
        base.atoms.p[0, 2:] = n - 0.25
    x = torch.randn(rows, n, device="cuda")
    dy = torch.randn_like(x)
    outputs = []
    saved_keys = []
    for _, plan in CASES:
        m = copy.deepcopy(base).cuda()
        m.selector = FixedSelector(plan, registry=REGISTRY)
        xx = x.clone().requires_grad_()

        def pack(t):
            if t.dtype == torch.int64 and t.shape == (32,):
                saved_keys.append(t.detach().cpu())
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = m(xx)
        outputs.append((y, *torch.autograd.grad(y, (xx, m.atoms.p), dy)))
    for candidate in outputs[1:]:
        for a, b in zip(outputs[0], candidate):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    if n == 16384:
        assert any(int(k.max()) > 2**31 for k in saved_keys)


@pytest.mark.parametrize("n", [2048, 8192, 16384])
def test_multirow_case_snapshot_and_linear_atom_count(n):
    from pathlib import Path

    from benchmarks.cuda.linear.manifest import decode_snapshot, load_run

    root = Path(__file__).resolve().parents[1] / "benchmarks/cuda/linear"
    run = load_run(
        root / f"cases/radius-multirow-{n}-r256-c8-xavier-q0.json",
        root / "plans-radius-multirow.json",
    )
    assert decode_snapshot(run.snapshot()) == run
    assert run.case.rows == 256 and run.case.atoms == 8 * n
    with pytest.raises(ValueError, match="A=8N"):
        replace(run.case, atoms=8 * n + 1)


@GPU
@pytest.mark.parametrize("freeze", [None, "input", "atoms"])
@pytest.mark.parametrize(
    "candidate", [LARGE_GEOMETRY, LARGE_STREAM64, LARGE_LEVELS, LARGE_LEVELS_STREAM64]
)
def test_cached_geometry_preserves_autograd_lifetime_and_gradient_branches(
    freeze, candidate
):
    from test_radius_packed import fixture

    torch.manual_seed(174)
    base = fixture(2048, 2048)  # variable amplitude, radius and periodic centers
    x = torch.randn(2048, 9, device="cuda").T  # non-contiguous input
    dy = torch.randn(9, 2048, device="cuda")
    values = []
    for plan in [LARGE_PACKED, candidate]:
        model = copy.deepcopy(base).cuda()
        model.atoms.p.requires_grad_(freeze != "atoms")
        model.selector = FixedSelector(plan, registry=REGISTRY)
        xx = x.detach().requires_grad_(freeze != "input")
        targets = tuple(t for t in [xx, model.atoms.p] if t.requires_grad)
        saved = []

        def pack(t, saved=saved):
            saved.append((tuple(t.shape), t.dtype))
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = model(xx)
        gradients = torch.autograd.grad(y, targets, dy, retain_graph=True)
        again = torch.autograd.grad(y, targets, dy, retain_graph=True)
        for a, b in zip(gradients, again):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        if plan == candidate:
            assert ((4096, 25), torch.float32) in saved
            assert all(dtype != torch.float64 for _, dtype in saved)
            if not candidate.recipe.save_response:
                assert not any(shape == (9, 4096) for shape, _ in saved)
        values.append((y, *gradients))
        y = model(xx)  # version checking without hooks
        with torch.no_grad():
            model.atoms.p.add_(0.0001)
        with pytest.raises(
            RuntimeError, match="modified (by an inplace operation|inplace)"
        ):
            torch.autograd.grad(y, targets, dy)
    for a, b in zip(*values):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


@GPU
@pytest.mark.parametrize(
    "plan", [LARGE_GEOMETRY, LARGE_STREAM64, LARGE_LEVELS, LARGE_LEVELS_STREAM64]
)
def test_geometry_hook_bad_shape_is_rejected_before_native_use(plan):
    from test_radius_packed import fixture

    model = fixture(2048, 2048).cuda()
    model.selector = FixedSelector(plan, registry=REGISTRY)
    x = torch.randn(9, 2048, device="cuda", requires_grad=True)

    def pack(t):
        return t[:, :24].contiguous() if t.shape == (4096, 25) else t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = model(x)
    with pytest.raises(RuntimeError, match="invalid packed saved tensor"):
        torch.autograd.grad(y.sum(), (x, model.atoms.p))
    torch.cuda.synchronize()
