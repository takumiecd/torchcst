"""Optional four-row responses reuse completed global hierarchical moments."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_radius_cell_mass_cuda import GPU, atoms, axis_plan
from test_radius_large_shapes import metadata_context

from benchmarks.cuda.linear.manifest import decode_snapshot, load_run
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor as ex
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    RadiusCellMassAlgorithm,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plan import (
    HierarchyPlan,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_BATCHED_CACHED,
    GLOBAL_BATCHED_RESPONSE4,
    HYBRID_SERIAL,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parallel_response_is_explicit_global_only_and_roundtrips():
    run = load_run(
        ROOT
        / "benchmarks/cuda/linear/cases/radius-cell-mass-1024-response4-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )
    assert run.baseline == "radius-global-batched-cached"
    assert run.entry("radius-global-batched-response4").plan == GLOBAL_BATCHED_RESPONSE4
    assert decode_snapshot(run.snapshot()) == run
    assert GLOBAL_BATCHED_RESPONSE4.recipe.schedule == "global_batched_cached_response4"
    assert GLOBAL_BATCHED_CACHED.recipe.schedule == "global_batched_cached"
    assert HYBRID_SERIAL.recipe.schedule == "sequential"
    assert (
        replace(GLOBAL_BATCHED_RESPONSE4.recipe, schedule="global_batched_cached")
        == GLOBAL_BATCHED_CACHED.recipe
    )
    algorithm = RadiusCellMassAlgorithm()
    assert algorithm.supports(
        metadata_context(), GLOBAL_BATCHED_RESPONSE4.recipe
    ).supported
    for value in (0, 1, None, "global_response4"):
        with pytest.raises(ValueError, match="schedule"):
            algorithm.validate_recipe(
                replace(GLOBAL_BATCHED_RESPONSE4.recipe, schedule=value)
            )
    for change in ({"native_autograd": True}, {"native_entry": True}):
        with pytest.raises(ValueError):
            algorithm.validate_recipe(
                replace(GLOBAL_BATCHED_RESPONSE4.recipe, **change)
            )


@pytest.mark.parametrize("selected", [GLOBAL_BATCHED_CACHED, GLOBAL_BATCHED_RESPONSE4])
@pytest.mark.parametrize("method", ["merge", "direct"])
@pytest.mark.parametrize("output_side", [False, True])
def test_cpu_stage_contract_completes_same_moments_before_parallel_response(
    monkeypatch, selected, method, output_side
):
    axis = axis_plan(1024)
    signal, p = torch.randn(5, 1024), torch.randn(8, 4)
    moments = torch.randn(
        5, axis.active_nodes(2), axis.basis_count, dtype=torch.float64
    )
    recipe = replace(selected.recipe, moment_method=method)
    order = []

    def build(actual_signal, actual_axis, **kwargs):
        assert actual_signal is signal and actual_axis is axis
        assert kwargs == {"method": method, "start_level": 2}
        order.append("moments")
        return moments

    def direct(sx, points, response, config, side, start):
        assert sx is signal and points is p and side == output_side and start == 2
        order.append("direct")

    def complete(points, z, response, config, side, start):
        assert points is p and z is moments and side == output_side and start == 2
        order.append("complete")
        response.fill_(17)

    def parallel(sx, points, z, response, config, side, start):
        assert sx is signal and points is p and z is moments
        assert side == output_side and start == 2 and config == axis.native_config()
        order.append("parallel")
        response.fill_(17)

    monkeypatch.setattr(ex, "build_moments", build)
    monkeypatch.setattr(
        ex,
        "_extension",
        lambda: SimpleNamespace(
            direct_response_into=direct,
            complete_response=complete,
            parallel_response_into=parallel,
        ),
    )
    z, h = ex._complete_hybrid_response(signal, p, axis, 2, recipe, output_side)
    assert z is moments
    assert h.shape == (5, 8) and h.dtype == torch.float64
    assert torch.equal(h, torch.full_like(h, 17))
    assert order == (
        ["moments", "parallel"]
        if recipe.schedule == "global_batched_cached_response4"
        else ["direct", "moments", "complete"]
    )


def assert_compact_response_roundoff(actual, expected, signal, axis):
    """A forward-error bound for two legal FP64 compact evaluations.

    46fe PTX contracts the harmonic sine recurrence's opposite product in the
    scalar and four-row kernels. The part/mode order is unchanged, but this
    makes bit equality inappropriate. Count 16 rounded operations per mode and
    window part (phase, complex dot and weighted accumulation). For normalized
    nonnegative Fourier masses, |c_k| <= c_0=1/P and the absolute Fourier-term
    sum is at most (2M+1)*max|X|. Two evaluations allow 2*gamma_n times this sum.
    Direct-response rows below start retain their separate exact comparison.
    """
    for name, tensor in (
        ("actual", actual),
        ("expected", expected),
        ("signal", signal),
    ):
        assert bool(torch.isfinite(tensor).all()), f"nonfinite {name} response input"
    assert actual.dtype == expected.dtype == torch.float64
    unit_roundoff = torch.finfo(torch.float64).eps / 2
    operations = 16 * max(bank.window_tiles for bank in axis.banks) * (axis.modes + 1)
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    scale = float(signal.abs().max()) if signal.numel() else 0.0
    absolute_bound = 2 * gamma * (2 * axis.modes + 1) * scale
    torch.testing.assert_close(actual, expected, atol=absolute_bound, rtol=0)
    return absolute_bound


def test_response_roundoff_budget_accepts_two_ulps_rejects_material_drift():
    axis = axis_plan(1024)
    expected = torch.ones(1, 1, dtype=torch.float64)
    actual = torch.nextafter(torch.nextafter(expected, expected * 2), expected * 2)
    allowed = assert_compact_response_roundoff(actual, expected, expected, axis)
    assert 0 < allowed < 1e-10
    with pytest.raises(AssertionError):
        assert_compact_response_roundoff(expected + 1e-9, expected, expected, axis)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["actual", "expected", "signal"])
def test_response_roundoff_budget_rejects_nonfinite(value, side):
    tensors = {
        name: torch.ones(1, 1, dtype=torch.float64)
        for name in ("actual", "expected", "signal")
    }
    tensors[side].fill_(value)
    with pytest.raises(AssertionError, match="nonfinite"):
        assert_compact_response_roundoff(**tensors, axis=axis_plan(1024))


def test_harmonic_sine_fma_contractions_can_differ_by_two_fp64_ulps():
    # A CPU exact-arithmetic illustration of the two PTX contractions; these
    # constants are unit-rotation inputs, not a claimed GPU failing H/G row.
    import math
    from decimal import Decimal, localcontext

    step_cos, step_sin = -0.5811357419978747, 0.8138066412684156
    previous_cos, previous_sin = -0.5019334727563961, -0.8649062312914066
    with localcontext() as context:
        context.prec = 200

        def fma(a, b, c):
            return float(
                Decimal.from_float(a) * Decimal.from_float(b) + Decimal.from_float(c)
            )

        scalar = fma(step_cos, previous_sin, step_sin * previous_cos)
        four_row = fma(step_sin, previous_cos, step_cos * previous_sin)
    assert scalar == 0.09415113087604247
    assert four_row == 0.09415113087604245
    assert abs(scalar - four_row) == 2 * math.ulp(scalar)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "batch,count,start",
    [(0, 8, 2), (4, 0, 2), (1, 8, 0), (3, 8, 2), (4, 8, 2), (5, 8, 2), (7, 8, 11)],
)
@pytest.mark.parametrize("output_side", [False, True])
def test_native_global_parallel_response_matches_serial_complete_rows(
    dtype, batch, count, start, output_side
):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    generator = torch.Generator().manual_seed(4242)
    signal = torch.randn(
        batch, 1024, dtype=dtype, generator=generator, device="cpu"
    ).cuda()
    p = atoms(plan, dtype=dtype)[:count].cuda()
    z = ex.build_moments(signal, axis, start_level=start)
    native = ex._extension()
    serial = torch.empty(batch, count, dtype=torch.float64, device="cuda")
    parallel = torch.empty_like(serial)
    native.direct_response_into(
        signal, p, serial, axis.native_config(), output_side, start
    )
    native.complete_response(p, z, serial, axis.native_config(), output_side, start)
    native.parallel_response_into(
        signal, p, z, parallel, axis.native_config(), output_side, start
    )
    # Direct rows retain exact parity. Compact phase recurrence has a different
    # legal FMA contraction in the qualified binary, so use a FP64 error bound.
    direct = (
        p[:, 1].double() <= axis.banks[2 * (start - 1)].tile_sites * axis.spacing
        if start
        else torch.zeros(count, device=p.device, dtype=torch.bool)
    )
    torch.testing.assert_close(parallel[:, direct], serial[:, direct], atol=0, rtol=0)
    assert_compact_response_roundoff(
        parallel[:, ~direct], serial[:, ~direct], signal, axis
    )


@GPU
@pytest.mark.parametrize(
    "wrong", ["moment_dtype", "moment_batch", "response_shape", "signal_stride"]
)
def test_native_global_parallel_response_rejects_changed_metadata_before_launch(wrong):
    axis = axis_plan(1024)
    plan = HierarchyPlan(input=axis, output=axis)
    signal = torch.randn(4, 1024, device="cuda")
    p = atoms(plan, dtype=torch.float32).cuda()
    z = ex.build_moments(signal, axis, start_level=2)
    h = torch.empty(4, 8, dtype=torch.float64, device="cuda")
    if wrong == "moment_dtype":
        z = z.float()
    elif wrong == "moment_batch":
        z = z[:3]
    elif wrong == "response_shape":
        h = torch.empty(4, 7, dtype=torch.float64, device="cuda")
    else:
        signal = torch.randn(4, 2048, device="cuda")[:, ::2]
    with pytest.raises(RuntimeError, match="compact|batch|response|contiguous"):
        ex._extension().parallel_response_into(
            signal, p, z, h, axis.native_config(), False, 2
        )
    torch.cuda.synchronize()
