import pytest
import torch
from test_quadratic_feature_gram import make_pair

from torchcst._derivatives import factored_taylor as ft


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_factor_contractions_match_visible_derivatives(gated, dtype):
    problem, _ = make_pair(dtype, gated)
    geometry = problem.context.geometry
    point = problem.context.current_point
    factors = ft.factor_derivatives(geometry.derivatives.factor_atoms, point)
    x, y = torch.randn_like(point) * 0.1, torch.randn_like(point) * 0.1
    j, h = geometry.local_quadratic_derivatives(point)
    torch.testing.assert_close(
        ft.tangent(factors, x).flatten(), torch.einsum("kmp,kp->m", j, x)
    )
    torch.testing.assert_close(
        ft.second(factors, x, y).flatten(), torch.einsum("kmpq,kp,kq->m", h, x, y)
    )
    torch.testing.assert_close(
        ft.displacement(factors, x), geometry.displacement(x, point=point)
    )
    force = torch.randn(geometry.visible_shape, dtype=dtype)
    torch.testing.assert_close(
        ft.pullback(factors, force, x),
        geometry.pullback(force, point=point, displacement=x),
    )
    constant, blocks = ft.affine_pullback(factors, force)
    torch.testing.assert_close(constant, torch.einsum("kmp,m->kp", j, force.flatten()))
    torch.testing.assert_close(blocks, torch.einsum("kmpq,m->kpq", h, force.flatten()))
    torch.testing.assert_close(
        ft.frame_gram(factors, x), geometry.gram(geometry.frame(point, x)).matrix
    )
    row, column, eps = problem.moments.second.metric.separable_weights()
    metric = problem.moments.second.metric.diagonal().flatten()
    torch.testing.assert_close(
        ft.metric_diagonal(factors, row, column, eps),
        torch.einsum("kmp,m,kmp->kp", j, metric, j),
    )


@pytest.mark.parametrize("corrections", [1, 4])
def test_factored_ray_preserves_original_quartic(corrections):
    from torchcst import DeviceRay
    from torchcst._derivatives.factored_frame import FactoredFrameGeometry
    from torchcst.optim import MomentContext, QuarticProblem
    from torchcst.optim.solvers._factored_ray import coefficients, value_gradient
    from torchcst.optim.solvers._ray import ray_coefficients

    original, _ = make_pair(torch.float64, True)
    geometry = FactoredFrameGeometry(original.context.geometry.derivatives)
    point = original.context.current_point
    factored = QuarticProblem(
        MomentContext(geometry, point),
        original.moments,
        learning_rate=original.learning_rate,
        evaluation="visible",
    )
    f = geometry.factor_local_derivatives(point)
    x, v = torch.randn_like(point) * 0.1, torch.randn_like(point) * 0.1
    metric = original.moments.second.metric.diagonal()
    a, b, inv = (
        original._first_constant,
        original._first_linear,
        1 / original.learning_rate,
    )
    torch.testing.assert_close(
        value_gradient(a, b, f, metric, inv, x), original.value_and_gradient(x)
    )
    j, h = original.context.geometry.local_quadratic_derivatives(point)
    torch.testing.assert_close(
        coefficients(a, b, f, metric, inv, x, v),
        ray_coefficients(a, b, j, h, metric.flatten(), inv, x, v),
    )
    solver = DeviceRay(corrections=corrections)
    expected = solver.solve(original, trust_radius=0.25)
    actual = solver.solve(factored, trust_radius=0.25)
    torch.testing.assert_close(actual.displacement, expected.displacement)
    torch.testing.assert_close(actual.objective, original.value(actual.displacement))
    source = geometry.frame(point + 0.02, v)
    expected_pullback = original.context.geometry.pullback_from_frame(
        current_point=point, source_frame=source, source_coefficients=x
    )
    actual_pullback = geometry.pullback_from_frame(
        current_point=point, source_frame=source, source_coefficients=x
    )
    torch.testing.assert_close(actual_pullback.constant, expected_pullback.constant)
    torch.testing.assert_close(actual_pullback.linear, expected_pullback.linear)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compact_cuda_graphs_match_visible_oracle(dtype):
    from torchcst import DeviceRay
    from torchcst._derivatives.factored_frame import FactoredFrameGeometry
    from torchcst._runtime.validation import device_checks
    from torchcst.optim import MomentContext, QuarticProblem

    original, _ = make_pair(dtype, True, device="cuda")
    geometry = FactoredFrameGeometry(original.context.geometry.derivatives)
    point = original.context.current_point
    compact = QuarticProblem(
        MomentContext(geometry, point),
        original.moments,
        learning_rate=original.learning_rate,
        evaluation="visible",
    )
    solver = DeviceRay(corrections=1)
    for i in range(2):
        out = solver.solve(compact, trust_radius=0.25)
        torch.testing.assert_close(out.objective, original.value(out.displacement))
        frame = geometry.frame(point + i * 0.01, out.displacement)
        torch.testing.assert_close(
            geometry.gram(frame).matrix, original.context.geometry.gram(frame).matrix
        )
        actual = geometry.pullback_from_frame(
            current_point=point,
            source_frame=frame,
            source_coefficients=out.displacement,
        )
        expected = original.context.geometry.pullback_from_frame(
            current_point=point,
            source_frame=frame,
            source_coefficients=out.displacement,
        )
        torch.testing.assert_close(actual.constant, expected.constant)
        torch.testing.assert_close(actual.linear, expected.linear)
    with device_checks():
        torch.cuda.set_sync_debug_mode("error")
        try:
            solver.solve(compact, trust_radius=0.25)
            geometry.gram(frame)
            geometry.pullback_from_frame(
                current_point=point,
                source_frame=frame,
                source_coefficients=out.displacement,
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")


def test_factored_optimizer_updates_without_visible_derivative_cache():
    from unittest.mock import patch

    from test_training_smoke import amplitude_bandwidth
    from torch.nn import functional

    from torchcst import (
        Chart,
        CSTLinear,
        CSTSecondOrderAdam,
        DeviceRay,
        SecondOrderAdamConfig,
    )
    from torchcst._derivatives.atoms import AtomDerivatives

    torch.manual_seed(99)
    model = CSTLinear(
        Chart.linspace(5),
        Chart.linspace(3),
        atoms=3,
        kernel=amplitude_bandwidth(),
        dtype=torch.float64,
        backend="factored",
    )
    optimizer = CSTSecondOrderAdam(
        model,
        cst=SecondOrderAdamConfig(
            lr=0.01, quartic=DeviceRay(corrections=1), factored_geometry=True
        ),
        dense=None,
    )
    inputs = torch.randn(8, 5, dtype=torch.float64)
    labels = torch.arange(8) % 3
    before = model.atoms.p.detach().clone()
    with (
        patch.object(
            model,
            "_materialize_atoms",
            side_effect=AssertionError("atom weights requested"),
        ),
        patch.object(
            AtomDerivatives,
            "materialized_local_derivatives",
            side_effect=AssertionError("visible derivatives requested"),
        ),
    ):
        for _ in range(3):
            optimizer.zero_grad()
            functional.cross_entropy(model(inputs), labels).backward()
            optimizer.step()
    assert not torch.equal(before, model.atoms.p)
    assert torch.isfinite(model.atoms.p).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_factor_observation_matches_visible_scalar_hessian(dtype):
    from torchcst._derivatives._captured import factor_observation

    original, _ = make_pair(dtype, True)
    geometry = original.context.geometry
    point = original.context.current_point
    x = torch.randn(7, 5, dtype=dtype)
    g = torch.randn(7, 4, dtype=dtype)
    j, h = geometry.local_quadratic_derivatives(point)
    force = (g.T @ x).flatten()
    actual = factor_observation(geometry.derivatives.factor_atoms, point, x, g)
    expected = (
        torch.einsum("kmp,m->kp", j, force),
        torch.einsum("kmpq,m->kpq", h, force),
    )
    torch.testing.assert_close(actual, expected)
