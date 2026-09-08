"""Dense independent oracles for the matrix-free trust-region update."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_tangent_ops import jacobian, site_for
from torchcst import CSTAdam, FirstOrderAdamConfig
from torchcst._derivatives.tangent_metric import from_prepared
from torchcst._runtime.validation import device_checks
from torchcst.optim._trust_pcg import solve
from torchcst.optim.moments import SeparableDiagonalMetric
from torchcst.optim.quadratic import TangentProblem, solve_tangent


class DenseAction:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, x, active=None):
        y = (self.matrix @ x.flatten()).reshape_as(x)
        return y if active is None else torch.where(active, y, 0)

    def blocks(self):
        return torch.diag_embed(self.matrix.diagonal()[None])


def reference(matrix, linear, radius):
    problem = object.__new__(TangentProblem)
    problem.matrix, problem.linear = matrix, linear
    problem.point_shape, problem.blocked = linear.shape, False
    return solve_tangent(problem, radius=radius)


@pytest.mark.parametrize("radius", [0.01, 0.25, 100.0])
@pytest.mark.parametrize("case", ["spd", "singular", "null_rhs", "zero"])
def test_trust_update_matches_spectral(case, radius):
    torch.manual_seed(22)
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    values = torch.linspace(0.1, 3, 8, dtype=torch.float64)
    if case != "spd":
        values[:3] = 0
    h = q @ torch.diag(values) @ q.T
    b = h @ torch.randn(8, dtype=h.dtype)
    if case == "null_rhs":
        b += q[:, 0]
    if case == "zero":
        b.zero_()
    b = b[None]
    got = solve(
        SimpleNamespace(linear=b, operator=DenseAction(h)),
        radius=radius,
        max_iter=64,
        shift_steps=48,
        rtol=1e-8,
    )
    expected = reference(h, b, radius)
    torch.testing.assert_close(
        got.displacement, expected.displacement, atol=3e-7, rtol=3e-7
    )
    torch.testing.assert_close(got.objective, expected.objective, atol=1e-8, rtol=1e-7)
    assert got.converged and got.relative_residual <= 1e-8
    assert got.relative_complementarity <= 1e-8
    assert got.displacement.norm() <= radius * (1 + 1e-14)
    if case == "singular" and radius == 100:
        assert (got.displacement @ q[:, :3]).norm() < 1e-10


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("zero_amplitude", [False, True])
def test_factor_update_oracle_and_replay(device, zero_amplitude):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    site = site_for("amplitude").to(device)
    p = site.atoms.p.detach().clone()
    if zero_amplitude:
        p[:, 0] = 0
    ops = site.cst_derivatives().tangent_ops(
        execution="triton" if device == "cuda" else "eager"
    )
    metric = SeparableDiagonalMetric(
        torch.ones(4, device=device, dtype=p.dtype),
        torch.ones(6, device=device, dtype=p.dtype),
        eps=0.1,
    )
    for turn in range(2):
        point = p + turn * 0.02
        weights = SeparableDiagonalMetric(
            metric.row, metric.column * (1 + turn), eps=0.1
        )
        action = from_prepared(ops.prepare(point), weights, 0.2)
        j = jacobian(site, point)
        h = j.T @ (weights.diagonal().flatten()[:, None] * j) / 0.2
        x = torch.randn_like(point)
        torch.testing.assert_close(
            action(x).flatten(), h @ x.flatten(), atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            action.diagonal().flatten(), h.diagonal(), atol=1e-10, rtol=1e-10
        )
        expected_blocks = torch.stack(
            [
                h[
                    a * p.shape[1] : (a + 1) * p.shape[1],
                    a * p.shape[1] : (a + 1) * p.shape[1],
                ]
                for a in range(p.shape[0])
            ]
        )
        torch.testing.assert_close(
            action.blocks(), expected_blocks, atol=1e-10, rtol=1e-10
        )
        linear = (
            j.T @ torch.randn(j.shape[0], device=device, dtype=p.dtype)
        ).reshape_as(p)
        got = solve(
            SimpleNamespace(operator=action, linear=linear),
            radius=0.1,
            max_iter=48,
            shift_steps=36,
            rtol=1e-7,
        )
        expected = reference(h, linear, 0.1)
        torch.testing.assert_close(
            got.displacement, expected.displacement, atol=1e-7, rtol=1e-5
        )
        assert got.relative_residual <= 1e-7 and got.relative_complementarity <= 1e-7


def test_budget_failure_is_explicit_and_deferred():
    h = torch.diag(torch.tensor([0.01, 1.0, 100.0], dtype=torch.float64))
    problem = SimpleNamespace(
        operator=DenseAction(h), linear=torch.ones(1, 3, dtype=h.dtype)
    )
    with pytest.raises(FloatingPointError, match="KKT"):
        solve(problem, radius=0.25, max_iter=1, shift_steps=1, rtol=1e-10)
    with device_checks() as checks:
        result = solve(problem, radius=0.25, max_iter=1, shift_steps=1, rtol=1e-10)
    assert not result.converged and not torch.stack(checks).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_matrix_free_and_atomic_failure(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from torchcst._derivatives.tangent import TangentGeometry

    site, baseline = site_for("amplitude").to(device), site_for("amplitude").to(device)
    options = {
        "lr": 0.05,
        "trust_radius": 0.01,
        "recompression": "pcg",
        "first_moment_damping": 0.1,
        "recompression_rtol": 1e-10,
        "recompression_max_iter": 48,
    }
    opt = CSTAdam(
        site,
        device_execution=True,
        update_solver="pcg",
        update_max_iter=48,
        update_shift_steps=36,
        update_rtol=1e-7,
        **options,
    )
    eager = CSTAdam(baseline, **options)
    x = torch.randn(3, 6, dtype=torch.float64, device=device)
    original = TangentGeometry.cross

    def forbid(*args, **kwargs):
        raise AssertionError("full Gram path was used")

    for _ in range(2):
        eager.zero_grad()
        baseline(x).square().mean().backward()
        eager.step()
        with monkeypatch.context() as patch:
            patch.setattr(TangentGeometry, "cross", forbid)
            opt.zero_grad()
            site(x).square().mean().backward()
            opt.step()
        opt.check_errors()
        torch.testing.assert_close(site.atoms.p, baseline.atoms.p, atol=1e-8, rtol=1e-6)
    assert TangentGeometry.cross is original
    before, saved = (
        site.atoms.p.detach().clone(),
        opt._sites[0].state.first.alpha.clone(),
    )
    opt.zero_grad()
    (site(x) * float("nan")).sum().backward()
    opt.step()
    torch.testing.assert_close(site.atoms.p, before, atol=0, rtol=0)
    torch.testing.assert_close(opt._sites[0].state.first.alpha, saved, atol=0, rtol=0)
    with pytest.raises(FloatingPointError):
        opt.check_errors()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"update_solver": "unknown"},
        {"update_max_iter": 0},
        {"update_shift_steps": True},
        {"update_rtol": float("nan")},
        {"update_solver": "pcg", "second_moment": "atom_diag"},
        {"update_solver": "pcg", "factored_geometry": False},
    ],
)
def test_update_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        FirstOrderAdamConfig(**kwargs)


def test_pcg_update_checkpoint_resume():
    import copy

    site = site_for("amplitude")
    options = {
        "device_execution": True,
        "update_solver": "pcg",
        "update_max_iter": 48,
        "update_shift_steps": 36,
        "update_rtol": 1e-7,
        "lr": 0.05,
        "trust_radius": 0.01,
        "recompression": "pcg",
        "first_moment_damping": 0.1,
    }
    optimizer = CSTAdam(site, **options)
    x = torch.randn(3, 6, dtype=torch.float64)
    optimizer.zero_grad()
    site(x).square().mean().backward()
    optimizer.step()
    optimizer.check_errors()
    model_state, state = (
        copy.deepcopy(site.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    restored = site_for("amplitude")
    restored.load_state_dict(model_state)
    resumed = CSTAdam(restored, **options)
    resumed.load_state_dict(state)
    for model, opt in ((site, optimizer), (restored, resumed)):
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
        opt.check_errors()
    torch.testing.assert_close(site.atoms.p, restored.atoms.p, atol=0, rtol=0)


def test_insufficient_update_budget_suppresses_joint_commit():
    site = site_for("amplitude")
    opt = CSTAdam(
        site,
        device_execution=True,
        update_solver="pcg",
        update_max_iter=1,
        update_shift_steps=1,
        update_rtol=1e-12,
        lr=0.05,
        trust_radius=0.01,
        recompression="pcg",
        first_moment_damping=0.1,
    )
    x = torch.randn(3, 6, dtype=torch.float64)
    p, alpha = site.atoms.p.detach().clone(), opt._sites[0].state.first.alpha.clone()
    opt.zero_grad()
    site(x).square().mean().backward()
    opt.step()
    assert not opt.last_step.site_results[0].converged
    with pytest.raises(FloatingPointError):
        opt.check_errors()
    torch.testing.assert_close(site.atoms.p, p, atol=0, rtol=0)
    torch.testing.assert_close(opt._sites[0].state.first.alpha, alpha, atol=0, rtol=0)


def test_near_boundary_refines_ambiguous_inner_tolerance(monkeypatch):
    import torchcst.optim._trust_pcg as module

    # The first close shift lands just outside the ball. A valid but coarse
    # inner residual cannot distinguish the radius side until refined.
    h = torch.diag(torch.tensor([0.0, 100.0], dtype=torch.float64))
    problem = SimpleNamespace(
        operator=DenseAction(h), linear=torch.tensor([[-0.000977, -1.0]], dtype=h.dtype)
    )
    tolerances = []

    def controlled_inner(
        action, rhs, blocks, shift, initial, enabled, *history, **kwargs
    ):
        tolerance = kwargs["tolerance"]
        tolerances.append(float(tolerance))
        if shift == 0:
            x = rhs.clone()  # inconsistent zero-shift system
        else:
            x = torch.linalg.solve(h + shift * torch.eye(2), rhs.flatten())[None]
            x[:, 1] += 0.9 * tolerance * rhs.norm() / (100 + shift)
        residual = action(x) + shift * x - rhs
        count = torch.tensor(1, dtype=torch.int32)
        ok = residual.norm() <= tolerance * rhs.norm()
        return x, residual, count, ok, count, -residual, torch.zeros_like(x), shift * 0

    monkeypatch.setattr(module, "linear_solve", controlled_inner)
    result = solve(problem, radius=1.0, max_iter=64, shift_steps=80, rtol=1e-5)
    assert result.converged
    assert min(tolerances) < 0.25e-5
    expected = reference(h, problem.linear, 1.0)
    torch.testing.assert_close(
        result.objective, expected.objective, atol=1e-5, rtol=1e-7
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_weighted_action_handles_row_and_feature_tails():
    from torchcst import Chart, CSTLinear
    from torchcst.kernels import Amplitude, Gaussian, Separable

    torch.manual_seed(14)
    site = CSTLinear(
        Chart.linspace(129),
        Chart.linspace(19),
        atoms=7,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.4), output_profile=Gaussian(0.5))
        ),
        dtype=torch.float64,
        device="cuda",
    )
    p = site.atoms.p.detach().clone()
    p[0, 0] = 0
    ops = site.cst_derivatives().tangent_ops(execution="triton")
    metric = SeparableDiagonalMetric(
        torch.rand(19, dtype=p.dtype, device=p.device),
        torch.rand(129, dtype=p.dtype, device=p.device),
        eps=0.13,
    )
    action = from_prepared(ops.prepare(p), metric, 0.3)
    x = torch.randn_like(p)
    j = jacobian(site, p)
    expected = j.T @ (metric.diagonal().flatten() * (j @ x.flatten())) / 0.3
    torch.testing.assert_close(action(x).flatten(), expected, atol=1e-9, rtol=1e-10)
    assert action(x, torch.tensor(False, device="cuda")).count_nonzero() == 0
