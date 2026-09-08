"""Independent device-execution oracles and failure-atomicity contracts."""

import pytest
import torch

from tests.test_tangent_ops import jacobian, site_for
from torchcst import CSTAdam
from torchcst._derivatives.tangent_device import block_inverse, solve
from torchcst._runtime.validation import device_checks
from torchcst.optim._tangent_device import spectral_ball
from torchcst.optim.quadratic import TangentProblem, solve_tangent


def test_block_inverse_matches_spd_solve_and_rejects_bad_pivot():
    torch.manual_seed(15)
    a = torch.randn(5, 4, 4, dtype=torch.float64)
    a = a @ a.transpose(-1, -2)
    inverse, valid = block_inverse(a, 0.01)
    assert valid
    torch.testing.assert_close(
        inverse,
        torch.linalg.inv(a + 0.01 * torch.eye(4, dtype=a.dtype)),
        rtol=1e-10,
        atol=1e-10,
    )
    _, bad = block_inverse(-torch.eye(3)[None], 0.01)
    assert not bad


@pytest.mark.parametrize("radius", [0.01, 10.0])
def test_tensor_secular_matches_reference_with_null_direction(radius):
    problem = object.__new__(TangentProblem)
    problem.point_shape = (1, 3)
    problem.matrix = torch.diag(torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64))
    problem.linear = torch.tensor([[1.0, -0.2, 0.1]], dtype=torch.float64)
    problem.blocked = False
    expected = solve_tangent(problem, radius=radius)
    solution, shift = spectral_ball(
        problem.matrix.diagonal(), -problem.linear.flatten(), radius
    )
    torch.testing.assert_close(
        solution, expected.displacement.flatten(), rtol=1e-10, atol=1e-10
    )
    assert shift >= 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_pcg_matches_damped_oracle(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    site = site_for("bandwidth", dtype).to(device)
    p = site.atoms.p.detach().clone()
    ops = site.cst_derivatives().tangent_ops(
        execution="triton" if device == "cuda" else "eager"
    )
    prepared = ops.prepare(p)
    rhs = torch.randn_like(p)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    result, count, residual, valid = solve(
        prepared, rhs, damping=0.1, max_iter=48, rtol=tolerance
    )
    j = jacobian(site, p)
    expected = torch.linalg.solve(
        j.T @ j + 0.1 * torch.eye(p.numel(), dtype=dtype, device=device), rhs.flatten()
    )
    assert valid and residual <= tolerance and count <= 48
    torch.testing.assert_close(
        result.flatten(), expected, rtol=10 * tolerance, atol=10 * tolerance
    )
    # Graph replay must use changing RHS, not cached solutions.
    result2, _, _, valid2 = solve(
        prepared, 2 * rhs, damping=0.1, max_iter=48, rtol=tolerance
    )
    assert valid2
    torch.testing.assert_close(
        result2, 2 * result, rtol=20 * tolerance, atol=20 * tolerance
    )
    zero, steps, err, ok = solve(
        prepared, torch.zeros_like(rhs), damping=0.1, max_iter=48, rtol=tolerance
    )
    assert ok and steps == 0 and err == 0 and zero.count_nonzero() == 0
    # A new point with the same layout must refresh every captured factor.
    changed = p + 0.03 * torch.randn_like(p)
    refreshed, _, _, valid = solve(
        ops.prepare(changed), rhs, damping=0.1, max_iter=48, rtol=tolerance
    )
    jc = jacobian(site, changed)
    expected = torch.linalg.solve(
        jc.T @ jc + 0.1 * torch.eye(p.numel(), dtype=dtype, device=device),
        rhs.flatten(),
    )
    assert valid
    torch.testing.assert_close(
        refreshed.flatten(), expected, rtol=10 * tolerance, atol=10 * tolerance
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "coeff", [[1.0, -0.2, 0.1], [0.0, 0.01, -0.01], [0.0, 0.0, 0.0]]
)
@pytest.mark.parametrize("radius", [0.01, 10.0])
def test_fused_secular_matches_tensor_oracle(coeff, radius):
    from torchcst._derivatives.tangent_triton import spectral_solution

    values = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64, device="cuda")
    coeff = torch.tensor(coeff, dtype=torch.float64, device="cuda")
    actual = spectral_solution(values, coeff, radius)
    expected = spectral_ball(values, coeff, radius)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["amplitude", "bandwidth"])
def test_fused_cross_and_weighted_match_autograd(kind):
    from torchcst.optim.moments import SeparableDiagonalMetric

    site = site_for(kind).cuda()
    p = site.atoms.p.detach().clone()
    ops = site.cst_derivatives().tangent_ops(execution="triton")
    old = p + 0.05 * torch.randn_like(p)
    a, b = ops.prepare(p), ops.prepare(old)
    x = torch.randn_like(p)
    j, jo = jacobian(site, p), jacobian(site, old)
    metric = SeparableDiagonalMetric(
        torch.rand(4, device="cuda", dtype=p.dtype),
        torch.rand(6, device="cuda", dtype=p.dtype),
        eps=0.12,
    )
    torch.testing.assert_close(
        a.cross_gram_matvec(b, x).flatten(),
        j.T @ jo @ x.flatten(),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        a.weighted_gram_matvec(metric, x).flatten(),
        j.T @ (metric.diagonal().flatten() * (j @ x.flatten())),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_optimizer_matches_eager_then_latches_failure(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    site = site_for("amplitude").to(device)
    baseline = site_for("amplitude").to(device)
    options = {
        "first_moment_damping": 0.1,
        "recompression": "pcg",
        "recompression_max_iter": 48,
        "recompression_rtol": 1e-10,
    }
    opt = CSTAdam(site, device_execution=True, **options)
    eager = CSTAdam(baseline, **options)
    x = torch.randn(3, 6, dtype=torch.float64, device=device)
    for _ in range(2):
        for model, optimizer in [(site, opt), (baseline, eager)]:
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
        opt.check_errors()
        torch.testing.assert_close(site.atoms.p, baseline.atoms.p, rtol=1e-7, atol=1e-9)
    before = site.atoms.p.detach().clone()
    saved = opt._sites[0].state.first.alpha.clone()
    opt.zero_grad()
    (site(x) * float("nan")).sum().backward()
    opt.step()
    torch.testing.assert_close(site.atoms.p, before, rtol=0, atol=0)
    torch.testing.assert_close(opt._sites[0].state.first.alpha, saved, rtol=0, atol=0)
    with pytest.raises(FloatingPointError):
        opt.check_errors()
    opt.zero_grad()
    site(x).square().mean().backward()
    opt.step()
    torch.testing.assert_close(site.atoms.p, before, rtol=0, atol=0)


def test_device_checks_defer_nonfinite_preparation():
    site = site_for("amplitude")
    ops = site.cst_derivatives().tangent_ops()
    with device_checks() as checks:
        ops.prepare(torch.full_like(site.atoms.p, float("nan")))
    assert not torch.stack(checks).all()
