import pytest
import torch
from test_quartic import make_problem


@pytest.mark.parametrize("corrections", [1, 2, 4])
def test_ray_solver_retains_original_feasible_objective(corrections):
    from torchcst import DeviceRay

    problem = make_problem()
    out = DeviceRay(corrections=corrections).solve(problem, trust_radius=0.25)
    assert out.displacement.norm() <= 0.25 * (1 + 1e-12)
    assert out.objective <= problem.value(torch.zeros_like(out.displacement))
    torch.testing.assert_close(out.objective, problem.value(out.displacement))
    assert out.evaluations == corrections + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_ray_result_matches_original_objective(dtype):
    from test_quadratic_feature_gram import make_pair

    from torchcst import DeviceRay

    problem, _ = make_pair(dtype, True, device="cuda")
    solver = DeviceRay(corrections=4)
    result = solver.solve(problem, trust_radius=0.25)
    expected, gradient = problem.value_and_gradient(result.displacement)
    torch.testing.assert_close(result.objective, expected)
    projected = result.displacement - gradient
    projected *= (0.25 / projected.norm().clamp_min(1e-30)).clamp(max=1)
    torch.testing.assert_close(
        result.projected_gradient_norm, (result.displacement - projected).norm()
    )
    assert result.displacement.norm() <= 0.25 * (1 + 10 * torch.finfo(dtype).eps)
