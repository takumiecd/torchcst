import torch

from experiments.unconstrained_update import solve


def test_direct_update_matches_regularized_oracle_without_clipping():
    metric = torch.tensor([[[2.0, 0.3], [0.3, 1.0]]], dtype=torch.float64)
    linear = torch.tensor([[10.0, -20.0]], dtype=torch.float64)
    result = solve(metric, linear, 0.05, 0.01)
    expected = torch.linalg.solve(
        metric[0] + 0.01 * torch.eye(2, dtype=torch.float64), -0.05 * linear[0]
    )
    torch.testing.assert_close(result.displacement[0], expected)
    assert result.displacement.norm() > 0.25
    assert bool(result.converged) and not result.on_boundary


def test_regularization_handles_null_metric_direction():
    metric = torch.diag_embed(torch.tensor([[0.0, 2.0]], dtype=torch.float64))
    linear = torch.ones(1, 2, dtype=torch.float64)
    result = solve(metric, linear, 0.01, 0.1)
    torch.testing.assert_close(
        result.displacement, torch.tensor([[-0.1, -0.01 / 2.1]], dtype=torch.float64)
    )
