import torch
from test_quartic import make_problem

from experiments.mnist_solver_comparison import fresh_problem


def test_replay_uses_fresh_cache_and_does_not_mutate_state():
    original = make_problem()
    point = original.context.current_point.clone()
    displacement = torch.zeros_like(point)
    expected = original.value(displacement).detach().clone()
    replay = fresh_problem(original, "visible")
    assert replay.context.geometry is not original.context.geometry
    assert replay.context.geometry._cache_point is None
    from torchcst import BallNewton
    BallNewton(max_iter=2, max_evaluations=10).solve(replay, trust_radius=0.25)
    torch.testing.assert_close(original.context.current_point, point)
    torch.testing.assert_close(original.value(displacement), expected)
