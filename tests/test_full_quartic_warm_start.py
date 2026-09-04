from __future__ import annotations

from typing import Any

import pytest
import torch

from torchcst.optim import FullQuartic, QuarticProblem, QuarticSolveResult


def test_full_quartic_can_ignore_a_supplied_warm_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = FullQuartic(starts=1, warm_start=False)
    problem = object.__new__(QuarticProblem)
    supplied = torch.full((2, 3), 0.1)
    displacement = torch.zeros_like(supplied)
    candidate = QuarticSolveResult(
        displacement=displacement,
        objective=torch.tensor(0.0),
        projected_gradient_norm=torch.tensor(0.0),
        start_index=0,
        iterations=0,
        evaluations=1,
        converged=True,
        on_boundary=False,
    )
    captured: dict[str, Any] = {}

    def initial_points(
        received_problem: QuarticProblem,
        radius: float,
        *,
        initial: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        captured.update(problem=received_problem, radius=radius, initial=initial)
        return (displacement,)

    def solve_one(*args: Any, **kwargs: Any) -> QuarticSolveResult:
        return candidate

    monkeypatch.setattr(solver, "_initial_points", initial_points)
    monkeypatch.setattr(solver, "_solve_one", solve_one)

    actual = solver.solve(problem, trust_radius=0.5, initial=supplied)

    assert actual is candidate
    assert captured == {"problem": problem, "radius": 0.5, "initial": None}


def test_full_quartic_rejects_non_boolean_warm_start() -> None:
    with pytest.raises(TypeError, match="warm_start must be a boolean"):
        FullQuartic(warm_start=1)  # type: ignore[arg-type]
