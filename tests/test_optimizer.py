from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from torchcst import (
    AdamWConfig,
    Amplitude,
    Chart,
    CSTLinear,
    CSTSecondOrderAdam,
    FullQuartic,
    Gaussian,
    SecondOrderAdamConfig,
    Separable,
)
from torchcst.optim import (
    DenseAdamWState,
    FunctionalAdamW,
    QuarticProblem,
    QuarticSolver,
    QuarticSolveResult,
)


def test_implicit_adam_keeps_full_quartic_as_its_default() -> None:
    assert isinstance(SecondOrderAdamConfig().quartic, FullQuartic)


class FixedDirectionSolver(QuarticSolver):
    def __init__(self, direction: Tensor) -> None:
        self.direction = direction

    def solve(
        self,
        problem: QuarticProblem,
        *,
        trust_radius: float,
    ) -> QuarticSolveResult:
        direction = self.direction.to(problem.context.current_point)
        norm = torch.linalg.vector_norm(direction)
        if norm > trust_radius:
            direction = direction * (trust_radius / norm)
        objective = problem.value(direction).detach()
        return QuarticSolveResult(
            displacement=direction.clone(),
            objective=objective,
            projected_gradient_norm=objective.new_zeros(()),
            start_index=0,
            iterations=0,
            evaluations=1,
            converged=True,
            on_boundary=bool(norm >= trust_radius),
        )


class MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cst = make_site()
        self.bias = nn.Parameter(torch.tensor([0.2], dtype=torch.float64))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.cst(inputs) + self.bias


def make_site() -> CSTLinear:
    torch.manual_seed(41)
    return CSTLinear(
        Chart.linspace(2, low=-1.0, high=1.0),
        Chart.linspace(1, low=-1.0, high=1.0),
        atoms=1,
        kernel=Amplitude(
            Separable(
                input_profile=Gaussian(0.8),
                output_profile=Gaussian(0.7),
            )
        ),
        backend="materialized",
        dtype=torch.float64,
    )


def fixed_config(direction: Tensor) -> SecondOrderAdamConfig:
    return SecondOrderAdamConfig(
        lr=0.05,
        betas=(0.0, 0.0),
        trust_radius=0.2,
        quartic=FixedDirectionSolver(direction),
    )


def batch() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([[1.0, -0.5], [0.2, 0.7]], dtype=torch.float64),
        torch.tensor([[0.4], [-0.1]], dtype=torch.float64),
    )


def test_functional_adamw_matches_one_torch_adamw_step_without_mutation() -> None:
    config = AdamWConfig(lr=0.03, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.2)
    parameter = nn.Parameter(torch.tensor([1.2, -0.4], dtype=torch.float64))
    reference = nn.Parameter(parameter.detach().clone())
    gradient = torch.tensor([0.3, -0.8], dtype=torch.float64)
    parameter.grad = gradient.clone()
    reference.grad = gradient.clone()
    engine = FunctionalAdamW(config)
    state = engine.initialize(parameter)
    before = parameter.detach().clone()

    proposal = engine.expand(parameter, state)
    torch.optim.AdamW(
        [reference],
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    ).step()

    torch.testing.assert_close(parameter, before)
    torch.testing.assert_close(before + proposal.displacement, reference)
    assert state.step == 0
    assert proposal.pending_state.step == 1


def test_step_commits_full_cst_and_dense_proposals_together() -> None:
    model = MixedModel()
    direction = torch.tensor([[0.02, -0.01, 0.01]], dtype=torch.float64)
    optimizer = CSTSecondOrderAdam(
        model,
        cst=fixed_config(direction),
        dense=AdamWConfig(lr=0.02, weight_decay=0.0),
    )
    inputs, targets = batch()
    atom_before = model.cst.atoms.p.detach().clone()
    bias_before = model.bias.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    returned = optimizer.step()
    state = optimizer.state_dict()

    assert returned is None
    assert optimizer.last_step is not None
    torch.testing.assert_close(model.cst.atoms.p, atom_before + direction)
    assert not torch.equal(model.bias, bias_before)
    assert state["cst"]["cst"].step == 1
    assert state["dense"]["bias"].step == 1


def test_step_compresses_at_the_full_solver_displacement() -> None:
    model = MixedModel()
    direction = torch.tensor([[0.02, -0.01, 0.01]], dtype=torch.float64)
    optimizer = CSTSecondOrderAdam(
        model,
        cst=fixed_config(direction),
        dense=AdamWConfig(lr=0.02, weight_decay=0.0),
    )
    inputs, targets = batch()
    atom_before = model.cst.atoms.p.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    state = optimizer.state_dict()["cst"]["cst"]

    torch.testing.assert_close(model.cst.atoms.p, atom_before + direction)
    torch.testing.assert_close(state.first.frame.displacement, direction)


def test_step_requires_zero_grad_to_open_the_observation_scope() -> None:
    model = make_site()
    optimizer = CSTSecondOrderAdam(
        model,
        cst=fixed_config(torch.zeros(1, 3, dtype=torch.float64)),
        dense=None,
    )

    with pytest.raises(RuntimeError, match="AtomGrad is not active"):
        optimizer.step()


def test_dense_parameters_require_an_explicit_dense_configuration() -> None:
    with pytest.raises(ValueError, match="dense=None"):
        CSTSecondOrderAdam(
            MixedModel(),
            cst=fixed_config(torch.zeros(1, 3, dtype=torch.float64)),
            dense=None,
        )


def test_state_dict_round_trip_restores_compact_and_dense_state() -> None:
    model = MixedModel()
    direction = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
    optimizer = CSTSecondOrderAdam(
        model,
        cst=fixed_config(direction),
        dense=AdamWConfig(),
    )
    inputs, targets = batch()

    def take_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()

    take_step()
    saved = optimizer.state_dict()
    saved_cst_step = saved["cst"]["cst"].step
    saved_dense_step = saved["dense"]["bias"].step
    take_step()

    optimizer.load_state_dict(saved)
    restored = optimizer.state_dict()

    assert restored["cst"]["cst"].step == saved_cst_step == 1
    assert restored["dense"]["bias"].step == saved_dense_step == 1


def test_state_dict_rejects_a_different_ownership_manifest() -> None:
    model = MixedModel()
    optimizer = CSTSecondOrderAdam(
        model,
        cst=fixed_config(torch.zeros(1, 3, dtype=torch.float64)),
        dense=AdamWConfig(),
    )
    state = optimizer.state_dict()
    state["manifest"][0]["shape"] = (999,)

    with pytest.raises(ValueError, match="manifest"):
        optimizer.load_state_dict(state)


@dataclass(frozen=True)
class _WrongDenseState:
    step: int


def test_dense_state_validation_rejects_the_wrong_state_type() -> None:
    model = MixedModel()
    optimizer = CSTSecondOrderAdam(
        model,
        cst=fixed_config(torch.zeros(1, 3, dtype=torch.float64)),
        dense=AdamWConfig(),
    )
    state = optimizer.state_dict()
    state["dense"]["bias"] = _WrongDenseState(step=0)

    with pytest.raises(TypeError, match="DenseAdamWState"):
        optimizer.load_state_dict(state)

    assert isinstance(optimizer.state_dict()["dense"]["bias"], DenseAdamWState)
