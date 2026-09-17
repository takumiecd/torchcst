from __future__ import annotations

import torch

from torchcst import Amplitude, Chart, CSTLinear, Gaussian, Separable
from torchcst.optim import (
    AtomGradRequest,
    CSTSGD,
    CSTMomentum,
    CSTNormalizedAdam,
    CSTRMSProp,
    LinearJGAtomGrad,
    LinearJGHAtomGrad,
    NormalizedBoxFixedPointSolver,
    NormalizedSolver,
    NormalizedSolveResult,
)


def make_site() -> CSTLinear:
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


def take_step(optimizer, model) -> None:
    inputs = torch.tensor([[1.0, -0.5], [0.2, 0.7]], dtype=torch.float64)
    targets = torch.tensor([[0.4], [-0.1]], dtype=torch.float64)
    optimizer.zero_grad()
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()


class FixedNormalizedSolver(NormalizedSolver):
    def __init__(self, displacement: torch.Tensor) -> None:
        self.displacement = displacement

    def solve(self, problem, *, trust_radius: float) -> NormalizedSolveResult:
        displacement = self.displacement.to(device=problem.device, dtype=problem.dtype)
        norm = torch.linalg.vector_norm(displacement)
        if norm > trust_radius:
            displacement = displacement * (trust_radius / norm)
        return NormalizedSolveResult(
            displacement=displacement.clone(),
            residual_norm=torch.zeros((), device=displacement.device, dtype=displacement.dtype),
            iterations=1,
            converged=True,
            on_boundary=bool(norm >= trust_radius),
        )


def test_all_wrappers_run_with_the_shared_normalized_coordinator() -> None:
    for optimizer_type in (CSTSGD, CSTMomentum, CSTRMSProp, CSTNormalizedAdam):
        model = make_site()
        optimizer = optimizer_type(
            model,
            lr=0.01,
            betas=(0.8, 0.9),
            trust_radius=0.2,
        )
        before = model.atoms.p.detach().clone()
        take_step(optimizer, model)

        assert optimizer.last_step is not None
        assert optimizer.last_step.site_results[0].converged
        assert optimizer._sites[0].state.step == 1
        assert not torch.equal(model.atoms.p, before)


def test_normalized_solver_is_injected_without_changing_the_coordinator() -> None:
    model = make_site()
    direction = torch.tensor([[0.01, -0.02, 0.03]], dtype=torch.float64)
    optimizer = CSTNormalizedAdam(
        model,
        lr=0.1,
        trust_radius=0.2,
        solver=FixedNormalizedSolver(direction),
    )
    before = model.atoms.p.detach().clone()
    take_step(optimizer, model)

    torch.testing.assert_close(model.atoms.p, before + direction)
    assert optimizer.state_dict()["cst"]["<root>"].step == 1


def test_numerator_and_denominator_states_are_kept_separately() -> None:
    model = make_site()
    optimizer = CSTNormalizedAdam(model, betas=(0.8, 0.9))
    take_step(optimizer, model)

    state = optimizer.state_dict()["cst"]["<root>"]
    assert state.step == 1
    assert state.numerator.beta_power == 0.8
    assert state.denominator.beta_power == 0.9
    assert state.numerator.m.shape == model.atoms.p.shape
    assert state.denominator.x.shape == model.atoms.p.shape


def test_initial_zero_step_is_used_once_before_implicit_solver() -> None:
    model = make_site()
    optimizer = CSTSGD(
        model,
        lr=0.01,
        trust_radius=0.2,
        solver=NormalizedBoxFixedPointSolver(),
        initial_zero_step=True,
    )

    take_step(optimizer, model)
    assert optimizer.last_step is not None
    assert optimizer.last_step.site_results[0].solver_mode == "zero_point"

    take_step(optimizer, model)
    assert optimizer.last_step is not None
    assert optimizer.last_step.site_results[0].solver_mode == "fixed_point"


def test_max_iter_one_uses_first_order_observations_only() -> None:
    model = make_site()
    optimizer = CSTNormalizedAdam(
        model,
        solver=NormalizedBoxFixedPointSolver(max_iter=1),
    )

    site = optimizer._sites[0]
    assert isinstance(site.atom_grad, LinearJGAtomGrad)
    assert site.atom_grad.observation_request == AtomGradRequest(jg=True)
    assert site.moments.numerator.observation_request == AtomGradRequest(jg=True)
    assert site.moments.denominator.observation_request == AtomGradRequest(jg=True)
    take_step(optimizer, model)
    assert optimizer.last_step is not None
    assert optimizer.last_step.site_results[0].iterations == 1


def test_max_iter_two_keeps_curvature_observations() -> None:
    model = make_site()
    optimizer = CSTNormalizedAdam(
        model,
        solver=NormalizedBoxFixedPointSolver(max_iter=2),
    )

    site = optimizer._sites[0]
    assert isinstance(site.atom_grad, LinearJGHAtomGrad)
    assert site.atom_grad.observation_request == AtomGradRequest(jg=True, gh=True)
    assert site.moments.numerator.observation_request == AtomGradRequest(jg=True, gh=True)
    assert site.moments.denominator.observation_request == AtomGradRequest(jg=True, gh=True)
