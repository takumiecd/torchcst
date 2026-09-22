from __future__ import annotations

import pytest
import torch

from torchcst import Amplitude, Chart, CSTLinear, Gaussian, Separable
from torchcst.optim import (
    AtomGradRequest,
    CSTNormalizedSGD,
    CSTQuadraticAdam,
    CSTQuadraticMomentum,
    CSTQuadraticRMSProp,
    CSTQuadraticSGD,
    CurvatureBlockMask,
    LinearJGAtomGrad,
    NormalizedOptimizerConfig,
    QuadraticGradientSolver,
    QuadraticOptimizerConfig,
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


def test_quadratic_wrappers_take_a_gradient_step_on_the_local_model() -> None:
    for optimizer_type in (
        CSTQuadraticSGD,
        CSTQuadraticMomentum,
        CSTQuadraticRMSProp,
        CSTQuadraticAdam,
    ):
        model = make_site()
        optimizer = optimizer_type(model, lr=0.01, betas=(0.8, 0.9), trust_radius=0.2)
        before = model.atoms.p.detach().clone()
        take_step(optimizer, model)

        assert optimizer.last_step is not None
        assert optimizer.last_step.site_results[0].solver_mode == "gradient"
        assert optimizer._sites[0].state.step == 1
        assert not torch.equal(model.atoms.p, before)


def test_quadratic_wrappers_use_the_normalized_gradient_solver() -> None:
    for optimizer_type in (
        CSTQuadraticSGD,
        CSTQuadraticMomentum,
        CSTQuadraticRMSProp,
        CSTQuadraticAdam,
    ):
        assert isinstance(
            optimizer_type(make_site()).cst_config.solver, QuadraticGradientSolver
        )


def test_quadratic_adam_max_iter_one_uses_first_order_observations_only() -> None:
    model = make_site()
    optimizer = CSTQuadraticAdam(
        model,
        solver=QuadraticGradientSolver(max_iter=1),
    )

    site = optimizer._sites[0]
    assert isinstance(site.atom_grad, LinearJGAtomGrad)
    assert site.atom_grad.observation_request == AtomGradRequest(jg=True)

    take_step(optimizer, model)
    assert optimizer.last_step is not None
    assert optimizer.last_step.site_results[0].iterations == 1


def test_quadratic_rmsprop_and_adam_keep_denominator_state() -> None:
    model = make_site()
    optimizer = CSTQuadraticAdam(model, betas=(0.8, 0.9), eps=1e-6)
    take_step(optimizer, model)

    state = optimizer.state_dict()["cst"]["<root>"]
    assert state.numerator.beta_power == 0.8
    assert state.denominator.beta_power == 0.9
    assert state.denominator.x.shape == model.atoms.p.shape


def test_quadratic_family_rejects_normalized_config() -> None:
    with pytest.raises(TypeError, match="QuadraticOptimizerConfig"):
        CSTQuadraticSGD(make_site(), cst=NormalizedOptimizerConfig())


def test_normalized_sgd_rejects_quadratic_config() -> None:
    with pytest.raises(TypeError, match="NormalizedOptimizerConfig"):
        CSTNormalizedSGD(make_site(), cst=QuadraticOptimizerConfig())


def test_quadratic_and_normalized_families_use_separate_configs() -> None:
    assert CSTQuadraticSGD.config_type is QuadraticOptimizerConfig
    assert CSTQuadraticMomentum.config_type is QuadraticOptimizerConfig
    assert CSTQuadraticRMSProp.config_type is QuadraticOptimizerConfig
    assert CSTQuadraticAdam.config_type is QuadraticOptimizerConfig
    assert CSTNormalizedSGD.config_type is NormalizedOptimizerConfig


def test_quadratic_and_normalized_families_share_the_nd_coordinator() -> None:
    from torchcst.optim.nd_optimizer import _NDModelOptimizer

    assert issubclass(CSTQuadraticSGD, _NDModelOptimizer)
    assert issubclass(CSTNormalizedSGD, _NDModelOptimizer)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
def test_kernel_step_size_must_be_positive(value) -> None:
    with pytest.raises(ValueError, match="kernel_step_size"):
        QuadraticOptimizerConfig(kernel_step_size=value)


def test_kernel_step_size_is_independent_of_inner_learning_rate() -> None:
    config = QuadraticOptimizerConfig(lr=0.00025, kernel_step_size=0.002)
    optimizer = CSTQuadraticAdam(make_site(), cst=config)

    assert optimizer.cst_config.lr == 0.00025
    assert optimizer.cst_config.kernel_step_size == 0.002


@pytest.mark.parametrize("mode", ["full", "no_m", "m_only", "none"])
def test_curvature_mask_is_applied_before_both_adam_moments(mode: str) -> None:
    model = make_site()
    mask = CurvatureBlockMask(split=1, mode=mode)
    optimizer = CSTQuadraticAdam(
        model,
        betas=(0.0, 0.0),
        curvature_mask=mask,
    )

    take_step(optimizer, model)

    observation = optimizer._sites[0].atom_grad.snapshot()
    assert observation.jg is not None
    assert observation.gh is not None
    expected_hessian = mask.apply(observation).gh
    assert expected_hessian is not None
    state = optimizer._sites[0].state
    torch.testing.assert_close(state.numerator.C, expected_hessian)
    torch.testing.assert_close(
        state.denominator.y,
        torch.einsum("ki,kip->kip", observation.jg, expected_hessian),
    )
    torch.testing.assert_close(
        state.denominator.Z,
        torch.einsum("kip,kiq->kipq", expected_hessian, expected_hessian),
    )


def test_curvature_mask_is_part_of_optimizer_state_contract() -> None:
    full = CSTQuadraticAdam(
        make_site(),
        curvature_mask=CurvatureBlockMask(split=1, mode="full"),
    )
    no_m = CSTQuadraticAdam(
        make_site(),
        curvature_mask=CurvatureBlockMask(split=1, mode="no_m"),
    )
    state = full.state_dict()

    with pytest.raises(ValueError, match="moment contract"):
        no_m.load_state_dict(state)


def test_quadratic_config_rejects_non_mask_curvature_mask() -> None:
    with pytest.raises(TypeError, match="curvature_mask"):
        QuadraticOptimizerConfig(curvature_mask="no_m")
