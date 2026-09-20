from __future__ import annotations

import pytest
import torch

from torchcst import Amplitude, Chart, CSTLinear, Gaussian, Separable
from torchcst.optim import (
    CSTNormalizedSGD,
    CSTQuadraticAdam,
    CSTQuadraticMomentum,
    CSTQuadraticRMSProp,
    CSTQuadraticSGD,
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
