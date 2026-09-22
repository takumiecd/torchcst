import pytest
import torch

from torchcst import Amplitude, Atoms, Chart, CSTLinear, Gaussian, Separable
from torchcst.atoms import AtomGrad
from torchcst.optim import (
    AtomGradientObservation,
    AtomGradRequest,
    CurvatureBlockMask,
    LinearJGAtomGrad,
    LinearJGHAtomGrad,
)


def make_site(*, backend: str = "factored") -> CSTLinear:
    torch.manual_seed(17)
    return CSTLinear(
        Chart.linspace(4, low=-1.0, high=1.0),
        Chart.linspace(3, low=-1.0, high=1.0),
        atoms=2,
        kernel=Amplitude(
            Separable(
                input_profile=Gaussian(0.8),
                output_profile=Gaussian(0.6),
            )
        ),
        backend=backend,
        dtype=torch.float64,
    )


class EmptyAtomGrad(AtomGrad):
    def _clear_values(self) -> None:
        pass

    def _complete_values(self) -> None:
        pass


def test_atoms_owns_but_does_not_serialize_optimizer_grad_program() -> None:
    atoms = Atoms(torch.ones(2, 3))
    grad = EmptyAtomGrad()
    atoms.set_grad(grad)

    assert atoms.grad is grad
    assert "grad" not in atoms.state_dict()


def test_atom_grad_has_an_explicit_scope_lifecycle() -> None:
    atoms = Atoms(torch.ones(2, 3))
    grad = EmptyAtomGrad()
    atoms.set_grad(grad)
    with pytest.raises(RuntimeError, match="not active"):
        grad.complete()
    grad.begin()
    with pytest.raises(RuntimeError, match="already active"):
        grad.begin()
    grad.complete()


def test_active_atom_grad_cannot_be_detached() -> None:
    atoms = Atoms(torch.ones(2, 3))
    grad = EmptyAtomGrad()
    atoms.set_grad(grad)
    grad.begin()
    with pytest.raises(RuntimeError, match="cannot detach"):
        atoms.set_grad(None)


@pytest.mark.parametrize("backend", ["factored", "materialized"])
@pytest.mark.parametrize("mode", ["custom", "hooks"])
@pytest.mark.parametrize("collector_type", [LinearJGAtomGrad, LinearJGHAtomGrad])
def test_nd_collectors_match_dense_derivative_oracles(
    backend: str, mode: str, collector_type
) -> None:
    site = make_site(backend=backend)
    collector = collector_type(mode=mode, factored=backend == "factored")
    site.atoms.set_grad(collector)
    inputs = torch.randn(2, 3, site.in_features, dtype=torch.float64)
    output_gradient = torch.randn(2, 3, site.out_features, dtype=torch.float64)
    point = site.atoms.p.detach().clone()

    collector.begin()
    (site(inputs) * output_gradient).sum().backward()
    collector.complete()

    represented_gradient = output_gradient.reshape(-1, site.out_features).T @ (
        inputs.reshape(-1, site.in_features)
    )
    derivatives = site.cst_derivatives()
    observation = collector.snapshot()
    torch.testing.assert_close(
        observation.jg,
        derivatives.pullback(represented_gradient, parameter_point=point),
    )
    if collector_type is LinearJGHAtomGrad:
        torch.testing.assert_close(
            observation.gh,
            derivatives.contracted_hessian(represented_gradient, parameter_point=point),
        )
        assert collector.observation_request == AtomGradRequest(jg=True, gh=True)
    else:
        assert observation.gh is None
        assert collector.observation_request == AtomGradRequest(jg=True)


def test_repeated_uses_accumulate_one_observation() -> None:
    site = make_site()
    collector = LinearJGAtomGrad(mode="custom", factored=True)
    site.atoms.set_grad(collector)
    x1 = torch.randn(4, site.in_features, dtype=torch.float64)
    x2 = torch.randn(2, site.in_features, dtype=torch.float64)
    g1 = torch.randn(4, site.out_features, dtype=torch.float64)
    g2 = torch.randn(2, site.out_features, dtype=torch.float64)

    collector.begin()
    ((site(x1) * g1).sum() + (site(x2) * g2).sum()).backward()
    collector.complete()

    represented_gradient = g1.T @ x1 + g2.T @ x2
    expected = site.cst_derivatives().pullback(represented_gradient)
    torch.testing.assert_close(collector.snapshot().jg, expected)
    assert collector.contributions == 2


def test_collector_rejects_trainable_charts() -> None:
    site = CSTLinear(
        Chart.linspace(4, low=-1.0, high=1.0, trainable=True),
        Chart.linspace(3, low=-1.0, high=1.0),
        atoms=2,
        kernel=Amplitude(
            Separable(
                input_profile=Gaussian(0.8),
                output_profile=Gaussian(0.6),
            )
        ),
        dtype=torch.float64,
    )
    collector = LinearJGAtomGrad()
    site.atoms.set_grad(collector)
    collector.begin()
    with pytest.raises(ValueError, match="frozen charts"):
        site(torch.randn(2, site.in_features, dtype=torch.float64)).sum().backward()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("full", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        ("no_m", [[1.0, 0.0, 0.0], [0.0, 5.0, 6.0], [0.0, 8.0, 9.0]]),
        ("m_only", [[0.0, 2.0, 3.0], [4.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        ("none", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_curvature_block_mask_selects_requested_hessian_blocks(
    mode: str, expected: list[list[float]]
) -> None:
    gradient = torch.tensor([[10.0, 11.0, 12.0]])
    hessian = torch.arange(1.0, 10.0).reshape(1, 3, 3)
    observation = AtomGradientObservation(
        jg=gradient,
        gh=hessian,
        contributions=2,
    )

    masked = CurvatureBlockMask(split=1, mode=mode).apply(observation)

    assert masked.jg is gradient
    assert masked.contributions == 2
    torch.testing.assert_close(masked.gh, torch.tensor([expected]))
    if mode != "full":
        assert masked.gh is not hessian


def test_curvature_block_mask_rejects_invalid_split_for_observation() -> None:
    observation = AtomGradientObservation(
        jg=torch.ones(1, 2),
        gh=torch.ones(1, 2, 2),
    )
    with pytest.raises(ValueError, match="smaller than P"):
        CurvatureBlockMask(split=2, mode="no_m").apply(observation)


@pytest.mark.parametrize("split", [True, 0, -1, 1.5])
def test_curvature_block_mask_rejects_invalid_constructor_split(split) -> None:
    error = TypeError if split in (True, 1.5) else ValueError
    with pytest.raises(error, match="split"):
        CurvatureBlockMask(split=split)
