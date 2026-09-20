import pytest
import torch

from torchcst import Amplitude, Atoms, Chart, CSTLinear, Gaussian, Separable
from torchcst.atoms import AtomGrad
from torchcst.optim import AtomGradRequest, LinearJGAtomGrad, LinearJGHAtomGrad


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
            derivatives.contracted_hessian(
                represented_gradient, parameter_point=point
            ),
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
