import pytest
import torch

from torchcst import Amplitude, Atoms, Chart, CSTLinear, Gaussian, Separable
from torchcst.atoms import AtomGrad
from torchcst.nn import LinearAtomGrad
from torchcst.optim import (
    AtomGradRequest,
    ImplicitLinearAtomGrad,
    LinearJGAtomGrad,
    LinearJGHAtomGrad,
)


def make_site(*, backend: str = "factored") -> CSTLinear:
    torch.manual_seed(17)
    return CSTLinear(
        Chart.linspace(4),
        Chart.linspace(3),
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


class HookOnlyLinearAtomGrad(LinearAtomGrad):
    def __init__(self, *, mode: str) -> None:
        super().__init__(mode=mode)
        self.calls = 0

    def _clear_values(self) -> None:
        self.calls = 0

    def _complete_values(self) -> None:
        pass

    def _accumulate_linear(
        self,
        site: CSTLinear,
        inputs: torch.Tensor,
        output_gradient: torch.Tensor,
        *,
        parameter_gradient: torch.Tensor | None,
        generation: int,
    ) -> None:
        if self.accepts(generation=generation):
            self.calls += 1


def test_atoms_owns_but_does_not_serialize_optimizer_grad_program() -> None:
    atoms = Atoms(torch.ones(2, 3))
    state_keys = tuple(atoms.state_dict())
    grad = EmptyAtomGrad()

    atoms.set_grad(grad)

    assert atoms.grad is grad
    assert grad.atoms is atoms
    assert tuple(atoms.state_dict()) == state_keys
    assert all("grad" not in key for key in state_keys)

    atoms.set_grad(None)
    with pytest.raises(RuntimeError, match="not attached"):
        _ = grad.atoms


def test_atom_grad_has_an_explicit_scope_lifecycle() -> None:
    atoms = Atoms(torch.ones(2, 3))
    grad = EmptyAtomGrad()
    atoms.set_grad(grad)

    with pytest.raises(RuntimeError, match="not active"):
        grad.complete()

    grad.begin()
    generation = grad.generation
    assert grad.active
    assert grad.accepts(generation=generation)

    grad.complete()
    assert grad.completed
    assert not grad.accepts(generation=generation)


def test_active_atom_grad_cannot_be_detached() -> None:
    atoms = Atoms(torch.ones(2, 3))
    grad = EmptyAtomGrad()
    atoms.set_grad(grad)
    grad.begin()

    with pytest.raises(RuntimeError, match="cannot detach"):
        atoms.set_grad(None)


@pytest.mark.parametrize("backend", ["factored", "materialized"])
def test_custom_and_hook_routes_match_dense_oracles(backend: str) -> None:
    custom_site = make_site(backend=backend)
    hook_site = make_site(backend=backend)
    hook_site.load_state_dict(custom_site.state_dict())
    custom_grad = ImplicitLinearAtomGrad(mode="custom", row_chunk_size=2)
    hook_grad = ImplicitLinearAtomGrad(mode="hooks", row_chunk_size=2)
    custom_site.atoms.set_grad(custom_grad)
    hook_site.atoms.set_grad(hook_grad)

    inputs = torch.randn(2, 3, custom_site.in_features, dtype=torch.float64)
    output_gradient = torch.randn(
        2, 3, custom_site.out_features, dtype=torch.float64
    )
    custom_inputs = inputs.clone().requires_grad_(True)
    hook_inputs = inputs.clone().requires_grad_(True)
    parameter_point = custom_site.atoms.p.detach().clone()

    custom_grad.begin()
    hook_grad.begin()
    (custom_site(custom_inputs) * output_gradient).sum().backward()
    (hook_site(hook_inputs) * output_gradient).sum().backward()
    custom_grad.complete()
    hook_grad.complete()

    represented_gradient = output_gradient.reshape(-1, custom_site.out_features).T @ (
        inputs.reshape(-1, custom_site.in_features)
    )
    derivatives = custom_site.cst_derivatives()
    expected_jg = derivatives.pullback(
        represented_gradient, parameter_point=parameter_point
    )
    expected_gh = derivatives.contracted_hessian(
        represented_gradient, parameter_point=parameter_point
    )
    expected_r = represented_gradient.square().mean(dim=1)
    expected_c = represented_gradient.square().mean(dim=0)

    assert custom_grad.last_route == "custom"
    assert hook_grad.last_route == "hooks"
    torch.testing.assert_close(custom_site.atoms.p.grad, expected_jg)
    torch.testing.assert_close(hook_site.atoms.p.grad, expected_jg)
    torch.testing.assert_close(custom_inputs.grad, hook_inputs.grad)
    torch.testing.assert_close(custom_grad.jg, expected_jg)
    torch.testing.assert_close(custom_grad.gh, expected_gh)
    torch.testing.assert_close(custom_grad.r, expected_r)
    torch.testing.assert_close(custom_grad.c, expected_c)
    torch.testing.assert_close(hook_grad.jg, custom_grad.jg)
    torch.testing.assert_close(hook_grad.gh, custom_grad.gh)
    torch.testing.assert_close(hook_grad.r, custom_grad.r)
    torch.testing.assert_close(hook_grad.c, custom_grad.c)


@pytest.mark.parametrize("backend", ["factored", "materialized"])
@pytest.mark.parametrize("mode", ["custom", "hooks"])
def test_linear_jgh_atom_grad_collects_only_jg_and_gh(backend: str, mode: str) -> None:
    site = make_site(backend=backend)
    collector = LinearJGHAtomGrad(
        mode=mode,
        factored=backend == "factored",
    )
    site.atoms.set_grad(collector)
    inputs = torch.randn(2, 3, site.in_features, dtype=torch.float64)
    output_gradient = torch.randn(
        2, 3, site.out_features, dtype=torch.float64
    )
    point = site.atoms.p.detach().clone()

    collector.begin()
    (site(inputs) * output_gradient).sum().backward()
    collector.complete()

    represented_gradient = output_gradient.reshape(-1, site.out_features).T @ (
        inputs.reshape(-1, site.in_features)
    )
    derivatives = site.cst_derivatives()
    expected_jg = derivatives.pullback(represented_gradient, parameter_point=point)
    expected_gh = derivatives.contracted_hessian(
        represented_gradient, parameter_point=point
    )
    observation = collector.snapshot()

    torch.testing.assert_close(observation.jg, expected_jg)
    torch.testing.assert_close(observation.gh, expected_gh)
    assert observation.row_square is None
    assert observation.column_square is None
    assert observation.atom_square is None
    assert observation.visible_gradient is None
    assert collector.contributions == 1


@pytest.mark.parametrize("backend", ["factored", "materialized"])
@pytest.mark.parametrize("mode", ["custom", "hooks"])
def test_linear_jg_atom_grad_skips_curvature(
    backend: str, mode: str
) -> None:
    site = make_site(backend=backend)
    collector = LinearJGAtomGrad(
        mode=mode,
        factored=backend == "factored",
    )
    site.atoms.set_grad(collector)
    inputs = torch.randn(2, 3, site.in_features, dtype=torch.float64)
    output_gradient = torch.randn(
        2, 3, site.out_features, dtype=torch.float64
    )
    point = site.atoms.p.detach().clone()

    collector.begin()
    (site(inputs) * output_gradient).sum().backward()
    collector.complete()

    represented_gradient = output_gradient.reshape(-1, site.out_features).T @ (
        inputs.reshape(-1, site.in_features)
    )
    expected_jg = site.cst_derivatives().pullback(
        represented_gradient,
        parameter_point=point,
    )
    observation = collector.snapshot()

    torch.testing.assert_close(observation.jg, expected_jg)
    assert observation.gh is None
    assert collector.observation_request == AtomGradRequest(jg=True)
    assert collector.contributions == 1


def test_auto_prefers_custom_autograd() -> None:
    site = make_site()
    grad = ImplicitLinearAtomGrad(mode="auto")
    site.atoms.set_grad(grad)

    grad.begin()
    site(torch.randn(2, site.in_features, dtype=torch.float64)).sum().backward()
    grad.complete()

    assert grad.last_route == "custom"


def test_auto_falls_back_to_hooks_and_custom_can_be_required() -> None:
    site = make_site()
    auto_grad = HookOnlyLinearAtomGrad(mode="auto")
    site.atoms.set_grad(auto_grad)
    auto_grad.begin()
    site(torch.randn(2, site.in_features, dtype=torch.float64)).sum().backward()
    auto_grad.complete()

    assert auto_grad.last_route == "hooks"
    assert auto_grad.calls == 1

    site.atoms.set_grad(None)
    required_grad = HookOnlyLinearAtomGrad(mode="custom")
    site.atoms.set_grad(required_grad)
    required_grad.begin()
    with pytest.raises(RuntimeError, match="does not implement custom autograd"):
        site(torch.randn(2, site.in_features, dtype=torch.float64))


def test_repeated_calls_square_the_aggregate_represented_gradient() -> None:
    site = make_site()
    grad = ImplicitLinearAtomGrad(mode="custom", row_chunk_size=1)
    site.atoms.set_grad(grad)
    x1 = torch.randn(4, site.in_features, dtype=torch.float64)
    x2 = torch.randn(2, site.in_features, dtype=torch.float64)
    g1 = torch.randn(4, site.out_features, dtype=torch.float64)
    g2 = torch.randn(2, site.out_features, dtype=torch.float64)

    grad.begin()
    ((site(x1) * g1).sum() + (site(x2) * g2).sum()).backward()
    grad.complete()

    represented_gradient = g1.T @ x1 + g2.T @ x2
    torch.testing.assert_close(grad.jg, site.atoms.p.grad)
    torch.testing.assert_close(grad.r, represented_gradient.square().mean(dim=1))
    torch.testing.assert_close(grad.c, represented_gradient.square().mean(dim=0))
    assert grad.contributions == 2


def test_visible_gradient_observation_sums_repeated_uses_exactly() -> None:
    site = make_site()
    grad = ImplicitLinearAtomGrad(
        mode="custom", request=AtomGradRequest(visible_gradient=True)
    )
    site.atoms.set_grad(grad)
    x1 = torch.randn(4, site.in_features, dtype=torch.float64)
    x2 = torch.randn(2, site.in_features, dtype=torch.float64)
    g1 = torch.randn(4, site.out_features, dtype=torch.float64)
    g2 = torch.randn(2, site.out_features, dtype=torch.float64)

    grad.begin()
    ((site(x1) * g1).sum() + (site(x2) * g2).sum()).backward()
    grad.complete()

    observation = grad.snapshot()
    assert observation.visible_gradient is not None
    torch.testing.assert_close(observation.visible_gradient, g1.T @ x1 + g2.T @ x2)
    assert observation.jg is None and observation.atom_square is None


def test_cancelled_scope_ignores_its_stale_hook() -> None:
    site = make_site()
    grad = ImplicitLinearAtomGrad(mode="hooks")
    site.atoms.set_grad(grad)
    old_inputs = torch.randn(2, site.in_features, dtype=torch.float64)
    new_inputs = torch.randn(3, site.in_features, dtype=torch.float64)

    grad.begin()
    old_outputs = site(old_inputs)
    grad.cancel()
    grad.begin()
    new_outputs = site(new_inputs)
    (old_outputs.sum() + new_outputs.sum()).backward()
    grad.complete()

    represented_gradient = torch.ones(
        site.out_features, new_inputs.shape[0], dtype=torch.float64
    ) @ new_inputs
    torch.testing.assert_close(grad.r, represented_gradient.square().mean(dim=1))
    torch.testing.assert_close(grad.c, represented_gradient.square().mean(dim=0))
    assert grad.contributions == 1


def test_linear_rejects_an_active_operation_incompatible_grad() -> None:
    site = make_site()
    grad = EmptyAtomGrad()
    site.atoms.set_grad(grad)
    grad.begin()

    with pytest.raises(TypeError, match="requires an active LinearAtomGrad"):
        site(torch.randn(2, site.in_features, dtype=torch.float64))


def test_implicit_linear_grad_rejects_trainable_charts() -> None:
    site = CSTLinear(
        Chart.linspace(4, trainable=True),
        Chart.linspace(3),
        atoms=2,
        kernel=Separable(
            input_profile=Gaussian(0.8),
            output_profile=Gaussian(0.6),
        ),
        dtype=torch.float64,
    )
    grad = ImplicitLinearAtomGrad()
    site.atoms.set_grad(grad)
    grad.begin()

    with pytest.raises(ValueError, match="frozen charts only"):
        site(torch.randn(2, site.in_features, dtype=torch.float64)).sum().backward()


def test_completed_values_are_isolated_snapshots() -> None:
    site = make_site()
    grad = ImplicitLinearAtomGrad()
    site.atoms.set_grad(grad)
    grad.begin()
    site(torch.randn(2, site.in_features, dtype=torch.float64)).sum().backward()
    grad.complete()

    jg = grad.jg
    jg.zero_()

    assert torch.count_nonzero(grad.jg) > 0
