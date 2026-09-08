"""Independent numerical oracles for the rebuilt first-order optimizer."""

import copy
from dataclasses import fields, is_dataclass

import pytest
import torch

from tests.test_moments import make_site
from tests.test_optimizer import MixedModel, batch
from torchcst import (
    AdamWConfig,
    CSTAdam,
    CSTSecondOrderAdam,
    FirstOrderAdamConfig,
    SecondOrderAdamConfig,
)
from torchcst._derivatives.atoms import AtomDerivatives
from torchcst._derivatives.tangent import TangentGeometry
from torchcst.optim.atom_grad import (
    AtomGradientObservation,
    AtomGradRequest,
    ImplicitLinearAtomGrad,
)
from torchcst.optim.moments import MomentContext, SeparableDiagonalMetric
from torchcst.optim.moments.atom_rms import AtomRMS
from torchcst.optim.moments.tangent import TangentFirstMoment
from torchcst.optim.quadratic import TangentProblem, solve_tangent


def dense_j(site, point):
    return torch.func.jacfwd(lambda p: site._materialize_atoms(p).sum(0))(
        point
    ).reshape(-1, point.numel())


@pytest.mark.parametrize("factored", [False, True])
def test_weighted_gram_and_cross_frame_match_dense_autograd(factored):
    site = make_site()
    geometry = TangentGeometry(site.cst_derivatives(), factored=factored)
    point = geometry.current_point()
    previous = point + 0.07 * torch.randn_like(point)
    j, old = dense_j(site, point), dense_j(site, previous)
    metric = SeparableDiagonalMetric(
        torch.rand(3, dtype=point.dtype), torch.rand(4, dtype=point.dtype), eps=1e-4
    )
    expected = j.T @ (metric.diagonal().flatten()[:, None] * old)
    torch.testing.assert_close(geometry.cross(point, previous, metric=metric), expected)
    blocks = expected.reshape(2, 3, 2, 3)
    torch.testing.assert_close(
        geometry.cross(point, previous, metric=metric, local=True),
        torch.stack([blocks[k, :, k, :] for k in range(2)]),
    )


@pytest.mark.parametrize("mode", ["custom", "hooks"])
@pytest.mark.parametrize("factored", [False, True])
def test_square_observation_uses_aggregate_gradient_before_squaring(mode, factored):
    site = make_site()
    point = site.atoms.p.detach().clone()
    j = dense_j(site, point).reshape(12, 2, 3)
    collector = ImplicitLinearAtomGrad(
        mode=mode,
        factored=factored,
        row_chunk_size=2,
        request=AtomGradRequest(jg=True, atom_square=True),
    )
    site.atoms.set_grad(collector)
    x1, x2 = torch.randn(5, 4, dtype=point.dtype), torch.randn(3, 4, dtype=point.dtype)
    y1, y2 = torch.randn(5, 3, dtype=point.dtype), torch.randn(3, 3, dtype=point.dtype)
    collector.begin()
    ((site(x1) * y1).sum() + (site(x2) * y2).sum()).backward()
    collector.complete()
    obs = collector.snapshot()
    g = y1.T @ x1 + y2.T @ x2
    expected = torch.einsum("mkp,m,mkq->kpq", j, g.flatten().square(), j)
    torch.testing.assert_close(obs.atom_square, expected)
    assert obs.gh is None and obs.row_square is None
    assert collector._terms == [] and collector._square_geometry is None


def test_tangent_moment_compresses_and_transports_the_visible_history():
    site = make_site()
    geometry = TangentGeometry(site.cst_derivatives())
    point = geometry.current_point()
    context = MomentContext(geometry, point)
    component = TangentFirstMoment(0.9, damping=1e-5)
    state = component.initialize(context)
    g = torch.randn(12, dtype=point.dtype)
    j = dense_j(site, point)
    obs = AtomGradientObservation(jg=(j.T @ g).reshape_as(point))
    expanded = component.expand(state, obs, context, next_step=1)
    saved = component.compress(expanded, torch.ones_like(point), context)
    expected = torch.linalg.solve(
        j.T @ j + 1e-5 * torch.eye(6, dtype=point.dtype), 0.1 * j.T @ g
    )
    torch.testing.assert_close(saved.alpha.flatten(), expected)
    assert torch.count_nonzero(saved.frame.displacement) == 0
    current = point + 0.08 * torch.randn_like(point)
    ctx = MomentContext(TangentGeometry(site.cst_derivatives()), current)
    jnew = dense_j(site, current)
    obs2 = AtomGradientObservation(jg=(jnew.T @ g).reshape_as(point))
    expanded2 = component.expand(saved, obs2, ctx, next_step=2)
    expected = 0.9 * jnew.T @ j @ expected + 0.1 * jnew.T @ g
    torch.testing.assert_close(expanded2.raw.constant.flatten(), expected)


@pytest.mark.parametrize("diagonal", [False, True])
def test_atom_rms_transport_matches_explicit_visible_operator(diagonal):
    site = make_site()
    component = AtomRMS(0.9, eps=1e-6, diagonal=diagonal)
    point = site.atoms.p.detach().clone()
    ctx = MomentContext(TangentGeometry(site.cst_derivatives()), point)
    state = component.initialize(ctx)
    for step in range(1, 4):
        j = dense_j(site, point).reshape(12, 2, 3).permute(1, 0, 2)
        g = torch.randn(12, dtype=point.dtype)
        observation = torch.einsum("kmp,m,kmq->kpq", j, g.square(), j)
        expanded = component.expand(
            state, AtomGradientObservation(atom_square=observation), ctx, next_step=step
        )
        saved = expanded.pending_state
        u = j @ saved.basis
        if step == 1:
            old_operator = torch.zeros(2, 12, 12, dtype=point.dtype)
        else:
            oldj = dense_j(site, state.point).reshape(12, 2, 3).permute(1, 0, 2)
            oldu = oldj @ state.basis
            oldc = torch.diag_embed(state.value) if diagonal else state.value
            old_operator = oldu @ oldc @ oldu.transpose(-1, -2)
        observed = torch.diag(g.square())[None]
        expected = u.transpose(-1, -2) @ (0.9 * old_operator + 0.1 * observed) @ u
        if diagonal:
            expected = expected.diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(saved.value, expected, atol=1e-9, rtol=1e-6)
        assert torch.linalg.eigvalsh(expanded.metric.blocks).min() > -1e-10
        state = saved
        point = point + 0.02 * torch.randn_like(point)
        ctx = MomentContext(TangentGeometry(site.cst_derivatives()), point)


def test_block_ball_solve_matches_full_and_kkt_including_null_force():
    linear = torch.tensor([[1.0, -2.0], [0.5, 1.0]], dtype=torch.float64)
    blocks = torch.tensor(
        [[[2.0, 0.4], [0.4, 1.0]], [[0.0, 0.0], [0.0, 3.0]]], dtype=linear.dtype
    )

    def problem(blocked):
        p = object.__new__(TangentProblem)
        p.point_shape = linear.shape
        p.linear = linear
        p.blocked = blocked
        p.matrix = blocks if blocked else torch.block_diag(*blocks)
        return p

    for radius in (0.1, 3.0):
        a, b = (
            solve_tangent(problem(True), radius=radius),
            solve_tangent(problem(False), radius=radius),
        )
        torch.testing.assert_close(a.displacement, b.displacement)
        assert a.converged and a.on_boundary
        _, grad = problem(True).value_and_gradient(a.displacement)
        lam = -(grad * a.displacement).sum() / a.displacement.square().sum()
        assert lam >= 0
        torch.testing.assert_close(
            grad + lam * a.displacement, torch.zeros_like(grad), atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("metric", ["separable", "atom_block", "atom_diag"])
@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("mode", ["custom", "hooks"])
def test_first_order_never_calls_second_derivatives(
    metric, factored, mode, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("second derivative called")

    for name in ("second", "contracted_hessian", "materialized_local_derivatives"):
        monkeypatch.setattr(AtomDerivatives, name, forbidden)
    site = make_site()
    opt = CSTAdam(
        site, second_moment=metric, factored_geometry=factored, atom_grad_mode=mode
    )
    x = torch.randn(5, 4, dtype=torch.float64)
    for _ in range(3):
        opt.zero_grad()
        site(x).square().mean().backward()
        opt.step()
    assert opt._sites[0].state.step == 3


def tensor_count(value):
    if isinstance(value, torch.Tensor):
        return value.numel()
    if is_dataclass(value):
        return sum(tensor_count(getattr(value, f.name)) for f in fields(value))
    return 0


@pytest.mark.parametrize("metric", ["separable", "atom_block", "atom_diag"])
def test_mixed_model_checkpoint_restarts_identically(metric):
    model = MixedModel()
    opt = CSTAdam(model, second_moment=metric, dense=AdamWConfig())
    x, y = batch()

    def step(m, o):
        o.zero_grad()
        (m(x) - y).square().mean().backward()
        o.step()

    step(model, opt)
    params = copy.deepcopy(model.state_dict())
    state = opt.state_dict()
    clone = MixedModel()
    clone.load_state_dict(params)
    resumed = CSTAdam(clone, second_moment=metric, dense=AdamWConfig())
    resumed.load_state_dict(state)
    step(model, opt)
    step(clone, resumed)
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    second = state["cst"]["cst"].second
    if metric == "atom_block":
        assert second.value.shape == (1, 3, 3)
    if metric == "atom_diag":
        assert second.value.shape == (1, 3)
    assert tensor_count(second) <= 27


def test_separate_api_and_checkpoint_contracts():
    with pytest.raises(TypeError):
        CSTAdam(make_site(), cst=SecondOrderAdamConfig())
    with pytest.raises(TypeError):
        CSTSecondOrderAdam(make_site(), cst=FirstOrderAdamConfig())
    with pytest.raises(TypeError):
        CSTAdam(make_site(), approximation_order=1)
    with pytest.raises(TypeError):
        CSTAdam(make_site(), quartic=object())
    with pytest.raises(TypeError):
        CSTAdam(make_site(), cst=FirstOrderAdamConfig(), lr=0.1)
    first = CSTAdam(make_site())
    second = CSTSecondOrderAdam(make_site())
    with pytest.raises(ValueError, match="algorithm"):
        second.load_state_dict(first.state_dict())
    block = CSTAdam(make_site(), second_moment="atom_block")
    with pytest.raises(ValueError, match="contract"):
        block.load_state_dict(first.state_dict())
    corrupted = first.state_dict()
    object.__setattr__(corrupted["cst"]["<root>"].first, "alpha", torch.zeros(99))
    with pytest.raises(ValueError, match="tensor"):
        first.load_state_dict(corrupted)


@pytest.mark.parametrize("metric", ["atom_block", "atom_diag"])
def test_zero_amplitude_and_zero_gradient_are_finite(metric):
    site = make_site()
    with torch.no_grad():
        site.atoms.p[:, 0].zero_()
    opt = CSTAdam(site, second_moment=metric)
    opt.zero_grad()
    (site(torch.zeros(2, 4, dtype=torch.float64)) * 0).sum().backward()
    opt.step()
    assert torch.isfinite(site.atoms.p).all()
    assert opt.last_step.site_results[0].displacement.count_nonzero() == 0
