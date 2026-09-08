"""PCG numerical, transport, checkpoint and atomic commit contracts."""

import copy

import pytest
import torch

from tests.test_tangent_ops import jacobian, site_for
from torchcst import CSTAdam, FirstOrderAdamConfig
from torchcst._derivatives.tangent import TangentGeometry
from torchcst._derivatives.tangent_solve import CompressionError, solve_compression


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["amplitude", "bandwidth"])
def test_pcg_matches_direct_damped_system_and_true_residual(kind, dtype):
    site = site_for(kind, dtype)
    p = site.atoms.p.detach()
    prepared = site.cst_derivatives().tangent_ops(atom_tile=3).prepare(p)
    j = jacobian(site, p)
    rhs = torch.randn_like(p)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    damping = 0.1
    alpha, result = solve_compression(
        prepared, rhs, damping=damping, max_iter=100, rtol=tolerance
    )
    matrix = j.T @ j + damping * torch.eye(p.numel(), dtype=dtype)
    direct = torch.linalg.solve(matrix, rhs.flatten())
    torch.testing.assert_close(
        alpha.flatten(), direct, rtol=3 * tolerance, atol=3 * tolerance
    )
    residual = rhs - (prepared.gram_matvec(alpha) + damping * alpha)
    relative = float(residual.double().norm() / rhs.double().norm())
    assert result.relative_residual == pytest.approx(relative)
    assert result.converged and relative <= tolerance
    zero, diag = solve_compression(prepared, torch.zeros_like(rhs), damping=damping)
    assert torch.count_nonzero(zero) == 0 and diag.iterations == 0
    with pytest.raises(CompressionError) as failed:
        solve_compression(prepared, rhs, damping=damping, max_iter=1, rtol=1e-12)
    assert not failed.value.result.converged


def test_pcg_transport_path_never_builds_full_gram(monkeypatch):
    site = site_for("amplitude")
    geometry = TangentGeometry(
        site.cst_derivatives(), recompression="pcg", recompression_rtol=1e-10
    )
    p = site.atoms.p.detach().clone()
    old = p + 0.03
    j, js = jacobian(site, p), jacobian(site, old)
    x = torch.randn_like(p)

    def forbidden(*args, **kwargs):
        raise AssertionError("global Gram called")

    monkeypatch.setattr(geometry, "cross", forbidden)
    monkeypatch.setattr(geometry, "gram", forbidden)
    pulled = geometry.pullback_from_frame(
        current_point=p, source_frame=geometry.frame(old), source_coefficients=x
    )
    torch.testing.assert_close(pulled.constant.flatten(), j.T @ js @ x.flatten())
    alpha = geometry.compress(
        frame=geometry.frame(p), pullback_numerator=pulled.constant, damping=0.1
    )
    torch.testing.assert_close(
        (j.T @ j + 0.1 * torch.eye(p.numel(), dtype=p.dtype)) @ alpha.flatten(),
        pulled.constant.flatten(),
        rtol=1e-8,
        atol=1e-9,
    )


def run_step(model, opt):
    opt.zero_grad()
    x = torch.linspace(-1.0, 1.0, 18, dtype=model.atoms.p.dtype).reshape(3, 6)
    model(x).square().sum().backward()
    opt.step()


def test_optimizer_pcg_matches_direct_over_multiple_steps_and_resume():
    model = site_for("amplitude")
    clone = copy.deepcopy(model)
    options = {
        "first_moment_damping": 0.1,
        "recompression_rtol": 1e-11,
        "recompression_max_iter": 100,
    }
    direct = CSTAdam(model, **options)
    iterative = CSTAdam(clone, recompression="pcg", **options)
    for _ in range(4):
        run_step(model, direct)
        run_step(clone, iterative)
        torch.testing.assert_close(model.atoms.p, clone.atoms.p, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(
            direct._sites[0].state.first.alpha,
            iterative._sites[0].state.first.alpha,
            rtol=1e-7,
            atol=1e-10,
        )
        assert iterative.last_step.compression_results[0].converged
    saved = iterative.state_dict()
    restart_model = site_for("amplitude")
    restart_model.load_state_dict(clone.state_dict())
    restarted = CSTAdam(restart_model, recompression="pcg", **options)
    restarted.load_state_dict(saved)
    run_step(clone, iterative)
    run_step(restart_model, restarted)
    torch.testing.assert_close(restart_model.atoms.p, clone.atoms.p, rtol=0, atol=0)
    assert "prepared" not in str(saved.keys())


def test_failed_recompression_does_not_commit_any_site_or_dense_state():
    from tests.test_optimizer import MixedModel, batch
    from torchcst import AdamWConfig

    model = MixedModel()
    opt = CSTAdam(
        model, dense=AdamWConfig(), recompression="pcg", first_moment_damping=0.01
    )
    before = copy.deepcopy(model.state_dict())
    saved = opt.state_dict()
    opt.zero_grad()
    x, y = batch()
    (model(x) - y).square().sum().backward()
    # Force an action failure at compression, after both CST and dense proposals.
    from unittest.mock import patch

    with (
        patch(
            "torchcst._derivatives.tangent_solve.solve_compression",
            side_effect=RuntimeError("forced compression failure"),
        ),
        pytest.raises(RuntimeError, match="forced compression"),
    ):
        opt.step()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    for site in opt._sites:
        assert site.state.step == 0
        assert site.atom_grad._terms == []
    assert all(value.step == 0 for value in opt._dense_states.values())
    torch.testing.assert_close(
        opt.state_dict()["cst"]["cst"].first.alpha, saved["cst"]["cst"].first.alpha
    )
    opt.zero_grad()
    (model(x) - y).square().sum().backward()
    opt.step()
    assert opt._sites[0].state.step == 1


def test_changed_chart_or_kernel_rejects_history_and_checkpoint():
    site = site_for("amplitude")
    opt = CSTAdam(site)
    saved = opt.state_dict()
    with torch.no_grad():
        site.input_chart.coordinates.add_(0.1)
    with pytest.raises(ValueError, match="configuration changed"):
        opt.zero_grad()
    fresh = site_for("amplitude")
    fresh.load_state_dict(site.state_dict())
    site = fresh
    rebuilt = CSTAdam(site)
    with pytest.raises(ValueError, match="descriptors"):
        rebuilt.load_state_dict(saved)
    # Replacing a whole module also invalidates the binding.
    site.kernel = copy.deepcopy(site.kernel)
    with pytest.raises(ValueError, match="configuration changed"):
        rebuilt.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recompression": "pcg"},
        {"recompression": "bad"},
        {"tangent_atom_tile": 0},
        {"recompression_max_iter": True},
        {"recompression_rtol": float("nan")},
        {"factored_geometry": False, "tangent_backend": "specialized"},
    ],
)
def test_invalid_options_fail_at_construction(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FirstOrderAdamConfig(**kwargs)
