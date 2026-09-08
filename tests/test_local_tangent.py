"""Dense masked-Gram oracles, global ball constraints and unchanged compression."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_tangent_ops import jacobian, site_for
from tests.test_trust_pcg import reference
from torchcst import CSTAdam, FirstOrderAdamConfig
from torchcst._derivatives.tangent import TangentGeometry
from torchcst._derivatives.tangent_metric import from_prepared
from torchcst._runtime.validation import device_checks
from torchcst.optim._local_tangent import solve
from torchcst.optim.moments import SeparableDiagonalMetric


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("approximation", ["diagonal", "atom_block"])
@pytest.mark.parametrize("kind", ["amplitude", "bandwidth"])
def test_masked_autograd_gram_and_changing_inputs(device, approximation, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    site = site_for(kind).to(device)
    point = site.atoms.p.detach().clone()
    point[0, 0] = 0  # No division by amplitude; singular atom block is allowed.
    ops = site.cst_derivatives().tangent_ops(
        execution="triton" if device == "cuda" else "eager"
    )
    for turn in range(2):
        p = point + turn * 0.02
        metric = SeparableDiagonalMetric(
            torch.rand(4, dtype=p.dtype, device=device) + 0.2,
            torch.rand(6, dtype=p.dtype, device=device) + 0.2,
            eps=0.01,
        )
        j = jacobian(site, p)
        h = j.T @ (metric.diagonal().flatten()[:, None] * j) / 0.3
        k, q = p.shape
        blocks = torch.stack(
            [h[a * q : (a + 1) * q, a * q : (a + 1) * q] for a in range(k)]
        )
        expected_h = (
            torch.diag(h.diagonal())
            if approximation == "diagonal"
            else torch.block_diag(*blocks)
        )
        action = from_prepared(ops.prepare(p), metric, 0.3)
        torch.testing.assert_close(action.diagonal().flatten(), h.diagonal())
        torch.testing.assert_close(action.blocks(), blocks)
        linear = torch.randn_like(p)
        for radius in (0.1, 10.0):
            with device_checks() as checks:
                got = solve(
                    SimpleNamespace(linear=linear, operator=action),
                    approximation=approximation,
                    radius=radius,
                    rtol=1e-8,
                )
            assert torch.stack(checks).all()
            expected = reference(expected_h.cpu(), linear.cpu(), radius)
            torch.testing.assert_close(
                got.displacement.cpu(), expected.displacement, atol=1e-8, rtol=1e-8
            )
            assert got.relative_residual <= 1e-8
            assert got.displacement.norm() <= radius * (1 + 1e-14)


@pytest.mark.parametrize("q", [1, 3, 4, 6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_eigensystems_and_null_linear_force(q, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from torchcst._derivatives.block_eigh import block_eigh

    torch.manual_seed(26)
    a = torch.randn(3, q, q, device=device, dtype=torch.float64)
    blocks = a @ a.transpose(-1, -2)
    blocks[0] = 0
    values, vectors = block_eigh(blocks)
    n = vectors.shape[-1]
    padded = torch.nn.functional.pad(blocks, (0, n - q, 0, n - q))
    torch.testing.assert_close(
        padded @ vectors, vectors * values[:, None, :], atol=1e-10, rtol=1e-10
    )
    linear = torch.randn(3, q, dtype=torch.float32, device=device)
    operator = SimpleNamespace(
        blocks=lambda: blocks, diagonal=lambda: blocks.diagonal(dim1=-2, dim2=-1)
    )
    for approximation in ("diagonal", "atom_block"):
        for b in (linear, torch.zeros_like(linear)):
            with device_checks() as checks:
                got = solve(
                    SimpleNamespace(linear=b, operator=operator),
                    approximation=approximation,
                    radius=0.2,
                    rtol=1e-5,
                )
            assert torch.stack(checks).all() and got.converged
            if torch.count_nonzero(b) == 0:
                assert torch.count_nonzero(got.displacement) == 0


@pytest.mark.parametrize("approximation", ["diagonal", "atom_block"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_preserves_full_compression_without_full_update_gram(
    approximation, device, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    site = site_for("amplitude").to(device)
    optimizer = CSTAdam(
        site,
        update_approximation=approximation,
        recompression="pcg",
        first_moment_damping=0.1,
        recompression_max_iter=128,
        recompression_rtol=1e-10,
        device_execution=device == "cuda",
    )
    monkeypatch.setattr(
        TangentGeometry, "cross", lambda *a, **kw: pytest.fail("full Gram constructed")
    )
    x = torch.randn(5, 6, dtype=site.atoms.p.dtype, device=device)
    y = torch.randn(5, 4, dtype=x.dtype, device=device)
    old_j = old_alpha = None
    for step in range(2):
        p = site.atoms.p.detach().clone()
        j = jacobian(site, p)
        optimizer.zero_grad()
        (site(x) * y).sum().backward()
        b = 0.1 * j.T @ (y.T @ x).flatten()
        if step:
            b += 0.9 * j.T @ old_j @ old_alpha
        optimizer.step()
        optimizer.check_errors()
        state = optimizer._sites[0].state.first
        expected = torch.linalg.solve(
            j.T @ j + 0.1 * torch.eye(p.numel(), dtype=p.dtype, device=device), b
        )
        torch.testing.assert_close(
            state.alpha.flatten(), expected, atol=1e-5, rtol=1e-5
        )
        old_j, old_alpha = j, state.alpha.flatten().clone()
    checkpoint = optimizer.state_dict()
    optimizer.load_state_dict(checkpoint)


@pytest.mark.parametrize(
    "options",
    [
        {"update_approximation": "invalid"},
        {"update_approximation": "diagonal", "update_solver": "pcg"},
        {"update_approximation": "atom_block", "second_moment": "atom_block"},
        {"update_approximation": "diagonal", "tangent_backend": "reference"},
    ],
)
def test_reject_ambiguous_approximation_options(options):
    with pytest.raises(ValueError):
        FirstOrderAdamConfig(**options)


@pytest.mark.parametrize("approximation", ["diagonal", "atom_block"])
def test_nonfinite_local_objective_latches_deferred_error(approximation):
    matrix = torch.eye(3, dtype=torch.float64)[None]
    operator = SimpleNamespace(
        blocks=lambda: matrix, diagonal=lambda: matrix.diagonal(dim1=-2, dim2=-1)
    )
    with device_checks() as checks:
        solve(
            SimpleNamespace(linear=torch.full((1, 3), float("nan")), operator=operator),
            approximation=approximation,
            radius=0.2,
            rtol=1e-5,
        )
    assert not torch.stack(checks).all()


@pytest.mark.parametrize("approximation", ["diagonal", "atom_block"])
def test_invalid_local_update_suppresses_parameter_and_moment_commits(
    approximation, monkeypatch
):
    from torchcst._derivatives.tangent_metric import FactorMetricAction

    site = site_for("amplitude")
    optimizer = CSTAdam(
        site,
        update_approximation=approximation,
        recompression="pcg",
        first_moment_damping=0.1,
        device_execution=True,
    )
    point = site.atoms.p.detach().clone()
    method = "diagonal" if approximation == "diagonal" else "blocks"
    original = getattr(FactorMetricAction, method)
    monkeypatch.setattr(
        FactorMetricAction,
        method,
        lambda self: torch.full_like(original(self), float("nan")),
    )
    optimizer.zero_grad()
    site(torch.ones(2, 6, dtype=point.dtype)).sum().backward()
    optimizer.step()
    with pytest.raises(FloatingPointError, match="deferred"):
        optimizer.check_errors()
    torch.testing.assert_close(site.atoms.p, point, atol=0, rtol=0)
    assert torch.count_nonzero(optimizer._sites[0].state.first.alpha) == 0
