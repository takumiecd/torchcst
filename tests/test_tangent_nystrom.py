"""Dense SPD oracle for the experimental low-rank inverse action."""

from types import SimpleNamespace

import pytest
import torch

from torchcst._derivatives.tangent_device import pcg
from torchcst._derivatives.tangent_nystrom import apply, build
from torchcst._derivatives.tangent_ops import PreparedFactors


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rank", [4, 12])
def test_nystrom_inverse_and_full_residual(device, rank):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(27)
    n = 12
    j = torch.randn(n, n, device=device, dtype=torch.float64)
    point = torch.zeros(1, n, device=device, dtype=j.dtype)
    prepared = PreparedFactors(
        SimpleNamespace(
            backend="specialized",
            execution="triton" if device == "cuda" else "eager",
            gram_action="jvp_vjp",
        ),
        point,
        (
            torch.zeros(1, n, device=device, dtype=j.dtype),
            torch.ones(1, 1, device=device, dtype=j.dtype),
            j[None],
            torch.zeros(1, 1, n, device=device, dtype=j.dtype),
        ),
    )
    omega = torch.linalg.qr(torch.randn(n, rank, device=device, dtype=j.dtype)).Q
    u, w, ok = build(prepared, omega, 0.1)
    assert ok
    h = j.T @ j
    # Independent unshifted Nyström construction (SPD core in this fixture).
    y = h @ omega
    approximation = y @ torch.linalg.solve(omega.T @ y, y.T)
    values, vectors = torch.linalg.eigh(approximation)
    values, vectors = values[-rank:].clamp_min(0), vectors[:, -rank:]
    expected = (
        torch.eye(n, device=device, dtype=j.dtype)
        + (vectors * ((values.min() + 0.1) / (values + 0.1) - 1)) @ vectors.T
    )
    inverse = torch.eye(n, device=device, dtype=j.dtype) + (u * w) @ u.T
    torch.testing.assert_close(inverse, expected, rtol=1e-9, atol=1e-9)
    rhs = torch.randn_like(point)
    torch.testing.assert_close(apply(rhs, u, w).flatten(), expected @ rhs.flatten())
    alpha, _, relative, valid = pcg(
        prepared, rhs, damping=0.1, max_iter=40, rtol=1e-9, nystrom=(u, w, ok)
    )
    assert valid and relative <= 1e-9
    assert (
        (h + 0.1 * torch.eye(n, device=device, dtype=j.dtype)) @ alpha.flatten()
        - rhs.flatten()
    ).norm() <= 1e-9 * rhs.norm()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nystrom_setup_captured_with_fresh_factors():
    from torchcst._runtime.graphs import CapturedCall

    torch.manual_seed(47)
    n = 10
    point = torch.zeros(1, n, device="cuda", dtype=torch.float64)
    omega = torch.linalg.qr(torch.randn(n, 4, device="cuda", dtype=point.dtype)).Q

    def run(j, rhs, sketch):
        p = PreparedFactors(
            SimpleNamespace(
                backend="specialized", execution="triton", gram_action="jvp_vjp"
            ),
            torch.zeros_like(rhs),
            (
                torch.zeros_like(rhs),
                torch.ones(1, 1, device=rhs.device, dtype=rhs.dtype),
                j[None],
                torch.zeros(1, 1, n, device=rhs.device, dtype=rhs.dtype),
            ),
        )
        return pcg(
            p,
            rhs,
            damping=0.1,
            max_iter=40,
            rtol=1e-8,
            compiled=True,
            nystrom=build(p, sketch, 0.1),
        )

    call = CapturedCall(run)
    for _ in range(2):
        j = torch.randn(n, n, device="cuda", dtype=point.dtype)
        rhs = torch.randn_like(point)
        alpha, _, _, valid = call(j, rhs, omega)
        assert valid
        residual = (
            j.T @ j + 0.1 * torch.eye(n, device="cuda", dtype=point.dtype)
        ) @ alpha.flatten() - rhs.flatten()
        assert residual.norm() <= 1e-8 * rhs.norm()
