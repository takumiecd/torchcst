import pytest
import torch

from torchcst import ImplicitAdamConfig
from torchcst._derivatives._cholesky import device_solve
from torchcst._derivatives.frame import GramSystem
from torchcst._runtime.validation import device_checks


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["full", "singular", "zero"])
def test_damped_cholesky_matches_dense_solve(dtype, kind):
    torch.manual_seed(17)
    a = torch.randn(8, 3 if kind == "singular" else 8, dtype=dtype)
    a = a @ a.T
    if kind == "zero":
        a.zero_()
    rhs = torch.randn(2, 4, dtype=dtype)
    system = GramSystem(a, (2, 4), damping=1e-3, device_solver="cholesky")
    actual = system.solve(rhs)
    damped = a + 1e-3 * torch.eye(8, dtype=dtype)
    expected = (
        torch.linalg.solve(damped.double(), rhs.flatten().double())
        .to(dtype)
        .reshape_as(rhs)
    )
    torch.testing.assert_close(actual, expected)


def test_cholesky_requires_explicit_damping():
    with pytest.raises(ValueError, match="positive"):
        ImplicitAdamConfig(gram_solver="cholesky")
    with pytest.raises(ValueError, match="positive"):
        GramSystem(torch.eye(2), (1, 2), device_solver="cholesky")


def test_non_positive_definite_system_returns_invalid():
    _, valid = device_solve(torch.diag(torch.tensor([-1.0, 1.0])), torch.ones(2))
    assert not valid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cholesky_graph_and_deferred_failure():
    torch.manual_seed(29)
    a = torch.randn(256, 256, device="cuda")
    a = a @ a.T + 0.01 * torch.eye(256, device="cuda")
    rhs = torch.randn(256, device="cuda")
    for offset in (0.0, 0.1):
        matrix = a + offset * torch.eye(256, device="cuda")
        actual, valid = device_solve(matrix, rhs)
        assert valid
        torch.testing.assert_close(
            actual, torch.linalg.solve(matrix.double(), rhs.double()).float()
        )
    torch.cuda.set_sync_debug_mode("error")
    try:
        device_solve(matrix, rhs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    # Same shape/strides replay, no host fallback on failure.
    bad = -torch.eye(256, device="cuda")
    with device_checks():
        _, valid = device_solve(bad, rhs)
    assert not valid


def test_output_overflow_returns_invalid():
    _, valid = device_solve(torch.eye(2) * 1e-10, torch.ones(2) * 1e38)
    assert not valid
