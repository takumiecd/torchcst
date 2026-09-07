import pytest
import torch

from torchcst._derivatives._jacobi import device_eigh, device_pinv_solve, pair_schedule


def test_round_robin_covers_every_pair_once():
    for n in (2, 4, 8, 16):
        pairs = [(i, j) for row in pair_schedule(n) for i, j in enumerate(row) if i < j]
        assert len(pairs) == len(set(pairs)) == n * (n - 1) // 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["indefinite", "singular", "zero", "repeated"])
def test_jacobi_eigen_and_pseudoinverse_oracles(dtype, kind):
    torch.manual_seed(13)
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=dtype))
    values = {
        "indefinite": [-3, -1, -0.1, 0, 0.2, 1, 2, 4],
        "singular": [0, 0, 0, 0, 0.2, 1, 2, 4],
        "zero": [0] * 8,
        "repeated": [1] * 8,
    }[kind]
    matrix = (q * q.new_tensor(values)) @ q.T
    w, v = device_eigh(matrix)
    tol = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close((v * w) @ v.T, matrix, atol=tol, rtol=tol)
    torch.testing.assert_close(v.T @ v, torch.eye(8, dtype=dtype), atol=tol, rtol=tol)
    rhs = torch.randn(8, dtype=dtype)
    solution, valid = device_pinv_solve(matrix, rhs)
    assert valid
    torch.testing.assert_close(
        solution, torch.linalg.pinv(matrix) @ rhs, atol=tol * 10, rtol=tol * 10
    )


@pytest.mark.parametrize("size", [1, 3, 5])
def test_padded_sizes(size):
    a = torch.eye(size, dtype=torch.float64)
    rhs = torch.arange(size, dtype=a.dtype)
    out, valid = device_pinv_solve(a, rhs)
    assert valid
    torch.testing.assert_close(out, rhs)


def test_float32_rank_cutoff_is_retained_with_double_accumulation():
    matrix = torch.diag(torch.tensor([1.0, 1e-8, 0.0, 0.0], dtype=torch.float32))
    solution, valid = device_pinv_solve(matrix, torch.ones(4))
    assert valid
    assert solution.dtype == torch.float32
    torch.testing.assert_close(solution, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_unconverged_decomposition_returns_invalid_status():
    matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    _, valid = device_pinv_solve(matrix, torch.ones(2, dtype=matrix.dtype), sweeps=0)
    assert not valid
