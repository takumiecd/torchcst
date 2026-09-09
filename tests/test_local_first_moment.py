from types import SimpleNamespace

import torch

from experiments.local_first_moment import pullback, solve_blocks


def test_batched_direct_solve_matches_block_diagonal_dense_oracle():
    gen = torch.Generator().manual_seed(8)
    jac = torch.randn(3, 7, 4, generator=gen, dtype=torch.float64)
    gram = jac.transpose(-1, -2) @ jac
    rhs = torch.randn(3, 4, generator=gen, dtype=torch.float64)
    actual, residual, valid = solve_blocks(gram, rhs, 0.01)
    matrix = torch.block_diag(*gram) + 0.01 * torch.eye(12, dtype=torch.float64)
    torch.testing.assert_close(
        actual.flatten(), torch.linalg.solve(matrix, rhs.flatten())
    )
    assert bool(valid) and residual < 1e-12


def test_local_transport_omits_other_atoms_even_when_they_overlap():
    eye = torch.eye(2, dtype=torch.float64)
    # Both atoms have identical visible directions; full transport would sum them.
    cross = eye.expand(2, 2, 2).clone()

    def local_cross(a, b, local=False):
        assert local
        return cross

    geometry = SimpleNamespace(_validate_frame=lambda _: None, cross=local_cross)
    point = torch.zeros(2, 2, dtype=torch.float64)
    coefficients = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    result = pullback(
        geometry,
        current_point=point,
        source_frame=SimpleNamespace(point=point),
        source_coefficients=coefficients,
    )
    torch.testing.assert_close(result.constant, coefficients)
    assert not torch.equal(result.constant, coefficients.sum(0).expand_as(coefficients))


def test_zero_gram_and_rhs_are_handled_by_positive_damping():
    solution, residual, valid = solve_blocks(
        torch.zeros(2, 4, 4), torch.zeros(2, 4), 0.01
    )
    assert bool(valid) and residual == 0 and solution.count_nonzero() == 0
