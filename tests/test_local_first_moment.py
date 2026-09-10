from types import SimpleNamespace

import torch

from torchcst._derivatives.local_solve import solve_blocks
from torchcst._derivatives.local_tangent import AtomLocalTangentGeometry

pullback = AtomLocalTangentGeometry.pullback_from_frame


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


def test_batched_direct_solve_uses_solve_ex_without_an_explicit_inverse(monkeypatch):
    calls = []
    original = torch.linalg.solve_ex

    def observed(matrix, rhs, *, check_errors):
        calls.append((matrix.shape, rhs.shape, check_errors))
        return original(matrix, rhs, check_errors=check_errors)

    def forbidden(*args, **kwargs):
        raise AssertionError("the atom-local solve must not form an inverse")

    monkeypatch.setattr(torch.linalg, "solve_ex", observed)
    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    monkeypatch.setattr(torch.linalg, "inv_ex", forbidden)
    gram = torch.eye(4, dtype=torch.float64).expand(3, 4, 4)
    rhs = torch.ones(3, 4, dtype=torch.float64)
    solution, residual, valid = solve_blocks(gram, rhs, 0.01)
    assert calls == [((3, 4, 4), (3, 4, 1), False)]
    assert bool(valid) and residual < 1e-12 and torch.isfinite(solution).all()


def test_batched_direct_solve_rejects_a_singular_local_system():
    gram = torch.diag_embed(torch.tensor([[-0.01, 1.0]], dtype=torch.float64))
    solution, residual, valid = solve_blocks(
        gram, torch.ones(1, 2, dtype=torch.float64), 0.01
    )
    assert not bool(valid)
    assert not torch.isfinite(solution).all() or not torch.isfinite(residual)


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
