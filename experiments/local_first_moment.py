"""Atom-local first-moment transport and direct damped recompression."""

from types import MethodType

import torch

from torchcst._derivatives import AffinePullback
from torchcst._derivatives.tangent_solve import CompressionResult
from torchcst._runtime.validation import require


def solve_blocks(gram, rhs, damping):
    matrix = (gram.double() + gram.double().transpose(-1, -2)) * 0.5
    matrix = matrix + damping * torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    )
    solution, info = torch.linalg.solve_ex(
        matrix,
        rhs.double().unsqueeze(-1),
        check_errors=False,
    )
    solution = solution.squeeze(-1).to(rhs)
    residual = (matrix @ solution.double().unsqueeze(-1)).squeeze(-1) - rhs.double()
    relative = residual.norm(dim=-1) / rhs.double().norm(dim=-1).clamp_min(1e-300)
    valid = (
        (info == 0).all() & torch.isfinite(solution).all() & (relative <= 1e-5).all()
    )
    return solution, relative.amax(), valid


def pullback(self, *, current_point, source_frame, source_coefficients):
    self._validate_frame(source_frame)
    cross = self.cross(current_point, source_frame.point, local=True)
    constant = (cross @ source_coefficients.unsqueeze(-1)).squeeze(-1)
    return AffinePullback(
        constant, current_point.new_zeros(*current_point.shape, current_point.shape[-1])
    )


def compress(self, *, frame, pullback_numerator, damping=0.0, rtol=None):
    self._validate_frame(frame)
    gram = self.cross(frame.point, frame.point, local=True)
    alpha, relative, valid = solve_blocks(gram, pullback_numerator, damping)
    require(valid, "atom-local direct recompression failed", FloatingPointError)
    self.compression_result = CompressionResult(0, relative, valid, "atom_block_direct")
    return alpha


def install(geometry):
    geometry.pullback_from_frame = MethodType(pullback, geometry)
    geometry.compress = MethodType(compress, geometry)
    return geometry
