"""Tangent history transport restricted to matching atoms."""

from torchcst._runtime.validation import require

from .frame import AffinePullback
from .local_solve import solve_blocks
from .tangent import TangentGeometry
from .tangent_solve import CompressionResult


class AtomLocalTangentGeometry(TangentGeometry):
    """Use K independent q-by-q blocks; never construct the full Gram."""

    def pullback_from_frame(self, *, current_point, source_frame, source_coefficients):
        self._validate_frame(source_frame)
        cross = self.cross(current_point, source_frame.point, local=True)
        constant = (cross @ source_coefficients.unsqueeze(-1)).squeeze(-1)
        return AffinePullback(
            constant,
            current_point.new_zeros(*current_point.shape, current_point.shape[-1]),
        )

    def compress(self, *, frame, pullback_numerator, damping=0.0, rtol=None):
        self._validate_frame(frame)
        gram = self.cross(frame.point, frame.point, local=True)
        alpha, relative, valid = solve_blocks(
            gram,
            pullback_numerator,
            damping,
            rtol=self.recompression_rtol if rtol is None else rtol,
        )
        require(valid, "atom-local direct recompression failed", FloatingPointError)
        self.compression_result = CompressionResult(
            0, relative, valid, "atom_block_direct"
        )
        return alpha
