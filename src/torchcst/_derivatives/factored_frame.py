"""Frame operations that never build visible Jacobians or Hessians."""

import torch

from . import factored_taylor as ft
from ._captured import call as captured
from .frame import AffinePullback, AutogradFrameGeometry, GramSystem


class FactoredFrameGeometry(AutogradFrameGeometry):
    def __init__(self, derivatives):
        if derivatives.factor_atoms is None:
            raise ValueError("factored geometry requires a factor-capable kernel")
        self.derivatives = derivatives
        self.device_solver = "jacobi"
        u, v = derivatives.factor_atoms(derivatives.current_point())
        self._visible_shape = (v.shape[0], u.shape[0])
        self._factor_point = None
        self._factors = None

    def factor_local_derivatives(self, point):
        self._validate_local(point, name="factor derivative point")
        if self._factor_point is not point:
            self._factors = captured(
                "factor_derivatives", self.derivatives.factor_atoms, point
            )
            self._factor_point = point
        return self._factors

    def _local_derivatives(self, point):
        return None

    def displacement(self, direction, *, point):
        self._validate_local(direction, name="displacement")
        return ft.displacement(self.factor_local_derivatives(point), direction)

    def pullback(self, cotangent, *, point, displacement=None):
        self._validate_visible(cotangent, name="cotangent")
        d = torch.zeros_like(point) if displacement is None else displacement
        self._validate_local(d, name="displacement")
        return ft.pullback(self.factor_local_derivatives(point), cotangent, d)

    def gram(self, frame, *, damping=0.0, rtol=None):
        self._validate_frame(frame)
        f = self.factor_local_derivatives(frame.point)
        matrix = ft.call("_gram_flat", *f, frame.displacement)[0]
        return GramSystem(
            matrix,
            self.point_shape,
            damping=damping,
            rtol=rtol,
            device_solver=self.device_solver,
        )

    def compress(self, *, frame, pullback_numerator, damping=0.0, rtol=None):
        if self.device_solver != "pcg":
            return super().compress(
                frame=frame,
                pullback_numerator=pullback_numerator,
                damping=damping,
                rtol=rtol,
            )
        from torchcst._runtime.validation import require

        from ._pcg import PCGOptions, solve

        if rtol is not None:
            raise ValueError("PCG does not implement a pseudoinverse rank cutoff")
        self._validate_frame(frame)
        self._validate_local(pullback_numerator, name="Gram rhs")
        solution, valid, _, _ = solve(
            self.factor_local_derivatives(frame.point),
            frame.displacement,
            pullback_numerator,
            damping=damping,
            options=getattr(self, "pcg_options", PCGOptions()),
        )
        require(valid, "PCG Gram solve failed residual validation", FloatingPointError)
        return solution

    def gram_matvec(self, frame, vector, *, block_size=32, damping=0.0):
        """Exact blocked Gram action; does not construct a GramSystem."""
        self._validate_frame(frame)
        self._validate_local(vector, name="Gram vector")
        return ft.frame_gram_matvec(
            self.factor_local_derivatives(frame.point),
            frame.displacement,
            vector,
            block_size=block_size,
            damping=damping,
        )

    def pullback_from_frame(self, *, current_point, source_frame, source_coefficients):
        self._validate_frame(source_frame)
        self._validate_local(source_coefficients, name="source coefficients")
        source = captured(
            "factor_derivatives", self.derivatives.factor_atoms, source_frame.point
        )
        current = self.factor_local_derivatives(current_point)
        constant, linear = ft.call(
            "_transport_flat",
            *current,
            *source,
            source_frame.displacement,
            source_coefficients,
        )
        return AffinePullback(constant, linear)
