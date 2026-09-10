"""First derivatives only, with separate full and atom-local contractions."""

import torch

from torchcst._runtime.validation import deferred

from .frame import AffinePullback, AutogradFrameGeometry, GramSystem


class TangentGeometry(AutogradFrameGeometry):
    """Eager tangent geometry. Full Gram solves remain correctness baselines.

    Factored kernels cache only first factor derivatives. Local square-gradient
    observations stream output rows and atoms rather than storing a visible J.
    """

    def __init__(
        self,
        derivatives,
        *,
        factored=True,
        ops=None,
        recompression="direct",
        recompression_max_iter=64,
        recompression_rtol=1e-5,
    ):
        self.ops = ops or derivatives.tangent_ops(
            backend="auto" if factored else "reference"
        )
        self.ops.check_configuration()
        self._prepared = []
        self.recompression = recompression
        self.recompression_max_iter = recompression_max_iter
        self.recompression_rtol = recompression_rtol
        self.compression_result = None
        self.factored = factored and self.ops.backend != "reference"
        if self.factored:
            self.derivatives = derivatives
            u, v = derivatives.factor_atoms(derivatives.current_point())
            self._visible_shape = (v.shape[0], u.shape[0])
            self.device_solver = "jacobi"
        else:
            super().__init__(derivatives)
        self._tangent_point = None
        self._columns = None
        self._columns_source = None
        self._columns_version = None

    def frame(self, point, displacement=None):
        frame = super().frame(point, displacement)
        # This clone is known to represent the same point. No value comparison.
        for original, version, prepared in self._prepared:
            if original is point and version == point._version:
                self._prepared = (
                    self._prepared + [(frame.point, frame.point._version, prepared)]
                )[-4:]
                break
        return frame

    def prepared(self, point):
        for original, version, prepared in self._prepared:
            if (original is point and version == point._version) or (
                not deferred() and torch.equal(prepared._point, point)
            ):
                return prepared
        result = self.ops.prepare(point)
        self._prepared = (self._prepared + [(point, point._version, result)])[-4:]
        return result

    def _parts(self, point):
        if self._columns_source is point and self._columns_version == point._version:
            return self._columns
        if (
            self._tangent_point is not None
            and not deferred()
            and torch.equal(self._tangent_point, point)
        ):
            return self._columns
        with torch.no_grad():
            if self.factored:
                prepared = self.prepared(point)
                u, v, du, dv = (prepared._u, prepared._v, prepared._du, prepared._dv)
                q = point.shape[-1]
                left = (dv.transpose(1, 2), v[:, None].expand(-1, q, -1))
                right = (u[:, None].expand(-1, q, -1), du.transpose(1, 2))
                columns = (left, right)
            else:

                def one(p):
                    return self.derivatives._materialize_atoms(p[None])[0]

                columns = torch.vmap(torch.func.jacfwd(one))(point).detach()
        self._tangent_point = point.detach().clone()
        self._columns = columns
        self._columns_source, self._columns_version = point, point._version
        return columns

    def _local_derivatives(self, point):
        raise RuntimeError("second derivatives are not part of TangentGeometry")

    @property
    def supports_quadratic_feature_gram(self):
        return False

    def cross(self, point, source_point, *, local=False, metric=None):
        """J(point)^T D J(source), optionally only matching atom blocks."""
        current = self._parts(point)
        previous = self._parts(source_point)
        if self.factored and (metric is None or hasattr(metric, "separable_weights")):
            left, right = current
            old_left, old_right = previous
            terms = [(None, None, 1.0)]
            if metric is not None:
                row, column, eps = metric.separable_weights()
                terms = [(row, column, 1.0), (None, None, eps)]
            k, q = point.shape
            shape = (k, q, q) if local else (k * q, k * q)
            result = point.new_zeros(shape)
            for row, column, scale in terms:
                for i in range(2):
                    for j in range(2):
                        l = left[i] if row is None else left[i] * row
                        r = right[i] if column is None else right[i] * column
                        if local:
                            value = (l @ old_left[j].transpose(-1, -2)) * (
                                r @ old_right[j].transpose(-1, -2)
                            )
                        else:
                            value = (l.flatten(0, 1) @ old_left[j].flatten(0, 1).T) * (
                                r.flatten(0, 1) @ old_right[j].flatten(0, 1).T
                            )
                        result.add_(value, alpha=scale)
            return result
        if self.factored:
            raise TypeError("factored cross requires a separable metric")
        weighted = (
            current if metric is None else current * metric.diagonal()[None, ..., None]
        )
        if local:
            return torch.einsum("koip,koiq->kpq", weighted, previous)
        a = weighted.permute(1, 2, 0, 3).flatten(0, 1).flatten(1)
        b = previous.permute(1, 2, 0, 3).flatten(0, 1).flatten(1)
        return a.T @ b

    def row_columns(self, point, start, stop, atom_start, atom_stop):
        """Bounded visible Jacobian tile [atoms, rows, inputs, coordinates]."""
        parts = self._parts(point)
        if not self.factored:
            return parts[atom_start:atom_stop, start:stop]
        left, right = parts
        return sum(
            torch.einsum(
                "kpr,kpi->krip",
                l[atom_start:atom_stop, :, start:stop],
                r[atom_start:atom_stop],
            )
            for l, r in zip(left, right)
        )

    def square_observation(self, point, squared_rows, start, *, atom_chunk=8):
        k, q = point.shape
        result = point.new_empty(k, q, q)
        for a in range(0, k, atom_chunk):
            j = self.row_columns(
                point, start, start + squared_rows.shape[0], a, a + atom_chunk
            )
            result[a : a + atom_chunk] = torch.einsum(
                "krip,ri,kriq->kpq", j, squared_rows, j
            )
        return result

    def local_diagonal_gram(
        self,
        point,
        diagonal,
        *,
        row_chunk=64,
        atom_chunk=8,
    ):
        """Return atom blocks of ``J.T @ Diag(diagonal) @ J``.

        Rows and atoms are streamed so a dense visible Jacobian is never
        materialized. ``diagonal`` may itself be dense in visible space.
        """
        self._validate_visible(diagonal, name="diagonal")
        if (
            isinstance(row_chunk, bool)
            or not isinstance(row_chunk, int)
            or row_chunk < 1
        ):
            raise ValueError("row_chunk must be a positive integer")
        if (
            isinstance(atom_chunk, bool)
            or not isinstance(atom_chunk, int)
            or atom_chunk < 1
        ):
            raise ValueError("atom_chunk must be a positive integer")
        k, q = point.shape
        rows = diagonal.shape[0]
        result = point.new_zeros(k, q, q)
        for start in range(0, rows, row_chunk):
            stop = min(start + row_chunk, rows)
            weights = diagonal[start:stop]
            for atom_start in range(0, k, atom_chunk):
                atom_stop = min(atom_start + atom_chunk, k)
                columns = self.row_columns(
                    point, start, stop, atom_start, atom_stop
                )
                result[atom_start:atom_stop].add_(
                    torch.einsum(
                        "krip,ri,kriq->kpq", columns, weights, columns
                    )
                )
        return result

    def displacement(self, direction, *, point):
        self._validate_local(direction, name="direction")
        return self.derivatives.jvp(direction, parameter_point=point).detach()

    def pullback(self, cotangent, *, point, displacement=None):
        self._validate_visible(cotangent, name="cotangent")
        del displacement
        return self.prepared(point).vjp(cotangent).detach()

    def pullback_from_frame(self, *, current_point, source_frame, source_coefficients):
        self._validate_frame(source_frame)
        constant = self.prepared(current_point).cross_gram_matvec(
            self.prepared(source_frame.point), source_coefficients
        )
        return AffinePullback(
            constant,
            current_point.new_zeros(*current_point.shape, current_point.shape[-1]),
        )

    def gram(self, frame, *, damping=0.0, rtol=None):
        return GramSystem(
            self.cross(frame.point, frame.point),
            self.point_shape,
            damping=damping,
            rtol=rtol,
        )

    def compress(self, *, frame, pullback_numerator, damping=0.0, rtol=None):
        if self.recompression == "direct":
            return super().compress(
                frame=frame,
                pullback_numerator=pullback_numerator,
                damping=damping,
                rtol=rtol,
            )
        if self.recompression != "pcg":
            raise ValueError("recompression must be direct or pcg")
        from .tangent_solve import solve_compression

        self._validate_frame(frame)
        alpha, self.compression_result = solve_compression(
            self.prepared(frame.point),
            pullback_numerator,
            damping=damping,
            max_iter=self.recompression_max_iter,
            rtol=self.recompression_rtol,
        )
        return alpha
