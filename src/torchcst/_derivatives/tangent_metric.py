"""Matrix-free, factor-owned weighted tangent action for iterative updates."""

from types import SimpleNamespace

import torch

from .tangent_ops import PreparedFactors


class FactorMetricAction:
    def __init__(self, point, u, v, du, dv, row, column, eps, rate, *, triton=False):
        self.prepared = PreparedFactors(
            SimpleNamespace(
                backend="specialized",
                execution="triton" if triton else "eager",
                atom_tile=32,
            ),
            point,
            (u, v, du, dv),
        )
        self.row, self.column, self.eps, self.rate = row, column, eps, rate
        self.triton = triton

    def __call__(self, x, active=None):
        p = self.prepared
        if self.triton:
            from .tangent_triton import cross

            return (
                cross(p, p, x, row=self.row, column=self.column, active=active)
                + self.eps * cross(p, p, x, active=active)
            ) / self.rate
        # CPU oracle contracts factors directly without forming J or a full Gram.
        from .tangent_ops import PreparedFactors

        class Weights:
            def separable_weights(_):
                return self.row, self.column, self.eps

        result = (
            PreparedFactors.cross_gram_matvec(p, p, x, metric=Weights()) / self.rate
        )
        return result if active is None else torch.where(active, result, 0)

    def blocks(self):
        p = self.prepared
        u, v, du, dv = p._u, p._v, p._du, p._dv

        def term(row, column):
            uv = (du * u[:, :, None] * column[None, :, None]).sum(1)
            vv = (dv * v[:, :, None] * row[None, :, None]).sum(1)
            return (
                (du.transpose(1, 2) @ (du * column[None, :, None]))
                * (v.square() * row).sum(1)[:, None, None]
                + (dv.transpose(1, 2) @ (dv * row[None, :, None]))
                * (u.square() * column).sum(1)[:, None, None]
                + uv[:, :, None] * vv[:, None, :]
                + vv[:, :, None] * uv[:, None, :]
            )

        return (
            term(self.row, self.column)
            + self.eps * term(torch.ones_like(self.row), torch.ones_like(self.column))
        ) / self.rate

    def diagonal(self):
        return self.blocks().diagonal(dim1=-2, dim2=-1)


def from_prepared(prepared, metric, rate):
    row, column, eps = metric.separable_weights()
    # Match the spectral solver's double-precision numerical solve. Casting the
    # prepared factors preserves the kernel's original derivative observations.
    tensors = tuple(
        t.detach().double().contiguous()
        for t in (
            prepared._point,
            prepared._u,
            prepared._v,
            prepared._du,
            prepared._dv,
            row,
            column,
        )
    )
    return FactorMetricAction(
        *tensors, eps, rate, triton=prepared._ops.execution == "triton"
    )
