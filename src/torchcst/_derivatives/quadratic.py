"""Exact quadratic-feature Gram contractions for rank-one atom kernels.

Only input/output factor derivatives are materialized. Each column of the
represented Taylor map is a sum of at most four outer products. Their weighted
inner products preserve every cross-atom interaction, including the metric eps.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.func import jacfwd, vmap

from .atoms import FactorAtoms


class QuadraticFeatureGram:
    """Squared represented norm in the features (d_i, d_i d_j for i <= j)."""

    def __init__(self, matrix: Tensor, point_shape: tuple[int, int]) -> None:
        self.matrix = matrix.detach()
        self.point_shape = point_shape
        self.left, self.right = torch.triu_indices(
            point_shape[1], point_shape[1], device=matrix.device
        )

    def features(self, displacement: Tensor) -> Tensor:
        return torch.cat(
            (displacement, displacement[:, self.left] * displacement[:, self.right]),
            dim=1,
        ).reshape(-1)

    def value(self, displacement: Tensor) -> Tensor:
        features = self.features(displacement)
        return features @ (self.matrix @ features)

    def value_and_gradient(self, displacement: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(displacement)
        force = self.matrix @ features
        local_force = force.reshape(self.point_shape[0], -1)
        parameters = self.point_shape[1]
        gradient = local_force[:, :parameters].clone()
        quadratic_force = local_force[:, parameters:]
        gradient.scatter_add_(
            1,
            self.left.expand(self.point_shape[0], -1),
            quadratic_force * displacement[:, self.right],
        )
        gradient.scatter_add_(
            1,
            self.right.expand(self.point_shape[0], -1),
            quadratic_force * displacement[:, self.left],
        )
        return features @ force, 2.0 * gradient

    def hessian(self, displacement: Tensor) -> Tensor:
        """Exact Hessian in the original monomial coordinates."""

        atoms, parameters = self.point_shape
        features = self.features(displacement)
        force = (self.matrix @ features).reshape(atoms, -1)
        local = displacement.new_zeros(atoms, force.shape[1], parameters)
        local[:, :parameters] = torch.eye(
            parameters, device=displacement.device, dtype=displacement.dtype
        )
        local[:, parameters:].scatter_add_(
            2,
            self.left[None, :, None].expand(atoms, -1, 1),
            displacement[:, self.right, None],
        )
        local[:, parameters:].scatter_add_(
            2,
            self.right[None, :, None].expand(atoms, -1, 1),
            displacement[:, self.left, None],
        )
        jacobian = torch.block_diag(*local.unbind())
        result = jacobian.T @ self.matrix @ jacobian
        for atom in range(atoms):
            block = result[
                atom * parameters : (atom + 1) * parameters,
                atom * parameters : (atom + 1) * parameters,
            ]
            block.index_put_(
                (self.left, self.right), force[atom, parameters:], accumulate=True
            )
            block.index_put_(
                (self.right, self.left), force[atom, parameters:], accumulate=True
            )
        return result + result.T


def factored_quadratic_gram(
    factor_atoms: FactorAtoms,
    point: Tensor,
    row_weight: Tensor,
    column_weight: Tensor,
    eps: float,
) -> QuadraticFeatureGram:
    """Build T.T D T without constructing visible Jacobians or Hessians.

    D_oi = row_weight[o] * column_weight[i] + eps. Factor functions return
    [input, atom] and [output, atom], as in Kernel.factors. Product-rule terms
    are assembled from Gram matrices of the factors and their derivatives.
    """

    atoms, parameters = point.shape
    left, right = torch.triu_indices(parameters, parameters, device=point.device)

    def one_atom(p: Tensor) -> tuple[Tensor, Tensor]:
        input_factor, output_factor = factor_atoms(p.unsqueeze(0))
        return input_factor[:, 0], output_factor[:, 0]

    with torch.no_grad():
        values = vmap(one_atom)(point)
        first = vmap(jacfwd(one_atom))(point)
        second = vmap(jacfwd(jacfwd(one_atom)))(point)
        # Primitive rows: value, first derivatives, upper-triangular Hessian.
        bundles = [
            torch.cat((v.unsqueeze(-1), j, h[..., left, right]), dim=-1)
            .transpose(1, 2)
            .flatten(0, 1)
            for v, j, h in zip(values, first, second)
        ]

        # Each feature column is the sum of four pairs of primitive factors.
        # Linear: u_i v + u v_i. Quadratic: u_ij v + u_i v_j
        # + u_j v_i + u v_ij, with a half for a diagonal monomial.
        count = parameters + left.numel()
        primitive_count = count + 1
        derivative = torch.arange(1, parameters + 1, device=point.device)
        hessian = torch.arange(parameters + 1, primitive_count, device=point.device)
        zero = torch.zeros(parameters, dtype=torch.long, device=point.device)
        input_indices = torch.stack(
            (
                torch.cat((derivative, hessian)),
                torch.cat((zero, left + 1)),
                torch.cat((zero, right + 1)),
                torch.zeros(count, dtype=torch.long, device=point.device),
            ),
            dim=1,
        )
        output_indices = torch.stack(
            (
                torch.zeros(count, dtype=torch.long, device=point.device),
                torch.cat((derivative, right + 1)),
                torch.cat((zero, left + 1)),
                torch.cat((zero, hessian)),
            ),
            dim=1,
        )
        offsets = (
            torch.arange(atoms, device=point.device)[:, None, None] * primitive_count
        )
        input_indices = (input_indices + offsets).reshape(-1, 4)
        output_indices = (output_indices + offsets).reshape(-1, 4)
        coefficients = point.new_ones(count, 4)
        coefficients[:parameters, 2:] = 0
        coefficients[parameters:] *= torch.where(left == right, 0.5, 1.0)[:, None]
        coefficients = coefficients.repeat(atoms, 1)

        input_bundle, output_bundle = bundles
        matrix = point.new_zeros(atoms * count, atoms * count)
        for input_weight, output_weight, scale in (
            (column_weight, row_weight, 1.0),
            (torch.ones_like(column_weight), torch.ones_like(row_weight), eps),
        ):
            input_gram = (input_bundle * input_weight) @ input_bundle.T
            output_gram = (output_bundle * output_weight) @ output_bundle.T
            for i in range(4):
                for j in range(4):
                    product = input_gram[
                        input_indices[:, i, None], input_indices[None, :, j]
                    ]
                    product = (
                        product
                        * output_gram[
                            output_indices[:, i, None], output_indices[None, :, j]
                        ]
                    )
                    product = (
                        product * coefficients[:, i, None] * coefficients[None, :, j]
                    )
                    matrix.add_(product, alpha=scale)
        # Remove roundoff asymmetry so analytic gradients match value autograd.
        matrix = 0.5 * (matrix + matrix.T)
    return QuadraticFeatureGram(matrix, (atoms, parameters))
