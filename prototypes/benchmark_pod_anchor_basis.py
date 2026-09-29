"""Screen offline POD/DEIM interpolation against independent CST atom seeds."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.diagnose_cst_sampled_blocks import (
    column_basis,
    evenly_spaced_indices,
    interpolation_matrix,
)
from torchcst.nn._backends._preparation import prepare


def blocks(weight):
    return weight.reshape(16, 64, 16, 64).permute(0, 2, 1, 3).reshape(256, 64, 64)


def dense(block_weights):
    return block_weights.reshape(16, 16, 64, 64).permute(0, 2, 1, 3).reshape(1024, 1024)


def get_weight(seed):
    torch.manual_seed(seed)
    layer = BlockStripLinear((1024, 1024), (64, 64), round(0.05 * 1024**2), device="cuda")
    with torch.no_grad():
        weight, canonical = mapped_control(
            layer, prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
            canonical_chunk=4,
        )
    if not canonical["passed"]:
        raise RuntimeError(f"CST control failed for seed {seed}: {canonical}")
    return blocks(weight)


def deim(basis):
    selected = []
    for column in range(basis.shape[1]):
        residual = basis[:, column]
        if selected:
            coeff = torch.linalg.solve(
                basis[selected, :column], basis[selected, column]
            )
            residual = residual - basis[:, :column] @ coeff
        residual = residual.abs().clone()
        if selected:
            residual[selected] = -1
        selected.append(int(residual.argmax()))
    return torch.tensor(selected, device=basis.device)


def basis_from_covariance(covariance, rank):
    _, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -rank:].flip(-1)
    indices = deim(basis)
    reconstruction = basis @ torch.linalg.inv(basis[indices, :])
    condition = float(torch.linalg.cond(basis[indices, :]))
    return basis, indices, reconstruction, condition


def reconstruct(original, row_indices, col_indices, row_reconstruction, col_reconstruction):
    sample = original.index_select(1, row_indices).index_select(2, col_indices)
    return row_reconstruction @ sample @ col_reconstruction.T


def relative(actual, reference):
    return float((actual - reference).norm() / reference.norm().clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = [get_weight(seed) for seed in (21, 37)]
    row_covariance = torch.zeros((64, 64), device="cuda")
    col_covariance = [torch.zeros((16, 16), device="cuda") for _ in range(4)]
    for batch in training:
        row_covariance += torch.einsum("bij,bkj->ik", batch, batch)
        for segment in range(4):
            part = batch[:, :, segment * 16 : (segment + 1) * 16]
            col_covariance[segment] += torch.einsum("bri,brj->ij", part, part)
    row_basis, row_indices, row_matrix, row_cond = basis_from_covariance(row_covariance, 8)
    column_bases = []
    column_indices = []
    column_matrices = []
    column_conditions = []
    for covariance in col_covariance:
        basis, indices, reconstruction, condition = basis_from_covariance(covariance, 4)
        column_bases.append(basis)
        column_indices.append(indices)
        column_matrices.append(reconstruction)
        column_conditions.append(condition)
    column_basis_full = torch.block_diag(*column_bases)
    column_reconstruction = torch.block_diag(*column_matrices)
    col_indices = torch.cat([indices + segment * 16 for segment, indices in enumerate(column_indices)])

    polynomial_row_indices = torch.tensor(evenly_spaced_indices(64, 8), device="cuda")
    polynomial_row_matrix = interpolation_matrix(64, polynomial_row_indices.tolist()).to("cuda")
    polynomial_column_matrix, polynomial_column_indices = column_basis(4)
    polynomial_column_matrix = polynomial_column_matrix.to("cuda")
    polynomial_column_indices = torch.tensor(polynomial_column_indices, device="cuda")

    holdout = {}
    for seed in (53, 71):
        reference = get_weight(seed)
        pod = reconstruct(reference, row_indices, col_indices, row_matrix, column_reconstruction)
        projection = (
            row_basis @ (row_basis.T @ reference @ column_basis_full)
            @ column_basis_full.T
        )
        polynomial = reconstruct(
            reference, polynomial_row_indices, polynomial_column_indices,
            polynomial_row_matrix, polynomial_column_matrix,
        )
        torch.manual_seed(seed + 1000)
        x16 = torch.randn(16, 1024, device="cuda")
        x128 = torch.randn(128, 1024, device="cuda")
        holdout[str(seed)] = {
            "projection_weight_l2": relative(projection, reference),
            "pod_weight_l2": relative(pod, reference),
            "polynomial_weight_l2": relative(polynomial, reference),
            "pod_output_l2_m16": relative(x16 @ dense(pod).T, x16 @ dense(reference).T),
            "pod_output_l2_m128": relative(x128 @ dense(pod).T, x128 @ dense(reference).T),
            "polynomial_output_l2_m16": relative(x16 @ dense(polynomial).T, x16 @ dense(reference).T),
            "polynomial_output_l2_m128": relative(x128 @ dense(polynomial).T, x128 @ dense(reference).T),
        }
        print(seed, holdout[str(seed)], flush=True)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "training_seeds": [21, 37],
        "holdout_seeds": [53, 71],
        "shape": [1024, 1024],
        "atom_density": 0.05,
        "row_rank": 8,
        "column_rank_per_segment": 4,
        "row_indices": row_indices.tolist(),
        "column_indices": col_indices.tolist(),
        "row_condition": row_cond,
        "column_conditions": column_conditions,
        "holdout": holdout,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
