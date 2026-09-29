"""Find the anchor count needed by shared POD/DEIM interpolation."""

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_pod_anchor_basis import (
    basis_from_covariance,
    dense,
    get_weight,
    reconstruct,
    relative,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row_covariance = torch.zeros((64, 64), device="cuda")
    col_covariance = [torch.zeros((16, 16), device="cuda") for _ in range(4)]
    for seed in (21, 37):
        batch = get_weight(seed)
        row_covariance += torch.einsum("bij,bkj->ik", batch, batch)
        for segment in range(4):
            part = batch[:, :, segment * 16 : (segment + 1) * 16]
            col_covariance[segment] += torch.einsum("bri,brj->ij", part, part)
    holdouts = {seed: get_weight(seed) for seed in (53, 71)}
    inputs = {}
    for seed in holdouts:
        torch.manual_seed(seed + 1000)
        inputs[seed] = torch.randn(16, 1024, device="cuda")
    results = []
    for row_rank in (8, 12, 16):
        row_basis, row_indices, row_matrix, row_condition = basis_from_covariance(
            row_covariance, row_rank
        )
        for col_rank in (4, 6, 8):
            column_bases = []
            column_indices = []
            column_matrices = []
            column_conditions = []
            for covariance in col_covariance:
                basis, indices, reconstruction, condition = basis_from_covariance(
                    covariance, col_rank
                )
                column_bases.append(basis)
                column_indices.append(indices)
                column_matrices.append(reconstruction)
                column_conditions.append(condition)
            column_basis = torch.block_diag(*column_bases)
            column_matrix = torch.block_diag(*column_matrices)
            col_indices = torch.cat([
                indices + segment * 16
                for segment, indices in enumerate(column_indices)
            ])
            checks = {}
            for seed, reference in holdouts.items():
                pod = reconstruct(
                    reference, row_indices, col_indices, row_matrix, column_matrix
                )
                projection = (
                    row_basis @ (row_basis.T @ reference @ column_basis)
                    @ column_basis.T
                )
                x = inputs[seed]
                checks[str(seed)] = {
                    "projection_weight_l2": relative(projection, reference),
                    "pod_weight_l2": relative(pod, reference),
                    "pod_output_l2_m16": relative(
                        x @ dense(pod).T, x @ dense(reference).T
                    ),
                }
            item = {
                "row_rank": row_rank,
                "column_rank_per_segment": col_rank,
                "sample_sites_per_block": row_rank * 4 * col_rank,
                "row_condition": row_condition,
                "column_conditions": column_conditions,
                "row_indices": row_indices.tolist(),
                "column_indices": col_indices.tolist(),
                "holdout": checks,
            }
            print(row_rank, col_rank, checks, flush=True)
            results.append(item)
    output = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "train_seeds": [21, 37],
        "holdout_seeds": [53, 71],
        "atom_density": 0.05,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
