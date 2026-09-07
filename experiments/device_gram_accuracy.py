"""Compare actual first-moment Gram solves against a float64 SVD oracle."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments import mnist_current_api as runner
from experiments.device_optimizer_benchmark import config
from torchcst._derivatives._jacobi import device_pinv_solve
from torchcst._derivatives.frame import GramSystem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=runner._default_data_root())
    parser.add_argument("--output", type=Path, default=Path("output/gram_accuracy.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = runner.load_mnist(
        args.data,
        train_size=8192,
        test_size=2000,
    )
    original = GramSystem.solve
    rows = []

    def compare(self, rhs):
        actual = original(self, rhs)
        a = self.matrix
        rtol = a.shape[0] * torch.finfo(a.dtype).eps
        reference = torch.linalg.pinv(a.double(), rtol=rtol) @ rhs.flatten().double()
        float_svd = torch.linalg.pinv(a) @ rhs.flatten()
        _, actual_valid = device_pinv_solve(a, rhs.flatten(), rtol=rtol)
        double_matrix = 0.5 * (a.double() + a.T.double())
        double_jacobi, valid = device_pinv_solve(
            double_matrix, rhs.flatten().double(), rtol=rtol
        )
        rows.append(
            {
                "relative_device_jacobi": float(
                    (actual.flatten().double() - reference).norm() / reference.norm()
                ),
                "relative_float_svd": float(
                    (float_svd.double() - reference).norm() / reference.norm()
                ),
                "relative_double_jacobi": float(
                    (double_jacobi - reference).norm() / reference.norm()
                ),
                "strict_float64_probe_valid": bool(valid),
                "actual_valid": bool(actual_valid),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        return actual

    for seed in (17, 29):
        cfg = config("device", seed=seed)
        model = runner.build_model(cfg, torch.device("cuda"))
        optimizer = runner.build_optimizer(model, cfg)
        with patch.object(GramSystem, "solve", compare):
            for i in range(4):
                x, y = (
                    data[0][i * 128 : (i + 1) * 128].cuda(),
                    data[1][i * 128 : (i + 1) * 128].cuda(),
                )
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()
                optimizer.check_errors()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
