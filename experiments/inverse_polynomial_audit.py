"""Freeze weighted tangent systems for a fixed-degree inverse accuracy audit."""

import argparse
import sys
from pathlib import Path

import torch

from experiments import local_tangent_learning as runner
from torchcst.optim.quadratic import MatrixFreeTangentProblem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    saved = []
    original = runner.TimedAdam._solve
    count = 0

    def record(self, context, expanded):
        nonlocal count
        count += 1
        if count in (32, 128, 384):
            problem = MatrixFreeTangentProblem(
                context, expanded, learning_rate=self.cst_config.lr
            )
            op = problem.operator
            f = op.prepared
            tensors = {
                k: getattr(f, "_" + k).detach().cpu()
                for k in ("point", "u", "v", "du", "dv")
            }
            tensors.update(
                row=op.row.cpu(),
                column=op.column.cpu(),
                rhs=-problem.linear.detach().double().cpu().flatten(),
            )
            # Store a true factor-owned action to check the independent dense oracle.
            x = torch.linspace(
                -1,
                1,
                problem.linear.numel(),
                device=problem.linear.device,
                dtype=torch.float64,
            ).reshape(problem.point_shape)
            tensors.update(probe=x.cpu().flatten(), action_probe=op(x).cpu().flatten())
            saved.append(
                {
                    "step": count,
                    "tensors": tensors,
                    "eps": op.eps,
                    "rate": op.rate,
                    "radius": self.cst_config.trust_radius,
                }
            )
        return original(self, context, expanded)

    runner.TimedAdam._solve = record
    sys.argv = [
        "runner",
        "--data",
        str(args.data),
        "--output",
        str(args.output.with_suffix(".training.json")),
        "--approximation",
        "full",
        "--seed",
        "17",
        "--steps",
        "384",
        "--eval-every",
        "8",
    ]
    runner.main()
    torch.save(saved, args.output)


if __name__ == "__main__":
    main()
