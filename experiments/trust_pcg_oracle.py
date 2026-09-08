"""Compare PCG updates with an explicit FP64 Jacobian/Gram oracle off the hot path."""

import argparse
import json

import torch

from torchcst import Chart, CSTAdam, CSTLinear
from torchcst._runtime.validation import device_checks
from torchcst.kernels import Amplitude, Gaussian, Separable
from torchcst.optim._tangent_device import solve as spectral_solve
from torchcst.optim.quadratic import MatrixFreeTangentProblem, TangentProblem


class AuditedAdam(CSTAdam):
    def _solve(self, context, expanded):
        c = self.cst_config
        problem = MatrixFreeTangentProblem(context, expanded, learning_rate=c.lr)
        result = super()._solve(context, expanded)
        action = problem.operator
        p = action.prepared
        # Independent dense contraction, used only in this correctness audit.
        j = (
            (
                torch.einsum("ko,kiq->oikq", p._v, p._du)
                + torch.einsum("koq,ki->oikq", p._dv, p._u)
            )
            .flatten(0, 1)
            .flatten(1)
        )
        weight = (action.row[:, None] * action.column[None, :] + action.eps).flatten()
        h = j.T @ (weight[:, None] * j) / c.lr
        oracle = object.__new__(TangentProblem)
        oracle.matrix, oracle.linear = (h + h.T) * 0.5, problem.linear.double()
        oracle.point_shape, oracle.blocked = problem.point_shape, False
        with device_checks() as checks:
            reference = spectral_solve(oracle, c.trust_radius)
        assert torch.stack(checks).all()
        objective, grad = oracle.value_and_gradient(result.displacement.double())
        residual = (grad + result.shift * result.displacement.double()).norm()
        norm_b = problem.linear.double().norm()
        gap = objective - reference.objective
        assert result.converged, {
            "step": len(self.audit_rows),
            "shift": result.shift.item(),
            "relative_residual": result.relative_residual.item(),
            "relative_complementarity": result.relative_complementarity.item(),
            "shift_iterations": getattr(
                result, "shift_iterations", torch.tensor(0)
            ).item(),
            "basis_dimension": getattr(
                result, "basis_dimension", torch.tensor(0)
            ).item(),
        }
        assert residual / norm_b <= c.update_rtol
        assert gap.abs() <= 3 * c.trust_radius * norm_b * c.update_rtol + 1e-12
        self.audit_rows.append(
            {
                "objective": objective.item(),
                "reference_objective": reference.objective.item(),
                "objective_gap": gap.item(),
                "relative_gap": (gap / reference.objective.abs()).item(),
                "dense_relative_residual": (residual / norm_b).item(),
                "reference_relative_residual": (
                    reference.projected_gradient_norm / norm_b
                ).item(),
                "relative_complementarity": result.relative_complementarity.item(),
                "displacement_difference_norm": (
                    result.displacement - reference.displacement
                )
                .norm()
                .item(),
                "displacement_norm": result.displacement.norm().item(),
                "shift": result.shift.item(),
                "solver_iterations": result.iterations.item(),
                "shift_iterations": getattr(
                    result, "shift_iterations", torch.tensor(0)
                ).item(),
                "basis_dimension": getattr(
                    result, "basis_dimension", torch.tensor(0)
                ).item(),
            }
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=85)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--solver", choices=("pcg", "krylov"), default="pcg")
    parser.add_argument("--basis-size", type=int, default=128)
    parser.add_argument(
        "--recompression-action", choices=("pair", "jvp_vjp"), default="pair"
    )
    parser.add_argument("--compression-max-iter", type=int, default=256)
    parser.add_argument("--update-shift-steps", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    site = CSTLinear(
        Chart.linspace(784),
        Chart.linspace(10),
        atoms=args.atoms,
        kernel=Amplitude(
            Separable(input_profile=Gaussian(0.25), output_profile=Gaussian(0.3))
        ),
        dtype=torch.float32,
        device="cuda",
    )
    optimizer = AuditedAdam(
        site,
        device_execution=True,
        update_solver=args.solver,
        update_basis_size=args.basis_size,
        recompression_action=args.recompression_action,
        recompression="pcg",
        first_moment_damping=0.01,
        recompression_max_iter=args.compression_max_iter,
        update_shift_steps=args.update_shift_steps,
    )
    optimizer.audit_rows = []
    x, target = torch.randn(32, 784, device="cuda"), torch.randn(32, 10, device="cuda")
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = (site(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.check_errors()
        losses.append(loss.item())
    print(
        json.dumps(
            {
                "atoms": args.atoms,
                "solver": args.solver,
                "basis_size": args.basis_size,
                "steps": args.steps,
                "compression_max_iter": args.compression_max_iter,
                "update_shift_steps": args.update_shift_steps,
                "losses": losses,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "updates": optimizer.audit_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
