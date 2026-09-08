"""Convex tangent objectives and direct full/block ball solves."""

import math

import torch

from .solvers.quartic import QuarticSolveResult


class TangentProblem:
    def __init__(self, context, moments, *, learning_rate):
        self.point_shape = context.current_point.shape
        self.linear = moments.first.corrected.constant.detach()
        metric = moments.second.metric
        self.blocked = hasattr(metric, "blocks")
        if self.blocked:
            matrix = metric.blocks
        else:
            matrix = context.geometry.cross(
                context.current_point, context.current_point, metric=metric
            )
        self.matrix = (0.5 / learning_rate) * (matrix + matrix.transpose(-1, -2))

    def action(self, d):
        if self.blocked:
            return torch.einsum("kpq,kq->kp", self.matrix, d)
        return (self.matrix @ d.flatten()).reshape(self.point_shape)

    def value_and_gradient(self, d):
        qd = self.action(d)
        return (self.linear * d).sum() + 0.5 * (d * qd).sum(), self.linear + qd


def solve_tangent(problem, *, radius):
    """One (batched for blocks) eigensolve plus one global secular equation.

    The same Euclidean ball couples the atom blocks; there is no per-atom
    clipping. Full matrices are never assembled from the block representation.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if (
        problem.matrix.dtype not in (torch.float32, torch.float64)
        or problem.matrix.device.type == "mps"
    ):
        raise ValueError("CSTAdam requires CPU/CUDA float32/float64")
    with torch.no_grad():
        matrix = problem.matrix.double()
        linear = problem.linear.double()
        if not bool(torch.isfinite(matrix).all() & torch.isfinite(linear).all()):
            raise FloatingPointError("non-finite tangent objective")
        values, vectors = torch.linalg.eigh(matrix)
        tolerance = 32 * torch.finfo(problem.matrix.dtype).eps * values.abs().max()
        if bool((values < -tolerance).any()):
            raise FloatingPointError("tangent metric is not positive semidefinite")
        values = values.clamp_min(0)
        rhs = -linear if problem.blocked else -linear.flatten()
        spectral = (vectors.transpose(-1, -2) @ rhs.unsqueeze(-1)).squeeze(-1)
        ev, coeff = torch.stack((values.flatten(), spectral.flatten())).cpu().tolist()
        cutoff = 8 * torch.finfo(matrix.dtype).eps * max(max(ev), 1e-30)
        norm_rhs = math.hypot(*coeff)
        null_ok = all(
            abs(b) <= 8 * torch.finfo(matrix.dtype).eps * max(norm_rhs, 1e-30)
            for v, b in zip(ev, coeff)
            if v <= cutoff
        )
        interior = [b / v if v > cutoff else 0.0 for v, b in zip(ev, coeff)]
        shift = 0.0
        if not null_ok or math.hypot(*interior) > radius:
            low, high = 0.0, norm_rhs / radius
            for _ in range(80):
                mid = (low + high) / 2
                norm = math.hypot(
                    *(b / max(v + mid, 1e-300) for v, b in zip(ev, coeff))
                )
                if norm > radius:
                    low = mid
                else:
                    high = mid
            shift = high
            interior = [b / max(v + shift, 1e-300) for v, b in zip(ev, coeff)]
        spectral_solution = values.new_tensor(interior).reshape_as(values)
        d = (
            (vectors @ spectral_solution.unsqueeze(-1))
            .squeeze(-1)
            .reshape(problem.point_shape)
            .to(problem.linear)
        )
        d *= min(1.0, radius / max(float(d.norm()), 1e-300))
        objective, gradient = problem.value_and_gradient(d)
        stationarity = (gradient + shift * d).norm()
        threshold = (
            128 * torch.finfo(d.dtype).eps * max(1.0, float(problem.linear.norm()))
        )
        return QuarticSolveResult(
            d.detach(),
            objective.detach(),
            stationarity.detach(),
            0,
            1,
            1,
            bool(stationarity <= threshold),
            bool(d.norm() >= radius * (1 - 1e-6)),
        )
