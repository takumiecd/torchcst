"""Compare P1, P2, model EMA, and step EMA without constructing dense W.

Run from the repository root with, for example::

    python -m experiments.cst_p2.dense_free_ablation --device cpu

The teacher and learner both use factored Gaussian evaluations.  The script
also replaces ``CSTLinear.dense_weight`` with a function that raises, making a
dense fallback a hard experiment failure rather than a silent memory change.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.compute import CSTLinear, Factored
from torchcst.optim import CSTP2TrustRegion
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


@dataclass(frozen=True)
class Arm:
    name: str
    curvature_scale: float
    model_beta: float
    step_beta: float | None = None
    adaptive_radius: bool = False


@dataclass(frozen=True)
class RunResult:
    arm: str
    seed: int
    initial_loss: float
    final_loss: float
    mean_loss: float
    relative_final_loss: float
    relative_mean_loss: float
    loss_increase_steps: int
    mean_update_ms: float
    mean_dimensionless_step_norm: float
    model_state_numel: int
    step_state_numel: int
    parameter_numel: int
    dense_weight_numel: int
    step_ema_fallbacks: int
    accepted_steps: int
    rejected_steps: int
    final_radius: float


ARMS = (
    Arm("p1_current", curvature_scale=0.0, model_beta=0.0),
    Arm("p2_current", curvature_scale=1.0, model_beta=0.0),
    Arm("p2_model_ema", curvature_scale=1.0, model_beta=0.9),
    Arm("p2_step_ema", curvature_scale=1.0, model_beta=0.0, step_beta=0.9),
    Arm(
        "p2_model_ema_adaptive",
        curvature_scale=1.0,
        model_beta=0.9,
        adaptive_radius=True,
    ),
)


def _normalized_gaussian_columns(mu: Tensor, positions: Tensor, sigma: float) -> Tensor:
    columns = torch.exp(-0.5 * ((mu[:, None] - positions[None, :]) / sigma).square())
    return columns / torch.linalg.vector_norm(columns, dim=0).clamp_min(1e-12)


def _factored_teacher(
    x: Tensor,
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    weights: Tensor,
    sigma: float,
) -> Tensor:
    k_in = _normalized_gaussian_columns(mu_in, source, sigma)
    k_out = _normalized_gaussian_columns(mu_out, target, sigma)
    return ((x @ k_in) * weights) @ k_out.T


def _dataset(seed: int, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    samples, in_features, out_features, teacher_atoms = 192, 16, 12, 12
    mu_in = torch.linspace(0, 1, in_features, device=device, dtype=dtype)
    mu_out = torch.linspace(0, 1, out_features, device=device, dtype=dtype)
    x = torch.randn(
        samples, in_features, generator=generator, device=device, dtype=dtype
    )
    source = torch.rand(teacher_atoms, generator=generator, device=device, dtype=dtype)
    target = torch.rand(teacher_atoms, generator=generator, device=device, dtype=dtype)
    weights = 0.6 * torch.randn(
        teacher_atoms, generator=generator, device=device, dtype=dtype
    )
    clean = _factored_teacher(x, mu_in, mu_out, source, target, weights, sigma=0.16)
    noise = 0.02 * torch.randn(
        clean.shape, generator=generator, device=device, dtype=dtype
    )
    return x, clean + noise, mu_in, mu_out, generator


def _learner(
    seed: int,
    mu_in: Tensor,
    mu_out: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> CSTLinear:
    generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    atoms = 10
    synapses = SynapseStore(
        f"dense-free-p2-{seed}",
        1,
        1,
        atoms,
        spec=RepresentationSpec.continuous(1, 1),
        device=device,
        dtype=dtype,
    )
    synapses.apply(
        [
            SynapseBirth(
                synapses.site,
                torch.rand(atoms, 1, generator=generator, device=device, dtype=dtype),
                torch.rand(atoms, 1, generator=generator, device=device, dtype=dtype),
                0.04
                * torch.randn(atoms, generator=generator, device=device, dtype=dtype),
                torch.arange(atoms, device=device, dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            f"dense-free-p2-{seed}-in",
            mu_in.numel(),
            mu=mu_in[:, None],
            initial_live=mu_in.numel(),
            device=device,
            dtype=dtype,
        ),
        NeuronStore(
            f"dense-free-p2-{seed}-out",
            mu_out.numel(),
            mu=mu_out[:, None],
            initial_live=mu_out.numel(),
            device=device,
            dtype=dtype,
        ),
        synapses,
        GaussianFactor(0.16, learnable=False).to(device=device, dtype=dtype),
        gauge=L2NormalizedColumns(),
        backend=Factored(),
        track_mass=False,
    )

    def dense_forbidden():
        raise AssertionError("dense_weight is forbidden in this experiment")

    layer.dense_weight = dense_forbidden
    return layer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_arm(
    arm: Arm,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    batch_size: int,
    radius: float,
) -> RunResult:
    x, targets, mu_in, mu_out, generator = _dataset(seed, device, dtype)
    layer = _learner(seed, mu_in, mu_out, device, dtype)
    optimizer = CSTP2TrustRegion(
        layer,
        radius=radius,
        beta=arm.model_beta,
        amplitude_scale=0.25,
        curvature_scale=arm.curvature_scale,
        step_beta=arm.step_beta,
        adaptive_radius=arm.adaptive_radius,
        root_iterations=48,
    )

    with torch.no_grad():
        initial_loss = float(F.mse_loss(layer(x), targets))
    losses = [initial_loss]
    update_times = []
    step_norms = []
    loss_increases = 0
    for update_id in range(steps):
        indices = torch.randint(
            x.shape[0],
            (batch_size,),
            generator=generator,
            device=device,
        )
        _synchronize(device)
        started = time.perf_counter()
        context = optimizer.capture_context(update_id)
        layer.set_backward_context(context)
        optimizer.zero_grad()
        batch_x = x.index_select(0, indices)
        batch_targets = targets.index_select(0, indices)
        batch_loss = F.mse_loss(layer(batch_x), batch_targets)
        batch_loss.backward()
        context.observe_microbatch()
        capture = context.finalize_capture()
        layer.set_backward_context(None)
        if arm.adaptive_radius:

            def loss_closure():
                return F.mse_loss(layer(batch_x), batch_targets)

            result = optimizer.step(
                capture,
                closure=loss_closure,
                current_loss=batch_loss.detach(),
            )
        else:
            result = optimizer.step(capture)
        _synchronize(device)
        update_times.append(1e3 * (time.perf_counter() - started))
        scales = optimizer._scales(result.step)
        step_norms.append(float(torch.linalg.vector_norm(result.step / scales)))
        with torch.no_grad():
            new_loss = float(F.mse_loss(layer(x), targets))
        loss_increases += int(new_loss > losses[-1])
        losses.append(new_loss)

    assert optimizer.model_ema.linear is not None
    assert optimizer.model_ema.curvature is not None
    model_state_numel = (
        optimizer.model_ema.linear.numel() + optimizer.model_ema.curvature.numel()
    )
    step_state_numel = 0
    if optimizer.step_ema is not None:
        assert optimizer.step_ema.dimensionless_step is not None
        step_state_numel = optimizer.step_ema.dimensionless_step.numel()
    parameter_numel = sum(
        parameter.numel() for parameter in optimizer.param_groups[0]["params"]
    )
    final_loss = losses[-1]
    mean_loss = float(sum(losses[1:]) / steps)
    return RunResult(
        arm=arm.name,
        seed=seed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        mean_loss=mean_loss,
        relative_final_loss=final_loss / initial_loss,
        relative_mean_loss=mean_loss / initial_loss,
        loss_increase_steps=loss_increases,
        mean_update_ms=float(statistics.mean(update_times)),
        mean_dimensionless_step_norm=float(statistics.mean(step_norms)),
        model_state_numel=model_state_numel,
        step_state_numel=step_state_numel,
        parameter_numel=parameter_numel,
        dense_weight_numel=mu_in.numel() * mu_out.numel(),
        step_ema_fallbacks=optimizer.step_ema_fallbacks,
        accepted_steps=optimizer.accepted_steps,
        rejected_steps=optimizer.rejected_steps,
        final_radius=optimizer.radius,
    )


def _aggregate(results: list[RunResult]) -> dict[str, dict[str, float]]:
    summary = {}
    for arm in ARMS:
        selected = [result for result in results if result.arm == arm.name]
        fields = (
            "relative_final_loss",
            "relative_mean_loss",
            "loss_increase_steps",
            "mean_update_ms",
            "mean_dimensionless_step_norm",
            "model_state_numel",
            "step_state_numel",
            "step_ema_fallbacks",
            "accepted_steps",
            "rejected_steps",
            "final_radius",
        )
        summary[arm.name] = {
            f"{field}_mean": float(
                statistics.mean(getattr(result, field) for result in selected)
            )
            for field in fields
        }
        for field in ("relative_final_loss", "relative_mean_loss", "mean_update_ms"):
            summary[arm.name][f"{field}_stdev"] = float(
                statistics.stdev(getattr(result, field) for result in selected)
                if len(selected) > 1
                else 0.0
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu":
        torch.set_num_threads(1)
    results = [
        run_arm(
            arm,
            seed,
            device=device,
            dtype=dtype,
            steps=args.steps,
            batch_size=args.batch_size,
            radius=args.radius,
        )
        for arm in ARMS
        for seed in range(args.seeds)
    ]
    print(
        json.dumps(
            {
                "config": vars(args),
                "runs": [asdict(result) for result in results],
                "summary": _aggregate(results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
