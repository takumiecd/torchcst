"""Fixed Sphere sharp/main study using the unchanged full-step comparison worker.

The generic worker CLI keeps its original sigma3/8 interface. This benchmark
wrapper declares sigma1.25/3 and requires explicit Plans for CST comparisons.
"""

import argparse
import json
from pathlib import Path

from .manifest import REGISTRY
from .scaling_comparison import worker


def route_diagnostics(size, sigma, plan):
    """Independent trajectory and eager GPU-event diagnostics, outside primary peaks.

    Preparation+CSR and entire fused forward are separate repeat experiments;
    their medians are not additive and do not alter the complete-step result.
    """
    import gc
    import statistics

    import torch

    from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_executor import (
        forward_snapshots,
        prepare_snapshots,
    )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.adaptive_kernels import (
        build_index,
    )

    from . import scaling_comparison as common

    gc.collect()
    torch.cuda.empty_cache()
    xx, target, dy = common.shared_inputs(size)
    x = xx.cuda().requires_grad_()
    target, dy = target.cuda(), dy.cuda()
    model, *_ = common.fixture("sphere", size, sigma)
    model = model.cuda()
    common.bind_plan(model, plan)
    step = common.TrainingStep(model, x, target, "sphere", "research_graph")

    def timed(call):
        hold = call()  # compile/warm outside the five recorded observations
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            hold = call()
            end.record()
            torch.cuda.synchronize()
            samples.append(begin.elapsed_time(end))
        if not all(value > 0 for value in samples):
            raise AssertionError("nonpositive diagnostic event timing")
        del hold
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    def checkpoint(clock):
        charts = model.cst_charts()
        # Use the local dict rather than exposing or retaining autograd state.
        diagnostics = {}
        with torch.no_grad():
            output, saved, floors = forward_snapshots(
                x,
                model.atoms.p,
                model.kernel,
                charts,
                plan.recipe,
                diagnostics=diagnostics,
            )
        torch.cuda.synchronize()
        common.require_finite("diagnostic forward", output)
        counts = [side[7].cpu().to(torch.int64) for side in diagnostics["sides"]]
        fallback = diagnostics["fallback"].cpu()
        reasons = diagnostics["reason"].cpu().to(torch.int64)
        if any(len(count) != model.atom_count or (count < 0).any() for count in counts):
            raise AssertionError("invalid complete support counts")
        if not torch.equal(fallback, (reasons != 0).any(dim=1)):
            raise AssertionError("fallback/reason disagreement")
        joint = (counts[0] <= 16) & (counts[1] <= 16)
        if ((~fallback) & ~joint).any():
            raise AssertionError("tiny route violates complete count threshold")
        sides = [
            {
                "mean_count": float(count.float().mean()),
                "max_count": int(count.max()),
                "empty_atoms": int((count == 0).sum()),
                "histogram": torch.bincount(count, minlength=size + 1).tolist(),
                "reason_counts": {
                    name: int(((reasons[:, axis] & bit) != 0).sum())
                    for bit, name in (
                        (1, "descriptor"),
                        (2, "bound"),
                        (4, "rows"),
                        (8, "candidate"),
                        (16, "count"),
                        (32, "normfloor"),
                    )
                },
            }
            for axis, count in enumerate(counts)
        ]
        del output, saved, floors, diagnostics

        def preparation_csr():
            prepared = prepare_snapshots(
                x, model.atoms.p, model.kernel, charts, plan.recipe
            )
            source, sides = prepared[1], prepared[-1]
            csr = [
                build_index(side[0], chart.geometry.radius.to(source))
                for side, chart in zip(sides, charts, strict=True)
            ]
            return prepared, csr

        with torch.no_grad():
            preparation = timed(preparation_csr)
            forward = timed(
                lambda: forward_snapshots(
                    x, model.atoms.p, model.kernel, charts, plan.recipe
                )
            )
        return {
            "step_counter": clock,
            "parameter_hash": common.sphere.tensor_hash(model.atoms.p),
            "oracle": common.oracle_check(model, x, dy, target, "sphere"),
            "atoms": model.atom_count,
            "sides": sides,
            "joint_count_le16_atoms": int(joint.sum()),
            "actual_tiny_atoms": int((~fallback).sum()),
            "fallback_atoms": int(fallback.sum()),
            "timings": {
                "prepare_plus_fresh_two_side_csr": preparation,
                "complete_adaptive_forward": forward,
            },
        }

    initial = checkpoint(0)
    graph, _ = common.capture(step)
    for _ in range(common.ROUNDS):
        graph.replay()
    torch.cuda.synchronize()
    clock = int(step.opt.state[next(model.parameters())]["step"])
    if clock != 24:
        raise AssertionError("diagnostic trajectory must execute exactly 24 updates")
    updated = checkpoint(clock)
    return {
        "scope": "independent initial/24-update trajectory after primary timing and peak; eager GPU-event blocks include launch gaps; medians are not additive",
        "tiny_fusion": "classification/Norm/H/Y are fused; no classification-only timing",
        "samples": 5,
        "warmup_steps": 2,
        "capture_steps": 1,
        "replay_steps": 21,
        "initial": initial,
        "updated": updated,
        "primary_parameter_hash_equality_required": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(1024, 2048), required=True)
    parser.add_argument("--sigma", type=float, choices=(1.25, 3), required=True)
    parser.add_argument(
        "--mode", choices=("research_graph", "dense_graph"), required=True
    )
    parser.add_argument("--linear-plan", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--route-diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "research_graph" and args.linear_plan is None:
        parser.error("research CST comparison requires an explicit Linear Plan")
    if args.mode == "dense_graph" and (args.linear_plan is not None or args.phases):
        parser.error("dense control has no CST Plan or phase diagnostics")
    plan = (
        None
        if args.linear_plan is None
        else REGISTRY.loads_plan(args.linear_plan.read_text())
    )
    if args.route_diagnostics and (
        args.verify_only
        or plan is None
        or plan.algorithm_id != "research_cuda_sphere_polar_support_adaptive"
    ):
        parser.error("route diagnostics require the timed adaptive candidate")
    result = worker(
        "sphere",
        args.size,
        args.sigma,
        args.mode,
        plan=plan,
        verify_only=args.verify_only,
        phases=args.phases,
    )
    result["study_condition"] = (
        "sharp diagnostic" if args.sigma == 1.25 else "ordinary sigma3 main"
    )
    result["route_diagnostics"] = (
        route_diagnostics(args.size, args.sigma, plan)
        if args.route_diagnostics
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "size",
                    "sigma_initial",
                    "study_condition",
                    "timing",
                    "peak_capture_replay",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
