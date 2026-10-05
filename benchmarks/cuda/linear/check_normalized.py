"""Installed-src integration: independent small truth; optional full-step timing.

Large benchmark is NOT an independent all-A oracle. No experiments imports.
Run from a checkout or an extracted wheel; fixtures and oracle are local modules.
"""

# Wheel selection deliberately precedes imports of torchcst.

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

wheel_root = os.environ.get("CST_PUBLIC_PACKAGE_ROOT") or os.environ.get(
    "CST_PUBLIC_WHEEL_ROOT"
)
if wheel_root:
    sys.path.insert(0, wheel_root)
import torch

from benchmarks.cuda.linear.fixtures import normalized_chart
from benchmarks.cuda.linear.reference import mixed, oracle
from torchcst import CSTLinear, presets
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.dispatch import FixedSelector


def algorithm_selector(algorithm_id):
    algorithm = REGISTRY.get(algorithm_id, revision="v1")
    from torchcst._backends.schema import ExecutionPlan

    plan = ExecutionPlan(algorithm.id, algorithm.revision, algorithm.recipe_type())
    return FixedSelector(plan, registry=REGISTRY)


def check(a, b, tol=3e-4):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a.double(), b.double(), atol=tol, rtol=tol)
    a, b = a.detach(), b.detach()
    return {
        "max": float((a.double() - b.double()).abs().max()),
        "rel_l2": float(
            (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)
        ),
    }


def small_gate(*, selector_factory=None):
    sizes = (1024, 4, 4)
    p = mixed(torch.float32, "cuda")
    p[0, 2] = 511.25
    p[1, 2] = 512.5
    p[2:4, 2] = 512.02978515625
    generator = torch.Generator(device="cuda").manual_seed(21)
    x = torch.randn(2, 3, 16, device="cuda", generator=generator)
    dy = torch.randn(2, 3, 1024, device="cuda", generator=generator)
    reports = {}
    for algorithm_id in ("normalized_full", "normalized_window"):
        model = CSTLinear(
            chart=normalized_chart(sizes, dtype=torch.float32, device="cuda"),
            atoms=p,
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
            selector=algorithm_selector(algorithm_id),
        )
        if selector_factory is not None:
            model.selector = selector_factory(model, algorithm_id)
        xx = x.clone().requires_grad_()
        actual = model(xx)
        tp = model.atoms.p.detach().double().requires_grad_()
        tx = x.double().requires_grad_()
        truth = tx @ oracle(tp, sizes, stored_dtype=torch.float32).T
        ag = torch.autograd.grad(actual, (xx, model.atoms.p), dy)
        tg = torch.autograd.grad(truth, (tx, tp), dy.double())
        reports[algorithm_id] = {
            "y": check(actual, truth),
            "dx": check(ag[0], tg[0]),
            "dp": check(ag[1], tg[1]),
            "w": check(
                model(torch.eye(16, device="cuda")).T,
                oracle(tp, sizes, stored_dtype=torch.float32),
                tol=2e-5,
            ),
        }
        # Release validation graphs before warming the captured training stream.
        # Otherwise leaf AccumulateGrad nodes retain their previous stream.
        del actual, truth, ag, tg, tp, tx, xx
        gc.collect()
        # Capture entire training step, then compare multiple replays from the
        # exact post-capture P/m/v/step state, with a current-centre mutation.
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            fused=True,
            capturable=True,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def step(m, o):
            o.zero_grad(set_to_none=True)
            loss = (m(x) * dy).sum() / dy.numel()
            loss.backward()
            o.step()

        with torch.cuda.stream(stream):
            for _ in range(3):
                step(model, opt)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step(model, opt)
        torch.cuda.synchronize()
        eager = CSTLinear(
            chart=normalized_chart(sizes, dtype=torch.float32, device="cuda"),
            atoms=model.atoms.p.detach(),
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
            selector=algorithm_selector(algorithm_id),
        )
        if selector_factory is not None:
            eager.selector = selector_factory(eager, algorithm_id)
        eo = torch.optim.AdamW(
            eager.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            fused=True,
            capturable=True,
        )
        eo.load_state_dict(copy.deepcopy(opt.state_dict()))
        with torch.no_grad():
            # Route/support changes across the row-window boundary after capture.
            model.atoms.p[0, 2].add_(1.0)
            eager.atoms.p[0, 2].add_(1.0)
            model.atoms.p[1, 1].fill_(math.log(0.199))
            eager.atoms.p[1, 1].fill_(math.log(0.199))
        for _ in range(3):
            graph.replay()
            step(eager, eo)
            torch.cuda.synchronize()
            check(model.atoms.p, eager.atoms.p)
            check(model.atoms.p.grad, eager.atoms.p.grad)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                check(opt.state[model.atoms.p][key], eo.state[eager.atoms.p][key])
        # Updated independent, unscaled all-five check, not just optimizer parity.
        xx = x.clone().requires_grad_()
        y = model(xx)
        tp = model.atoms.p.detach().double().requires_grad_()
        tx = x.double().requires_grad_()
        truth = tx @ oracle(tp, sizes, stored_dtype=torch.float32).T
        aa = torch.autograd.grad(y, (xx, model.atoms.p), dy)
        bb = torch.autograd.grad(truth, (tx, tp), dy.double())
        reports[algorithm_id]["updated"] = {
            "y": check(y, truth),
            "dx": check(aa[0], bb[0]),
            "dp": check(aa[1], bb[1]),
            "w": check(
                model(torch.eye(16, device="cuda")).T,
                oracle(tp, sizes, stored_dtype=torch.float32),
                tol=2e-5,
            ),
        }
    return reports


def benchmark(n, profile, algorithm_id, dense=False, *, selector_factory=None):
    h, j = (32, 32) if n == 1024 else (64, 128)
    sizes = (n, h, j)
    origin = (-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4)
    a = round(0.05 * n * n)
    gen = torch.Generator(device="cpu").manual_seed(21)
    p = torch.empty(a, 5)
    p[:, 0] = torch.rand(a, generator=gen) - 0.5
    p[:, 1] = math.log(3.0 if profile == "broad" else 0.199)
    p[:, 2:] = torch.rand(a, 3, generator=gen) * torch.tensor(
        [(count - 1) * s for count, s in zip(sizes, (1.0, 0.5, 0.5))]
    ) + torch.tensor(origin)
    if profile == "sharp":
        for axis, (spacing, o) in enumerate(zip((1.0, 0.5, 0.5), origin)):
            u = (p[:, axis + 2] - o) / spacing
            near = torch.floor(u + 0.5)
            p[:, axis + 2] = o + near * spacing + (u - near) * 0.04
    sha = hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest()
    p = p.cuda()
    torch.manual_seed(21)
    x = torch.randn(128, n, device="cuda", requires_grad=True)
    target = torch.randn(128, n, device="cuda")
    if dense:
        model = torch.nn.Linear(n, n, bias=False, device="cuda")
        model.weight.data.uniform_(-0.01, 0.01)
    else:
        model = CSTLinear(
            chart=normalized_chart(sizes, origin, dtype=torch.float32, device="cuda"),
            atoms=p,
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
            selector=algorithm_selector(algorithm_id),
        )
        if selector_factory is not None:
            model.selector = selector_factory(model, algorithm_id)
    del p
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        fused=True,
        capturable=True,
    )

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        loss = (model(x) * target).sum() / (128 * n)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()

    def measured(call):
        samples = []
        for _ in range(7):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            call()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - begin) * 1000)
        return {
            "median_ms": sorted(samples)[3],
            "samples_ms": samples,
            "clock": "perf_counter with CUDA synchronization; no event instrumentation",
        }

    eager = measured(step)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    ms = measured(graph.replay)
    torch.cuda.synchronize()
    return {
        "n": n,
        "a": None if dense else a,
        "m": 128,
        "profile": None if dense else profile,
        "algorithm_id": algorithm_id if not dense else "dense",
        "dense": dense,
        "initial_p_sha256": None if dense else sha,
        "eager_ms": eager["median_ms"],
        "graph_ms": ms["median_ms"],
        "eager_wall": eager,
        "graph_wall": ms,
        "allocated_before_capture": baseline,
        "max_allocated_capture_replay": torch.cuda.max_memory_allocated(),
        "max_reserved_capture_replay": torch.cuda.max_memory_reserved(),
        "optimizer_steps": float(next(iter(opt.state.values()))["step"]),
        "scope": "full-shape timing, no independent all-A oracle",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1024, 8192])
    ap.add_argument("--profiles", nargs="+", default=["broad", "sharp"])
    ap.add_argument(
        "--algorithm", nargs="+", default=["normalized_full", "normalized_window"]
    )
    ap.add_argument("--dense", action="store_true")
    ap.add_argument(
        "--dense-only",
        action="store_true",
        help="Skip CST cases for isolated dense measurement",
    )
    ap.add_argument("--skip-small", action="store_true")
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name() != "NVIDIA L4":
        raise RuntimeError("this protocol requires NVIDIA L4")
    import torchcst

    module_path = str(Path(torchcst.__file__).resolve())
    if wheel_root and not Path(module_path).is_relative_to(Path(wheel_root).resolve()):
        raise RuntimeError("torchcst was not imported from requested extracted wheel")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tf32": False,
        "torchcst_file": module_path,
    }
    if not args.skip_small:
        result["small_independent"] = small_gate()
        gc.collect()
        torch.cuda.empty_cache()
        Path(args.output).write_text(json.dumps(result, indent=2))
    if args.bench:
        result["benchmarks"] = []
        for n in args.sizes:
            if n not in (1024, 8192):
                raise ValueError("sizes must be1024 or8192")
            for profile in [] if args.dense_only else args.profiles:
                if profile not in ("broad", "sharp"):
                    raise ValueError("unknown profile")
                for algorithm_id in args.algorithm:
                    result["benchmarks"].append(benchmark(n, profile, algorithm_id))
                    gc.collect()
                    torch.cuda.empty_cache()
                    Path(args.output).write_text(json.dumps(result, indent=2))
            if args.dense or args.dense_only:
                result["benchmarks"].append(benchmark(n, "broad", None, True))
                gc.collect()
                torch.cuda.empty_cache()
                Path(args.output).write_text(json.dumps(result, indent=2))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
