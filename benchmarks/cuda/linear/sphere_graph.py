"""Complete research Sphere Graph steps with an explicit declared coordinate plan.

This measures loss, all backward gradients, AdamW moments/clock and the exact
intrinsic retraction. Public CSTOptimizer remains eager and is the trajectory
oracle. Intrinsic Sphere gradient projection and vector transport are identity.
Public synchronous finite-value guards are not part of the research Graph route;
we validate finite initial values and oracle/state equivalence around replay.
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.cuda.linear import sphere_baseline as baseline
from torchcst import AtomUpdateBinding, AtomUpdateInputs, CSTOptimizer
from torchcst._backends.cuda.algorithms.sphere_polar_update.algorithm import (
    SpherePolarUpdateAlgorithm,
)
from torchcst._backends.cuda.algorithms.sphere_polar_update.plans import FUSED
from torchcst._backends.dispatch import Dispatcher
from torchcst._backends.registry import Registry


def update_dispatcher():
    registry = Registry()
    registry.register(SpherePolarUpdateAlgorithm())
    return Dispatcher(registry=registry)


def validate_geometry_scalars(model):
    """Reject malformed live radius/margin buffers before any base proposal."""
    p = model.atoms.p
    if any(
        not isinstance(s, torch.Tensor)
        or s.dtype != p.dtype
        or s.device != p.device
        or s.numel() != 1
        for chart in model.cst_charts()
        for s in (chart.geometry.radius, chart.geometry.chart_margin)
    ):
        raise ValueError(
            "requires scalar Sphere radius/margin on the atom device and dtype"
        )


def apply_linear_plan(model, plan):
    from benchmarks.cuda.linear.manifest import REGISTRY
    from torchcst._backends.dispatch import FixedSelector

    REGISTRY.validate_plan(plan)
    model.selector = FixedSelector(plan, registry=REGISTRY)
    return "planned"


class ResearchStep:
    """Benchmark-only explicit base proposal; no public optimizer bypass."""

    def __init__(
        self, model, x, target, route="weight-64", *, lr=1e-4, capturable=True
    ):
        if not model.atoms.p.is_cuda or model.atoms.p.dtype != torch.float32:
            raise ValueError("research Sphere steps require CUDA FP32")
        validate_geometry_scalars(model)
        if any(chart.trainable for chart in model.cst_charts()):
            raise ValueError("research Sphere steps require fixed charts")
        if not all(bool(torch.isfinite(t).all()) for t in (model.atoms.p, x, target)):
            raise FloatingPointError("non-finite research initial state")
        self.model, self.x, self.target, self.route, self.lr = (
            model,
            x,
            target,
            route,
            lr,
        )
        self.binding = AtomUpdateBinding(model.operator)
        self.dispatch = update_dispatcher()
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            fused=True,
            capturable=capturable,
        )
        # Metadata selection happens before AdamW's first proposal.
        inputs = AtomUpdateInputs(model.atoms.p.detach().clone(), lr)
        self.binding.validate_inputs(inputs)
        self.dispatch.select(self.binding.build_context(inputs), plan=FUSED)

    def __call__(self):
        validate_geometry_scalars(self.model)
        self.opt.zero_grad(set_to_none=True)
        self.x.grad = None
        y = baseline.forward(self.model, self.x, self.route)
        loss = (y - self.target).square().mean()
        loss.backward()
        previous = self.model.atoms.p.detach().clone()
        self.opt.step()
        self.dispatch.run(self.binding, AtomUpdateInputs(previous, self.lr), plan=FUSED)
        return y, loss


def capture(step, *, warmups=3):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = step()
    torch.cuda.synchronize()
    return graph, outputs


def trajectory_gate(size=32, sigma=3.0, steps=20, route="weight-64", linear_plan=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    base, xx, target, _ = baseline.fixture(size, sigma, batch=4)
    actual = base.cuda()
    if linear_plan is not None:
        route = apply_linear_plan(actual, linear_plan)
    x, target = xx.cuda().requires_grad_(), target.cuda()
    step = ResearchStep(actual, x, target, route)
    graph, (cy, closs) = capture(step)
    reference = copy.deepcopy(actual)
    reference.selector = None  # Independent Torch factored linear trajectory.
    opt = torch.optim.AdamW(
        reference.parameters(), lr=step.lr, weight_decay=0.01, fused=True
    )
    opt.load_state_dict(copy.deepcopy(step.opt.state_dict()))
    for group in opt.param_groups:
        group["capturable"] = False
    public = CSTOptimizer(opt, model=reference)
    rx = x.detach().clone().requires_grad_()
    maxima = {
        key: 0.0
        for key in ("Y", "loss", "dX", "all_dP", "parameters", "exp_avg", "exp_avg_sq")
    }
    initial_width = actual.atoms.p[:, :2].detach().square().sum(-1)
    for _ in range(steps):
        public.zero_grad(set_to_none=True)
        rx.grad = None
        ry = baseline.forward(reference, rx, "factored")
        rloss = (ry - target).square().mean()
        rloss.backward()
        grad = reference.atoms.p.grad.clone()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        for key, aa, bb in (
            ("Y", cy, ry),
            ("loss", closs, rloss),
            ("dX", x.grad, rx.grad),
            ("all_dP", actual.atoms.p.grad, grad),
            ("parameters", actual.atoms.p, reference.atoms.p),
            (
                "exp_avg",
                step.opt.state[actual.atoms.p]["exp_avg"],
                opt.state[reference.atoms.p]["exp_avg"],
            ),
            (
                "exp_avg_sq",
                step.opt.state[actual.atoms.p]["exp_avg_sq"],
                opt.state[reference.atoms.p]["exp_avg_sq"],
            ),
        ):
            e = baseline.error(aa, bb)
            maxima[key] = max(maxima[key], e["max_abs"])
            # Near-zero moment norms make relative errors ill-conditioned; the
            # existing complete optimizer gate uses absolute state tolerances.
            limit = 2e-6 if key == "parameters" else 4e-4
            if e["max_abs"] > limit:
                raise AssertionError((key, e))
        assert torch.equal(
            step.opt.state[actual.atoms.p]["step"], opt.state[reference.atoms.p]["step"]
        )
    changed = bool(
        torch.any(initial_width != actual.atoms.p[:, :2].detach().square().sum(-1))
    )
    if not changed:
        raise AssertionError("Polar width activity clock did not evolve")
    return {
        "status": "PASS",
        "steps": steps,
        "max_abs": maxima,
        "live_width_changed": changed,
    }


def timed(call, samples=21):
    values = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def measure(size, sigma, route, mode, linear_plan=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    serialized_linear_plan = None
    if linear_plan is not None:
        from benchmarks.cuda.linear.manifest import REGISTRY

        serialized_linear_plan = REGISTRY.dump_plan(linear_plan)
    base, xx, target, dy = baseline.fixture(size, sigma)
    fixture_hashes = {
        "parameters": baseline.tensor_hash(base.atoms.p),
        "x": baseline.tensor_hash(xx),
        "target": baseline.tensor_hash(target),
        "sites": [
            baseline.tensor_hash(chart.coordinates) for chart in base.cst_charts()
        ],
    }
    dense = mode.startswith("dense")
    x, target = xx.cuda().requires_grad_(), target.cuda()
    if dense:
        model = torch.nn.Linear(size, size, bias=False, device="cuda")
        before = None
        del base, dy
    else:
        model, dy = base.cuda(), dy.cuda()
        if linear_plan is not None:
            route = apply_linear_plan(model, linear_plan)
        del base
        before = baseline.correctness(model, x, dy, route)
    initial_hashes = {
        "parameters": baseline.tensor_hash(next(model.parameters())),
        "x": baseline.tensor_hash(x),
        "target": baseline.tensor_hash(target),
    }
    if dense:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            fused=True,
            capturable=mode.endswith("graph"),
        )

        def call():
            opt.zero_grad(set_to_none=True)
            x.grad = None
            y = model(x)
            loss = (y - target).square().mean()
            loss.backward()
            opt.step()
            return y, loss
    elif mode == "public_eager":
        opt = baseline.optimizer(model)

        def call():
            return baseline.step(model, opt, x, target, route)
    else:
        call = ResearchStep(model, x, target, route, capturable=mode.endswith("graph"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if mode.endswith("graph"):
        graph, _ = capture(call, warmups=2)
        timing = timed(graph.replay)
    else:
        for _ in range(3):
            call()
        timing = timed(call)
    peaks = {
        "peak_allocated_training_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_training_bytes": torch.cuda.max_memory_reserved(),
    }
    after = None if dense else baseline.correctness(model, x, dy, route)
    return {
        "status": "PASS",
        "size": size,
        "sigma": sigma,
        "batch": 32,
        "atoms": None if dense else model.atom_count,
        "route": route,
        "mode": mode,
        "linear_plan": serialized_linear_plan,
        "protocol": "complete loss/backward/dX/AdamW + declared Sphere update"
        if mode.startswith("research")
        else mode,
        "public_optimizer_guards": mode == "public_eager",
        "timing": timing,
        **peaks,
        "initial_oracle": before,
        "updated_oracle": after,
        "initial_hashes": initial_hashes,
        "shared_fixture_hashes": fixture_hashes,
        "final_parameter_hash": baseline.tensor_hash(next(model.parameters())),
        "warmup_steps": 2 if mode.endswith("graph") else 3,
        "capture_steps": 1 if mode.endswith("graph") else 0,
        "timed_steps": 21,
        "atom_fraction": None if dense else model.atom_count / (size * size),
        "triton": __import__("triton").__version__,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--route", default="weight-64")
    p.add_argument(
        "--mode",
        choices=(
            "public_eager",
            "research_eager",
            "research_graph",
            "dense_eager",
            "dense_graph",
        ),
        default="research_graph",
    )
    p.add_argument(
        "--linear-plan", type=Path, help="registered benchmark ExecutionPlan JSON"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = None
    if args.linear_plan is not None:
        from benchmarks.cuda.linear.manifest import REGISTRY

        plan = REGISTRY.loads_plan(args.linear_plan.read_text())
    result = measure(args.size, args.sigma, args.route, args.mode, plan)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
