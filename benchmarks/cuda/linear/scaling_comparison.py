"""Matched research complete-step protocol for Sphere and Strip/Torus.

Each worker owns one fresh model/optimizer. Geometry contracts remain distinct:
equal N, batch and initial sigma do not imply equal mathematical work. Public
eager guards remain public; the explicit research proposal/update Graph has the
same boundary as the existing geometry-specific benchmark wrappers.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from benchmarks.cuda.linear import sphere_baseline as sphere
from benchmarks.cuda.linear import torus_profile_product as torus
from benchmarks.cuda.linear.manifest import REGISTRY, load_run
from torchcst import AtomUpdateBinding, AtomUpdateInputs, CSTLinear, CSTOptimizer
from torchcst._backends.dispatch import Dispatcher, FixedSelector
from torchcst._backends.registry import Registry
from torchcst._backends.schema import DefaultRecipe, ExecutionPlan

MODES = ("research_graph", "research_eager", "public_eager", "dense_graph")
LR, DECAY, ROUNDS = 1e-4, 0.01, 21
ROOT = Path(__file__).resolve().parents[3]


def shared_inputs(size, *, batch=32, seed=41):
    """Independent generator: model initialization cannot consume this stream."""
    gen = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch, size, generator=gen) for _ in range(3))


def dense_fixture(size, *, seed=41):
    model = torch.nn.Linear(size, size, bias=False)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.weight.uniform_(-(size**-0.5), size**-0.5, generator=gen)
    return model


def fixture(geometry, size, sigma, *, batch=32, atoms=None):
    if geometry == "sphere":
        model, *_ = sphere.fixture(size, sigma, batch=batch, atoms=atoms)
    elif geometry == "torus":
        # Reuse the frozen physical Strip fixture and activity initialization.
        case = load_run(
            ROOT
            / f"benchmarks/cuda/linear/cases/torus-sparse-weight-1024-sigma{int(sigma)}.json",
            ROOT / "benchmarks/cuda/linear/plans-torus-sparse-weight.json",
        ).case
        case = replace(
            case,
            size=size,
            rows=batch,
            atoms=int(size * size * 0.05) if atoms is None else atoms,
        )
        op = torus.fixture_operator(case)
        model = CSTLinear(
            chart=op.charts[0], kernel=op.kernel, atoms=torus.initialize(case)
        )
    else:
        raise ValueError("requires sphere or torus geometry")
    return model, *shared_inputs(size, batch=batch)


def baseline_plan(geometry, sigma):
    if geometry == "torus":
        from torchcst._backends.cuda.algorithms.linear.torus_profile_product.sparse_weight.algorithm import (
            SparseWeightRecipe,
        )

        return ExecutionPlan(
            "research_cuda_torus_profile_product_sparse_weight",
            "v1",
            SparseWeightRecipe(),
        )
    if sigma == 3:
        from torchcst._backends.cuda.algorithms.linear.sphere_polar.weight_algorithm import (
            SphereWeightRecipe,
        )

        return ExecutionPlan(
            "research_cuda_sphere_polar_fused_weight", "v1", SphereWeightRecipe(64, 32)
        )
    from torchcst._backends.cuda.algorithms.linear.sphere_polar.algorithm import (
        SphereRecipe,
    )

    return ExecutionPlan(
        "research_cuda_sphere_polar_blocked", "v1", SphereRecipe(4096, True)
    )


def update_plan(geometry):
    registry = Registry()
    if geometry == "sphere":
        from torchcst._backends.cuda.algorithms.sphere_polar_update.algorithm import (
            SpherePolarUpdateAlgorithm,
        )
        from torchcst._backends.cuda.algorithms.sphere_polar_update.plans import FUSED

        registry.register(SpherePolarUpdateAlgorithm())
        plan = FUSED
    else:
        from torchcst._backends.torch.algorithms.torus_profile_product_update.algorithm import (
            TorusProfileProductUpdateAlgorithm,
        )

        registry.register(TorusProfileProductUpdateAlgorithm())
        plan = ExecutionPlan(
            "research_torch_torus_profile_product_update", "v1", DefaultRecipe()
        )
    return Dispatcher(registry=registry), plan


def bind_plan(model, plan):
    REGISTRY.validate_plan(plan)
    model.selector = FixedSelector(plan, registry=REGISTRY)


class ProposalUpdate:
    """Explicit research AdamW proposal and declared intrinsic coordinate law."""

    def __init__(self, model, geometry, *, capturable):
        self.model = model
        self.binding = AtomUpdateBinding(model.operator)
        self.dispatch, self.plan = update_plan(geometry)
        if geometry == "sphere":
            from .sphere_graph import validate_geometry_scalars

            validate_geometry_scalars(model)
        if any(chart.trainable for chart in model.cst_charts()):
            raise ValueError("research protocol requires fixed charts")
        if not bool(torch.isfinite(model.atoms.p).all()):
            raise FloatingPointError("nonfinite initial parameters")
        inp = AtomUpdateInputs(model.atoms.p.detach().clone(), LR)
        self.binding.validate_inputs(inp)
        self.dispatch.select(self.binding.build_context(inp), plan=self.plan)
        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=DECAY,
            fused=True,
            capturable=capturable,
        )

    def __call__(self, events=None):
        if events is not None:
            events[0].record()
        previous = self.model.atoms.p.detach().clone()
        if events is not None:
            events[1].record()
        self.opt.step()
        if events is not None:
            events[2].record()
        self.dispatch.run(self.binding, AtomUpdateInputs(previous, LR), plan=self.plan)
        if events is not None:
            events[3].record()


class TrainingStep:
    def __init__(self, model, x, target, geometry, mode):
        self.model, self.x, self.target, self.mode = model, x, target, mode
        self.update = None
        if mode == "public_eager":
            dispatch, plan = update_plan(geometry)
            self.opt = CSTOptimizer(
                torch.optim.AdamW(
                    model.parameters(), lr=LR, weight_decay=DECAY, fused=True
                ),
                model=model,
                update_selector=FixedSelector(plan, registry=dispatch.registry),
            )
        elif mode == "dense_graph":
            self.opt = torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                weight_decay=DECAY,
                fused=True,
                capturable=True,
            )
        else:
            self.update = ProposalUpdate(
                model, geometry, capturable=mode.endswith("graph")
            )
            self.opt = self.update.opt

    def __call__(self, events=None, update_events=None):
        if events is not None:
            events[0].record()
        self.opt.zero_grad(set_to_none=True)
        self.x.grad = None
        y = self.model(self.x)
        loss = (y - self.target).square().mean()
        if events is not None:
            events[1].record()
        loss.backward()
        if events is not None:
            events[2].record()
        if self.update is None:
            self.opt.step()
        else:
            self.update(update_events)
        if events is not None:
            events[3].record()
        return y, loss


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = call()
    torch.cuda.synchronize()
    return graph, outputs


def support_summary(model, geometry, plan, *, chunk=128):
    """Untimed bounded complete-axis oracle factors, never a runtime cache."""
    capacity = (
        (getattr(plan.recipe, "support_capacity", None),) * 2
        if geometry == "sphere"
        else (
            getattr(plan.recipe, "circle_capacity", None),
            getattr(plan.recipe, "section_capacity", None),
        )
    )
    counts = [{"max_count": 0, "overflow_atoms": 0, "empty_atoms": 0} for _ in range(2)]
    ref = copy.deepcopy(model).double()
    with torch.no_grad():
        for start in range(0, ref.atom_count, chunk):
            p = ref.atoms.p[start : start + chunk]
            factors = (
                sphere.oracle_factors(ref, p, scale_amplitude=False)
                if geometry == "sphere"
                else torus.oracle_factors(ref.kernel, p, ref.chart)[:2]
            )
            for side, values in enumerate(factors):
                num = values.count_nonzero(dim=0)
                counts[side]["max_count"] = max(
                    counts[side]["max_count"], int(num.max())
                )
                counts[side]["empty_atoms"] += int((num == 0).sum())
                if capacity[side] is not None:
                    counts[side]["overflow_atoms"] += int((num > capacity[side]).sum())
    return {
        "scope": "independent FP64 full-axis positive raw supports; FP32 boundary rounding may change runtime counts",
        "capacity": capacity,
        "sides": counts,
    }


def oracle_check(model, x, dy, target, geometry):
    """Independent fixed cotangent, all atoms/sites, no sampled accuracy gate."""
    truth = (
        sphere.oracle_vjp(model, x, dy, chunk=128)
        if geometry == "sphere"
        else torus.oracle_vjp(
            copy.deepcopy(model.kernel).double(),
            model.atoms.p,
            x,
            dy,
            model.chart,
            chunk=128,
        )
    )
    xx = x.detach().clone().requires_grad_()
    y = model(xx)
    gx, gp = torch.autograd.grad(y, (xx, model.atoms.p), dy)
    errors = {}
    for name, actual, expected in zip(
        ("Y", "dX", "all_dP"), (y, gx, gp), truth, strict=True
    ):
        metrics = sphere.error(actual, expected)
        if metrics["max_abs"] > 4e-4 or metrics["relative_l2"] > 4e-4:
            raise AssertionError((name, metrics))
        if geometry == "torus":
            torch.testing.assert_close(actual.double(), expected, atol=4e-4, rtol=4e-4)
        errors[name] = metrics
    actual_loss = float((y.detach().double() - target.double()).square().mean())
    oracle_loss = float((truth[0] - target.double()).square().mean())
    return {
        "status": "PASS",
        "scope": "independent FP64 full sites and all atoms, fixed dy",
        "errors": errors,
        "mse_loss": actual_loss,
        "oracle_mse_loss": oracle_loss,
    }


def state_trajectory(geometry, sigma, *, steps=20):
    """Same cotangent public oracle isolates proposal, moments and geometry law."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # Torus benchmark declarations accept N1024/2048; this state-only gate does
    # not evaluate a large Linear and therefore needs only seventeen atoms.
    model, *_ = fixture(
        geometry, 64 if geometry == "sphere" else 1024, sigma, batch=4, atoms=17
    )
    model = model.cuda()
    actual = ProposalUpdate(model, geometry, capturable=True)
    gradient = torch.full_like(model.atoms.p, 0.03)
    model.atoms.p.grad = gradient
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual()
    reference = copy.deepcopy(model)
    base = torch.optim.AdamW(
        reference.parameters(), lr=LR, weight_decay=DECAY, fused=True
    )
    base.load_state_dict(copy.deepcopy(actual.opt.state_dict()))
    for group in base.param_groups:
        group["capturable"] = False
    public = CSTOptimizer(base, model=reference)
    maxima = {name: 0.0 for name in ("parameters", "exp_avg", "exp_avg_sq")}
    moment_tol = 4e-4 if geometry == "sphere" else 2e-5
    for index in range(steps):
        cotangent = (
            torch.arange(gradient.numel(), device="cuda").reshape_as(gradient) * 0.13
            + index
        ).sin() * 0.03
        gradient.copy_(cotangent)
        reference.atoms.p.grad = cotangent.clone()
        public.step()
        graph.replay()
        torch.cuda.synchronize()
        for name, aa, bb in (
            ("parameters", model.atoms.p, reference.atoms.p),
            (
                "exp_avg",
                actual.opt.state[model.atoms.p]["exp_avg"],
                base.state[reference.atoms.p]["exp_avg"],
            ),
            (
                "exp_avg_sq",
                actual.opt.state[model.atoms.p]["exp_avg_sq"],
                base.state[reference.atoms.p]["exp_avg_sq"],
            ),
        ):
            metric = sphere.error(aa, bb)["max_abs"]
            maxima[name] = max(maxima[name], metric)
            if metric > (2e-6 if name == "parameters" else moment_tol):
                raise AssertionError((name, metric))
        if not torch.equal(
            actual.opt.state[model.atoms.p]["step"],
            base.state[reference.atoms.p]["step"],
        ):
            raise AssertionError("step counter differs")
    return {
        "status": "PASS",
        "steps": steps,
        "same_cotangent": True,
        "exact_step_counter": True,
        "max_abs": maxima,
    }


def phase_diagnostics(step, *, samples=5):
    """Separate bounded Graph after primary peaks; phases are not additive."""
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
    updates = (
        None
        if step.update is None
        else [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(events, updates)
    phases = {name: [] for name in ("forward_loss", "backward", "optimizer")}
    components = (
        {name: [] for name in ("snapshot", "adamw", "geometry_update")}
        if updates
        else {}
    )
    for _ in range(samples):
        graph.replay()
        torch.cuda.synchronize()
        for i, name in enumerate(phases):
            phases[name].append(events[i].elapsed_time(events[i + 1]))
        for i, name in enumerate(components):
            components[name].append(updates[i].elapsed_time(updates[i + 1]))

    def compact(values):
        return {
            name: {"median_ms": statistics.median(items), "samples_ms": items}
            for name, items in values.items()
        }

    return {
        "scope": "separate instrumented Graph after 24-step oracle; not additive complete-step estimates",
        "samples": samples,
        "phases": compact(phases),
        "optimizer_components": compact(components),
    }


def source_metadata():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(
        Path(__file__).parent.glob("*.py")
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        # Frozen pool archives deliberately have no .git. The pool spec owns
        # its provenance; a caller hint is separate from the actual file hashes.
        commit = None
    return {
        "source_commit": commit,
        "source_commit_hint": os.environ.get("CST_FROZEN_SOURCE_COMMIT"),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }


def fixture_metadata(model):
    declaration = asdict(model.execution_declaration())
    payload = json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()
    return {
        "operator_declaration": declaration,
        "declaration_sha256": hashlib.sha256(payload).hexdigest(),
    }


def widths(model, geometry):
    from torchcst._backends.torch.parameterizations import polar_amp_width as polar

    with torch.no_grad():
        amp, alpha = polar._amplitude_and_alpha(model.kernel, model.atoms.p[:, :2])
        values = (
            polar._bandwidth_sigmas(model.kernel, amp, alpha)
            if geometry == "sphere"
            else (polar._sigma_bounds(model.kernel, amp, alpha, side="input")[0],)
        )
        return [value.cpu() for value in values]


def worker(geometry, size, sigma, mode, *, plan=None, verify_only=False, phases=False):
    if mode not in MODES:
        raise ValueError("unsupported mode")
    if phases and mode == "public_eager":
        raise ValueError(
            "public synchronous guards cannot be captured for phase diagnostics"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dense = mode == "dense_graph"
    xx, target, dy = shared_inputs(size)
    x, target = xx.cuda().requires_grad_(), target.cuda()
    if dense:
        model = dense_fixture(size).cuda()
        dy = None
    else:
        model, *_ = fixture(geometry, size, sigma)
        model = model.cuda()
        plan = plan or baseline_plan(geometry, sigma)
        bind_plan(model, plan)
        dy = dy.cuda()
    initial_hashes = {
        "parameters": sphere.tensor_hash(next(model.parameters())),
        "x": sphere.tensor_hash(x),
        "target": sphere.tensor_hash(target),
    }
    geometry_metadata = None if dense else fixture_metadata(model)
    width0 = None if dense else widths(model, geometry)
    before = None if dense else oracle_check(model, x, dy, target, geometry)
    support0 = None if dense else support_summary(model, geometry, plan)
    step = TrainingStep(model, x, target, geometry, mode)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    graph = None
    captured = mode.endswith("graph")
    if mode.endswith("graph"):
        graph, _ = capture(step)
        call = graph.replay
    else:
        for _ in range(3):
            step()
        call = step
    samples = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        if not verify_only:
            samples.append((time.perf_counter() - start) * 1000)
    peaks = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    after = None if dense else oracle_check(model, x, dy, target, geometry)
    support1 = None if dense else support_summary(model, geometry, plan)
    width_changes = None
    if not dense:
        width1 = widths(model, geometry)
        width_changes = [
            {
                "changed_atoms": int((a != b).sum()),
                "max_abs_change": float((a - b).abs().max()),
                "final_min": float(b.min()),
                "final_max": float(b.max()),
            }
            for a, b in zip(width0, width1, strict=True)
        ]
        if not all(change["changed_atoms"] for change in width_changes):
            raise AssertionError("live activity widths did not evolve")
    final_hash = sphere.tensor_hash(next(model.parameters()))
    final_step = int(step.opt.state[next(model.parameters())]["step"])
    if final_step != 24:
        raise AssertionError(("expected exactly 24 updates", final_step))
    phase = None
    if phases:
        # Diagnostic state is independent, including its proposal/moments/clock.
        diagnostic_model = copy.deepcopy(model)
        diagnostic = TrainingStep(
            diagnostic_model,
            x.detach().clone().requires_grad_(),
            target.clone(),
            geometry,
            mode,
        )
        diagnostic.opt.load_state_dict(copy.deepcopy(step.opt.state_dict()))
        phase = phase_diagnostics(diagnostic)
        phase["initial_step_counter"] = final_step
        phase["capture_steps"] = 1
        phase["replay_steps"] = 5
    return {
        "status": "PASS",
        "geometry": geometry,
        "size": size,
        "sigma_initial": sigma,
        "batch": 32,
        "atoms": None if dense else model.atom_count,
        "atom_fraction": None if dense else model.atom_count / size**2,
        "mode": mode,
        "public_optimizer_guards": mode == "public_eager",
        "mathematical_work": "geometry-specific; equal initial sigma is not equal work",
        "geometry_fixture": geometry_metadata,
        "linear_plan": None if dense else REGISTRY.dump_plan(plan),
        "loss": "mean((Y-target)^2)",
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": DECAY, "fused": True},
        "timing": None
        if verify_only
        else {"median_ms": statistics.median(samples), "samples_ms": samples},
        "warmup_steps": 2 if captured else 3,
        "capture_steps": 1 if captured else 0,
        "replay_or_timed_steps": ROUNDS,
        "total_updates": final_step,
        "initial_oracle": before,
        "updated_oracle": after,
        "initial_support": support0,
        "updated_support": support1,
        "live_width_changes": width_changes,
        "initial_hashes": initial_hashes,
        "final_parameter_hash": final_hash,
        "peak_capture_replay": peaks,
        "gpu_process_bytes": None,
        "memory_scope": "own model/optimizer/input/target/dy; capture/replay included; oracle scratch excluded",
        "phase_diagnostics": phase,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
        },
        **source_metadata(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("sphere", "torus"), required=True)
    parser.add_argument("--size", type=int, choices=(1024, 2048), required=True)
    parser.add_argument("--sigma", type=float, choices=(3, 8), required=True)
    parser.add_argument("--mode", choices=MODES, default="research_graph")
    parser.add_argument("--linear-plan", type=Path, action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--state-gate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.linear_plan) > 1:
        if args.state_gate:
            parser.error("state gate has no Linear plan cohort")
        labels = [p.stem for p in args.linear_plan]
        if len(set(labels)) != len(labels):
            parser.error("plan filenames must have unique stems")
        records = []
        for plan in args.linear_plan:
            output = args.output.parent / f"{args.output.stem}-{plan.stem}.json"
            command = [
                sys.executable,
                "-m",
                __spec__.name,
                "--geometry",
                args.geometry,
                "--size",
                str(args.size),
                "--sigma",
                str(args.sigma),
                "--mode",
                args.mode,
                "--linear-plan",
                str(plan),
                "--output",
                str(output),
            ]
            command += [
                flag
                for flag, enabled in (
                    ("--verify-only", args.verify_only),
                    ("--phases", args.phases),
                )
                if enabled
            ]
            subprocess.run(command, check=True)
            records.append(
                {"plan_file": str(plan), "result_file": str(output), "status": "PASS"}
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"status": "PASS", "isolated_plan_workers": records}, indent=2)
            + "\n"
        )
        return
    if args.state_gate:
        result = state_trajectory(args.geometry, args.sigma)
    else:
        plan = (
            None
            if not args.linear_plan
            else REGISTRY.loads_plan(args.linear_plan[0].read_text())
        )
        result = worker(
            args.geometry,
            args.size,
            args.sigma,
            args.mode,
            plan=plan,
            verify_only=args.verify_only,
            phases=args.phases,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "geometry",
                    "size",
                    "sigma_initial",
                    "mode",
                    "timing",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
