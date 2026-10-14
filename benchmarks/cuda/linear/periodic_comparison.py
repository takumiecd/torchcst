"""Matched periodic v1 and Fourier rev4/rev5 complete-step research protocol.

This standalone protocol JSON is not the benchmark submission/DB schema.
Numerical gates cover full axes and all atoms before and after 24 real updates.
"""

import argparse
import copy
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from benchmarks.cuda.linear import periodic_profile_product as fixture
from benchmarks.cuda.linear.scaling_comparison import (
    DECAY,
    LR,
    ROUNDS,
    capture,
    dense_fixture,
    finite_error,
    fixture_metadata,
    require_finite,
    shared_inputs,
    source_metadata,
    validate_measurement,
)
from torchcst import CSTOptimizer

KINDS = ("matrix-torch", "matrix-triton", "factor", "torch-factored", "dense")
PRIMARY = ("matrix-torch", "factor", "dense")
CATALOG = Path(__file__).with_name("plans-periodic-profile-product.json")


def case_definition(size, rho):
    return {
        "schema": "periodic-case.research.v1",
        "size": size,
        "rho_initial": float(rho),
        "batch": 32,
        "seed": 41,
        "atoms": int(0.05 * size * size),
        "periods": [size, size],
        "origin": [0.0, 0.0],
        "spacing": [1.0, 1.0],
        "dtype": "float32",
        "tf32": False,
        "width_bounds": {
            "minimum": 1.0,
            "birth": 1.0,
            "maximum": 16.0,
            "upper_floor": 1.0,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": LR,
            "weight_decay": DECAY,
            "fused": True,
            "capturable": True,
        },
        "schedule": {
            "eager_warmups": 2,
            "initial_replays": 1,
            "timed_replays": 21,
            "actual_updates_required": 24,
        },
        "oracle": {
            "full_sites": True,
            "all_atoms": True,
            "max_abs": 4e-4,
            "relative_l2": 4e-4,
        },
    }


def load_case(path):
    value = json.loads(Path(path).read_text())
    if (
        type(value.get("size")) is not int
        or value["size"] not in (1024, 2048)
        or type(value.get("rho_initial")) not in (int, float)
        or value["rho_initial"] not in (3, 8)
    ):
        raise ValueError("only the four predeclared periodic cases are accepted")
    if value != case_definition(value["size"], value["rho_initial"]):
        raise ValueError("case differs from the frozen periodic protocol")
    return value


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def bind_plan(model, kind):
    if kind == "torch-factored":
        return None
    from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog
    from torchcst._backends.cuda.algorithms.linear.periodic_product.recipe import (
        PeriodicRecipe,
    )
    from torchcst._backends.dispatch import FixedSelector

    entries = decode_catalog(json.loads(CATALOG.read_text()))
    plan = next(entry.plan for entry in entries if entry.id == kind)
    expected_id = (
        "research_cuda_periodic_prepared_factor"
        if kind == "factor"
        else "research_cuda_periodic_grouped_matrix"
    )
    if (
        plan.algorithm_id != expected_id
        or plan.algorithm_revision != "v1"
        or plan.recipe
        != PeriodicRecipe(gemm="triton" if kind == "matrix-triton" else "torch")
    ):
        raise ValueError(
            "Plan differs from the frozen grouped preparation/contraction recipe"
        )
    REGISTRY.validate_plan(plan)
    model.selector = FixedSelector(plan, registry=REGISTRY)
    return REGISTRY.dump_plan(plan)


class TrainingStep:
    def __init__(self, model, x, target, *, dense=False):
        self.model, self.x, self.target, self.dense = model, x, target, dense
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=DECAY, fused=True, capturable=True
        )

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
        if self.dense:
            self.opt.step()
        else:
            fixture.graph_update(self.model, self.opt, events=update_events)
        if events is not None:
            events[3].record()
        return y, loss


def oracle_check(model, x, dy):
    expected = fixture.oracle_vjp(model, x, dy)
    xx = x.detach().clone().requires_grad_()
    y = model(xx)
    gx, gp = torch.autograd.grad(y, (xx, model.atoms.p), dy)
    errors = {}
    for name, actual, truth in zip(
        ("Y", "dX", "all_dP"), (y, gx, gp), expected, strict=True
    ):
        metric = finite_error(name, actual, truth)
        if metric["max_abs"] > 4e-4 or metric["relative_l2"] > 4e-4:
            raise AssertionError((name, metric))
        errors[name] = metric
    return {
        "status": "PASS",
        "scope": "independent FP64 complete axes/all atoms/all four cotangents",
        "errors": errors,
    }


def state_gate(*, device="cuda", steps=20, fourier_case=None):
    """Same-cotangent public optimizer truth, including seam crossings and moments."""
    if fourier_case is None:
        model = fixture.fixture(32, 3, atoms=17).to(device)
    else:
        ff = profile_protocol(fourier_case)
        model = ff.fixture(fourier_case).to(device)
    with torch.no_grad():
        model.atoms.p[:, 2] = model.chart.geometry.periods[0] - 1e-5
        model.atoms.p[:, 3] = 1e-5
    reference = copy.deepcopy(model)
    actual = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=DECAY,
        fused=True,
        capturable=device == "cuda",
    )
    base = torch.optim.AdamW(
        reference.parameters(), lr=LR, weight_decay=DECAY, fused=True
    )
    public = CSTOptimizer(base, model=reference)
    gradient = torch.full_like(model.atoms.p, 0.03)
    model.atoms.p.grad = gradient
    reference.atoms.p.grad = gradient.clone()
    fixture.graph_update(model, actual)
    public.step()
    one_step = finite_error("public one-step", model.atoms.p, reference.atoms.p)
    if one_step["max_abs"] > 2e-6:
        raise AssertionError(one_step)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fixture.graph_update(model, actual)
        torch.cuda.synchronize()
        if int(actual.state[model.atoms.p]["step"]) != 1:
            raise AssertionError("capture must not advance optimizer clock")
        call = graph.replay
    else:
        call = lambda: fixture.graph_update(model, actual)
    maxima = {name: 0.0 for name in ("parameters", "exp_avg", "exp_avg_sq")}
    for index in range(steps):
        cotangent = (
            torch.arange(gradient.numel(), device=device).reshape_as(gradient) * 0.13
            + index
        ).sin() * 0.03
        gradient.copy_(cotangent)
        reference.atoms.p.grad = cotangent.clone()
        public.step()
        call()
        if device == "cuda":
            torch.cuda.synchronize()
        for name, aa, bb in (
            ("parameters", model.atoms.p, reference.atoms.p),
            (
                "exp_avg",
                actual.state[model.atoms.p]["exp_avg"],
                base.state[reference.atoms.p]["exp_avg"],
            ),
            (
                "exp_avg_sq",
                actual.state[model.atoms.p]["exp_avg_sq"],
                base.state[reference.atoms.p]["exp_avg_sq"],
            ),
        ):
            error = finite_error(name, aa, bb)["max_abs"]
            maxima[name] = max(maxima[name], error)
            if error > (2e-6 if name == "parameters" else 2e-5):
                raise AssertionError((name, error))
        if int(actual.state[model.atoms.p]["step"]) != int(
            base.state[reference.atoms.p]["step"]
        ):
            raise AssertionError("public/Graph clock mismatch")
    if int(actual.state[model.atoms.p]["step"]) != steps + 1:
        raise AssertionError("wrong actual state-gate update count")
    return {
        "status": "PASS",
        "same_cotangent_updates": steps,
        "one_step_truth": one_step,
        "max_abs": maxima,
        "final_clock": steps + 1,
    }


def forward_stages(step, kind):
    """Fixed post-primary forward copy; these stages are not a training-step sum."""
    if kind not in ("matrix-torch", "matrix-triton", "factor"):
        return None
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.periodic_product import kernels
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _axis,
        _matmul,
        _prepare,
        _scatter,
    )
    from torchcst._backends.cuda.algorithms.linear.periodic_product.recipe import (
        PeriodicRecipe,
    )

    model = copy.deepcopy(step.model)
    x = step.x.detach().clone()
    p = model.atoms.p.detach()
    n = model.in_features
    sizes = (len(x), n, n, float(n), float(n), 0.0, 0.0)
    recipe = PeriodicRecipe(gemm="triton" if kind == "matrix-triton" else "torch")
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]

    def call():
        events[0].record()
        source = p.contiguous().clone()
        packed = _prepare(source, model.kernel, sizes, recipe)
        events[1].record()
        if kind.startswith("matrix"):
            w = x.new_zeros((n, n))
            kernels.assemble[(tr.cdiv(len(p), recipe.atom_group),)](
                packed,
                w,
                len(p),
                n,
                n,
                float(n),
                float(n),
                0.0,
                0.0,
                recipe.patch_sites,
                recipe.atom_group,
                num_warps=4,
                enable_fp_fusion=False,
            )
            events[2].record()
            y = _matmul(x, w.T, recipe)
        else:
            h = _axis(x, packed, sizes, recipe, output=False)
            events[2].record()
            y = _scatter(h, packed, sizes, recipe, output=True)
        events[3].record()
        return y

    call()  # Warm compilation/preparation outside diagnostic capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = call()
    graph.replay()
    torch.cuda.synchronize()
    names = (
        ("snapshot_and_preparation", "W_zero_and_assembly", "forward_GEMM")
        if kind.startswith("matrix")
        else ("snapshot_and_preparation", "support_local_H", "support_scatter_Y")
    )
    phases = {name: [] for name in names}
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        for i, name in enumerate(phases):
            phases[name].append(events[i].elapsed_time(events[i + 1]))
    require_finite("fixed-stage Y", y)
    return {
        "scope": "fixed post24 forward copy; no optimizer; nonadditive diagnostic",
        "samples_ms": phases,
        "parameters_unchanged": tensor_hash(p) == tensor_hash(step.model.atoms.p),
    }


def phase_diagnostics(step):
    """Separate copy; record one initial and five diagnostic replays, no timing sum."""
    model = copy.deepcopy(step.model)
    xx = step.x.detach().clone().requires_grad_()
    diagnostic = TrainingStep(model, xx, step.target.clone(), dense=step.dense)
    diagnostic.opt.load_state_dict(copy.deepcopy(step.opt.state_dict()))
    model(xx)  # Fresh live binding/metadata initialization outside capture.
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
    updates = (
        None
        if step.dense
        else [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        diagnostic(events, updates)
    graph.replay()
    torch.cuda.synchronize()
    phases = {name: [] for name in ("forward_loss", "backward", "optimizer")}
    components = (
        {}
        if updates is None
        else {
            name: [] for name in ("snapshot", "adamw", "polar_and_periodic_retraction")
        }
    )
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        for i, name in enumerate(phases):
            phases[name].append(events[i].elapsed_time(events[i + 1]))
        for i, name in enumerate(components):
            components[name].append(updates[i].elapsed_time(updates[i + 1]))
    clock = int(diagnostic.opt.state[next(model.parameters())]["step"])
    if clock != 30:
        raise AssertionError(("diagnostic expected clock30", clock))
    return {
        "scope": "separate instrumented copy after primary24; not additive",
        "phases_ms": phases,
        "optimizer_components_ms": components,
        "initial_clock": 24,
        "final_clock": clock,
    }


def worker(size, rho, kind, *, phases=False, verify_only=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result = {
        "status": "RUNNING",
        "kind": kind,
        "size": size,
        "rho_initial": rho,
        "batch": 32,
        "seed": 41,
        "atoms": None if kind == "dense" else int(0.05 * size * size),
        **source_metadata(),
    }
    case = case_definition(size, rho)
    result["case_definition"] = case
    result["plan_catalog_sha256"] = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
    result["case_sha256"] = hashlib.sha256(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stage = "fixture"
    try:
        xx, target, dy = shared_inputs(size)
        x, target, dy = xx.cuda().requires_grad_(), target.cuda(), dy.cuda()
        dense = kind == "dense"
        model = (
            dense_fixture(size).cuda() if dense else fixture.fixture(size, rho).cuda()
        )
        result["plan"] = None if dense else bind_plan(model, kind)
        result["initial_hashes"] = {
            "parameters": tensor_hash(next(model.parameters())),
            "x": tensor_hash(x),
            "target": tensor_hash(target),
            "dy": tensor_hash(dy),
        }
        result["fixture"] = None if dense else fixture_metadata(model)
        result["runtime"] = {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "tf32": False,
            "dtype": "float32",
        }
        width0 = None if dense else fixture.widths(model).detach().cpu()
        stage = "initial-full-oracle"
        result["initial_oracle"] = None if dense else oracle_check(model, x, dy)
        step = TrainingStep(model, x, target, dense=dense)
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stage = "primary-capture-and-replay"
        graph, outputs = capture(step)
        samples = []
        for _ in range(ROUNDS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        import statistics

        timing = (
            None
            if verify_only
            else {"median_ms": statistics.median(samples), "samples_ms": samples}
        )
        peaks = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        validate_measurement(timing, peaks)
        clock = int(step.opt.state[next(model.parameters())]["step"])
        if clock != 24:
            raise AssertionError(("primary expected actual clock24", clock))
        for value in outputs:
            require_finite("final output/loss", value)
        for parameter in model.parameters():
            require_finite("final parameter", parameter)
            if parameter.grad is not None:
                require_finite("final parameter gradient", parameter.grad)
            for value in step.opt.state[parameter].values():
                if isinstance(value, torch.Tensor):
                    require_finite("optimizer state", value)
        require_finite("final dX", x.grad)
        result.update(
            timing=timing,
            peak_capture_replay=peaks,
            actual_updates=clock,
            schedule={
                "eager_warmups": 2,
                "initial_replays": 1,
                "timed_replays": 21,
                "capture_executes_update": False,
            },
            optimizer={
                "name": "AdamW",
                "lr": LR,
                "weight_decay": DECAY,
                "fused": True,
                "capturable": True,
            },
            gpu_process_bytes=None,
            memory_scope="own model/optimizer/X/target/dy; capture/replay included; oracle/diagnostic scratch excluded",
        )
        stage = "updated-full-oracle"
        result["updated_oracle"] = None if dense else oracle_check(model, x, dy)
        if int(step.opt.state[next(model.parameters())]["step"]) != 24:
            raise AssertionError("correctness reads changed the primary clock")
        if not dense:
            width1 = fixture.widths(model).detach().cpu()
            changed = int((width0 != width1).sum())
            if not changed:
                raise AssertionError("live widths did not evolve")
            result["width_changes"] = {
                "changed_atoms": changed,
                "max_abs_change": float((width0 - width1).abs().max()),
                "initial_range": [float(width0.min()), float(width0.max())],
                "final_range": [float(width1.min()), float(width1.max())],
                "initial_hash": tensor_hash(width0),
                "final_hash": tensor_hash(width1),
            }
        result["final_parameter_hash"] = tensor_hash(next(model.parameters()))
        stage = "separate-phase-diagnostics"
        result["phase_diagnostics"] = phase_diagnostics(step) if phases else None
        result["forward_stage_diagnostics"] = (
            forward_stages(step, kind) if phases else None
        )
        result["verify_only"] = verify_only
        result["status"] = "PASS"
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="OOM", failed_stage=stage, error=str(error))
    except Exception as error:  # noqa: BLE001 - Persist every failed worker before its nonzero exit.
        result.update(
            status="FAIL",
            failed_stage=stage,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    return result


def validate_record(record):
    """Independent local research schema gate; never a submission-schema claim."""
    if record["kind"] not in KINDS or record["status"] not in ("PASS", "OOM", "FAIL"):
        raise ValueError("invalid periodic worker state")
    if record["status"] != "PASS":
        if not record.get("failed_stage") or not record.get("error"):
            raise ValueError("failure must retain stage and error")
        return
    if record["actual_updates"] != 24 or record["batch"] != 32:
        raise ValueError("requires the frozen batch/update schedule")
    if record["runtime"]["tf32"] or record["runtime"]["dtype"] != "float32":
        raise ValueError("requires IEEE FP32")
    validate_measurement(record["timing"], record["peak_capture_replay"])
    if record["kind"] != "dense":
        if record["atoms"] != int(0.05 * record["size"] ** 2):
            raise ValueError("requires fixed five-percent atom count")
        chart = record["fixture"]["operator_declaration"]["layout"]["chart"]
        n = record["size"]
        if (
            chart["kind"] != "periodic_grid"
            or tuple(chart["grid_shape"]) != (n, n)
            or tuple(chart["geometry"]["periods"]) != (n, n)
            or tuple(chart["origin"]) != (0.0, 0.0)
            or chart["output_dims"] != 1
        ):
            raise ValueError(
                "comparison requires canonical D2 unit-spacing periodic lattice"
            )
        for name in ("initial_oracle", "updated_oracle"):
            oracle = record[name]
            if oracle["status"] != "PASS" or set(oracle["errors"]) != {
                "Y",
                "dX",
                "all_dP",
            }:
                raise ValueError("incomplete full numerical gate")
            for error in oracle["errors"].values():
                if not all(
                    0 <= error[key] <= 4e-4 for key in ("max_abs", "relative_l2")
                ):
                    raise ValueError("invalid numerical gate")
        if not 0 < record["width_changes"]["changed_atoms"] <= record["atoms"]:
            raise ValueError("live widths must evolve")


def assess(records):
    """Keep matrix, support-factor and dense outcomes distinct; never select OOM."""
    kinds = {r["kind"]: r for r in records}
    for key in PRIMARY:
        if key not in kinds or kinds[key]["status"] != "PASS":
            return {
                "status": "FAIL",
                "reason": f"required primary {key} missing or not PASS",
            }
    valid = [r for r in records if r["status"] == "PASS"]
    for record in records:
        validate_record(record)
    for key in ("x", "target", "dy"):
        if len({r["initial_hashes"][key] for r in valid}) != 1:
            raise AssertionError(f"unmatched {key}")
    if (
        len({r["initial_hashes"]["parameters"] for r in valid if r["kind"] != "dense"})
        != 1
    ):
        raise AssertionError("unmatched CST parameters")
    for key in (
        "source_hashes",
        "source_commit",
        "source_commit_hint",
        "case_sha256",
        "plan_catalog_sha256",
    ):
        if len({json.dumps(r[key], sort_keys=True) for r in valid}) != 1:
            raise AssertionError(f"unmatched {key}")
    cst = [r for r in valid if r["kind"] != "dense"]
    for key in ("declaration_sha256", "tensor_hashes", "tensor_specs"):
        if len({json.dumps(r["fixture"][key], sort_keys=True) for r in cst}) != 1:
            raise AssertionError(f"unmatched fixture {key}")
    if any(r["status"] == "FAIL" for r in records):
        return {
            "status": "FAIL",
            "reason": "worker numerical/runtime failure preserved",
        }
    candidate = kinds["factor"]
    if any(r.get("verify_only", False) for r in valid):
        return {"status": "PASS", "verification_only": True, "qualifies": False}
    controls = [
        kinds[key]
        for key in ("matrix-torch", "matrix-triton")
        if key in kinds and kinds[key]["status"] == "PASS"
    ]
    comparisons = {}
    for control in controls:
        ratio = candidate["timing"]["median_ms"] / control["timing"]["median_ms"]
        memory = (
            candidate["peak_capture_replay"]["allocated_bytes"]
            / control["peak_capture_replay"]["allocated_bytes"]
        )
        comparisons[control["kind"]] = {
            "time_ratio": ratio,
            "allocated_ratio": memory,
            "qualifies": (ratio < 0.97 and memory <= 1)
            or (memory <= 0.95 and ratio <= 1.03),
        }
    return {
        "status": "PASS",
        "gate": ">3% faster/no allocated increase OR >=5% lower allocated/<=3% slower against every valid matrix control",
        "comparisons": comparisons,
        "qualifies": all(v["qualifies"] for v in comparisons.values()),
        "dense_scope": "different parameterization; reported engineering control",
    }


# Rev4 extends this runner with an explicitly different mathematical fixture.
# Legacy periodic v1 cases and qualification rules above remain unchanged.
FOURIER_KINDS = (
    "fourier-shared",
    "fourier-direct",
    "torch-factored",
    "torch-shared",
    "dense",
)
FOURIER_PRIMARY = ("fourier-shared", "fourier-direct", "torch-factored", "dense")
FOURIER_DENSITY_PRIMARY = ("fourier-shared", "fourier-direct", "dense")
FOURIER_CATALOG = Path(__file__).with_name("plans-fourier-tiles.json")


MATERN_PRIMARY = ("matern-shared", "matern-direct", "dense")
MATERN_CATALOG = Path(__file__).with_name("plans-matern-mixture.json")


def profile_protocol(case):
    if case is not None and case.get("schema") == "matern-mixture-case.research.v7":
        from benchmarks.cuda.linear import matern_mixture_product

        return matern_mixture_product
    from benchmarks.cuda.linear import fourier_profile_product

    return fourier_profile_product


def profile_catalog(case):
    return MATERN_CATALOG if fourier_revision(case) == "v7" else FOURIER_CATALOG


def fourier_primary(case):
    if case["schema"] == "matern-mixture-case.research.v7":
        return MATERN_PRIMARY
    return (
        FOURIER_DENSITY_PRIMARY
        if case["schema"] == "fourier-case.research.v5"
        else FOURIER_PRIMARY
    )


def fourier_revision(case):
    if case["schema"] == "matern-mixture-case.research.v7":
        return "v7"
    return "v5" if case["schema"] == "fourier-case.research.v5" else "v4"


def bind_fourier_plan(model, kind):
    if kind in ("matern-shared", "matern-direct"):
        from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog
        from torchcst import FixedSelector

        entries = decode_catalog(json.loads(MATERN_CATALOG.read_text()))
        plan = next(entry.plan for entry in entries if entry.id == kind)
        REGISTRY.validate_plan(plan)
        model.selector = FixedSelector(plan, registry=REGISTRY)
        return REGISTRY.dump_plan(plan)
    if kind == "torch-factored":
        return None
    from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog
    from torchcst import (
        DefaultRecipe,
        ExecutionPlan,
        FixedSelector,
        make_algorithm_registry,
    )

    if kind == "torch-shared":
        plan = ExecutionPlan("torch_fourier_tiles", "v1", DefaultRecipe())
        registry = make_algorithm_registry()
    else:
        from torchcst._backends.cuda.algorithms.linear.fourier_tiles.recipe import (
            FourierTilesRecipe,
        )

        entries = decode_catalog(json.loads(FOURIER_CATALOG.read_text()))
        plan = next(entry.plan for entry in entries if entry.id == kind)
        if (
            plan.algorithm_id != "research_cuda_fourier_tiles_triton"
            or plan.algorithm_revision != "v1"
        ):
            raise ValueError("requires the frozen Fourier Triton revision")
        if plan.recipe != FourierTilesRecipe(
            strategy="shared" if kind == "fourier-shared" else "direct",
            block_atoms=32,
            block_batch=16,
        ):
            raise ValueError("requires the frozen shared/direct recipe")
        registry = REGISTRY
    registry.validate_plan(plan)
    model.selector = FixedSelector(plan, registry=registry)
    return registry.dump_plan(plan)


def fourier_oracle_check(model, x, dy, *, case=None):
    ff = profile_protocol(case)

    if case is not None and fourier_revision(case) == "v7":
        truth = ff.oracle_vjp(model, x, dy, chunk=case["oracle"]["chunk_atoms"])
        scope = "independent CPU FP64 native periodic image sum at all sites; actual sampled norm derivatives and all atom cotangents; chunks64"
    elif case is not None and fourier_revision(case) == "v5":
        from benchmarks.cuda.linear.fourier_native_oracle import oracle_vjp

        truth = oracle_vjp(model, x, dy, chunk=case["oracle"]["chunk_atoms"])
        scope = (
            "independent CPU FP64 native H squared at every global site; all atom "
            "cotangents; vectorized native H generation; full-site norm derivative; "
            "atom chunks64"
        )
    else:
        truth = ff.oracle_vjp(model, x, dy)
        scope = "independent CPU FP64 native H squared at every global site; all atom cotangents; atom chunks64"
    xx = x.detach().clone().requires_grad_()
    y = model(xx)
    gx, gp = torch.autograd.grad(y, (xx, model.atoms.p), dy)
    errors = {}
    for name, actual, expected in zip(
        ("Y", "dX", "all_dP"), (y, gx, gp), truth, strict=True
    ):
        metric = finite_error(name, actual.detach().cpu(), expected)
        if metric["max_abs"] > 4e-4 or metric["relative_l2"] > 4e-4:
            raise AssertionError((name, metric))
        errors[name] = metric
    radial = (gp[:, :2] * model.atoms.p[:, :2]).sum(-1)
    require_finite("task stop-gradient radial cotangent", radial)
    return {
        "status": "PASS",
        "scope": scope,
        "errors": errors,
        "max_abs_radial_task_gradient": float(radial.detach().abs().max()),
    }


def fourier_worker(case, kind, *, phases=False, verify_only=False):
    ff = profile_protocol(case)

    ff.load_case(case)
    if fourier_revision(case) in ("v5", "v7") and kind not in fourier_primary(case):
        raise ValueError("density cohort permits only shared/direct/dense workers")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    size, batch = case["size"], case["batch"]
    result = {
        "schema": f"fourier-worker.research.{fourier_revision(case)}",
        "status": "RUNNING",
        "kind": kind,
        "size": size,
        "batch": batch,
        "atoms": None if kind == "dense" else case["atoms"],
        "weight_shape": [size, size],
        "case_definition": case,
        **source_metadata(),
    }
    result["plan_catalog_sha256"] = hashlib.sha256(
        profile_catalog(case).read_bytes()
    ).hexdigest()
    result["case_sha256"] = hashlib.sha256(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stage = "fixture"
    try:
        xx, target, dy = shared_inputs(size, batch=batch, seed=case["seed"])
        x, target, dy = xx.cuda().requires_grad_(), target.cuda(), dy.cuda()
        dense = kind == "dense"
        model = (
            dense_fixture(size, seed=case["seed"]).cuda()
            if dense
            else ff.fixture(case).cuda()
        )
        result["plan"] = None if dense else bind_fourier_plan(model, kind)
        result["initial_hashes"] = {
            "parameters": tensor_hash(next(model.parameters())),
            "x": tensor_hash(x),
            "target": tensor_hash(target),
            "dy": tensor_hash(dy),
        }
        result["fixture"] = None if dense else fixture_metadata(model)
        result["runtime"] = {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "tf32": False,
            "dtype": "float32",
        }
        width0 = None if dense else ff.widths(model).detach().cpu()
        stage = "initial-full-oracle"
        result["initial_oracle"] = (
            None if dense else fourier_oracle_check(model, x, dy, case=case)
        )
        step = TrainingStep(model, x, target, dense=dense)
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_start = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        stage = "primary-capture-and-replay"
        graph, outputs = capture(step)
        peaks_capture = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        samples = []
        for _ in range(ROUNDS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        import statistics

        timing = (
            None
            if verify_only
            else {"median_ms": statistics.median(samples), "samples_ms": samples}
        )
        peaks = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        validate_measurement(timing, peaks)
        clock = int(step.opt.state[next(model.parameters())]["step"])
        if clock != 24:
            raise AssertionError(("primary expected actual clock24", clock))
        for value in outputs:
            require_finite("final output/loss", value)
        require_finite("final dX", x.grad)
        for parameter in model.parameters():
            require_finite("final parameter", parameter)
            require_finite("final gradient", parameter.grad)
            for value in step.opt.state[parameter].values():
                if isinstance(value, torch.Tensor):
                    require_finite("optimizer state", value)
        result.update(
            timing=timing,
            peak_capture_replay=peaks,
            peak_warmup_capture=peaks_capture,
            memory_before_warmup=memory_start,
            actual_updates=clock,
            schedule=case["schedule"],
            optimizer=case["optimizer"],
            gpu_process_bytes=None,
            memory_scope="own model/optimizer/X/target/dy; warmup/capture/replay included; independent CPU oracle and diagnostics excluded",
        )
        stage = "updated-full-oracle"
        result["updated_oracle"] = (
            None if dense else fourier_oracle_check(model, x, dy, case=case)
        )
        if int(step.opt.state[next(model.parameters())]["step"]) != 24:
            raise AssertionError("oracle changed primary optimizer clock")
        if not dense:
            width1 = ff.widths(model).detach().cpu()
            changed = int((width0 != width1).sum())
            if not changed:
                raise AssertionError("live widths did not evolve")
            result["width_changes"] = {
                "changed_atoms": changed,
                "max_abs_change": float((width0 - width1).abs().max()),
                "initial_range": [float(width0.min()), float(width0.max())],
                "final_range": [float(width1.min()), float(width1.max())],
                "initial_hash": tensor_hash(width0),
                "final_hash": tensor_hash(width1),
            }
        result["final_parameter_hash"] = tensor_hash(next(model.parameters()))
        stage = "separate-phase-diagnostics"
        result["phase_diagnostics"] = phase_diagnostics(step) if phases else None
        result["verify_only"] = verify_only
        result["status"] = "PASS"
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="OOM", failed_stage=stage, error=str(error))
    except Exception as error:  # noqa: BLE001 - Preserve failed validation before nonzero exit.
        result.update(
            status="FAIL",
            failed_stage=stage,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    if result["status"] != "PASS":
        # Qualification failures must never publish a timing that could be selected.
        result["timing"] = None
    return result


def validate_fourier_record(record):
    ff = profile_protocol(record.get("case_definition"))

    if record["kind"] not in (*FOURIER_KINDS, *MATERN_PRIMARY) or record[
        "status"
    ] not in (
        "PASS",
        "FAIL",
        "OOM",
    ):
        raise ValueError("invalid Fourier worker state")
    if record["status"] != "PASS":
        if (
            not record.get("failed_stage")
            or not record.get("error")
            or record.get("timing") is not None
        ):
            raise ValueError("failure must retain stage/error without timings")
        return
    case = ff.load_case(record["case_definition"])
    if fourier_revision(case) in ("v5", "v7") and (
        record["schema"] != f"fourier-worker.research.{fourier_revision(case)}"
        or record["kind"] not in fourier_primary(case)
        or record["weight_shape"] != [1024, 1024]
    ):
        raise ValueError("requires the exact 1024 density cohort and weight shape")
    if (
        record["actual_updates"] != 24
        or record["batch"] != case["batch"]
        or record["size"] != case["size"]
    ):
        raise ValueError("requires frozen Fourier dimensions and actual updates")
    if record["runtime"]["tf32"] or record["runtime"]["dtype"] != "float32":
        raise ValueError("requires IEEE FP32")
    validate_measurement(record["timing"], record["peak_capture_replay"])
    if record["kind"] != "dense":
        if record["atoms"] != case["atoms"]:
            raise ValueError("atom count differs from Fourier case")
        for name in ("initial_oracle", "updated_oracle"):
            gate = record[name]
            if gate["status"] != "PASS" or set(gate["errors"]) != {"Y", "dX", "all_dP"}:
                raise ValueError("incomplete full Fourier gate")
            if any(
                not all(0 <= error[key] <= 4e-4 for key in ("max_abs", "relative_l2"))
                for error in gate["errors"].values()
            ):
                raise ValueError("failed numerical gate")
        if not 0 < record["width_changes"]["changed_atoms"] <= record["atoms"]:
            raise ValueError("live widths must evolve")


def assess_fourier(records):
    for r in records:
        validate_fourier_record(r)
    kinds = {r["kind"]: r for r in records}
    cases = [r["case_definition"] for r in records if "case_definition" in r]
    primary = fourier_primary(cases[0]) if cases else FOURIER_PRIMARY
    if any(key not in kinds or kinds[key]["status"] != "PASS" for key in primary):
        return {
            "status": "FAIL",
            "reason": f"required {'/'.join(primary)} worker missing or not PASS",
        }
    if any(r["status"] == "FAIL" for r in records):
        return {"status": "FAIL", "reason": "numerical/runtime failure retained"}
    valid = [r for r in records if r["status"] == "PASS"]
    for key in ("x", "target", "dy"):
        if len({r["initial_hashes"][key] for r in valid}) != 1:
            raise ValueError(f"unmatched {key}")
    cst = [r for r in valid if r["kind"] != "dense"]
    if len({r["initial_hashes"]["parameters"] for r in cst}) != 1:
        raise ValueError("unmatched CST parameters")
    for key in (
        "source_hashes",
        "source_commit",
        "source_commit_hint",
        "case_sha256",
        "plan_catalog_sha256",
    ):
        if len({json.dumps(r[key], sort_keys=True) for r in valid}) != 1:
            raise ValueError(f"unmatched {key}")
    for key in ("declaration_sha256", "tensor_hashes", "tensor_specs"):
        if len({json.dumps(r["fixture"][key], sort_keys=True) for r in cst}) != 1:
            raise ValueError(f"unmatched Fourier fixture {key}")
    if any(r.get("verify_only") for r in valid):
        return {"status": "PASS", "verification_only": True, "qualifies": False}
    is_matern = bool(cases) and fourier_revision(cases[0]) == "v7"
    shared_name = "matern-shared" if is_matern else "fourier-shared"
    direct_name = "matern-direct" if is_matern else "fourier-direct"
    shared = kinds[shared_name]
    comparisons = {}
    for name in (direct_name, "torch-factored", "torch-shared", "dense"):
        if name in kinds and kinds[name]["status"] == "PASS":
            control = kinds[name]
            comparisons[name] = {
                "time_ratio": shared["timing"]["median_ms"]
                / control["timing"]["median_ms"],
                "allocated_ratio": shared["peak_capture_replay"]["allocated_bytes"]
                / control["peak_capture_replay"]["allocated_bytes"],
                "reserved_ratio": shared["peak_capture_replay"]["reserved_bytes"]
                / control["peak_capture_replay"]["reserved_bytes"],
            }
    direct = comparisons[direct_name]
    return {
        "status": "PASS",
        "comparisons": comparisons,
        "qualifies": (direct["time_ratio"] < 0.97 and direct["allocated_ratio"] <= 1)
        or (direct["allocated_ratio"] <= 0.95 and direct["time_ratio"] <= 1.03),
        "gate": ">3% faster/no allocated increase OR >=5% lower allocated/<=3% slower versus native direct",
        "dense_scope": "different parameterization; engineering control",
        "independent_runs": 1,
    }


def fourier_main(args, parser, case):
    if args.size is not None and args.size != case["size"]:
        parser.error("explicit size differs from Fourier case")
    if args.rho is not None:
        parser.error("Fourier case uses a variable width distribution, not rho")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state_gate:
        result = state_gate(fourier_case=case)
    elif args.worker:
        if args.worker not in (*FOURIER_KINDS, *MATERN_PRIMARY):
            parser.error("requires a Fourier worker alias")
        result = fourier_worker(
            case, args.worker, phases=args.phases, verify_only=args.verify_only
        )
        validate_fourier_record(result)
    else:
        # Torch shared tiles is a CPU FP64 reference and remains an explicit
        # diagnostic alias; CUDA cohorts default to the four executable controls.
        primary = fourier_primary(case)
        allowed = primary if fourier_revision(case) in ("v5", "v7") else FOURIER_KINDS
        plans = list(primary) if args.plans is None else args.plans
        if (
            len(set(plans)) != len(plans)
            or not set(primary) <= set(plans)
            or not set(plans) <= set(allowed)
        ):
            parser.error(f"cohort requires unique {'/'.join(primary)} workers")
        records = []
        for kind in reversed(plans) if args.reverse else plans:
            output = args.output.parent / f"{args.output.stem}-{kind}.json"
            command = [
                sys.executable,
                "-m",
                __spec__.name,
                "--worker",
                kind,
                "--case",
                str(args.case),
                "--output",
                str(output),
            ]
            if args.phases:
                command.append("--phases")
            if args.verify_only:
                command.append("--verify-only")
            try:
                completed = subprocess.run(
                    command, check=False, timeout=args.worker_timeout
                )
            except subprocess.TimeoutExpired:
                record = {
                    "schema": f"fourier-worker.research.{fourier_revision(case)}",
                    "kind": kind,
                    "status": "FAIL",
                    "failed_stage": "worker-deadline",
                    "error": f"worker exceeded predeclared {args.worker_timeout}s",
                    "timing": None,
                }
                output.write_text(json.dumps(record, indent=2))
                records.append(record)
                break
            if not output.exists():
                raise RuntimeError(f"worker {kind} exited without artifact")
            record = json.loads(output.read_text())
            if completed.returncode and record["status"] == "PASS":
                raise RuntimeError("nonzero exit cannot certify PASS")
            records.append(record)
            if record["status"] != "PASS" and kind in primary:
                break
        result = {
            "schema": f"fourier-comparison.research.{fourier_revision(case)}",
            "records": records,
            "assessment": assess_fourier(records),
            "independent_runs": 1,
            "submission_schema": False,
        }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.worker and result["status"] == "OOM":
        return
    if result.get("status", result.get("assessment", {}).get("status")) != "PASS":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(300, 700, 2000, 1024, 2048))
    parser.add_argument("--rho", type=float, choices=(3, 8))
    parser.add_argument("--case", type=Path)
    parser.add_argument(
        "--plans",
        nargs="+",
        choices=tuple(dict.fromkeys(KINDS + FOURIER_KINDS + MATERN_PRIMARY)),
        default=None,
    )
    parser.add_argument(
        "--worker", choices=tuple(dict.fromkeys(KINDS + FOURIER_KINDS + MATERN_PRIMARY))
    )
    parser.add_argument("--state-gate", action="store_true")
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--worker-timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case:
        case_data = json.loads(args.case.read_text())
        if case_data.get("schema") in (
            "fourier-case.research.v4",
            "fourier-case.research.v5",
            "matern-mixture-case.research.v7",
        ):
            ff = profile_protocol(case_data)
            fourier_main(args, parser, ff.load_case(case_data))
            return
        case = load_case(args.case)
        if args.size is not None and args.size != case["size"]:
            parser.error("explicit size differs from frozen case")
        if args.rho is not None and args.rho != case["rho_initial"]:
            parser.error("explicit rho differs from frozen case")
        args.size, args.rho = case["size"], case["rho_initial"]
    else:
        args.size = 1024 if args.size is None else args.size
        args.rho = 3 if args.rho is None else args.rho
    if args.size not in (1024, 2048):
        parser.error("nonlegacy sizes require an explicit Fourier rev4 case")
    if args.plans is None:
        args.plans = list(KINDS)
    if not set(args.plans) <= set(KINDS) or (args.worker and args.worker not in KINDS):
        parser.error("Fourier plans require a Fourier rev4 case")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state_gate:
        result = state_gate()
    elif args.worker:
        result = worker(
            args.size,
            args.rho,
            args.worker,
            phases=args.phases,
            verify_only=args.verify_only,
        )
        validate_record(result)
    else:
        if len(set(args.plans)) != len(args.plans) or not set(PRIMARY) <= set(
            args.plans
        ):
            parser.error("cohort requires unique matrix-torch/factor/dense workers")
        records = []
        for kind in reversed(args.plans) if args.reverse else args.plans:
            output = args.output.parent / f"{args.output.stem}-{kind}.json"
            command = [
                sys.executable,
                "-m",
                __spec__.name,
                "--worker",
                kind,
                "--size",
                str(args.size),
                "--rho",
                str(args.rho),
                "--output",
                str(output),
            ]
            if args.phases:
                command.append("--phases")
            if args.verify_only:
                command.append("--verify-only")
            try:
                completed = subprocess.run(
                    command, check=False, timeout=args.worker_timeout
                )
            except subprocess.TimeoutExpired:
                record = {
                    "kind": kind,
                    "status": "FAIL",
                    "failed_stage": "worker-deadline",
                    "error": f"worker exceeded predeclared {args.worker_timeout}s",
                }
                output.write_text(json.dumps(record, indent=2))
                records.append(record)
                break
            if not output.exists():
                raise RuntimeError(
                    f"worker {kind} exited {completed.returncode} without artifact"
                )
            record = json.loads(output.read_text())
            if completed.returncode and record["status"] == "PASS":
                raise RuntimeError("nonzero worker exit cannot certify PASS")
            records.append(record)
            if records[-1]["status"] != "PASS" and kind in PRIMARY:
                break
        result = {
            "schema": "periodic-comparison.research.v1",
            "records": records,
            "assessment": assess(records),
            "independent_runs": 1,
            "submission_schema": False,
        }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.worker and result["status"] == "OOM":
        return  # Parent retains an honest OOM record; required plans then fail.
    if result.get("status", result.get("assessment", {}).get("status")) != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
