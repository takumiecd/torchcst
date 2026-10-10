"""Isolated matched periodic matrix/factor complete-step research protocol.

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

LEGACY_KINDS = ("matrix-torch", "matrix-triton", "factor", "torch-factored", "dense")
KINDS = (*LEGACY_KINDS, "onchip-h", "output-owned-h")
REGULAR_PRIMARY = ("matrix-torch", "factor", "onchip-h", "dense")
PRIMARY = ("matrix-torch", "factor", "dense")
CATALOG = Path(__file__).with_name("plans-periodic-profile-product.json")


def case_definition(size, rho, *, regular_grid=False):
    value = {
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
    if regular_grid:
        value.update(
            schema="regular-grid-h-case.research.v1", grid_shape=[[size], [size]]
        )
    return value


def load_case(path):
    value = json.loads(Path(path).read_text())
    if (
        type(value.get("size")) is not int
        or value["size"] not in (1024, 2048)
        or type(value.get("rho_initial")) not in (int, float)
        or value["rho_initial"] not in (3, 8)
    ):
        raise ValueError("only the four predeclared periodic cases are accepted")
    if value != case_definition(
        value["size"], value["rho_initial"], regular_grid="grid_shape" in value
    ):
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

    regular_grid = model.chart.spec.kind == "regular_grid"
    catalog = (
        CATALOG.with_name("plans-regular-grid-h.json") if regular_grid else CATALOG
    )
    entries = decode_catalog(json.loads(catalog.read_text()))
    plan = next(entry.plan for entry in entries if entry.id == kind)
    if regular_grid:
        from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
            OnchipHRecipe,
            OutputOwnedHRecipe,
        )

        expected_id = {
            "factor": "research_cuda_regular_grid_saved_factor",
            "onchip-h": "research_cuda_regular_grid_onchip_h",
            "output-owned-h": "research_cuda_regular_grid_output_owned_h",
        }.get(kind, "research_cuda_regular_grid_matrix")
        expected_recipe = (
            {"onchip-h": OnchipHRecipe, "output-owned-h": OutputOwnedHRecipe}[kind]()
            if kind in ("onchip-h", "output-owned-h")
            else PeriodicRecipe(gemm="triton" if kind == "matrix-triton" else "torch")
        )
    else:
        expected_recipe = PeriodicRecipe(
            gemm="triton" if kind == "matrix-triton" else "torch"
        )
        expected_id = (
            "research_cuda_periodic_prepared_factor"
            if kind == "factor"
            else "research_cuda_periodic_grouped_matrix"
        )
    if (
        plan.algorithm_id != expected_id
        or plan.algorithm_revision != "v1"
        or plan.recipe != expected_recipe
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


def isolated_oracle_check(model, x, dy, *, size, rho, directory, label):
    """Validate exact CPU snapshots in a separate, untimed CUDA process."""
    xx = x.detach().clone().requires_grad_()
    y = model(xx)
    gx, gp = torch.autograd.grad(y, (xx, model.atoms.p), dy)
    snapshot = {
        "size": size,
        "rho": rho,
        "model": {
            key: (
                value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
            )
            for key, value in model.state_dict().items()
        },
        "x": x.detach().cpu(),
        "dy": dy.detach().cpu(),
        "actual": [value.detach().cpu() for value in (y, gx, gp)],
    }
    directory.mkdir(parents=True, exist_ok=True)
    path, result = directory / f"{label}.pt", directory / f"{label}.json"
    torch.save(snapshot, path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.periodic_comparison",
            "--oracle-snapshot",
            str(path),
            "--output",
            str(result),
        ],
        check=True,
        timeout=180,
    )
    record = json.loads(result.read_text())
    if record["status"] != "PASS":
        raise AssertionError(record)
    record["snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def oracle_snapshot(path):
    """Independent FP64 whole-axis/all-atom truth; no candidate execution."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = fixture.fixture(
        data["size"],
        data["rho"],
        atoms=len(data["model"]["atom_state.atoms.p"]),
        regular_grid=True,
    ).cuda()
    model.load_state_dict(data["model"])
    expected = fixture.oracle_vjp(model, data["x"].cuda(), data["dy"].cuda())
    errors = {}
    for name, actual, truth in zip(
        ("Y", "dX", "all_dP"), data["actual"], expected, strict=True
    ):
        metric = finite_error(name, actual, truth.detach().cpu())
        if metric["max_abs"] > 4e-4 or metric["relative_l2"] > 4e-4:
            raise AssertionError((name, metric))
        errors[name] = metric
    return {
        "status": "PASS",
        "scope": "isolated FP64 complete axes/all atoms/all four cotangents",
        "errors": errors,
    }


def state_gate(*, device="cuda", steps=20, regular_grid=False):
    """Same-cotangent public optimizer truth, including seam crossings and moments."""
    model = fixture.fixture(32, 3, atoms=17, regular_grid=regular_grid).to(device)
    with torch.no_grad():
        model.atoms.p[:, 2] = 32 - 1e-5
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
    if kind in ("onchip-h", "output-owned-h"):
        return onchip_stages(step, output_owned=kind == "output-owned-h")
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


def onchip_stages(step, *, output_owned=False):
    """Fused H-to-Y cannot be decomposed by timestamps inside one kernel."""
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.executor import (
        forward_contraction,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
        OnchipHRecipe,
        OutputOwnedHRecipe,
    )

    model = copy.deepcopy(step.model)
    x, p = step.x.detach().clone(), model.atoms.p.detach()
    n = model.in_features
    sizes = (len(x), n, n, float(n), float(n), 0.0, 0.0)
    recipe = OutputOwnedHRecipe() if output_owned else OnchipHRecipe()
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]

    def call():
        events[0].record()
        packed = _prepare(p.contiguous().clone(), model.kernel, sizes, recipe)
        events[1].record()
        y = forward_contraction(x, packed, sizes, recipe)
        events[2].record()
        return y

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = call()
    graph.replay()
    torch.cuda.synchronize()
    phases = {
        name: []
        for name in (
            "snapshot_and_preparation",
            "routing_and_output_owned_H_to_Y"
            if output_owned
            else "Y_zero_and_fused_H_to_Y",
        )
    }
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        for i, name in enumerate(phases):
            phases[name].append(events[i].elapsed_time(events[i + 1]))
    require_finite("fused-stage Y", y)
    return {
        "scope": "fixed post24 forward copy; nonadditive diagnostic; fused H/Y unsplit",
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


def worker(
    size,
    rho,
    kind,
    *,
    phases=False,
    verify_only=False,
    regular_grid=False,
    oracle_directory=None,
):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result = {
        "status": "RUNNING",
        "regular_grid": regular_grid,
        "kind": kind,
        "size": size,
        "rho_initial": rho,
        "batch": 32,
        "seed": 41,
        "atoms": None if kind == "dense" else int(0.05 * size * size),
        **source_metadata(),
        "measurement_protocol": (
            "isolated-oracle-v2" if oracle_directory else "inprocess-oracle-v1"
        ),
    }
    case = case_definition(size, rho, regular_grid=regular_grid)
    result["case_definition"] = case
    result["plan_catalog_sha256"] = hashlib.sha256(
        (
            CATALOG.with_name("plans-regular-grid-h.json") if regular_grid else CATALOG
        ).read_bytes()
    ).hexdigest()
    result["case_sha256"] = hashlib.sha256(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stage = "fixture"
    try:
        xx, target, dy = shared_inputs(size)
        x, target, dy = xx.cuda().requires_grad_(), target.cuda(), dy.cuda()
        dense = kind == "dense"
        model = (
            dense_fixture(size).cuda()
            if dense
            else fixture.fixture(size, rho, regular_grid=regular_grid).cuda()
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

        def check(label):
            if oracle_directory is not None:
                return isolated_oracle_check(
                    model,
                    x,
                    dy,
                    size=size,
                    rho=rho,
                    directory=oracle_directory,
                    label=label,
                )
            return oracle_check(model, x, dy)

        result["initial_oracle"] = None if dense else check("initial")
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
            memory_scope=(
                "own model/optimizer/X/target/dy and candidate warmup allocations; capture/replay included; FP64 oracle in separate process"
                if oracle_directory
                else "own model/optimizer/X/target/dy; capture/replay included; oracle/diagnostic scratch excluded; persistent library allocations may remain"
            ),
        )
        stage = "updated-full-oracle"
        result["updated_oracle"] = None if dense else check("post24")
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
            chart["kind"]
            != ("regular_grid" if record.get("regular_grid") else "periodic_grid")
            or (
                tuple(map(tuple, chart["grid_shape"]))
                if record.get("regular_grid")
                else tuple(chart["grid_shape"])
            )
            != (((n,), (n,)) if record.get("regular_grid") else (n, n))
            or tuple(chart["geometry"]["periods"]) != (n, n)
            or tuple(chart["origin"]) != (0.0, 0.0)
            or (chart.get("output_dims", 1) != 1)
            or (record.get("regular_grid") and tuple(chart["spacing"]) != (1.0, 1.0))
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
    primary = (
        REGULAR_PRIMARY if any(r.get("regular_grid") for r in records) else PRIMARY
    )
    for key in primary:
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
    if len({r.get("measurement_protocol", "inprocess-oracle-v1") for r in valid}) != 1:
        raise AssertionError("unmatched measurement protocol")
    cst = [r for r in valid if r["kind"] != "dense"]
    for key in ("declaration_sha256", "tensor_hashes", "tensor_specs"):
        if len({json.dumps(r["fixture"][key], sort_keys=True) for r in cst}) != 1:
            raise AssertionError(f"unmatched fixture {key}")
    if any(r["status"] == "FAIL" for r in records):
        return {
            "status": "FAIL",
            "reason": "worker numerical/runtime failure preserved",
        }
    if primary == REGULAR_PRIMARY:
        return assess_regular(kinds, valid)
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


def assess_regular(kinds, valid):
    """Both allocator peaks must fit dense before a candidate can be selected."""
    if any(r.get("verify_only", False) for r in valid):
        return {"status": "PASS", "verification_only": True, "qualifies": False}
    if len({json.dumps(r["runtime"], sort_keys=True) for r in valid}) != 1:
        raise AssertionError("unmatched runtime")
    dense, saved = kinds["dense"], kinds["factor"]
    candidates = {}
    for name, r in kinds.items():
        if name == "dense" or r["status"] != "PASS":
            continue
        fits = all(
            r["peak_capture_replay"][key] <= dense["peak_capture_replay"][key]
            for key in ("allocated_bytes", "reserved_bytes")
        )
        candidates[name] = {
            "fits_dense_allocated_and_reserved": fits,
            "time_ratio_to_dense": r["timing"]["median_ms"]
            / dense["timing"]["median_ms"],
            "time_ratio_to_saved_factor": r["timing"]["median_ms"]
            / saved["timing"]["median_ms"],
            "allocated_ratio_to_saved_factor": r["peak_capture_replay"][
                "allocated_bytes"
            ]
            / saved["peak_capture_replay"]["allocated_bytes"],
        }
    fitting = [
        name for name, r in candidates.items() if r["fits_dense_allocated_and_reserved"]
    ]
    return {
        "status": "PASS",
        "candidates": candidates,
        "fastest_fitting_cst": min(
            fitting, key=lambda name: kinds[name]["timing"]["median_ms"]
        )
        if fitting
        else None,
        "gate": "both total allocated/reserved <= dense; choose fastest passing CST",
        "adopted": False,
        "scope": "single cohort; research baseline; dispatcher adoption is separate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(1024, 2048))
    parser.add_argument("--rho", type=float, choices=(3, 8))
    parser.add_argument("--case", type=Path)
    parser.add_argument("--plans", nargs="+", choices=KINDS, default=None)
    parser.add_argument("--worker", choices=KINDS)
    parser.add_argument("--regular-grid", action="store_true")
    parser.add_argument("--isolated-oracle", action="store_true")
    parser.add_argument("--oracle-snapshot", type=Path)
    parser.add_argument("--state-gate", action="store_true")
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--worker-timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.oracle_snapshot:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(oracle_snapshot(args.oracle_snapshot), indent=2)
        )
        return
    if args.case:
        case = load_case(args.case)
        args.regular_grid = "grid_shape" in case
        if args.size is not None and args.size != case["size"]:
            parser.error("explicit size differs from frozen case")
        if args.rho is not None and args.rho != case["rho_initial"]:
            parser.error("explicit rho differs from frozen case")
        args.size, args.rho = case["size"], case["rho_initial"]
    else:
        args.size = 1024 if args.size is None else args.size
        args.rho = 3 if args.rho is None else args.rho
    if args.plans is None:
        args.plans = (
            [
                "matrix-torch",
                "matrix-triton",
                "factor",
                "onchip-h",
                "output-owned-h",
                "dense",
            ]
            if args.regular_grid
            else list(LEGACY_KINDS)
        )
    if not args.regular_grid and (
        args.worker in ("onchip-h", "output-owned-h")
        or any(kind in args.plans for kind in ("onchip-h", "output-owned-h"))
    ):
        parser.error("H recipes require --regular-grid")
    primary = REGULAR_PRIMARY if args.regular_grid else PRIMARY
    if args.isolated_oracle and not args.regular_grid:
        parser.error("isolated oracle snapshot currently requires --regular-grid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state_gate:
        result = state_gate(regular_grid=args.regular_grid)
    elif args.worker:
        result = worker(
            args.size,
            args.rho,
            args.worker,
            phases=args.phases,
            verify_only=args.verify_only,
            regular_grid=args.regular_grid,
            oracle_directory=(
                args.output.parent / f"{args.output.stem}-oracle"
                if args.isolated_oracle
                else None
            ),
        )
        validate_record(result)
    else:
        if len(set(args.plans)) != len(args.plans) or not set(primary) <= set(
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
            if args.regular_grid:
                command.append("--regular-grid")
            if args.isolated_oracle:
                command.append("--isolated-oracle")
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
            if records[-1]["status"] != "PASS" and kind in primary:
                break
        result = {
            "schema": "regular-grid-h-comparison.research.v1"
            if args.regular_grid
            else "periodic-comparison.research.v1",
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
