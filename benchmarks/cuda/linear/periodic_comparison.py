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
KINDS = (
    *LEGACY_KINDS,
    "onchip-h",
    "output-owned-h",
    "reused-h",
    "reused-h16",
    "reused-h32",
    "prepared-h16",
    "prepared-h32",
    "prepared-g32-h16",
    "prepared-g32-h32",
    "input-owned-h16",
    "input-owned-h32",
    "stream-g8-h16",
    "stream-g8-h32",
    "site-routed-h16",
    "site-routed-h32",
    "site-owner-bm16-h16",
    "site-owner-bm16-h32",
    "site-stream-g8-h16",
    "site-stream-g8-h32",
    "site-owner-bm16-stream-g8-h16",
    "site-owner-bm16-stream-g8-h32",
    "input-site-stream-g8-h32",
    "input-order-stream-g8-h32",
)
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
            GroupedOutputHRecipe,
            InputOrderStreamingHRecipe,
            InputOwnedHRecipe,
            InputSiteStreamingHRecipe,
            OnchipHRecipe,
            OutputOwnedHRecipe,
            OwnerBatchHRecipe,
            OwnerBatchStreamingHRecipe,
            ParallelReusedHRecipe,
            PreparedReusedHRecipe,
            ReusedHRecipe,
            SiteRoutedHRecipe,
            SiteRoutedStreamingHRecipe,
            StreamingInputHRecipe,
        )

        expected_id = {
            "factor": "research_cuda_regular_grid_saved_factor",
            "onchip-h": "research_cuda_regular_grid_onchip_h",
            "output-owned-h": "research_cuda_regular_grid_output_owned_h",
            "reused-h": "research_cuda_regular_grid_reused_h",
            "reused-h16": "research_cuda_regular_grid_parallel_reused_h",
            "reused-h32": "research_cuda_regular_grid_parallel_reused_h",
            "prepared-h16": "research_cuda_regular_grid_prepared_reused_h",
            "prepared-h32": "research_cuda_regular_grid_prepared_reused_h",
            "prepared-g32-h16": "research_cuda_regular_grid_grouped_output_h",
            "prepared-g32-h32": "research_cuda_regular_grid_grouped_output_h",
            "input-owned-h16": "research_cuda_regular_grid_input_owned_h",
            "input-owned-h32": "research_cuda_regular_grid_input_owned_h",
            "stream-g8-h16": "research_cuda_regular_grid_streaming_input_h",
            "stream-g8-h32": "research_cuda_regular_grid_streaming_input_h",
            "site-routed-h16": "research_cuda_regular_grid_site_routed_h",
            "site-routed-h32": "research_cuda_regular_grid_site_routed_h",
            "site-owner-bm16-h16": "research_cuda_regular_grid_site_owner_batch_h",
            "site-owner-bm16-h32": "research_cuda_regular_grid_site_owner_batch_h",
            "site-stream-g8-h16": "research_cuda_regular_grid_site_routed_streaming_h",
            "site-stream-g8-h32": "research_cuda_regular_grid_site_routed_streaming_h",
            "site-owner-bm16-stream-g8-h16": "research_cuda_regular_grid_site_owner_batch_streaming_h",
            "site-owner-bm16-stream-g8-h32": "research_cuda_regular_grid_site_owner_batch_streaming_h",
            "input-site-stream-g8-h32": "research_cuda_regular_grid_input_site_streaming_h",
            "input-order-stream-g8-h32": "research_cuda_regular_grid_input_order_streaming_h",
        }.get(kind, "research_cuda_regular_grid_matrix")
        expected_recipe = (
            {
                "onchip-h": OnchipHRecipe,
                "output-owned-h": OutputOwnedHRecipe,
                "reused-h": ReusedHRecipe,
                "reused-h16": lambda: ParallelReusedHRecipe(h_batch=16),
                "reused-h32": lambda: ParallelReusedHRecipe(h_batch=32),
                "prepared-h16": lambda: PreparedReusedHRecipe(h_batch=16),
                "prepared-h32": lambda: PreparedReusedHRecipe(h_batch=32),
                "prepared-g32-h16": lambda: GroupedOutputHRecipe(h_batch=16),
                "prepared-g32-h32": lambda: GroupedOutputHRecipe(h_batch=32),
                "input-owned-h16": lambda: InputOwnedHRecipe(h_batch=16),
                "input-owned-h32": lambda: InputOwnedHRecipe(h_batch=32),
                "stream-g8-h16": lambda: StreamingInputHRecipe(h_batch=16),
                "stream-g8-h32": lambda: StreamingInputHRecipe(h_batch=32),
                "site-routed-h16": lambda: SiteRoutedHRecipe(h_batch=16),
                "site-routed-h32": lambda: SiteRoutedHRecipe(h_batch=32),
                "site-owner-bm16-h16": lambda: OwnerBatchHRecipe(h_batch=16),
                "site-owner-bm16-h32": OwnerBatchHRecipe,
                "site-stream-g8-h16": lambda: SiteRoutedStreamingHRecipe(h_batch=16),
                "site-stream-g8-h32": lambda: SiteRoutedStreamingHRecipe(h_batch=32),
                "site-owner-bm16-stream-g8-h16": lambda: OwnerBatchStreamingHRecipe(
                    h_batch=16
                ),
                "site-owner-bm16-stream-g8-h32": OwnerBatchStreamingHRecipe,
                "input-site-stream-g8-h32": InputSiteStreamingHRecipe,
                "input-order-stream-g8-h32": InputOrderStreamingHRecipe,
            }[kind]()
            if kind
            in (
                "onchip-h",
                "output-owned-h",
                "reused-h",
                "reused-h16",
                "reused-h32",
                "prepared-h16",
                "prepared-h32",
                "prepared-g32-h16",
                "prepared-g32-h32",
                "input-owned-h16",
                "input-owned-h32",
                "stream-g8-h16",
                "stream-g8-h32",
                "site-routed-h16",
                "site-routed-h32",
                "site-owner-bm16-h16",
                "site-owner-bm16-h32",
                "site-stream-g8-h16",
                "site-stream-g8-h32",
                "site-owner-bm16-stream-g8-h16",
                "site-owner-bm16-stream-g8-h32",
                "input-site-stream-g8-h32",
                "input-order-stream-g8-h32",
            )
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
    if kind in (
        "output-owned-h",
        "reused-h",
        "reused-h16",
        "reused-h32",
        "prepared-h16",
        "prepared-h32",
        "prepared-g32-h16",
        "prepared-g32-h32",
        "input-owned-h16",
        "input-owned-h32",
        "stream-g8-h16",
        "stream-g8-h32",
        "site-routed-h16",
        "site-routed-h32",
        "site-owner-bm16-h16",
        "site-owner-bm16-h32",
        "site-stream-g8-h16",
        "site-stream-g8-h32",
        "site-owner-bm16-stream-g8-h16",
        "site-owner-bm16-stream-g8-h32",
        "input-site-stream-g8-h32",
        "input-order-stream-g8-h32",
    ):
        return output_owner_stages(
            step,
            reused=kind != "output-owned-h",
            h_batch={
                "reused-h16": 16,
                "reused-h32": 32,
                "prepared-h16": 16,
                "prepared-h32": 32,
                "prepared-g32-h16": 16,
                "prepared-g32-h32": 32,
                "input-owned-h16": 16,
                "input-owned-h32": 32,
                "stream-g8-h16": 16,
                "stream-g8-h32": 32,
                "site-routed-h16": 16,
                "site-routed-h32": 32,
                "site-owner-bm16-h16": 16,
                "site-owner-bm16-h32": 32,
                "site-stream-g8-h16": 16,
                "site-stream-g8-h32": 32,
                "site-owner-bm16-stream-g8-h16": 16,
                "site-owner-bm16-stream-g8-h32": 32,
                "input-site-stream-g8-h32": 32,
                "input-order-stream-g8-h32": 32,
            }.get(kind),
            prepared=kind.startswith(
                (
                    "prepared-",
                    "input-owned-",
                    "stream-",
                    "site-",
                    "input-site-",
                    "input-order-",
                )
            ),
            grouped=kind.startswith(
                (
                    "prepared-g32-",
                    "input-owned-",
                    "stream-",
                    "site-",
                    "input-site-",
                    "input-order-",
                )
            ),
            site_routed=kind.startswith(("site-", "input-site-", "input-order-")),
            owner_bm16=kind.startswith(
                ("site-owner-bm16-", "input-site-", "input-order-")
            ),
        )
    if kind == "onchip-h":
        return onchip_stages(step)
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


def routing_bounds_census(bounds, distance, *, sites, tile, group, site_routed):
    """CPU counts of actual candidate ranges; no positive-support or traffic claim."""
    from itertools import pairwise

    bins = (sites + tile - 1) // tile
    visits = iterations = checks = 0
    coarse_iterations = [0]
    if not site_routed:
        for left, right in pairwise(bounds):
            coarse_iterations.append(
                coarse_iterations[-1] + (right - left + group - 1) // group
            )
    for owner in range(bins):
        start = owner * tile
        width = min(tile, sites - start)
        if site_routed:
            count = min(width + 2 * distance, sites)
            first = 0 if count == sites else (start - distance) % sites
            end = first + count
            ranges = [(first, min(end, sites))]
            if end > sites:
                ranges.append((0, end - sites))
        else:
            radius = (distance + tile - 1) // tile + int(sites % tile != 0)
            count = min(2 * radius + 1, bins)
            first = 0 if count == bins else (owner - radius) % bins
            end = first + count
            ranges = [(first, min(end, bins))]
            if end > bins:
                ranges.append((0, end - bins))
        owner_visits = 0
        for low, high in ranges:
            size = bounds[high] - bounds[low]
            owner_visits += size
            iterations += (
                (size + group - 1) // group
                if site_routed
                else coarse_iterations[high] - coarse_iterations[low]
            )
        visits += owner_visits
        checks += width * owner_visits
    return {
        "scope": "candidate ranges over output owners; excludes batch multiplicity and padded lanes",
        "prefix_entries": len(bounds),
        "guarded_max_distance": distance,
        "atom_group": group,
        "candidate_atom_visits": visits,
        "atom_group_iterations": iterations,
        "candidate_live_site_checks": checks,
    }


def output_owner_stages(
    step,
    *,
    reused,
    h_batch=None,
    prepared=False,
    grouped=False,
    site_routed=False,
    owner_bm16=False,
):
    """Timestamp the actual staged forward on a fixed post24 copy.

    Candidate-index construction is separate. Exact support checks remain in
    the Y kernel; the fused control cannot separate H work from Y aggregation.
    Diagnostic event nodes are absent from the primary complete-step graph.
    """
    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        aggregate_chunk,
        allocate_h,
        h_capacity,
        owner_batch_tile,
        prepare_output_fields,
        prepare_routing,
        produce_h_chunk,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
        GroupedOutputHRecipe,
        OutputOwnedHRecipe,
        OwnerBatchHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
        ReusedHRecipe,
        SiteRoutedHRecipe,
    )

    model = copy.deepcopy(step.model)
    x, p = step.x.detach().clone(), model.atoms.p.detach()
    n = model.in_features
    sizes = (len(x), n, n, float(n), float(n), 0.0, 0.0)
    recipe = (
        (
            OwnerBatchHRecipe
            if owner_bm16
            else SiteRoutedHRecipe
            if site_routed
            else GroupedOutputHRecipe
            if grouped
            else PreparedReusedHRecipe
            if prepared
            else ParallelReusedHRecipe
        )(h_batch=h_batch)
        if h_batch is not None
        else (ReusedHRecipe() if reused else OutputOwnedHRecipe())
    )
    chunks = list(range(0, len(x), h_capacity(recipe))) if reused else [0]
    offset = int(prepared)
    events = [
        torch.cuda.Event(enable_timing=True, external=True)
        for _ in range(3 + offset + (2 * len(chunks) if reused else 1))
    ]

    observed = {}

    def call():
        events[0].record()
        packed = _prepare(p.contiguous().clone(), model.kernel, sizes, recipe)
        events[1].record()
        y = x.new_empty((len(x), n))
        routing = prepare_routing(packed, sizes, recipe)
        h = allocate_h(x, len(p), recipe) if reused else None
        events[2].record()
        hot = prepare_output_fields(packed, routing) if prepared else None
        observed["routing"] = routing
        observed["hot"] = hot
        if prepared:
            events[3].record()
        if reused:
            for i, start in enumerate(chunks):
                produce_h_chunk(x, packed, sizes, recipe, routing, h, start)
                events[3 + offset + 2 * i].record()
                aggregate_chunk(
                    x,
                    packed,
                    sizes,
                    recipe,
                    routing,
                    y,
                    h=h,
                    batch_start=start,
                    hot=hot,
                )
                events[4 + offset + 2 * i].record()
        else:
            aggregate_chunk(x, packed, sizes, recipe, routing, y)
            events[3].record()
        return y

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = call()
    graph.replay()
    torch.cuda.synchronize()
    phases = {"snapshot_and_preparation": [], "candidate_index": []}
    if prepared:
        phases["output_fields_preparation"] = []
    if reused:
        phases.update(H_generation=[], Y_aggregation_with_support_checks=[])
    else:
        phases["fused_H_and_Y_with_support_checks"] = []
    per_chunk = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        phases["snapshot_and_preparation"].append(events[0].elapsed_time(events[1]))
        phases["candidate_index"].append(events[1].elapsed_time(events[2]))
        if prepared:
            phases["output_fields_preparation"].append(
                events[2].elapsed_time(events[3])
            )
        if reused:
            sample = [
                {
                    "batch_start": start,
                    "H_generation": events[2 + offset + 2 * i].elapsed_time(
                        events[3 + offset + 2 * i]
                    ),
                    "Y_aggregation_with_support_checks": events[
                        3 + offset + 2 * i
                    ].elapsed_time(events[4 + offset + 2 * i]),
                }
                for i, start in enumerate(chunks)
            ]
            per_chunk.append(sample)
            for name in ("H_generation", "Y_aggregation_with_support_checks"):
                phases[name].append(sum(chunk[name] for chunk in sample))
        else:
            phases["fused_H_and_Y_with_support_checks"].append(
                events[2].elapsed_time(events[3])
            )
    require_finite("output-owner-stage Y", y)
    # Independently compare the instrumented schedule to its uninstrumented
    # forward on the same immutable snapshot, outside all timing intervals.
    torch.testing.assert_close(y, model(x), rtol=4e-4, atol=4e-4)
    # Read the exact routing produced by the captured diagnostic call only
    # after all event samples. These copies are outside primary graphs/peaks.
    routing, hot = observed["routing"], observed["hot"]
    unsafe = bool(hot[1].detach().cpu().item()) if hot is not None else False
    group = (
        recipe.atom_group
        if unsafe
        else getattr(recipe, "output_group", recipe.atom_group)
    )
    census = routing_bounds_census(
        routing[1].detach().cpu().tolist(),
        int(routing[2].detach().cpu().item()),
        sites=n,
        tile=recipe.output_tile,
        group=group,
        site_routed=site_routed,
    )
    batch_ctas = (
        sum(
            (min(h_capacity(recipe), len(x) - start) + owner_batch_tile(recipe) - 1)
            // owner_batch_tile(recipe)
            for start in chunks
        )
        if reused
        else (len(x) + owner_batch_tile(recipe) - 1) // owner_batch_tile(recipe)
    )
    census.update(
        unsafe_beta_fallback=unsafe,
        owner_batch_tile=owner_batch_tile(recipe),
        batch_ctas_per_output_owner=batch_ctas,
        active_y_ctas=batch_ctas * ((n + recipe.output_tile - 1) // recipe.output_tile),
        launched_y_ctas=(2 if prepared else 1)
        * batch_ctas
        * ((n + recipe.output_tile - 1) // recipe.output_tile),
    )
    return {
        "routing_census": census,
        "scope": "fixed post24 forward copy; no optimizer; nonadditive to primary step",
        "candidate_scope": (
            "site-key prefix construction; conservative ranges; exact support checks in Y kernel"
            if site_routed
            else "coarse index construction; exact support checks in Y kernel"
        ),
        "candidate_prefix_entries": n + 1
        if site_routed
        else (n + recipe.output_tile - 1) // recipe.output_tile + 1,
        "samples_ms": phases,
        "per_chunk_samples_ms": per_chunk,
        "output_fields_bytes": 12 * len(p) + 4 if prepared else 0,
        "H_scratch_bytes": 4 * len(p) * h_capacity(recipe) if reused else 0,
        "H_batch_capacity": h_capacity(recipe) if reused else 0,
        "H_producer_batch_tile": recipe.batch_tile,
        "Y_batch_tile": owner_batch_tile(recipe),
        "parameters_unchanged": tensor_hash(p) == tensor_hash(step.model.atoms.p),
        "same_uninstrumented_forward": True,
    }


STREAMING_STAGE_KINDS = (
    "stream-g8-h16",
    "stream-g8-h32",
    "site-stream-g8-h16",
    "site-stream-g8-h32",
    "site-owner-bm16-stream-g8-h16",
    "site-owner-bm16-stream-g8-h32",
    "input-site-stream-g8-h32",
    "input-order-stream-g8-h32",
)


def streaming_stage_recipe(kind):
    """Resolve the actual registered streaming plan without importing GPU code."""
    if kind not in STREAMING_STAGE_KINDS:
        return None
    from benchmarks.cuda.linear.manifest import decode_catalog

    entries = decode_catalog(
        json.loads(CATALOG.with_name("plans-regular-grid-h.json").read_text())
    )
    return next(entry.plan.recipe for entry in entries if entry.id == kind)


def streaming_backward_layout(recipe, *, batch, atoms):
    """Logical tensor sizes, not allocator peaks or hardware traffic."""
    tiles = (min(batch, recipe.g_batch) + recipe.batch_tile - 1) // recipe.batch_tile
    return {
        "forward_H_batch_capacity": recipe.h_batch,
        "G_batch_capacity": recipe.g_batch,
        "G_producer_batch_tile": recipe.batch_tile,
        "dX_owner_batch_tile": recipe.batch_tile,
        "dX_owner_site_tile": recipe.input_tile,
        "G_scratch_bytes": 4 * atoms * recipe.g_batch,
        "parameter_partial_bytes": 4 * tiles * 3 * atoms,
        "physical_accumulator_bytes": 4 * 3 * atoms,
        "input_fields_bytes": 4 * 3 * atoms + 4,
        "chunk_starts": list(range(0, batch, recipe.g_batch)),
        "backward_saved_H_bytes": 0,
        "backward_saved_G_bytes": 0,
    }


def streaming_backward_stages(step, kind):
    """Fixed post24 all-gradient backward; measured separately from primary peaks.

    Use production contractions and routing on immutable forward snapshots.
    Event intervals are diagnostics, never an additive primary-step estimate.
    """
    recipe = streaming_stage_recipe(kind)
    if recipe is None:
        return None
    import triton as tr

    from torchcst._backends.cuda.algorithms.linear.periodic_product.executor import (
        _prepare,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        input_kernels,
        kernels,
        output_kernels,
    )
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h.output_owner import (
        active_h_tiles,
        prepare_output_fields,
        prepare_routing,
        uses_input_secondary_order,
        uses_input_site_routing,
    )

    model = copy.deepcopy(step.model)
    x = step.x.detach().clone().requires_grad_()
    source = model.atoms.p.detach().clone(memory_format=torch.contiguous_format)
    ampmax = model.kernel.scalar("amplitude_max").detach().clone()
    # Explicit cotangent from the same post24 loss as the production training
    # step. Freeze it before timing; autograd reference consumes the same dy.
    y = model(x)
    loss = (y - step.target.detach()).square().mean()
    dy = torch.autograd.grad(loss, y, retain_graph=True)[0].detach().clone()
    expected = tuple(
        t.detach().clone() for t in torch.autograd.grad(y, (x, model.atoms.p), dy)
    )
    del y, loss
    b, ni, no = len(x), model.in_features, model.out_features
    chart = model.chart.spec
    sizes = (
        b,
        ni,
        no,
        chart.geometry.periods[1],
        chart.geometry.periods[0],
        chart.origin[1],
        chart.origin[0],
    )
    packed = _prepare(source, model.kernel, sizes, recipe)
    # Diagnostic-only immutable audit copy; never part of primary peaks.
    packed_audit = packed.clone()
    a = len(source)
    if not a or not b:
        raise ValueError(
            "streaming stage diagnostics require nonempty benchmark batch and atoms"
        )
    layout = streaming_backward_layout(recipe, batch=b, atoms=a)
    starts = layout["chunk_starts"]
    events = [
        torch.cuda.Event(enable_timing=True, external=True)
        for _ in range(3 + 3 * len(starts))
    ]
    observed = {}
    launch = {"num_warps": 4, "enable_fp_fusion": False}

    def call():
        events[0].record()
        dx, dp = torch.zeros_like(x), torch.empty_like(source)
        partial_tiles = tr.cdiv(min(b, recipe.g_batch), recipe.batch_tile)
        partial = x.new_empty((partial_tiles, 3, a))
        physical = x.new_empty((3, a))
        routing = prepare_routing(
            packed, sizes, recipe, output=False, tile=recipe.input_tile
        )
        hot = prepare_output_fields(packed, routing, output=False)
        g = x.new_empty((recipe.g_batch // recipe.batch_tile, a, recipe.batch_tile))
        observed["routing"], observed["hot"] = routing, hot
        events[1].record()
        swapped = (b, no, ni, sizes[4], sizes[3], sizes[6], sizes[5])
        for i, start in enumerate(starts):
            tiles = active_h_tiles(g, b, start, recipe.batch_tile)
            input_kernels.produce_g_parameters[(tr.cdiv(a, recipe.atom_group), tiles)](
                x,
                dy,
                packed,
                routing[0],
                g,
                partial,
                a,
                *sizes,
                *x.stride(),
                *dy.stride(),
                True,
                start,
                recipe.batch_tile,
                recipe.patch_sites,
                recipe.atom_group,
                STREAM_PARTIAL=True,
                **launch,
            )
            events[2 + 3 * i].record()
            for fallback in (False, True):
                group = recipe.atom_group if fallback else recipe.output_group
                output_kernels.prepared_output_owned[
                    (tr.cdiv(ni, recipe.input_tile), tiles)
                ](
                    dy,
                    packed,
                    *routing,
                    dx,
                    a,
                    *swapped,
                    *dy.stride(),
                    recipe.batch_tile,
                    recipe.patch_sites,
                    group,
                    recipe.input_tile,
                    H=g,
                    Hot=hot[0],
                    Unsafe=hot[1],
                    BSTART=start,
                    FALLBACK=fallback,
                    PROFILE_OUTPUT=False,
                    SITE_ROUTING=uses_input_site_routing(recipe),
                    **launch,
                )
            events[3 + 3 * i].record()
            input_kernels.accumulate_physical[(tr.cdiv(a, recipe.prep_group),)](
                partial,
                physical,
                a,
                tiles,
                tr.next_power_of_2(g.shape[0]),
                recipe.prep_group,
                start == 0,
                **launch,
            )
            events[4 + 3 * i].record()
        kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
            physical[None],
            dp,
            source,
            ampmax,
            a,
            1,
            1,
            recipe.prep_group,
            **launch,
        )
        events[-1].record()
        return dx, dp

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(
            {
                "setup_input_routing_and_fields": events[0].elapsed_time(events[1]),
                "chunks": [
                    {
                        "batch_start": start,
                        "produce_G_and_parameter_partials": events[
                            1 + 3 * i
                        ].elapsed_time(events[2 + 3 * i]),
                        "dX_owner": events[2 + 3 * i].elapsed_time(events[3 + 3 * i]),
                        "physical_accumulation": events[3 + 3 * i].elapsed_time(
                            events[4 + 3 * i]
                        ),
                    }
                    for i, start in enumerate(starts)
                ],
                "source_VJP": events[-2].elapsed_time(events[-1]),
            }
        )
    errors = {}
    for name, got, wanted in zip(("dX", "all_dP"), actual, expected, strict=True):
        require_finite(name, got)
        require_finite(name + " reference", wanted)
        torch.testing.assert_close(got, wanted, rtol=4e-4, atol=4e-4)
        errors[name] = float((got - wanted).abs().max().cpu())
    # Only after every timed event: read the exact captured routing and flag.
    routing, hot = observed["routing"], observed["hot"]
    unsafe = bool(hot[1].detach().cpu().item())
    group = recipe.atom_group if unsafe else recipe.output_group
    census = routing_bounds_census(
        routing[1].detach().cpu().tolist(),
        int(routing[2].detach().cpu().item()),
        sites=ni,
        tile=recipe.input_tile,
        group=group,
        site_routed=uses_input_site_routing(recipe),
    )
    batch_ctas = sum(
        tr.cdiv(min(recipe.g_batch, b - start), recipe.batch_tile) for start in starts
    )
    census.update(
        scope="candidate ranges over input owners; excludes batch multiplicity and padded lanes",
        unsafe_beta_fallback=unsafe,
        input_site_routed=uses_input_site_routing(recipe),
        output_site_secondary_order=uses_input_secondary_order(recipe),
        batch_ctas_per_input_owner=batch_ctas,
        active_dX_ctas=tr.cdiv(ni, recipe.input_tile) * batch_ctas,
        launched_dX_ctas=2 * tr.cdiv(ni, recipe.input_tile) * batch_ctas,
    )
    unchanged = (
        torch.equal(model.atoms.p.detach(), source)
        and torch.equal(step.model.atoms.p.detach(), source)
        and torch.equal(packed, packed_audit)
    )
    if not unchanged:
        raise AssertionError("backward diagnostic changed its forward snapshot")
    return {
        "scope": "fixed post24 all-gradient backward; no optimizer; nonadditive to primary; excluded from primary peaks",
        "snapshot_preparation_timed": False,
        "packed_audit_copy_bytes": packed_audit.numel() * packed_audit.element_size(),
        "dy_source": "fixed explicit cotangent of post24 mean squared loss",
        "samples_ms": samples,
        "layout": layout,
        "input_routing_census": census,
        "same_uninstrumented_backward": True,
        "same_snapshot": unchanged,
        "max_abs_vs_uninstrumented": errors,
        "lifetime": "G and partials overwritten after each chunk; physical accumulator lives to source VJP; no H/G saved from forward",
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
    aggregation_diagnostics=False,
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
        stage = "separate-streaming-backward-diagnostics"
        result["backward_stage_diagnostics"] = (
            streaming_backward_stages(step, kind)
            if phases and kind in STREAMING_STAGE_KINDS
            else None
        )
        if aggregation_diagnostics and kind in (
            "reused-h16",
            "reused-h32",
            "prepared-h16",
            "prepared-h32",
        ):
            stage = "separate-aggregation-diagnostics"
            from benchmarks.cuda.linear.aggregation_diagnostics import (
                prepared_snapshot_diagnostics,
                snapshot_diagnostics,
            )
            from torchcst._backends.cuda.algorithms.linear.regular_grid_h.recipe import (
                ParallelReusedHRecipe,
                PreparedReusedHRecipe,
            )

            prepared = kind.startswith("prepared-")
            diagnostic_recipe = (
                PreparedReusedHRecipe if prepared else ParallelReusedHRecipe
            )(h_batch=16 if kind.endswith("16") else 32)
            diagnose = (
                prepared_snapshot_diagnostics if prepared else snapshot_diagnostics
            )
            result["aggregation_diagnostics"] = {
                label: diagnose(
                    oracle_directory / f"{label}.pt",
                    diagnostic_recipe,
                    directory=oracle_directory / (label + "-aggregation"),
                    timing=label == "post24",
                )
                for label in ("initial", "post24")
            }
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
    parser.add_argument("--aggregation-diagnostics", action="store_true")
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
                "reused-h",
                "reused-h16",
                "reused-h32",
                "prepared-h16",
                "prepared-h32",
                "prepared-g32-h16",
                "prepared-g32-h32",
                "input-owned-h16",
                "input-owned-h32",
                "stream-g8-h16",
                "stream-g8-h32",
                "site-routed-h16",
                "site-routed-h32",
                "site-owner-bm16-h16",
                "site-owner-bm16-h32",
                "site-stream-g8-h16",
                "site-stream-g8-h32",
                "site-owner-bm16-stream-g8-h16",
                "site-owner-bm16-stream-g8-h32",
                "input-site-stream-g8-h32",
                "input-order-stream-g8-h32",
                "dense",
            ]
            if args.regular_grid
            else list(LEGACY_KINDS)
        )
    if not args.regular_grid and (
        args.worker
        in (
            "onchip-h",
            "output-owned-h",
            "reused-h",
            "reused-h16",
            "reused-h32",
            "prepared-h16",
            "prepared-h32",
            "prepared-g32-h16",
            "prepared-g32-h32",
            "input-owned-h16",
            "input-owned-h32",
            "stream-g8-h16",
            "stream-g8-h32",
            "site-routed-h16",
            "site-routed-h32",
            "site-owner-bm16-h16",
            "site-owner-bm16-h32",
            "site-stream-g8-h16",
            "site-stream-g8-h32",
            "site-owner-bm16-stream-g8-h16",
            "site-owner-bm16-stream-g8-h32",
            "input-site-stream-g8-h32",
            "input-order-stream-g8-h32",
        )
        or any(
            kind in args.plans
            for kind in (
                "onchip-h",
                "output-owned-h",
                "reused-h",
                "reused-h16",
                "reused-h32",
                "prepared-h16",
                "prepared-h32",
                "prepared-g32-h16",
                "prepared-g32-h32",
                "input-owned-h16",
                "input-owned-h32",
                "stream-g8-h16",
                "stream-g8-h32",
                "site-routed-h16",
                "site-routed-h32",
                "site-owner-bm16-h16",
                "site-owner-bm16-h32",
                "site-stream-g8-h16",
                "site-stream-g8-h32",
                "site-owner-bm16-stream-g8-h16",
                "site-owner-bm16-stream-g8-h32",
                "input-site-stream-g8-h32",
                "input-order-stream-g8-h32",
            )
        )
    ):
        parser.error("H recipes require --regular-grid")
    primary = REGULAR_PRIMARY if args.regular_grid else PRIMARY
    if args.isolated_oracle and not args.regular_grid:
        parser.error("isolated oracle snapshot currently requires --regular-grid")
    if args.aggregation_diagnostics and (
        not args.regular_grid or not args.isolated_oracle
    ):
        parser.error("aggregation diagnostics require regular grid and isolated oracle")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.state_gate:
        result = state_gate(regular_grid=args.regular_grid)
    elif args.worker:
        result = worker(
            args.size,
            args.rho,
            args.worker,
            phases=args.phases,
            aggregation_diagnostics=args.aggregation_diagnostics,
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
            if args.aggregation_diagnostics:
                command.append("--aggregation-diagnostics")
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
