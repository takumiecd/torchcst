"""Linear plan verification and complete-step comparison.

A fresh subprocess owns each correctness/timing/peak measurement. The baseline
is the same normalized operator with its full recipe; dense is a separate
performance reference. This runner does not certify or promote plans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from benchmarks.cuda.linear import reference
from benchmarks.cuda.linear.fixtures import operator_spec
from benchmarks.cuda.linear.manifest import (
    DEFAULT_PLANS,
    REGISTRY,
    decode_catalog,
    load_run,
    load_snapshot,
    read_json,
)
from torchcst import Dispatcher, LinearInputs
from torchcst._backends.torch.algorithms.linear.normalized_radial.layout import (
    geometry,
)
from torchcst.operators.execution import LinearBinding


class PlanLinear(nn.Module):
    """Execute precisely the forced plan; never fall back to another algorithm."""

    def __init__(self, p, operator, plan):
        super().__init__()
        self.p = nn.Parameter(p.detach().clone().contiguous())
        self.operator, self.plan = operator, plan
        self._algorithm_states = []
        self.workspace_limit_bytes = None
        self.local_state = None
        self.persistent_layout = None
        self.update_binding = None
        if plan.algorithm_id == "research_local_product":
            from benchmarks.cuda.linear.local_product import runtime

            self.local_state, domain = runtime(operator, self.p.device)
            from torchcst import Atoms, AtomUpdateBinding, Operator

            self.update_binding = AtomUpdateBinding(
                Operator(
                    charts=domain.charts(device=self.p.device, dtype=self.p.dtype),
                    kernel=self.local_state,
                    atoms=Atoms(self.p),
                )
            )
            if plan.recipe.cached_order:
                from torchcst._backends.cuda.algorithms.linear.local_product.order_cache import (
                    OrderKeyCache,
                )

                self.persistent_layout = OrderKeyCache(
                    self.p, self.local_state, domain, plan.recipe
                )
            elif plan.recipe.execution_route == "hybrid_persistent":
                from torchcst._backends.cuda.algorithms.linear.local_product.persistent import (
                    PersistentLayout,
                )

                self.persistent_layout = PersistentLayout(
                    self.p, self.local_state, domain, plan.recipe
                )

    input_type = LinearInputs

    execution_declaration = LinearBinding.execution_declaration

    def execution_parameters(self):
        return self.p

    validate_inputs = LinearBinding.validate_inputs

    def build_context(self, inputs):
        from torchcst.operators.context import context_from_tensors

        return context_from_tensors(self.operator, inputs.x, self.p)

    def state_signature(self):
        from torchcst._backends.state import tensor_signature

        return tensor_signature(self.p), self.operator

    algorithm_state = LinearBinding.algorithm_state

    def forward(self, x):
        flat = x.reshape(-1, self.operator.in_features).contiguous()
        y = Dispatcher(registry=REGISTRY).run(self, LinearInputs(flat), plan=self.plan)
        return y.reshape(*x.shape[:-1], self.operator.out_features)


def _metadata(args, run):
    prop = torch.cuda.get_device_properties(args.device)
    import triton

    import torchcst

    source_files = sorted(Path(torchcst.__file__).parent.rglob("*.py"))
    source_files += [
        Path(__file__),
        Path(reference.__file__),
        Path(__file__).with_name("fixtures.py"),
        Path(__file__).with_name("check_normalized.py"),
        Path(__file__).with_name("manifest.py"),
        Path(__file__).with_name("local_product.py"),
        Path(__file__).parent.parent / "polar_update.py",
    ]
    return {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "source_hashes": {
            str(p.relative_to(Path.cwd()))
            if p.is_relative_to(Path.cwd())
            else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        "gpu": prop.name,
        "device_index": args.device,
        "compute_capability": [prop.major, prop.minor],
        "sm_count": prop.multi_processor_count,
        "total_device_bytes": prop.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "tf32": False,
        "dtype": "float32",
        "seed": run.case.seed,
        "initialization": {
            "method": "torch.cpu.v1",
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
        "case": asdict(run.case),
        "input_hashes": run.input_hashes,
        "snapshot_sha256": args.snapshot_sha256,
        "worker": args.worker,
        "plan_id": args.plan_id,
        "polar_update": args.polar_update,
        "plan": None
        if args.worker == "dense"
        else REGISTRY.dump_plan(run.entry(args.plan_id).plan),
    }


def correctness(args, run):
    if run.case.fixture == "local_polar_product":
        from benchmarks.cuda.linear.local_product import correctness as local_check

        return local_check(args, run, PlanLinear)
    from benchmarks.cuda.linear.check_normalized import check

    sizes = (1024, 4, 4)
    p = reference.mixed(torch.float32, "cuda")
    p[0, 2], p[1, 2] = 511.25, 512.5
    p[2:4, 2] = 512.02978515625
    op = operator_spec(sizes=sizes, origin=(0.0, 0.0, 0.0), spacing=(1.0, 0.5, 0.5))
    model = PlanLinear(p, op, run.entry(args.plan_id).plan)
    gen = torch.Generator(device="cpu").manual_seed(run.case.seed)
    x = (
        torch.randn(2, 3, 16, device="cpu", dtype=torch.float32, generator=gen)
        .cuda()
        .requires_grad_()
    )
    dy = torch.randn(
        2, 3, 1024, device="cpu", dtype=torch.float32, generator=gen
    ).cuda()
    y = model(x)
    tp = model.p.detach().double().requires_grad_()
    tx = x.detach().double().requires_grad_()
    w = reference.oracle(tp, sizes, stored_dtype=torch.float32)
    truth = tx @ w.T
    ga = torch.autograd.grad(y, (x, model.p), dy)
    gt = torch.autograd.grad(truth, (tx, tp), dy.double())
    return {
        "status": "PASS",
        "scope": "small mixed fixture; independent FP64 all-atom oracle",
        "fixture_sizes": sizes,
        "fixture_atoms": len(p),
        "y": check(y, truth),
        "dx": check(ga[0], gt[0]),
        "dp": check(ga[1], gt[1]),
        "w": check(model(torch.eye(16, device="cuda")).T, w, tol=2e-5),
    }


def measure(args, run):
    case = run.case
    n, m = case.size, case.rows
    local = case.fixture == "local_polar_product"
    support_report = None
    if local:
        from benchmarks.cuda.linear.local_product import (
            fixture_operator,
            fixture_state,
            initialize,
        )
        from torchcst._backends.cuda.algorithms.linear.local_product.contract import (
            Domain,
        )
        from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
            decode,
        )
        from torchcst._backends.cuda.algorithms.linear.local_product.support import (
            summarize,
        )

        op, p = fixture_operator(case), initialize(case)
        support_report = summarize(decode(fixture_state(case), p), Domain(n, n))
    else:
        h, j = (32, 32) if n == 1024 else (64, 128)
        sizes = (n, h, j)
        origin = (-(n - 1) / 2, -(h - 1) / 4, -(j - 1) / 4)
        op = operator_spec(sizes=sizes, origin=origin, spacing=(1.0, 0.5, 0.5))
        sites = geometry(op)
        gen = torch.Generator(device="cpu").manual_seed(run.case.seed)
        p = torch.empty(case.atoms, 5)
        p[:, 0] = torch.rand(len(p), generator=gen) - 0.5
        p[:, 1] = math.log(3.0 if case.profile == "broad" else 0.199)
        p[:, 2:] = torch.rand(len(p), 3, generator=gen) * torch.tensor(
            [(count - 1) * spacing for count, spacing in zip(sizes, sites.spacing)]
        ) + torch.tensor(origin)
        if case.profile == "sharp":
            for axis, (spacing, o) in enumerate(zip(sites.spacing, sites.origin)):
                u = (p[:, axis + 2] - o) / spacing
                near = torch.floor(u + 0.5)
                p[:, axis + 2] = o + near * spacing + (u - near) * 0.04
    initial_p_hash = hashlib.sha256(p.numpy().tobytes()).hexdigest()
    torch.manual_seed(run.case.seed)
    cpu_x, cpu_target = generate_inputs(run.case.seed, m, n)
    x = cpu_x.cuda().requires_grad_()
    target = cpu_target.cuda()
    del cpu_x, cpu_target
    initial_inputs = {
        "x_sha256": hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
    }
    if args.worker == "dense":
        model = nn.Linear(n, n, bias=False, device="cuda")
        weight_gen = torch.Generator(device="cpu").manual_seed(run.case.seed)
        weight = torch.empty(n, n).uniform_(-0.01, 0.01, generator=weight_gen)
        model.weight.data.copy_(weight.cuda())
        del weight
    else:
        model = PlanLinear(p.cuda(), op, run.entry(args.plan_id).plan)
    del p
    initial_sigma = None
    initial_precision = None
    if local and args.worker != "dense":
        from torchcst._backends.cuda.algorithms.linear.local_product.preparation import (
            decode,
        )

        initial_precision = decode(model.local_state, model.p).detach()[:, 1].cpu()
        initial_sigma = initial_precision.rsqrt()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=case.optimizer.fused,
        capturable=case.optimizer.capturable,
    )

    def step(events=None, update_events=None):
        if events is not None:
            events[0].record()
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        loss = (model(x) * target).sum() / (m * n)
        if events is not None:
            events[1].record()
        loss.backward()
        if events is not None:
            events[2].record()
        if local and args.worker != "dense":
            from benchmarks.cuda.polar_update import optimizer_step

            optimizer_step(
                model.update_binding,
                optimizer,
                step_size=case.optimizer.lr,
                polar_update=args.polar_update,
                events=update_events,
            )
        else:
            optimizer.step()
        if events is not None:
            events[3].record()

    def timed(call):
        samples = []
        for _ in range(case.rounds):
            torch.cuda.synchronize()
            start = time.perf_counter()
            call()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {"median_ms": statistics.median(samples), "samples_ms": samples}

    for _ in range(case.warmup):
        step()
    eager = timed(step)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(case.warmup):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    graph_times = timed(graph.replay)
    torch.cuda.synchronize()
    # Save the measured peaks before post-run diagnostics allocate buffers.
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    sigma_updates = None
    hybrid_routing = None
    if initial_sigma is not None:
        final_precision = decode(model.local_state, model.p).detach()[:, 1].cpu()
        final_sigma = final_precision.rsqrt()
        change = final_sigma - initial_sigma
        sigma_updates = {
            "fixed": False,
            "source": "current polar activity every forward/replay",
            "initial": initial_sigma.tolist(),
            "final": final_sigma.tolist(),
            "changed_atoms": int((change != 0).sum()),
            "max_abs_change": float(change.abs().max()),
        }
        if not sigma_updates["changed_atoms"]:
            raise AssertionError("dynamic-width fixture did not update sigma")
        recipe = run.entry(args.plan_id).plan.recipe
        if recipe.execution_route in (
            "hybrid",
            "hybrid_support",
            "hybrid_three",
            "hybrid_singletons",
            "hybrid_packed",
            "hybrid_persistent",
        ):
            inclusive = recipe.execution_route in (
                "hybrid_three",
                "hybrid_singletons",
                "hybrid_packed",
                "hybrid_persistent",
            )
            limit = recipe.rho_upper[1 if inclusive else 0]
            initial_wide = (
                initial_precision <= limit**-2
                if inclusive
                else initial_precision < limit**-2
            )
            final_wide = (
                final_precision <= limit**-2
                if inclusive
                else final_precision < limit**-2
            )
            hybrid_routing = {
                "rho_limit": limit,
                "diagnostic_basis": "production Torch decode outside timing; exact boundary comparisons can differ by FP32 rounding",
                "initial_local_atoms": int((~initial_wide).sum()),
                "initial_saved_atoms": int(initial_wide.sum()),
                "final_local_atoms": int((~final_wide).sum()),
                "final_saved_atoms": int(final_wide.sum()),
                "changed_routes": int((initial_wide != final_wide).sum()),
                "scratch_capacity_elements": 0
                if recipe.recompute_h
                else m * case.atoms,
                "capacity_policy": "H recomputed within output owners and parameter VJP; no saved H"
                if recipe.recompute_h
                else "fixed B*A; only wide H lanes written/read; not compacted",
            }
            if inclusive:
                for label, precision in (
                    ("initial", initial_precision),
                    ("final", final_precision),
                ):
                    rho = precision.rsqrt()
                    hybrid_routing[label + "_three_band_atoms"] = [
                        int((rho < 1).sum()),
                        int(((rho >= 1) & (rho < limit)).sum()),
                        int((rho >= limit).sum()),
                    ]
                hybrid_routing["band_boundaries"] = [1, limit]
                hybrid_routing["kernel_saved_comparison"] = (
                    "rho >= mid; CPU diagnostics may differ at FP32 boundaries"
                )
        final_support = summarize(
            decode(model.local_state, model.p).detach().cpu(), Domain(n, n)
        )
        if recipe.execution_route in (
            "hybrid_singletons",
            "hybrid_packed",
            "hybrid_persistent",
        ):
            hybrid_routing["singleton_direct_initial_atoms"] = support_report[
                "onehot_both_live_atoms"
            ]
            hybrid_routing["singleton_direct_final_atoms"] = final_support[
                "onehot_both_live_atoms"
            ]
            hybrid_routing["saved_scope"] = (
                "none; H recomputed"
                if recipe.recompute_h
                else (
                    "rho>=mid excluding full-domain live singletons; width-only saved counts above are pre-exclusion"
                )
            )
    else:
        final_support = None
    if local:
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if args.worker != "dense":
            radius = model.p.detach()[:, :2].square().sum(1)
            assert ((radius >= 1 - 1e-5) & (radius <= 4 + 1e-5)).all()
    layout_report = None
    if args.worker != "dense" and model.persistent_layout is not None:
        layout_report = model.persistent_layout.report()
    phases = None
    if args.phase_diagnostics or args.kernel_diagnostics:
        # A separate instrumented training graph after the primary timing/memory
        # measurement. External event nodes add overhead; these are diagnostics,
        # not replacements for uninstrumented complete-step graph timings.
        events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
        update_events = (
            [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
            if local and args.worker != "dense"
            else None
        )
        diagnostic = torch.cuda.CUDAGraph()
        backend_events = None
        if args.kernel_diagnostics and local and args.worker != "dense":
            from torchcst._backends.cuda.algorithms.linear.local_product import executor

            backward_components = (
                ("dx_source_vjp", "source_partial_reduce")
                if run.entry(args.plan_id).plan.recipe.fused_backward
                else ("dx", "parameters")
            )
            backend_events = {
                name: [
                    torch.cuda.Event(enable_timing=True, external=True)
                    for _ in range(2)
                ]
                for name in ("prepare", "layout", "h", "output", *backward_components)
            }
            executor.DIAGNOSTIC_EVENTS = backend_events
        try:
            with torch.cuda.graph(diagnostic, stream=stream):
                step(events, update_events)
        finally:
            if backend_events is not None:
                executor.DIAGNOSTIC_EVENTS = None
        samples = {name: [] for name in ("forward_loss", "backward", "optimizer")}
        update_samples = {name: [] for name in ("old_snapshot", "adamw", "polar")}
        backend_samples = {name: [] for name in backend_events or {}}
        for _ in range(case.rounds):
            diagnostic.replay()
            torch.cuda.synchronize()
            for index, values in enumerate(samples.values()):
                values.append(events[index].elapsed_time(events[index + 1]))
            if update_events is not None:
                for index, values in enumerate(update_samples.values()):
                    values.append(
                        update_events[index].elapsed_time(update_events[index + 1])
                    )
            for name, values in backend_samples.items():
                pair = backend_events[name]
                values.append(pair[0].elapsed_time(pair[1]))
        phases = {
            "scope": "separate graph with external CUDA events; sigma continues updating; after primary timing/memory measurement",
            "phases": {
                name: {"median_ms": statistics.median(values), "samples_ms": values}
                for name, values in samples.items()
            },
            "optimizer_components": None
            if update_events is None
            else {
                name: {"median_ms": statistics.median(values), "samples_ms": values}
                for name, values in update_samples.items()
            },
            "backend_components": {
                name: {"median_ms": statistics.median(values), "samples_ms": values}
                for name, values in backend_samples.items()
            },
        }
    core = None
    if args.core_diagnostics:
        from benchmarks.cuda.linear.local_product import measure_prepared_forward

        core = measure_prepared_forward(
            case,
            None if args.worker == "dense" else run.entry(args.plan_id).plan.recipe,
        )
    return {
        "status": "PASS",
        "reference": "dense_linear" if args.worker == "dense" else args.plan_id,
        "operator": asdict(op),
        "rows": m,
        "atoms": None if args.worker == "dense" else case.atoms,
        "profile": None if args.worker == "dense" else case.profile,
        "initial_p_sha256": initial_p_hash if args.worker != "dense" else None,
        "initial_inputs": initial_inputs,
        "eager": eager,
        "graph": graph_times,
        "phase_diagnostics": phases,
        "prepared_forward_diagnostics": core,
        "allocated_before_capture_bytes": before,
        "peak_allocated_capture_replay_bytes": peak_allocated,
        "peak_reserved_capture_replay_bytes": peak_reserved,
        "total_gpu_process_bytes": None,
        "memory_scope": "isolated process; warmed model, gradients and AdamW state; peak includes capture/replay; process usage unmeasured",
        "optimizer": asdict(case.optimizer),
        "scope": "complete-step performance; no independent full-shape all-atom gradient oracle",
        "optimizer_policy": "euclidean polar finite_chord; AdamW proposal + activity/radial update"
        if local and args.worker != "dense"
        else "ordinary AdamW",
        "initial_support": support_report if args.worker != "dense" else None,
        "final_support": final_support,
        "sigma_updates": sigma_updates,
        "persistent_layout": layout_report,
        "hybrid_routing": hybrid_routing,
        "h_policy": run.entry(args.plan_id).plan.recipe.execution_route
        if local and args.worker != "dense"
        else None,
        "compiler_reports": __import__(
            "torchcst._backends.cuda.algorithms.linear.local_product.executor",
            fromlist=["COMPILER_REPORTS"],
        ).COMPILER_REPORTS
        if local and args.worker != "dense"
        else None,
    }


def generate_inputs(seed, rows, features):
    """Independent CPU generator; CUDA device scheduling cannot change the batch."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        torch.randn(
            rows, features, device="cpu", dtype=torch.float32, generator=generator
        )
        for _ in range(2)
    )


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    ap.add_argument("--case", type=Path)
    ap.add_argument("--list-plans", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--source-commit", default="unrecorded")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--phase-diagnostics", action="store_true")
    ap.add_argument("--core-diagnostics", action="store_true")
    ap.add_argument("--kernel-diagnostics", action="store_true")
    ap.add_argument(
        "--polar-update",
        choices=("torch", "fused"),
        default="torch",
        help="local-product research polar update; AdamW is unchanged",
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--worker", choices=("correctness", "measure", "dense"), help=argparse.SUPPRESS
    )
    ap.add_argument("--snapshot", type=Path, help="execute a frozen run snapshot")
    ap.add_argument("--snapshot-sha256", help="required SHA256 of the frozen snapshot")
    ap.add_argument("--plan-id", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.device < 0:
        ap.error("device must be nonnegative")
    if args.list_plans:
        entries = decode_catalog(read_json(args.plans)[0])
        print(
            json.dumps(
                [
                    {"id": entry.id, "plan": REGISTRY.dump_plan(entry.plan)}
                    for entry in entries
                ],
                indent=2,
            )
        )
        return
    if args.worker:
        if not args.snapshot or not args.snapshot_sha256 or not args.output:
            ap.error("worker needs snapshot, snapshot hash and output")
        run = load_snapshot(args.snapshot, expected_hash=args.snapshot_sha256)
        if args.worker != "dense":
            run.entry(args.plan_id)
        elif not run.dense:
            ap.error("dense reference was not selected")
    else:
        if args.snapshot:
            if args.case or not args.snapshot_sha256:
                ap.error(
                    "--snapshot requires its hash and cannot be combined with --case"
                )
            run = load_snapshot(args.snapshot, expected_hash=args.snapshot_sha256)
        else:
            if not args.case or args.snapshot_sha256:
                ap.error("--case or a frozen --snapshot with its hash is required")
            run = load_run(args.case, args.plans)
        if args.validate_only:
            print(json.dumps(run.snapshot(), indent=2))
            return
    if args.core_diagnostics and run.case.fixture != "local_polar_product":
        ap.error("--core-diagnostics requires the local product fixture")
    if args.polar_update == "fused" and run.case.fixture != "local_polar_product":
        ap.error("--polar-update fused requires the local product fixture")
    if args.kernel_diagnostics and (
        run.case.fixture != "local_polar_product"
        or any(entry.plan.recipe.execution_route == "torch" for entry in run.plans)
    ):
        ap.error("--kernel-diagnostics requires CUDA local product routes")
    if not args.output:
        ap.error("--output is required for GPU execution")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = {"metadata": {"worker": args.worker, "plan_id": args.plan_id}}
        try:
            torch.cuda.set_device(args.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            result["metadata"] = _metadata(args, run)
            result["result"] = (
                correctness(args, run)
                if args.worker == "correctness"
                else measure(args, run)
            )
        except Exception as error:
            result["result"] = {"status": "FAIL", "error": repr(error)}
            _write(args.output, result)
            raise
        _write(args.output, result)
        return

    artifacts = Path(
        tempfile.mkdtemp(prefix=args.output.stem + "-", dir=args.output.parent)
    )
    snapshot = artifacts / "run.json"
    _write(snapshot, run.snapshot())
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    combined = {
        "schema_version": 1,
        "execution_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "certification": "not assessed",
        "run": run.snapshot(),
        "snapshot_sha256": snapshot_hash,
        "artifacts_directory": str(artifacts),
        "records": [],
    }
    _write(args.output, combined)

    def run_worker(kind, plan_id=None):
        path = artifacts / f"{kind}-{plan_id or 'dense'}.json"
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.cuda.linear.run",
            "--worker",
            kind,
            "--snapshot",
            str(snapshot),
            "--snapshot-sha256",
            snapshot_hash,
            "--output",
            str(path),
            "--device",
            str(args.device),
            "--source-commit",
            args.source_commit,
        ]
        if plan_id is not None:
            cmd += ["--plan-id", plan_id]
        if args.phase_diagnostics:
            cmd += ["--phase-diagnostics"]
        if args.core_diagnostics:
            cmd += ["--core-diagnostics"]
        if args.kernel_diagnostics:
            cmd += ["--kernel-diagnostics"]
        cmd += ["--polar-update", args.polar_update]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        returncode = None
        try:
            completed = subprocess.run(cmd, env=env, check=False)
            returncode = completed.returncode
            if not path.exists():
                raise ValueError("worker did not produce a record")
            record = read_json(path)[0]
            if (
                type(record) is not dict
                or type(record.get("result")) is not dict
                or record["result"].get("status") not in ("PASS", "FAIL")
            ):
                raise ValueError("worker produced an invalid record")
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
            record = {
                "metadata": {"worker": kind, "plan_id": plan_id},
                "result": {"status": "FAIL", "error": repr(error)},
            }
        combined["records"].append(record)
        if returncode != 0 or record["result"]["status"] != "PASS":
            combined["status"] = "FAIL"
            combined["error"] = (
                f"{kind}/{plan_id} exited {returncode}: {record['result'].get('error', 'worker failure')}"
            )
        _write(args.output, combined)
        if combined["status"] == "FAIL":
            raise RuntimeError(combined["error"])
        return record["result"]

    for entry in run.plans:
        run_worker("correctness", entry.id)
    if not args.correctness_only:
        measured = {entry.id: run_worker("measure", entry.id) for entry in run.plans}
        baseline = measured[run.baseline]
        try:
            for result in measured.values():
                if (result["initial_p_sha256"], result["initial_inputs"]) != (
                    baseline["initial_p_sha256"],
                    baseline["initial_inputs"],
                ):
                    raise ValueError("candidate initialization differs from baseline")
            combined["comparisons"] = {
                name: {
                    mode + "_time_ratio_to_baseline": result[mode]["median_ms"]
                    / baseline[mode]["median_ms"]
                    for mode in ("eager", "graph")
                }
                for name, result in measured.items()
            }
            if run.dense:
                dense = run_worker("dense")
                if dense["initial_inputs"] != baseline["initial_inputs"]:
                    raise ValueError("dense inputs differ from baseline")
        except Exception as error:
            combined["status"] = "FAIL"
            combined["error"] = repr(error)
            _write(args.output, combined)
            raise
    combined["status"] = "PASS"
    combined["comparison_scope"] = (
        "same fixture initialization and optimizer; separate processes; sequential sessions, not paired alternating timing"
    )
    _write(args.output, combined)
    print(json.dumps({"status": combined["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
