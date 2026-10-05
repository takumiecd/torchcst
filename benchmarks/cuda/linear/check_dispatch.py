"""Write real dispatch JSON, reload it, and validate the public CUDA execution path.

Policies here are explicit demonstration fixtures, not measured winners. The
optional 1024-square training benchmarks run in separate processes per Plan.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from benchmarks.cuda.linear.check_normalized import benchmark, check, small_gate
from benchmarks.cuda.linear.fixtures import normalized_chart
from benchmarks.cuda.linear.manifest import DEFAULT_PLANS, decode_catalog, read_json
from benchmarks.cuda.linear.reference import mixed, oracle
from torchcst import CSTLinear, presets
from torchcst._backends.catalog import REGISTRY
from torchcst._backends.dispatch import ExactSelector, load_selector
from torchcst._backends.dispatch.conditions import dump_condition
from torchcst._backends.serialization import encode_json
from torchcst.operators.context import context_from_tensors


class RecordingSelector(ExactSelector):
    """Test instrumentation: keep only primitive diagnostics, never tensors."""

    def select(self, context):
        decision = super().select(context)
        self.trace.append(
            {
                "shape": list(context.input_shape),
                "mode": context.execution_mode,
                "algorithm_id": decision.plan.algorithm_id,
                "path": list(decision.matched_path),
            }
        )
        return decision


class PolicyFiles:
    def __init__(self, directory, catalog, *, measured=False):
        if len(catalog) != 2:
            raise ValueError("this demonstration compares exactly two catalog Plans")
        self.directory, self.catalog, self.measured = directory, catalog, measured
        self.selectors, self.contexts = [], {}

    def __call__(self, model, algorithm_id):
        primary = next(e for e in self.catalog if e.plan.algorithm_id == algorithm_id)
        fallback = next(e for e in self.catalog if e.id != primary.id)
        shapes = ((128, model.in_features),) if self.measured else ((6, 16), (16, 16))
        contexts = []
        for shape in shapes:
            for input_grad in (False, True):
                x = torch.zeros(
                    shape,
                    dtype=model.atoms.p.dtype,
                    device=model.atoms.p.device,
                    requires_grad=input_grad,
                )
                ctx = context_from_tensors(model.declaration(), x, model.atoms.p)
                del x
                contexts.extend((ctx, replace(ctx, execution_mode="cuda_graph")))
        import triton

        # Construct the JSON fields explicitly, instead of an in-memory-only
        # selector. Human-readable catalog IDs remain local aliases for Plans.
        document = {
            "schema_version": 1,
            "selector": {"kind": "exact_table", "revision": "json-demo-v1"},
            "score_policy": {
                "id": "demonstration-only",
                "revision": "v1",
                "parameters": {},
            },
            "dataset_snapshot": "demonstration-fixture-not-a-leaderboard",
            "runtime_versions": {
                "torch": str(torch.__version__),
                "triton": triton.__version__,
            },
            "plans": {
                entry.id: REGISTRY.dump_plan(entry.plan) for entry in self.catalog
            },
            "entries": [
                {
                    "condition": dump_condition(ctx),
                    "plan_id": primary.id,
                    "evidence_ids": [f"demonstration-only:{primary.id}"],
                }
                for ctx in contexts
            ],
            "fallback_plan_id": fallback.id,
        }
        path = self.directory / f"dispatch-{primary.id}.json"
        raw = encode_json(document).encode()
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("same fixture generated different policy bytes")
        else:
            path.write_bytes(raw)
        selector = RecordingSelector.loads(path.read_bytes(), registry=REGISTRY)
        assert selector.dumps().encode() == raw
        selector.trace = []
        self.selectors.append(selector)
        self.contexts[primary.id] = contexts[0]
        return selector

    def report(self):
        trace = [item for selector in self.selectors for item in selector.trace]
        if not trace or any(item["path"][0] != "exact_table" for item in trace):
            raise AssertionError(
                "the public path did not exclusively use exact entries"
            )
        return {
            "trace": trace,
            "policies": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.directory.glob("dispatch-*.json"))
            },
        }


def correctness(directory, catalog):
    factory = PolicyFiles(directory, catalog)
    reports = small_gate(selector_factory=factory)
    verification = factory.report()
    original = directory / f"dispatch-{catalog[0].id}.json"
    loaded = load_selector(original.read_bytes(), registry=REGISTRY)
    ctx = factory.contexts[catalog[0].id]
    changed = replace(ctx, input_shape=(7, 16))
    decision = loaded.select(changed)
    assert decision.plan == catalog[1].plan
    assert not decision.evidence_ids and decision.matched_path[0] == "fallback"
    model = CSTLinear(
        chart=normalized_chart((1024, 4, 4), dtype=torch.float32, device="cuda"),
        atoms=mixed(torch.float32, "cuda"),
        kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
        selector=loaded,
    )
    x = torch.randn(7, 16, device="cuda", requires_grad=True)
    y = model(x)
    pp = model.atoms.p.detach().double().requires_grad_()
    xx = x.detach().double().requires_grad_()
    truth = xx @ oracle(pp, (1024, 4, 4), stored_dtype=torch.float32).T
    actual_grads = torch.autograd.grad(y.sum(), (x, model.atoms.p))
    truth_grads = torch.autograd.grad(truth.sum(), (xx, pp))
    fallback = {
        name: check(a, b)
        for name, a, b in zip(
            ("y", "dx", "dp"),
            (y, *actual_grads),
            (truth, *truth_grads),
        )
    }
    # Exercise file-based failures independently of the successful policies.
    failures = {}
    data = json.loads(original.read_bytes())
    invalid = json.loads(original.read_bytes())
    invalid["entries"][0]["plan_id"] = "missing"
    bad_path = directory / "invalid-plan.json"
    bad_path.write_text(encode_json(invalid), encoding="utf-8")
    try:
        load_selector(bad_path.read_bytes(), registry=REGISTRY)
    except ValueError as error:
        failures["unknown_plan"] = str(error)
    else:
        raise AssertionError("invalid Plan was accepted")
    invalid["entries"][0]["plan_id"] = data["entries"][0]["plan_id"]
    invalid["runtime_versions"]["torch"] = "0.0"
    bad_path = directory / "invalid-runtime.json"
    bad_path.write_text(encode_json(invalid), encoding="utf-8")
    try:
        load_selector(bad_path.read_bytes(), registry=REGISTRY)
    except ValueError as error:
        failures["runtime_version"] = str(error)
    else:
        raise AssertionError("incompatible runtime was accepted")
    return {
        "status": "PASS",
        "scope": "independent small mixed fixture; y/dX/all-five atom gradients; AdamW Graph replay",
        "correctness": reports,
        "selection": verification,
        "fallback": {"algorithm_id": decision.plan.algorithm_id, "checks": fallback},
        "rejected_files": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument(
        "--worker", choices=("measure", "dense"), help=argparse.SUPPRESS
    )
    parser.add_argument("--plan-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    catalog = decode_catalog(read_json(args.plans)[0])
    import triton

    prop = torch.cuda.get_device_properties(args.device)
    runtime = {
        "gpu": prop.name,
        "compute_capability": [prop.major, prop.minor],
        "sm_count": prop.multi_processor_count,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "tf32": False,
    }
    if args.worker:
        dense = args.worker == "dense"
        factory = (
            None if dense else PolicyFiles(args.output_dir, catalog, measured=True)
        )
        entry = None if dense else next(e for e in catalog if e.id == args.plan_id)
        report = benchmark(
            1024,
            "broad",
            None if dense else entry.plan.algorithm_id,
            dense=dense,
            selector_factory=factory,
        )
        report["status"] = "PASS"
        if factory is not None:
            report["selection"] = factory.report()
    else:
        report = correctness(args.output_dir, catalog)
        if args.bench:
            commands = [(entry.id, "measure") for entry in catalog] + [
                ("dense", "dense")
            ]
            report["benchmarks"] = {}
            for name, worker in commands:
                path = args.output_dir / name
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.cuda.linear.check_dispatch",
                        "--output-dir",
                        str(path),
                        "--plans",
                        str(args.plans),
                        "--device",
                        str(args.device),
                        "--worker",
                        worker,
                        "--plan-id",
                        name,
                    ],
                    check=True,
                )
                report["benchmarks"][name] = json.loads(
                    (path / "result.json").read_bytes()
                )
    report["runtime"] = runtime
    report["process_memory_usage"] = "unmeasured"
    (args.output_dir / "result.json").write_text(encode_json(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output_dir),
                "gpu": prop.name,
            }
        )
    )


if __name__ == "__main__":
    main()
