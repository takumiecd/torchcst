"""Finite, sequential one-L4 HIGGS screening campaign with correctness gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def qualification_oracles(output):
    import torch

    from benchmarks.training.higgs_density import make_model
    from benchmarks.training.owner_window_fp32 import errors, quadrature
    from benchmarks.training.radius_local_reference import local_reference
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    torch.set_num_threads(1)
    torch.manual_seed(413)
    rows = []
    for name, independent in (("full_min_density", False), ("quadrature31", True)):
        model = make_model(1024, 200, 17, "cst")[2]
        parameters = model.atoms.p.detach().clone()
        if independent:
            parameters = parameters[:31].clone()
        x, dy = torch.randn(4, 1024), torch.randn(4, 1024)
        reference_model = copy.deepcopy(model).double()
        rx = x.double().requires_grad_()
        rp = parameters.double().requires_grad_()
        expected_y = (
            quadrature(reference_model, rx, rp)
            if independent
            else local_reference(reference_model, rx, rp)
        )
        expected = (
            expected_y.detach(),
            *torch.autograd.grad(expected_y, (rx, rp), dy.double()),
        )
        layout = hierarchy_from_operator(model.declaration())
        xx = x.cuda().requires_grad_()
        pp = parameters.cuda().requires_grad_()
        actual_y = (
            executor._extension().research_owner_window_fp32_geometry_sort32_flat_c(
                xx,
                pp,
                layout.input.native_config(),
                layout.output.native_config(),
                False,
            )
        )
        actual = (
            actual_y.detach().cpu(),
            *[v.cpu() for v in torch.autograd.grad(actual_y, (xx, pp), dy.cuda())],
        )
        record = {
            key: errors(a, b) for key, a, b in zip(("Y", "dX", "dP"), actual, expected)
        }
        record["atoms"] = len(parameters)
        torch.save(
            {"actual": actual, "expected": expected, "P": parameters, "X": x, "dY": dy},
            output / (name + ".pt"),
        )
        rows.append({"name": name, **record})
        write(output / "oracles.json", rows)
        assert all(record[key]["pass"] for key in ("Y", "dX", "dP")), record
    return rows


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cells = args.output / "cells"
    cells.mkdir()
    state = {"status": "qualifying", "completed": [], "selected_lr": {}}
    state_path = args.output / "campaign.json"
    source = Path(__file__).resolve().parents[2]
    write(
        args.output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "sources": {
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.rglob("*.py"))
                if ".higgs-job-venv" not in p.parts and ".job-deps" not in p.parts
            },
        },
    )

    def persist():
        write(state_path, state)

    def invoke(
        name,
        kind,
        width,
        divisor=20,
        lr=0.001,
        epochs=1,
        steps=0,
        mode="graph",
        validation_rows=8192,
        checkpoints="final",
    ):
        destination = cells / name
        command = [
            sys.executable,
            "-m",
            "benchmarks.training.higgs_density",
            "--data",
            str(args.data.resolve()),
            "--output",
            str(destination.resolve()),
            "--kind",
            kind,
            "--width",
            str(width),
            "--divisor",
            str(divisor),
            "--lr",
            str(lr),
            "--epochs",
            str(epochs),
            "--steps-per-epoch",
            str(steps),
            "--mode",
            mode,
            "--validation-rows",
            str(validation_rows),
            "--checkpoint-policy",
            checkpoints,
        ]
        print("START " + name, flush=True)
        state["running"] = {"name": name, "command": command, "started": time.time()}
        persist()
        with (cells / (name + ".log")).open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800
            )
        result = json.loads((destination / "learning.json").read_text())
        state["completed"].append(
            {
                "name": name,
                "kind": kind,
                "width": width,
                "alpha": 1 / divisor if kind == "cst" else None,
                "lr": lr,
                "epochs": epochs,
                "steps_per_epoch": steps,
                "validation": result["epochs"][-1]["validation"],
                "training_seconds": result["training_seconds"],
                "adoption_seconds": result["adoption_seconds"],
                "allocated": result["training_peak_allocated_bytes"],
                "reserved": result["training_peak_reserved_bytes"],
            }
        )
        state["running"] = None
        persist()
        print("DONE " + json.dumps(state["completed"][-1]), flush=True)
        return result

    try:
        persist()
        with (args.output / "gpu-tests.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_higgs_graph_cohort.py",
                    "tests/test_higgs_density.py",
                    "tests/test_graph_topology_guard.py",
                    "tests/test_graph_stock_qualification.py",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        state["oracles"] = qualification_oracles(args.output)
        # Independent fresh processes; compare actual model/moments/clock/grad
        # tensors after both epoch boundaries, not just validation scores.
        import torch

        from benchmarks.training.full_step_graph import exact, tensors

        comparisons = []
        for divisor in (200, 10):
            for kind in ("cst", "dense"):
                for mode in ("eager", "graph"):
                    invoke(
                        f"qual-{divisor}-{kind}-{mode}",
                        kind,
                        1024,
                        divisor,
                        epochs=2,
                        steps=16,
                        mode=mode,
                        validation_rows=32,
                        checkpoints="full",
                    )
                for epoch in (1, 2):
                    left = torch.load(
                        cells / f"qual-{divisor}-{kind}-eager" / f"epoch-{epoch}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    right = torch.load(
                        cells / f"qual-{divisor}-{kind}-graph" / f"epoch-{epoch}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    comparison = exact(dict(tensors(left)), dict(tensors(right)))
                    assert comparison["all_byte_equal"], comparison
                    assert left["updates"] == right["updates"] == 16 * epoch
                    comparisons.append(
                        {"divisor": divisor, "kind": kind, "epoch": epoch, **comparison}
                    )
                    write(args.output / "eager-graph.json", comparisons)
        state["status"] = "lr_screen"
        persist()
        for kind in ("dense", "cst"):
            candidates = []
            for lr in (0.0001, 0.001, 0.003):
                result = invoke(f"lr-{kind}-{lr}", kind, 1024, lr=lr, steps=5000)
                candidates.append((result["epochs"][-1]["validation"]["loss"], lr))
            state["selected_lr"][kind] = min(candidates)[1]
            persist()
        state["status"] = "density_scale_screen"
        persist()
        for width in (1024, 2048, 4096, 8192):
            # Reverse dense/CST position across widths to expose obvious order
            # dependence; single runs still do not establish timing uncertainty.
            if width in (1024, 4096):
                invoke(
                    f"grid-{width}-dense",
                    "dense",
                    width,
                    lr=state["selected_lr"]["dense"],
                )
            for divisor in (200, 100, 50, 20, 10) if width != 8192 else (100, 20):
                invoke(
                    f"grid-{width}-cst-{divisor}",
                    "cst",
                    width,
                    divisor,
                    lr=state["selected_lr"]["cst"],
                )
            if width in (2048, 8192):
                invoke(
                    f"grid-{width}-dense",
                    "dense",
                    width,
                    lr=state["selected_lr"]["dense"],
                )
        state["status"] = "complete_screen_only"
        persist()
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = type(error).__name__ + ": " + str(error)
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("CST_JOB_OUTPUT", "output/higgs-campaign")),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
