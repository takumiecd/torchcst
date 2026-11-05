"""Numerical/trajectory admission before the long low-density HIGGS study."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    import torch

    from benchmarks.training.full_step_graph import exact, tensors
    from benchmarks.training.higgs_confirmation import make_model
    from benchmarks.training.higgs_confirmation_campaign import experiment_identity
    from benchmarks.training.owner_window_fp32 import errors, quadrature
    from benchmarks.training.radius_local_reference import local_reference
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    output, state = Path(args.output), Path(args.campaign_state)
    output.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.get_device_name() != "NVIDIA L4":
        raise ValueError("this confirmation campaign requires the requested L4")
    source = Path(__file__).resolve().parents[2]
    record = {
        "status": "running",
        "identity": experiment_identity(args.data, args.source_identity),
        "stage": "qualify",
        "tests": None,
        "oracles": [],
        "comparisons": [],
        "pilots": [],
        "test_evaluated": False,
    }
    _write(output / "qualification.json", record)
    with (output / "gpu-tests.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_higgs_confirmation.py",
                "tests/test_higgs_graph_cohort.py",
                "tests/test_graph_topology_guard.py",
                "tests/test_graph_stock_qualification.py",
            ],
            cwd=source,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1200,
            check=True,
        )
    record["tests"] = (output / "gpu-tests.log").read_text().splitlines()[-1]
    torch.manual_seed(20261105)
    for width in (1024, 2048):
        for divisor in (1000, 500, 200, 100):
            model = make_model(width, divisor, 17, "cst")[2]
            parameters = model.atoms.p.detach().clone()
            x, dy = torch.randn(4, width), torch.randn(4, width)
            reference = copy.deepcopy(model).double()
            rx, rp = x.double().requires_grad_(), parameters.double().requires_grad_()
            expected_y = local_reference(reference, rx, rp)
            expected = (
                expected_y.detach(),
                *torch.autograd.grad(expected_y, (rx, rp), dy.double()),
            )
            layout = hierarchy_from_operator(model.declaration())
            xx, pp = x.cuda().requires_grad_(), parameters.cuda().requires_grad_()
            actual_y = executor._extension().research_higgs_confirmation_owner_window_fp32_geometry_sort32_flat_c(
                xx,
                pp,
                layout.input.native_config(),
                layout.output.native_config(),
                True,
            )
            actual = (
                actual_y.detach().cpu(),
                *[
                    value.cpu()
                    for value in torch.autograd.grad(actual_y, (xx, pp), dy.cuda())
                ],
            )
            label = f"N{width}-D{divisor}"
            metrics = {
                key: errors(a, r)
                for key, a, r in zip(("Y", "dX", "dP"), actual, expected)
            }
            torch.save(
                {
                    "actual": actual,
                    "expected": expected,
                    "P": parameters,
                    "X": x,
                    "dY": dy,
                },
                output / (label + "-vjp.pt"),
            )
            record["oracles"].append(
                {
                    "name": label,
                    "atoms": len(parameters),
                    "reference": "public_CPU_FP64_local_Fourier",
                    **metrics,
                }
            )
            _write(output / "qualification.json", record)
            assert all(metrics[key]["pass"] for key in ("Y", "dX", "dP")), metrics
            if divisor == 1000:
                qp = parameters[:31].double().requires_grad_()
                qx = x.double().requires_grad_()
                qy = quadrature(reference, qx, qp)
                qexpected = (
                    qy.detach(),
                    *torch.autograd.grad(qy, (qx, qp), dy.double()),
                )
                qxx, qpp = (
                    x.cuda().requires_grad_(),
                    parameters[:31].cuda().requires_grad_(),
                )
                qactual_y = executor._extension().research_owner_window_fp32_geometry_sort32_flat_c(
                    qxx,
                    qpp,
                    layout.input.native_config(),
                    layout.output.native_config(),
                    False,
                )
                qactual = (
                    qactual_y.detach().cpu(),
                    *[
                        v.cpu()
                        for v in torch.autograd.grad(qactual_y, (qxx, qpp), dy.cuda())
                    ],
                )
                qmetrics = {
                    key: errors(a, r)
                    for key, a, r in zip(("Y", "dX", "dP"), qactual, qexpected)
                }
                torch.save(
                    {
                        "actual": qactual,
                        "expected": qexpected,
                        "P": parameters[:31],
                        "X": x,
                        "dY": dy,
                    },
                    output / f"N{width}-quadrature31.pt",
                )
                record["oracles"].append(
                    {
                        "name": f"N{width}-quadrature31",
                        "atoms": 31,
                        "reference": "independent_quadrature48",
                        **qmetrics,
                    }
                )
                _write(output / "qualification.json", record)
                assert all(qmetrics[key]["pass"] for key in ("Y", "dX", "dP")), qmetrics
            del model, reference, xx, pp, rx, rp, actual_y, expected_y
    cells = output / "cells"
    cells.mkdir(exist_ok=False)

    def invoke(name, width, divisor, kind, mode, *, pilot=False):
        destination = cells / name
        command = [
            sys.executable,
            "-m",
            "benchmarks.training.higgs_confirmation",
            "--data",
            str(Path(args.data).resolve()),
            "--output",
            str(destination),
            "--kind",
            kind,
            "--width",
            str(width),
            "--divisor",
            str(divisor),
            "--seed",
            "17",
            "--lr",
            "0.001",
            "--mode",
            mode,
            "--epochs",
            "1" if pilot else "2",
        ]
        if not pilot:
            command += [
                "--steps-per-epoch",
                "16",
                "--development-rows",
                "32",
                "--retain-epoch-checkpoints",
            ]
        begin = time.perf_counter()
        print("START " + name, flush=True)
        with (cells / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        learning = json.loads((destination / "learning.json").read_text())
        return {
            "name": name,
            "wall_seconds": time.perf_counter() - begin,
            "learning": learning,
        }

    for width in (1024, 2048):
        for divisor in (1000, 100):
            for kind in ("cst", "dense"):
                for mode in ("eager", "graph"):
                    invoke(
                        f"qual-{width}-{divisor}-{kind}-{mode}",
                        width,
                        divisor,
                        kind,
                        mode,
                    )
                for epoch in (1, 2):
                    left = torch.load(
                        cells
                        / f"qual-{width}-{divisor}-{kind}-eager"
                        / f"epoch-{epoch}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    right = torch.load(
                        cells
                        / f"qual-{width}-{divisor}-{kind}-graph"
                        / f"epoch-{epoch}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    comparison = exact(dict(tensors(left)), dict(tensors(right)))
                    assert comparison["all_byte_equal"], comparison
                    assert left["updates"] == right["updates"] == 16 * epoch
                    record["comparisons"].append(
                        {
                            "width": width,
                            "divisor": divisor,
                            "kind": kind,
                            "epoch": epoch,
                            **comparison,
                        }
                    )
                    _write(output / "qualification.json", record)
    for kind in ("dense", "cst"):
        pilot = invoke(
            f"budget-pilot-2048-{kind}", 2048, 1000, kind, "graph", pilot=True
        )
        record["pilots"].append(pilot)
        _write(output / "qualification.json", record)
    record["status"] = "passed"
    _write(output / "qualification.json", record)
    marker = state / "qualification.json"
    if marker.exists() and json.loads(marker.read_text()) != record:
        raise ValueError(
            "refusing to overwrite an existing different qualification marker"
        )
    _write(marker, record)
    return record
