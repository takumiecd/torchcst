"""Compare CPU and GPU variance calibration for the width-2048 QKV study."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from torchcst import CSTLinear, ChebyshevSpec, chart_presets, kernels
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    physical_operator,
    width_extension,
)
from torchcst._backends.torch import initialization
from torchcst._backends.torch.kernels.radius_cell_mass import atom_energies
from torchcst._backends.torch.parameterizations import radius_envelope

WIDTH = 2048
TAU = 0.03
AMPLITUDE_MAX = 32.0
VARIANCE = 0.02**2
ATOM_COUNTS = {"qkv-c64": 131072, "qkv-c128": 262144}
ROLES = {"query": "q", "key": "k", "value": "v"}


def site_seed(seed, site):
    block, role = site.split(".")
    name = f"block{block}.attention.{ROLES[role]}"
    return (
        seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    ) % (2**63 - 1)


def make_layer(count, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CSTLinear(
            chart_presets.regular_grid(WIDTH, spacing=1.0, geometry="flat_torus"),
            chart_presets.regular_grid(WIDTH, spacing=1.0, geometry="flat_torus"),
            atoms=count,
            dtype=torch.float32,
            backend="factored",
            kernel=kernels.product(
                kernels.cell_mass(ChebyshevSpec(modes=8), merge_factor=2),
                coordinates=kernels.radius_envelope(
                    radius_min=0.5,
                    radius_max=512,
                    tau=TAU,
                    radius_fraction_init=0,
                    amplitude_max=AMPLITUDE_MAX,
                ),
            ),
        )


def calibrated(layer, seed, *, energy_evaluator=None):
    return initialization.match_variance(
        layer.kernel,
        layer.cst_charts(),
        layer.atoms.p,
        variance=VARIANCE,
        q=0,
        generator=torch.Generator().manual_seed(seed + 1),
        energy_evaluator=energy_evaluator,
    )


def physical_rows_and_banks(kernel, raw, boundaries):
    # R = 0.5 belongs to the two-cell bank. A radius just above 0.5 can
    # round to it in FP32, so classify before converting the physical rows.
    physical = radius_envelope.from_state(kernel, raw.double())
    radius = physical[:, 1].contiguous()
    level = torch.bucketize(radius, boundaries, out_int32=True)
    banks = 2 * level + (radius > 0.5).to(torch.int32)
    return physical, banks


def gpu_evaluator(native, layer):
    op = physical_operator(layer.execution_declaration())
    hierarchy = hierarchy_from_operator(op)
    axes = (hierarchy.input.native_config(),)
    if hierarchy.input.site_count != hierarchy.output.site_count:
        axes += (hierarchy.output.native_config(),)
    boundaries = torch.tensor([1 << level for level in range(9)], dtype=torch.float64)

    def energies(kernel, p):
        physical, banks = physical_rows_and_banks(kernel, p, boundaries)
        return native.packed_initialization_energies(
            physical.to(device="cuda", dtype=torch.float32), banks.cuda(), *axes
        )

    def evaluate(kernel, _charts, p):
        return float(energies(kernel, p).sum())

    return energies, evaluate


def timed_calibration(layer, seed, *, energy_evaluator=None):
    torch.cuda.synchronize()
    start = time.perf_counter()
    parameters, report = calibrated(layer, seed, energy_evaluator=energy_evaluator)
    torch.cuda.synchronize()
    return parameters, report, time.perf_counter() - start


def reconstruct_original(layer, seed, report):
    raw = layer.atoms.p.detach().clone()
    noise = torch.randn(
        len(raw), dtype=raw.dtype, generator=torch.Generator().manual_seed(seed + 1)
    )
    ratio = (noise * (report["amplitude_scale"] / AMPLITUDE_MAX)).clamp(
        -1 + 1e-6, 1 - 1e-6
    )
    raw[:, 0] = ratio * AMPLITUDE_MAX
    raw[:, 1] = 0
    return raw


def projection_energy_error(energies, layer, raw):
    sample = 4096
    state = copy.deepcopy(layer.kernel).double()
    expected = atom_energies(
        state,
        *(copy.deepcopy(chart).double() for chart in layer.cst_charts()),
        raw[:sample].double(),
    )
    actual = energies(state, raw)[:sample].cpu().double()
    radius = radius_envelope.from_state(state, raw.double())[:, 1]
    difference = actual - expected
    return {
        "sample_atoms": sample,
        "relative_l2": float(difference.norm() / expected.norm()),
        "max_abs": float(difference.abs().max()),
        "radii_rounded_to_half_from_above": int(
            ((radius > 0.5) & (radius.float() == 0.5)).sum()
        ),
    }


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(args):
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    assert torch.cuda.get_device_capability() == (8, 9)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(radius_envelope, "USE_RATIONAL_LOWER"):
        radius_envelope.USE_RATIONAL_LOWER = False

    evidence = json.loads(args.study.read_text())
    assert evidence["state"] == "completed"
    assert evidence["identity"]["atoms_per_qkv_projection"] == ATOM_COUNTS
    native = width_extension()
    result = {
        "state": "running",
        "gpu": torch.cuda.get_device_name(),
        "study_sha256": digest(args.study),
        "driver_sha256": digest(Path(__file__)),
        "native_sha256": digest(Path(native.__file__)),
        "controlled_timings": {},
        "projection_checks": [],
    }
    save(args.output, result)
    for arm, count in ATOM_COUNTS.items():
        seed = site_seed(17, "0.query")
        layer = make_layer(count, seed)
        energies, evaluate = gpu_evaluator(native, layer)
        evaluate(layer.kernel, None, layer.atoms.p.detach())
        cpu, cpu_report, cpu_seconds = timed_calibration(layer, seed)
        gpu, gpu_report, gpu_seconds = timed_calibration(
            layer, seed, energy_evaluator=evaluate
        )
        check = projection_energy_error(energies, layer, gpu)
        result["controlled_timings"][arm] = {
            "cpu_seconds": cpu_seconds,
            "gpu_seconds": gpu_seconds,
            "bitwise_parameters_equal": torch.equal(cpu, gpu),
            "cpu_report": cpu_report,
            "gpu_report": gpu_report,
            "per_atom_energy_error": check,
        }
        print(arm, result["controlled_timings"][arm], flush=True)
        save(args.output, result)

    for seed, arms in evidence["seeds"].items():
        for arm in ATOM_COUNTS:
            for site, entry in arms[arm]["sites"].items():
                parameter_seed = site_seed(int(seed), site)
                layer = make_layer(ATOM_COUNTS[arm], parameter_seed)
                _, evaluate = gpu_evaluator(native, layer)
                cpu_report = entry["initialization"]
                cpu = reconstruct_original(layer, parameter_seed, cpu_report)
                candidate, report, seconds = timed_calibration(
                    layer, parameter_seed, energy_evaluator=evaluate
                )
                result["projection_checks"].append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        "site": site,
                        "seconds": seconds,
                        "bitwise_parameters_equal": torch.equal(cpu, candidate),
                        "reference_scale": cpu_report["amplitude_scale"],
                        "candidate_scale": report["amplitude_scale"],
                        "relative_energy_difference": abs(
                            report["expected_energy"] / cpu_report["expected_energy"]
                            - 1
                        ),
                    }
                )
            save(args.output, result)
            print("CHECKED", seed, arm, flush=True)
    assert len(result["projection_checks"]) == 72
    result["state"] = "completed"
    save(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
