"""MNIST learning/width-distribution study through the public CSTLinear API.

This is a learning experiment, not the Linear benchmark runner or a DB result
adapter. Epoch timings include input gathering, loss/backward and the actual
optimizer workflow. Validation, sigma diagnostics and checkpoints are separate.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import struct
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    FixedSelector,
    chart_presets,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.cell_mass_hierarchical.plans import (
    SITE_SCALED,
    VALIDATED_128,
    VALIDATED_1024,
)
from torchcst.kernels import presets

# The official torchvision MNIST loader publishes these mirror/checksum pairs:
# https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_mnist(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, hashes = {}, {}
    for name, expected in MNIST_FILES.items():
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(
                "https://ossci-datasets.s3.amazonaws.com/mnist/" + name, timeout=120
            ) as stream:
                path.write_bytes(stream.read())
        data = path.read_bytes()
        if hashlib.md5(data, usedforsecurity=False).hexdigest() != expected:
            raise ValueError("MNIST checksum mismatch: " + name)
        hashes[name] = hashlib.sha256(data).hexdigest()
        raw = gzip.decompress(data)
        magic, count = struct.unpack(">II", raw[:8])
        if "images" in name:
            if (
                magic != 2051
                or struct.unpack(">II", raw[8:16]) != (28, 28)
                or len(raw) != 16 + count * 784
            ):
                raise ValueError("invalid MNIST images")
            value = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).reshape(
                count, 784
            )
        else:
            if magic != 2049 or len(raw) != 8 + count:
                raise ValueError("invalid MNIST labels")
            value = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).long()
        tensors[name.split("-")[0] + ("_x" if "images" in name else "_y")] = value
    if len(tensors["train_x"]) != 60000 or len(tensors["t10k_x"]) != 10000:
        raise ValueError("expected complete MNIST train/test splits")
    train = tensors["train_x"].float() / 255
    mean, std = train.mean().item(), train.std(unbiased=False).item()
    return tensors, {"sha256": hashes, "mean": mean, "std": std}


def make_cst(width, atoms, *, dtype=torch.float32):
    chart = chart_presets.regular_grid(
        ((width,),), spacing=128 / width, geometry="flat_torus"
    )
    kernel = presets.cell_mass(
        input_min_tile_sites=width // 32,
        output_min_tile_sites=width // 16,
        sigma_init=(2.0, 40.0),
    )
    plan = VALIDATED_128 if width == 128 else VALIDATED_1024
    return CSTLinear(
        chart,
        chart,
        atoms=atoms,
        kernel=kernel,
        dtype=dtype,
        selector=FixedSelector(plan, registry=make_algorithm_registry()),
    )


def polar_bounds(width, *, site_scaled):
    spacing = 128 / width
    if site_scaled:
        minimum = spacing
        maximum = min(100 * minimum, 128 / 3)
        # Scalar buffers are FP32 in training. Round the nonrepresentable
        # Q/3 ceiling down so a later FP64 decode still fits the domain.
        rounded = torch.tensor(maximum, dtype=torch.float32)
        if float(rounded) > maximum:
            rounded = torch.nextafter(rounded, torch.zeros_like(rounded))
        return BandwidthBounds(
            minimum=minimum,
            birth=8 * minimum,
            maximum=float(rounded),
            upper_floor=2 * minimum,
        )
    return BandwidthBounds(minimum=2.0, birth=16.0, maximum=40.0, upper_floor=4.0)


def make_polar_cst(
    width,
    atoms,
    *,
    amplitude_scale=1.0,
    dtype=torch.float32,
    site_scaled=False,
    bounds=None,
):
    chart = chart_presets.regular_grid(
        ((width,),), spacing=128 / width, geometry="flat_torus"
    )
    kernel = presets.polar_cell_mass(
        input_min_tile_sites=4 if site_scaled else width // 32,
        output_min_tile_sites=8 if site_scaled else width // 16,
        amplitude_max=amplitude_scale,
        w_c=0.3 * amplitude_scale,
        input_bounds=bounds or polar_bounds(width, site_scaled=site_scaled),
        alpha_init=0.5,
    )
    plan = (
        SITE_SCALED
        if site_scaled
        else (VALIDATED_128 if width == 128 else VALIDATED_1024)
    )
    return CSTLinear(
        chart,
        chart,
        atoms=atoms,
        kernel=kernel,
        dtype=dtype,
        selector=FixedSelector(plan, registry=make_algorithm_registry()),
    )


def make_polar_initialization(width, seed=120, *, site_scaled=True):
    """Random angles, initial alpha=.5, calibrated amplitude units.

    Scaling amplitude_max AND w_c together preserves w/w_c, hence every
    initial width. This is explicit initialization, not a new update rule.
    """
    if width not in (128, 1024):
        raise ValueError("this learning study qualifies widths 128 and 1024")
    torch.manual_seed(seed)
    count = math.floor(0.05 * width * width)
    atoms = torch.randn(count, 4)
    atoms[:, :2] *= math.sqrt(2.5) / atoms[:, :2].norm(dim=1, keepdim=True)
    atoms[:, 2:].uniform_(0.0, 128.0)
    reference = make_polar_cst(
        width, atoms.double(), dtype=torch.float64, site_scaled=site_scaled
    )
    with torch.no_grad():
        gain = float(
            (1 / math.sqrt(width)) / reference.dense_weight().square().mean().sqrt()
        )
        # Round scalar buffers as FP32 training does, before the FP64 export.
        reference = make_polar_cst(
            width, atoms.clone(), amplitude_scale=gain, site_scaled=site_scaled
        ).double()
        weight = reference.dense_weight().float()
    from torchcst._backends.torch.kernels.polar_cell_mass import decoded_values

    _, si, so = decoded_values(reference.kernel, atoms)
    torch.manual_seed(seed + 1)
    shared_in, shared_out = nn.Linear(784, width), nn.Linear(width, 10)
    bounds = polar_bounds(width, site_scaled=site_scaled)
    return {
        "width": width,
        "seed": seed,
        "atoms": atoms,
        "weight": weight,
        "amplitude_gain": gain,
        "parameterization": "polar",
        "cohort": "site_scaled" if site_scaled else "fixed_coordinates",
        "sigma_initial": {"input": si.detach(), "output": so.detach()},
        "input": shared_in.state_dict(),
        "output": shared_out.state_dict(),
        "kernel_settings": {
            "amplitude_max": gain,
            "w_c": 0.3 * gain,
            "minimum": bounds.minimum,
            "birth": bounds.birth,
            "maximum": bounds.maximum,
            "upper_floor": bounds.upper_floor,
            "spacing": 128 / width,
            "input_min_tile_sites": 4 if site_scaled else width // 32,
            "output_min_tile_sites": 8 if site_scaled else width // 16,
            "alpha_init": 0.5,
            "kappa": 3.0,
            "lower_kappa": 3.0,
            "upper_decay_power": 1.0,
            "radial_regularization": 0.1,
            "activity_gain": 1.0,
            "activity_mode": "finite_chord",
            "dormant_expansion_rate": 0.0,
        },
    }


def make_initialization(width, seed=120):
    """Share initial effective W; calibrate only the initial amplitude gain."""
    if width not in (128, 1024):
        raise ValueError("this learning study currently qualifies widths 128 and 1024")
    torch.manual_seed(seed)
    count = math.floor(0.05 * width * width)
    atoms = torch.randn(count, 4)
    atoms[:, 1].uniform_(2.0, 40.0)
    atoms[:, 2:].uniform_(0.0, 128.0)
    reference = make_cst(width, atoms.double(), dtype=torch.float64)
    with torch.no_grad():
        weight = reference.dense_weight()
        gain = float((1 / math.sqrt(width)) / weight.square().mean().sqrt())
        atoms[:, 0] *= gain
        reference.atoms.p.copy_(atoms.double())
        weight = reference.dense_weight().float()
    torch.manual_seed(seed + 1)
    shared_in, shared_out = nn.Linear(784, width), nn.Linear(width, 10)
    return {
        "width": width,
        "seed": seed,
        "atoms": atoms,
        "weight": weight,
        "amplitude_gain": gain,
        "input": shared_in.state_dict(),
        "output": shared_out.state_dict(),
    }


def make_model(initialization, kind):
    n = initialization["width"]
    if kind not in ("cst", "dense"):
        raise ValueError("kind must be cst or dense")
    hidden = (
        (
            make_polar_cst(
                n,
                initialization["atoms"].clone(),
                amplitude_scale=initialization["amplitude_gain"],
                site_scaled=initialization.get("cohort") == "site_scaled",
                # Frozen fixtures carry their own bounds. In particular,
                # earlier min/spacing=.25 runs must remain reproducible.
                bounds=BandwidthBounds(
                    **{
                        name: initialization["kernel_settings"][name]
                        for name in ("minimum", "birth", "maximum", "upper_floor")
                    }
                ),
            )
            if initialization.get("parameterization") == "polar"
            else make_cst(n, initialization["atoms"].clone())
        )
        if kind == "cst"
        else nn.Linear(n, n, bias=False)
    )
    if kind == "dense":
        with torch.no_grad():
            hidden.weight.copy_(initialization["weight"])
    model = nn.Sequential(
        nn.Linear(784, n), nn.ReLU(), hidden, nn.ReLU(), nn.Linear(n, 10)
    )
    model[0].load_state_dict(copy.deepcopy(initialization["input"]))
    model[4].load_state_dict(copy.deepcopy(initialization["output"]))
    return model


def sigma_statistics(layer, initial_sigma):
    p = layer.atoms.p.detach().cpu().double()
    plan = hierarchy_from_operator(layer.declaration())
    is_polar = plan.input.polar is not None
    if is_polar:
        from torchcst._backends.torch.kernels.polar_cell_mass import decoded_values

        state = copy.deepcopy(layer.kernel).cpu().double()
        amplitude, si, so = decoded_values(state, layer.atoms.p.detach().cpu())
        sigmas = {"input": si, "output": so}
    else:
        amplitude = p[:, 0]
        sigmas = {"input": p[:, 1], "output": p[:, 1]}
    sigma = sigmas["input"]
    initial_input = (
        initial_sigma["input"] if isinstance(initial_sigma, dict) else initial_sigma
    )
    result = {
        "quantiles": torch.quantile(
            sigma,
            torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64),
        ).tolist(),
        "absolute_change_quantiles": torch.quantile(
            (sigma - initial_input.double()).abs(),
            torch.tensor([0.0, 0.5, 0.9, 1.0], dtype=torch.float64),
        ).tolist(),
        "valid": bool(
            torch.isfinite(p).all()
            and all(
                bool((v > 0).all() and (v <= 128 / 3).all()) for v in sigmas.values()
            )
        ),
    }
    result["parameterization"] = "polar" if is_polar else "physical"
    if is_polar:
        result["alpha_quantiles"] = torch.quantile(
            ((p[:, :2].square().sum(1) - 1) / 3).clamp(0, 1),
            torch.tensor([0.0, 0.5, 0.9, 1.0], dtype=torch.float64),
        ).tolist()
        result["output_sigma_quantiles"] = torch.quantile(
            sigmas["output"],
            torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64),
        ).tolist()
    nodes, periods = {}, {}
    for side, column in (("input", 3), ("output", 2)):
        from torchcst._backends.torch.profiles.fourier import effective_radius

        sigma = sigmas[side]
        if is_polar:
            column = 2 if side == "input" else 3
        axis = getattr(plan, side)
        remaining = torch.ones(len(p), dtype=torch.bool)
        ids = torch.empty(len(p), dtype=torch.int64)
        atom_period = torch.empty(len(p), dtype=torch.float64)
        radii = torch.empty(len(p), dtype=torch.float64)
        banks = {}
        center = (p[:, column] - axis.origin).remainder(axis.period)
        for bank in axis.banks:
            selected = (
                remaining
                if bank.full_chart
                else remaining & (sigma <= bank.tile_sites * axis.spacing)
            )
            owner = (
                (center + axis.spacing / 2) / (bank.tile_sites * axis.spacing)
            ).floor().long() % bank.tile_count
            ids[selected] = bank.node_offset + owner[selected]
            atom_period[selected] = bank.period
            radii[selected] = effective_radius(
                axis.profile, sigma[selected], bank.period
            )
            banks[str(bank.level)] = {
                "atoms": int(selected.sum()),
                "amplitude_abs_sum": float(amplitude[selected].abs().sum()),
                "tile_width": bank.tile_sites * axis.spacing,
            }
            remaining = remaining & ~selected
        assert not remaining.any()
        counts = torch.bincount(ids, minlength=axis.node_count)
        occupied = counts[counts > 0].double()
        result[side] = {
            "spacing": axis.spacing,
            "sigma_over_spacing_quantiles": torch.quantile(
                sigma / axis.spacing,
                torch.tensor(
                    [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64
                ),
            ).tolist(),
            "effective_radius_over_spacing_quantiles": torch.quantile(
                radii / axis.spacing,
                torch.tensor(
                    [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64
                ),
            ).tolist(),
            "banks": banks,
            "owner_counts": counts.tolist(),
            "max_owner_fraction": float(counts.max() / len(p)),
            "max_to_mean_occupied": float(occupied.max() / occupied.mean()),
        }
        nodes[side], periods[side] = ids, atom_period
    group = nodes["output"] * plan.input.node_count + nodes["input"]
    _, counts = torch.unique(group, return_counts=True)
    result["groups"] = {
        "nonempty": len(counts),
        "max_atoms": int(counts.max()),
        "max_to_mean": float(counts.max() / counts.double().mean()),
    }
    result["same_shape_fraction"] = float(
        (
            (periods["input"] == periods["output"])
            & (sigmas["input"] == sigmas["output"])
        )
        .double()
        .mean()
    )
    return result


@torch.no_grad()
def evaluate(model, x, y, batch_size):
    model.eval()
    total_loss = x.new_zeros(())
    correct = torch.zeros((), device=x.device, dtype=torch.int64)
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size])
        targets = y[start : start + batch_size]
        total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum")
        correct += (logits.argmax(1) == targets).sum()
    return {"loss": float(total_loss / len(x)), "accuracy": float(correct / len(x))}


def run_training(
    initialization, kind, data_directory, output, *, epochs=3, batch_size=32, lr=1e-3
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, data_info = load_mnist(data_directory)
    device = torch.device("cuda")
    model = make_model(initialization, kind).to(device)
    tx = ((data["train_x"].to(device).float() / 255) - data_info["mean"]) / data_info[
        "std"
    ]
    vx = ((data["t10k_x"].to(device).float() / 255) - data_info["mean"]) / data_info[
        "std"
    ]
    ty, vy = data["train_y"].to(device), data["t10k_y"].to(device)
    base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
    optimizer = CSTOptimizer(base, model=model) if kind == "cst" else base
    initial_test = evaluate(model, vx, vy, batch_size)
    # Load/compile the implementation and exercise backward before timing.
    # Warmup does not update any Parameter or optimizer state.
    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(tx[:batch_size]), ty[:batch_size]).backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    rows = []
    if kind == "cst":
        (output / "sigma-initial.json").write_text(
            json.dumps(
                sigma_statistics(
                    model[2],
                    initialization.get("sigma_initial", initialization["atoms"][:, 1]),
                ),
                indent=2,
            )
        )
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(
            len(tx),
            generator=torch.Generator().manual_seed(
                initialization["seed"] + 100 + epoch
            ),
        )
        order_sha = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        order = order.to(device)
        total_loss = tx.new_zeros(())
        correct = torch.zeros((), device=device, dtype=torch.int64)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for start in range(0, len(tx), batch_size):
            indices = order[start : start + batch_size]
            inputs = tx[indices]
            target = ty[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(indices)
            correct += (logits.detach().argmax(1) == target).sum()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        row = {
            "epoch": epoch + 1,
            "training_seconds": seconds,
            "steps": math.ceil(len(tx) / batch_size),
            "mean_step_ms": seconds * 1000 / math.ceil(len(tx) / batch_size),
            "training_loss": float(total_loss / len(tx)),
            "training_accuracy": float(correct / len(tx)),
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "data_order_sha256": order_sha,
        }
        row["test"] = evaluate(model, vx, vy, batch_size)
        if kind == "cst":
            row["sigma"] = sigma_statistics(
                model[2],
                initialization.get("sigma_initial", initialization["atoms"][:, 1]),
            )
        rows.append(row)
        (output / "epochs.json").write_text(json.dumps(rows, indent=2))
        print(
            f"MNIST {initialization['width']} {kind} epoch {epoch + 1}: {row['test']}",
            flush=True,
        )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "width": initialization["width"],
            "kind": kind,
            "epoch": epochs,
        },
        output / "final.pt",
    )
    report = {
        "cohort": initialization.get("cohort", "fixed_coordinates"),
        "parameterization": initialization.get("parameterization", "physical"),
        "kernel_settings": initialization.get("kernel_settings"),
        "kind": kind,
        "initial_test": initial_test,
        "width": initialization["width"],
        "atom_count": len(initialization["atoms"]),
        "epochs": rows,
        "batch_size": batch_size,
        "lr": lr,
        "seed": initialization["seed"],
        "amplitude_gain": initialization["amplitude_gain"],
        "data": data_info,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "scope": "Fresh process; full MNIST train/test splits, one seed, same initial effective hidden W and other layers, same data order; epoch training wall includes gathering/loss/all backward/actual optimizer, excludes evaluation/diagnostics/checkpoints; allocator peaks include resident data and model; no Graph, TF32 off; learning experiment, not DB/Linear benchmark output",
    }
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--kind", choices=["cst", "dense"], required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    initialization = torch.load(
        args.initialization, map_location="cpu", weights_only=True
    )
    run_training(
        initialization,
        args.kind,
        args.data_directory,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
