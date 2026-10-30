"""MNIST learning/width-distribution study through the public CSTLinear API.

This is a learning experiment, not the Linear benchmark runner or a DB result
adapter. Epoch timings include input gathering, loss/backward and the actual
optimizer workflow. Validation, kernel diagnostics and checkpoints are separate.
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
    ChebyshevRadiusSpec,
    CSTAdamW,
    CSTLinear,
    CSTOptimizer,
    FixedSelector,
    RadiusBounds,
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

RADIUS_COHORT = "polar_radius_native_128_v1"


def radius_optimizer_contract():
    return {
        "lr": 1e-4,
        "ordinary_weight_decay": 0.01,
        "dense_weight_decay": 0.01,
        "atom_raw_weight_decay": 0.0,
        "amplitude_weight_decay": 0.01,
        "fused": True,
        "capturable": False,
        "check_finite": False,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }


def radius_plan(execution="cached", *, direct_levels=2):
    """The explicit recipe in plans-radius-cell-mass.json, without dispatch changes."""
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.recipe import (
        RadiusCellMassRecipe,
    )
    from torchcst._backends.schema import ExecutionPlan

    if execution not in ("cached", "batched"):
        raise ValueError("radius_execution must be cached or batched")
    if type(direct_levels) is not int or not 0 <= direct_levels <= 8:
        raise ValueError("radius_direct_levels must be an integer in [0,8]")
    return ExecutionPlan(
        "cuda_radius_cell_mass",
        "v1",
        RadiusCellMassRecipe(
            input_direct_levels=direct_levels,
            output_direct_levels=direct_levels,
            schedule="block_local_cached_parallel"
            if execution == "cached"
            else "block_local_batched_cached_parallel",
            output_tile_sites=32,
            native_autograd=True,
            native_entry=True,
        ),
    )


def radius_settings():
    return {
        "profile": "ChebyshevRadiusSpec",
        "spacing": 1.0,
        "modes": 8,
        "base_tile_sites": 1,
        "merge_factor": 2,
        "amplitude_max": 1.0,
        "w_c": 0.3,
        "radius_bounds": {
            "minimum": 0.5,
            "birth": 4.0,
            "maximum": 50.0,
            "upper_floor": 1.0,
        },
        "alpha_init": 0.0,
        "kappa": 3.0,
        "lower_kappa": 3.0,
        "upper_decay_power": 1.0,
        "radial_regularization": 0.1,
        "activity_gain": 1.0,
        "activity_mode": "finite_chord",
        "dormant_expansion_rate": 0.0,
    }


def make_radius_cst(
    width,
    atoms,
    *,
    dtype=torch.float32,
    radius_execution="cached",
    radius_direct_levels=2,
):
    """Current N128 public radius fixture, with no sigma/radius conversion."""
    if width != 128:
        raise ValueError("native radius MNIST currently qualifies only width 128")
    chart = chart_presets.regular_grid(((width,),), spacing=1.0, geometry="flat_torus")
    kernel = presets.polar_radius_cell_mass(
        amplitude_max=1.0,
        w_c=0.3,
        input_radius_bounds=RadiusBounds(
            minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0
        ),
        profile=ChebyshevRadiusSpec(modes=8),
        merge_factor=2,
    )
    return CSTLinear(
        chart,
        chart,
        atoms=atoms,
        kernel=kernel,
        dtype=dtype,
        selector=FixedSelector(
            radius_plan(radius_execution, direct_levels=radius_direct_levels),
            registry=make_algorithm_registry(),
        ),
    )


def tensor_sha256(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def buffer_metadata(module):
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().reshape(-1).tolist(),
        }
        for name, value in module.named_buffers()
    }


def make_radius_initialization(width=128, seed=41):
    """Export exact public radius atoms/effective W and shared ordinary layers."""
    torch.manual_seed(seed)
    reference = make_radius_cst(width, math.floor(0.05 * width * width))
    atoms = reference.atoms.p.detach().clone()
    with torch.no_grad():
        weight = reference.dense_weight().detach().clone()
    torch.manual_seed(seed + 1)
    shared_in, shared_out = nn.Linear(784, width), nn.Linear(width, 10)
    initialization = {
        "schema_version": 1,
        "cohort": RADIUS_COHORT,
        "parameterization": "polar_radius",
        "width": width,
        "seed": seed,
        "atoms": atoms,
        "weight": weight,
        "amplitude_gain": 1.0,
        "input": shared_in.state_dict(),
        "output": shared_out.state_dict(),
        "kernel_settings": radius_settings(),
        "kernel_buffers_fp32": buffer_metadata(reference.kernel),
        "optimizer_contract": radius_optimizer_contract(),
        "linear_plan": make_algorithm_registry().dump_plan(radius_plan()),
        "update_plan": "cuda_radius_flat_torus_polar_update_fused/v1",
        "initialization_policy": "public FP32 radius initialization; no gain calibration; no optimizer updates",
        "architecture": "Linear(784,128)-ReLU-radius(128,128)-ReLU-Linear(128,10)",
    }
    initialization["initial_hashes"] = {
        "atoms": tensor_sha256(atoms),
        "weight": tensor_sha256(weight),
        "input": {
            name: tensor_sha256(value)
            for name, value in initialization["input"].items()
        },
        "output": {
            name: tensor_sha256(value)
            for name, value in initialization["output"].items()
        },
        "kernel_buffers_fp32": hashlib.sha256(
            json.dumps(initialization["kernel_buffers_fp32"], sort_keys=True).encode()
        ).hexdigest(),
    }
    initialization["radius_initial"] = radius_statistics(reference)
    return initialization


def validate_radius_initialization(initialization):
    required = {
        "cohort",
        "parameterization",
        "schema_version",
        "width",
        "seed",
        "atoms",
        "weight",
        "input",
        "output",
        "kernel_settings",
        "kernel_buffers_fp32",
        "optimizer_contract",
        "linear_plan",
        "update_plan",
        "initial_hashes",
    }
    if not required <= initialization.keys():
        raise ValueError("radius initialization is missing frozen cohort fields")
    if (
        initialization.get("cohort") != RADIUS_COHORT
        or initialization.get("parameterization") != "polar_radius"
        or initialization.get("schema_version") != 1
        or initialization["width"] != 128
        or tuple(initialization["atoms"].shape) != (819, 4)
        or tuple(initialization["weight"].shape) != (128, 128)
        or any(
            initialization[name].dtype != torch.float32 for name in ("atoms", "weight")
        )
        or initialization["kernel_settings"] != radius_settings()
        or initialization["optimizer_contract"] != radius_optimizer_contract()
        or initialization["update_plan"]
        != "cuda_radius_flat_torus_polar_update_fused/v1"
        or initialization["linear_plan"]
        != make_algorithm_registry().dump_plan(radius_plan())
    ):
        raise ValueError("radius initialization differs from the frozen N128 cohort")
    for name in ("atoms", "weight"):
        if (
            tensor_sha256(initialization[name])
            != initialization["initial_hashes"][name]
        ):
            raise ValueError("radius initialization tensor hash differs: " + name)
    for name in ("input", "output"):
        if {
            key: tensor_sha256(value) for key, value in initialization[name].items()
        } != initialization["initial_hashes"][name]:
            raise ValueError(
                "radius initialization ordinary-layer hash differs: " + name
            )
    if (
        hashlib.sha256(
            json.dumps(initialization["kernel_buffers_fp32"], sort_keys=True).encode()
        ).hexdigest()
        != initialization["initial_hashes"]["kernel_buffers_fp32"]
    ):
        raise ValueError("radius initialization FP32 buffer metadata differs")


def make_optimizer(model, initialization, kind, *, lr=None):
    if kind not in ("cst", "dense"):
        raise ValueError("kind must be cst or dense")
    if initialization.get("parameterization") != "polar_radius":
        base = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3 if lr is None else lr,
            weight_decay=0.0,
            fused=True,
        )
        return CSTOptimizer(base, model=model) if kind == "cst" else base
    validate_radius_initialization(initialization)
    contract = initialization["optimizer_contract"]
    if lr is not None and lr != contract["lr"]:
        raise ValueError("radius lr must match the frozen initialization contract")
    ordinary = [*model[0].parameters(), *model[4].parameters()]
    groups = [
        {
            "params": ordinary,
            "weight_decay": contract["ordinary_weight_decay"],
            "role": "ordinary",
        },
        {
            "params": list(model[2].parameters()),
            "weight_decay": contract["atom_raw_weight_decay"]
            if kind == "cst"
            else contract["dense_weight_decay"],
            "role": "middle",
        },
    ]
    options = {
        name: contract[name]
        for name in ("lr", "fused", "capturable", "betas", "eps", "amsgrad", "maximize")
    }
    if kind == "dense":
        return torch.optim.AdamW(groups, **options)
    from torchcst._backends.cuda.algorithms.polar_update.plans import (
        RADIUS_FLAT_TORUS_FUSED,
    )

    groups[1]["amplitude_weight_decay"] = contract["amplitude_weight_decay"]
    return CSTAdamW(
        groups,
        model=model,
        check_finite=contract["check_finite"],
        update_selector=FixedSelector(
            RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
        ),
        **options,
    )


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


def normalize_images(images, device, mean, std, *, data_normalization="original"):
    """Normalize MNIST pixels; the opt-in allocation variant never aliases input."""
    if data_normalization == "original":
        return ((images.to(device).float() / 255) - mean) / std
    if data_normalization == "inplace":
        return (
            images.to(device=device, dtype=torch.float32, copy=True)
            .div_(255)
            .sub_(mean)
            .div_(std)
        )
    raise ValueError("data_normalization must be original or inplace")


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


def make_model(
    initialization, kind, *, radius_execution="cached", radius_direct_levels=2
):
    n = initialization["width"]
    if kind not in ("cst", "dense"):
        raise ValueError("kind must be cst or dense")
    radius = initialization.get("parameterization") == "polar_radius"
    radius_plan(radius_execution, direct_levels=radius_direct_levels)
    if not radius and radius_execution != "cached":
        raise ValueError("radius_execution overrides require a radius initialization")
    if not radius and radius_direct_levels != 2:
        raise ValueError(
            "radius_direct_levels overrides require a radius initialization"
        )
    if radius:
        validate_radius_initialization(initialization)
    hidden = (
        make_radius_cst(
            n,
            initialization["atoms"].clone(),
            radius_execution=radius_execution,
            radius_direct_levels=radius_direct_levels,
        )
        if radius and kind == "cst"
        else (
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
    )
    if kind == "dense":
        with torch.no_grad():
            hidden.weight.copy_(initialization["weight"])
    elif (
        radius
        and buffer_metadata(hidden.kernel) != initialization["kernel_buffers_fp32"]
    ):
        raise ValueError("radius initialization live FP32 buffers differ")
    model = nn.Sequential(
        nn.Linear(784, n), nn.ReLU(), hidden, nn.ReLU(), nn.Linear(n, 10)
    )
    model[0].load_state_dict(copy.deepcopy(initialization["input"]))
    model[4].load_state_dict(copy.deepcopy(initialization["output"]))
    return model


def radius_statistics(layer):
    """Physical radii and exact whole-tile hierarchy membership, never sigma."""
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator as radius_hierarchy,
    )
    from torchcst._backends.torch.parameterizations.radius_cell_mass import (
        decoded_values,
    )

    point = layer.atoms.p.detach().cpu()
    state = copy.deepcopy(layer.kernel).cpu()
    amplitude, ri, ro = decoded_values(state, point)
    plan = radius_hierarchy(layer.execution_declaration())
    quantiles = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64)
    result = {
        "parameterization": "polar_radius",
        "units": "physical support radius",
        "valid": bool(torch.isfinite(point).all()),
        "amplitude_quantiles": torch.quantile(amplitude, quantiles).tolist(),
        "alpha_quantiles": torch.quantile(
            ((point.double()[:, :2].square().sum(1) - 1) / 3).clamp(0, 1), quantiles
        ).tolist(),
    }
    for side, radii, column in (("input", ri, 2), ("output", ro, 3)):
        axis = getattr(plan, side)
        centers = (point[:, column].double() - axis.origin).remainder(axis.period)
        remaining = torch.ones(len(point), dtype=torch.bool)
        owners = torch.empty(len(point), dtype=torch.int64)
        banks = []
        for bank in axis.banks:
            tile = bank.tile_sites * axis.spacing
            selected = (
                remaining & (radii <= tile) & ((2 * radii / tile).ceil() + 1 == bank.k)
            )
            first = (
                ((centers + axis.spacing / 2) / tile - (bank.k - 1) / 2).floor().long()
            )
            owners[selected] = bank.node_offset + first[selected].remainder(
                bank.tile_count
            )
            banks.append(
                {
                    "level": bank.level,
                    "window_tiles": bank.k,
                    "tile_sites": bank.tile_sites,
                    "physical_tile_width": tile,
                    "atoms": int(selected.sum()),
                }
            )
            remaining &= ~selected
        if remaining.any():
            raise ValueError("radius diagnostics could not classify every atom")
        counts = torch.bincount(owners, minlength=axis.node_count)
        result["valid"] &= bool(
            torch.isfinite(radii).all()
            and (radii > 0).all()
            and (radii <= axis.period / 2).all()
        )
        result[side] = {
            "radius_quantiles": torch.quantile(radii, quantiles).tolist(),
            "radius_over_spacing_quantiles": torch.quantile(
                radii / axis.spacing, quantiles
            ).tolist(),
            "spacing": axis.spacing,
            "chart_period": axis.period,
            "banks": banks,
            "owner_counts": counts.tolist(),
            "max_owner_fraction": float(counts.max() / len(point)),
        }
    return result


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
    initialization,
    kind,
    data_directory,
    output,
    *,
    epochs=3,
    batch_size=32,
    lr=None,
    allow_radius_batch32=False,
    initialization_sha256=None,
    data_normalization="original",
    radius_execution="cached",
    radius_direct_levels=2,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if data_normalization not in ("original", "inplace"):
        raise ValueError("data_normalization must be original or inplace")
    radius = initialization.get("parameterization") == "polar_radius"
    radius_plan(radius_execution, direct_levels=radius_direct_levels)
    if not radius and radius_execution != "cached":
        raise ValueError("radius_execution overrides require a radius initialization")
    if not radius and radius_direct_levels != 2:
        raise ValueError(
            "radius_direct_levels overrides require a radius initialization"
        )
    if radius:
        validate_radius_initialization(initialization)
        if batch_size != 4 and not (batch_size == 32 and allow_radius_batch32):
            raise ValueError(
                "radius requires true batch 4; batch 32 needs explicit allow_radius_batch32"
            )
        if lr is not None and lr != initialization["optimizer_contract"]["lr"]:
            raise ValueError("radius lr must match the frozen initialization contract")
    lr = (
        (initialization["optimizer_contract"]["lr"] if radius else 1e-3)
        if lr is None
        else lr
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, data_info = load_mnist(data_directory)
    device = torch.device("cuda")
    model = make_model(
        initialization,
        kind,
        radius_execution=radius_execution,
        radius_direct_levels=radius_direct_levels,
    ).to(device)
    execution_linear_plan = (
        make_algorithm_registry().dump_plan(
            radius_plan(radius_execution, direct_levels=radius_direct_levels)
        )
        if radius and kind == "cst"
        else None
    )
    tx = normalize_images(
        data["train_x"],
        device,
        data_info["mean"],
        data_info["std"],
        data_normalization=data_normalization,
    )
    vx = normalize_images(
        data["t10k_x"],
        device,
        data_info["mean"],
        data_info["std"],
        data_normalization=data_normalization,
    )
    ty, vy = data["train_y"].to(device), data["t10k_y"].to(device)
    optimizer = make_optimizer(model, initialization, kind, lr=lr)
    if radius:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    initial_test = evaluate(model, vx, vy, batch_size)
    # Load/compile the implementation and exercise backward before timing.
    # Warmup does not update any Parameter or optimizer state.
    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(tx[:batch_size]), ty[:batch_size]).backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    warmup_peak = (
        (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        if radius
        else None
    )
    rows = []
    if kind == "cst":
        (
            output / ("radius-initial.json" if radius else "sigma-initial.json")
        ).write_text(
            json.dumps(
                radius_statistics(model[2])
                if radius
                else sigma_statistics(
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
        if radius:
            row.update(
                training_only_peak_allocated_bytes=row["peak_allocated_bytes"],
                training_only_peak_reserved_bytes=row["peak_reserved_bytes"],
                peak_allocated_bytes=max(row["peak_allocated_bytes"], warmup_peak[0]),
                peak_reserved_bytes=max(row["peak_reserved_bytes"], warmup_peak[1]),
                optimizer_cold_start_included=epoch == 0,
                timing_scope="complete epoch; first optimizer invocation/JIT/native preparation included"
                if epoch == 0
                else "complete later epoch after earlier training; no independent cold start",
            )
        row["test"] = evaluate(model, vx, vy, batch_size)
        if kind == "cst":
            if radius:
                row["radius"] = radius_statistics(model[2])
            else:
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
            "radius_execution": radius_execution if radius else None,
            "radius_direct_levels": radius_direct_levels if radius else None,
            "execution_linear_plan": execution_linear_plan,
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
        "initialization_sha256": initialization_sha256,
        "initial_hashes": initialization.get("initial_hashes"),
        "optimizer_contract": initialization.get("optimizer_contract"),
        "kernel_buffers_fp32": initialization.get("kernel_buffers_fp32"),
        "optimizer_implementation": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "optimizer_effective_groups": [
            {
                key: group.get(key, 0.0)
                for key in ("role", "lr", "weight_decay", "amplitude_weight_decay")
            }
            for group in optimizer.param_groups
        ],
        "linear_plan": initialization.get("linear_plan"),
        "radius_execution": radius_execution if radius else None,
        "radius_direct_levels": radius_direct_levels if radius else None,
        "execution_linear_plan": execution_linear_plan,
        "linear_plan_scope": "linear_plan is the frozen initialization contract; execution_linear_plan is the actual CST execution override (None for dense)"
        if radius
        else None,
        "update_plan": initialization.get("update_plan") if kind == "cst" else None,
        "batch_policy": "true batch; no gradient accumulation",
        "batch32_separately_opted_in": radius and batch_size == 32,
        "warmup_policy": "one forward/backward followed by zero_grad; no optimizer step or hidden parameter/state update; first training epoch includes optimizer cold start",
        "memory_scope": "max(initial evaluation/backward-only warmup, complete training epoch); resident data/model included; eager/no Graph"
        if radius
        else "complete training epoch; resident data/model included",
        "data": data_info,
        "data_normalization": data_normalization,
        "data_normalization_policy": "original transfer/float32 conversion and out-of-place div/sub/div"
        if data_normalization == "original"
        else "opt-in independent float32 transfer (copy=True) and in-place div/sub/div; same pixel arithmetic",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "scope": "Fresh process; full MNIST train/test splits, one seed, same initial effective hidden W and other layers, same data order; epoch training wall includes gathering/loss/all backward/actual optimizer, excludes evaluation/diagnostics/checkpoints; allocator peaks include resident data and model; no Graph, TF32 off; learning experiment, not DB/Linear benchmark output",
    }
    (output / "training.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--create-radius-initialization", type=Path)
    parser.add_argument(
        "--seed", type=int, default=41, help="seed for a new radius initialization"
    )
    parser.add_argument("--kind", choices=["cst", "dense"])
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument(
        "--data-normalization",
        choices=["original", "inplace"],
        default="original",
        help="resident MNIST allocation policy; original protocol is the default",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--radius-execution",
        choices=["cached", "batched"],
        default="cached",
        help="opt-in radius execution Plan; initialization contract is unchanged",
    )
    parser.add_argument(
        "--radius-direct-levels",
        type=int,
        choices=range(9),
        default=2,
        help="opt-in number of direct radius levels for both axes; frozen initialization stays at 2",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-radius-batch32",
        action="store_true",
        help="explicit separate B32 qualification; no gradient accumulation",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.create_radius_initialization is not None:
        if any(
            value is not None
            for value in (
                args.initialization,
                args.kind,
                args.data_directory,
                args.output,
            )
        ):
            parser.error("initialization creation is separate from training")
        initialization = make_radius_initialization(seed=args.seed)
        args.create_radius_initialization.parent.mkdir(parents=True, exist_ok=True)
        with args.create_radius_initialization.open("xb") as stream:
            torch.save(initialization, stream)
        print(
            json.dumps(
                {
                    "initialization": str(args.create_radius_initialization),
                    "cohort": RADIUS_COHORT,
                    "initial_hashes": initialization["initial_hashes"],
                },
                indent=2,
            )
        )
        return
    if any(
        value is None
        for value in (args.initialization, args.kind, args.data_directory, args.output)
    ):
        parser.error(
            "training requires --initialization, --kind, --data-directory and --output"
        )
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
        allow_radius_batch32=args.allow_radius_batch32,
        initialization_sha256=hashlib.sha256(
            args.initialization.read_bytes()
        ).hexdigest(),
        data_normalization=args.data_normalization,
        radius_execution=args.radius_execution,
        radius_direct_levels=args.radius_direct_levels,
    )


if __name__ == "__main__":
    main()
