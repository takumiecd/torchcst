"""One untimed all-atom VJP/update cell for diagnostic A; no speed verdict."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.training.cell_mass_mnist import (
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from benchmarks.training.full_step_graph import ordinary_step, restore, snapshot
from benchmarks.training.full_step_graph_comparison import hashes
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
    _extension,
)


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
    init = torch.load(
        args.inputs / "initialization.pt", map_location="cpu", weights_only=True
    )
    data, info = load_mnist(args.inputs / "data")
    tx = normalize_images(
        data["train_x"],
        "cuda",
        info["mean"],
        info["std"],
        data_normalization="original",
    )
    ty = data["train_y"].cuda()
    order = torch.randperm(
        len(tx), generator=torch.Generator().manual_seed(init["seed"] + 100)
    ).cuda()
    base = plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
    model = make_model(init, "cst", execution_plan=base).cuda()
    checkpoint = (
        torch.load(args.inputs / "fp32_final.pt", map_location="cpu", weights_only=True)
        if args.state == "final"
        else None
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    opt = make_optimizer(model, init, "cst")
    if checkpoint is not None:
        opt.load_state_dict(checkpoint["optimizer"])
        wanted = snapshot(model, opt)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
    for i in range(8):
        ordinary_step(model, opt, tx, ty, order[4 * i : 4 * i + 4], totals)
    if checkpoint is not None:
        restore(model, opt, wanted)
    start = snapshot(model, opt)
    assert len(start) == 42
    if args.variant == "shared":
        model[2].selector = FixedSelector(
            plans.GLOBAL_OWNER_WINDOW_FP32_SHARED_RADIAL,
            registry=make_algorithm_registry(),
        )
    opt.zero_grad(set_to_none=True)
    p = model[2].atoms.p
    payloads = {}

    def pack(t):
        if t.dtype == torch.int64 and tuple(t.shape) == (52428,):
            payloads["keys"] = t.detach().cpu().clone()
        if t.dtype == torch.float64 and tuple(t.shape) == (2, 52428):
            payloads["geometry"] = t.detach().cpu().clone()
        return t

    if args.reference is not None:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        assert hashes(start) == hashes(reference["start"])
        x = reference["x"].cuda().requires_grad_()
        dy = reference["dy"].cuda()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = model[2](x)
    else:
        x = torch.nn.functional.relu(model[0](tx[order[:4]])).detach().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = model[2](x)
        loss = torch.nn.functional.cross_entropy(
            model[4](torch.nn.functional.relu(y)), ty[order[:4]]
        )
        dy = torch.autograd.grad(loss, y, retain_graph=True)[0].detach()
    dx, dp = torch.autograd.grad(y, (x, p), dy)
    assert set(payloads) == {"keys", "geometry"}
    values = {
        "x": x.detach().cpu(),
        "dy": dy.cpu(),
        "y": y.detach().cpu(),
        "dx": dx.cpu(),
        "dp": dp.cpu(),
        **payloads,
    }
    assert (
        dp.shape == (52428, 4) and torch.isfinite(dp).all() and bool(dp.abs().max() > 0)
    )
    clock = int(opt.state[p]["step"].item())
    assert clock == (8 if args.state == "initial" else 45000)
    p.grad = dp.detach().clone()
    opt.step()
    end = snapshot(model, opt)
    assert int(opt.state[p]["step"].item()) == clock + 1
    assert all(
        torch.isfinite(v).all()
        for v in [*values.values(), *start.values(), *end.values()]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"start": start, "end": end, **values}, args.output.with_suffix(".pt"))
    loaded = Path(_extension().__file__)
    report = {
        "state": args.state,
        "variant": args.variant,
        "scope": "untimed actual model X/dY, all atoms, one site-only native optimizer update",
        "adoption_timing": False,
        "clock_before": clock,
        "clock_after": clock + 1,
        "start": hashes(start),
        "end": hashes(end),
        "vjp": hashes(values),
        "loaded_extension": str(loaded),
        "extension_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest(),
        "build_ninja": (loaded.parent / "build.ninja").read_text(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "unexpected_fallbacks": 0,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", choices=("initial", "final"), required=True)
    parser.add_argument("--variant", choices=("base", "shared"), required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
