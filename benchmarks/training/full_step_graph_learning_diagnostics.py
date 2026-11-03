"""Untimed VJP and learned-Adam sensitivity at FP32's own final checkpoint."""

import argparse
import copy
import json
from pathlib import Path

import torch

from benchmarks.cuda.linear.radius_cell_mass import stratified_subset
from benchmarks.training.cell_mass_mnist import (
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from benchmarks.training.full_step_graph import tensors
from benchmarks.training.owner_window_fp32 import adam_direction, errors, vjp_reference
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans


def load_cohort(init, checkpoint):
    model = make_model(
        init, "cst", execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32
    ).cuda()
    model.load_state_dict(checkpoint["model"])
    opt = make_optimizer(model, init, "cst")
    opt.load_state_dict(checkpoint["optimizer"])
    return model, opt


def next_update(init, checkpoint, gradient):
    # Build and load through the public optimizer lifecycle; deepcopy of a
    # trained site's opaque CPU optimizer state is not a valid cold control.
    model, opt = load_cohort(init, checkpoint)
    p = model[2].atoms.p
    assert opt.state[p]["step"].device == p.device
    p.grad = gradient.to(p.device).clone()
    opt.step()
    point = p.detach().cpu().clone()
    state = {
        key: value.detach().cpu().clone()
        for key, value in opt.state[p].items()
        if isinstance(value, torch.Tensor)
    }
    assert all(torch.isfinite(v).all() for v in [point, *state.values()])
    return point, state


def byte_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    init = torch.load(
        args.inputs / "initialization.pt", map_location="cpu", weights_only=True
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, opt = load_cohort(init, checkpoint)
    cpu_net = make_model(init, "cst", execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32)
    cpu_net.load_state_dict(checkpoint["model"])
    cpu_layer = cpu_net[2]
    cpu_layer.selector = None
    cpu_layer.backend = "factored"
    data, info = load_mnist(args.inputs / "data")
    train = normalize_images(
        data["train_x"],
        "cuda",
        info["mean"],
        info["std"],
        data_normalization="original",
    )
    test = normalize_images(
        data["t10k_x"], "cuda", info["mean"], info["std"], data_normalization="original"
    )
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(init["seed"] + 100)
    )[:4]
    samples = [
        (
            "first_training_minibatch",
            train[order.cuda()],
            data["train_y"][order].cuda(),
            order.tolist(),
        ),
        ("first_test_minibatch", test[:4], data["t10k_y"][:4].cuda(), [0, 1, 2, 3]),
    ]
    rows = []
    p = model[2].atoms.p
    learned = opt.state[p]
    m, v = (
        learned["exp_avg"].detach().cpu().clone(),
        learned["exp_avg_sq"].detach().cpu().clone(),
    )
    step = int(learned["step"].detach().cpu())
    assert step == checkpoint["updates"]
    model.train()
    for name, xx, target, indices in samples:
        # Actual learned front/tail layers supply the CST input and upstream dY.
        x = torch.nn.functional.relu(model[0](xx)).detach().requires_grad_()
        y = model[2](x)
        logits = model[4](torch.nn.functional.relu(y))
        loss = torch.nn.functional.cross_entropy(logits, target)
        dy = torch.autograd.grad(loss, y, retain_graph=True)[0].detach()
        dx, dp = torch.autograd.grad(y, (x, p), dy)
        actual = (y.detach().cpu(), dx.detach().cpu(), dp.detach().cpu())
        reference = vjp_reference(
            cpu_layer, x.detach().cpu(), dy.cpu(), independent=False
        )
        row = {
            "sample": name,
            "indices": indices,
            "reference": "all-atom independent public CPU FP64 Fourier",
            "vjp": {
                key: errors(a, r)
                for key, a, r in zip(("Y", "dX", "dP"), actual, reference)
            },
            "actual_loss": float(loss.detach().cpu()),
            "learned_adam_step": step,
        }
        assert all(value["pass"] for value in row["vjp"].values()), row["vjp"]
        subset, subset_indices = stratified_subset(cpu_layer, count=8)
        sub_reference = vjp_reference(
            subset, x.detach().cpu(), dy.cpu(), independent=True
        )
        sub_gpu = copy.deepcopy(subset).cuda()
        sub_gpu.selector = FixedSelector(
            plans.GLOBAL_OWNER_WINDOW_FP32, registry=make_algorithm_registry()
        )
        sx = x.detach().clone().requires_grad_()
        sy = sub_gpu(sx)
        sdx, sdp = torch.autograd.grad(sy, (sx, sub_gpu.atoms.p), dy)
        row["quadrature"] = {
            key: errors(a, r)
            for key, a, r in zip(("Y", "dX", "dP"), (sy, sdx, sdp), sub_reference)
        }
        row["quadrature_subset_indices"] = subset_indices
        assert all(value["pass"] for value in row["quadrature"].values())
        g, r = actual[2].float(), reference[2].float()
        u, ur = adam_direction(g, m, v, step), adam_direction(r, m, v, step)
        delta = (u - ur).abs()
        row["learned_moment_direction"] = {
            "max_abs": float(delta.max()),
            "rms_by_column": delta.double().square().mean(0).sqrt().tolist(),
            "scope": "same actual m/v/step, reference dP rounded once to FP32; diagnostic, no retroactive cold-gate pass",
        }
        updates = [next_update(init, checkpoint, grad) for grad in (g, r, r)]
        a, b, c = [entry[0] for entry in updates]
        distance = (a.double() - b.double()).abs()
        distance[:, 2:] = (
            (a[:, 2:].double() - b[:, 2:].double() + 512).remainder(1024) - 512
        ).abs()
        row["next_native_CSTAdamW_update"] = {
            "post_P_max_abs_by_column": distance.max(0).values.tolist(),
            "state_errors": {
                key: errors(updates[0][1][key], updates[1][1][key])
                for key in updates[0][1]
            },
            "same_gradient_P_bytes": byte_equal(b, c),
            "same_gradient_state_bytes": all(
                byte_equal(updates[1][1][key], updates[2][1][key])
                for key in updates[1][1]
            ),
        }
        assert row["next_native_CSTAdamW_update"]["same_gradient_P_bytes"]
        assert row["next_native_CSTAdamW_update"]["same_gradient_state_bytes"]
        torch.save(
            {
                "input": x.detach().cpu(),
                "dy": dy.cpu(),
                "P": p.detach().cpu(),
                "candidate": actual,
                "reference": reference,
                "learned_m": m,
                "learned_v": v,
                "learned_step": step,
                "updates": updates,
            },
            args.output / (name + ".pt"),
        )
        rows.append(row)
        (args.output / "diagnostics.json").write_text(
            json.dumps(
                {"samples": rows, "all_vjp_and_quadrature_gates_pass": True}, indent=2
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "sample": name,
                    "all_vjp_pass": True,
                    "step": step,
                    "max_du": float(delta.max()),
                }
            ),
            flush=True,
        )
    assert all(torch.isfinite(t).all() for _, t in tensors(checkpoint))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
