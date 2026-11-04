"""One Stage-1 qualification/numerical cell; no performance/adoption claim."""

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import torch

from benchmarks.training.cell_mass_mnist import (
    load_mnist,
    make_model,
    make_optimizer,
    normalize_images,
)
from benchmarks.training.flat_step_graph import FlatPreparedStep
from benchmarks.training.full_step_graph import (
    PreparedStep,
    exact,
    ordinary_step,
    restore,
    snapshot,
)
from benchmarks.training.full_step_graph_comparison import hashes
from benchmarks.training.full_step_graph_learning import renew_certificates
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor, plans

ENTRY = "research_owner_window_fp32_geometry_sort32_flat_c"
SCHEDULE = "global_owner_window_fp32_geometry_sort32_flat_c"
READ_LEASE = "torch::autograd::CppNode<(anonymous namespace)::ReadLease>"


def native_read_lease_path(output, parameter):
    """Inspect only the fixed output/native/P lease route, before backward.

    ReadLease has two Tensor inputs. Its non-differentiable lifetime marker
    still contributes a (None, 0) edge; it is not a unary autograd node.
    The caller separately matches the native name to the direct entry result.
    """

    def borrowed_input(node):
        assert node is not None and node.name() == READ_LEASE
        edges = node.next_functions
        assert len(edges) == 2 and edges[0][0] is not None
        assert edges[0][1] == 0 and edges[1] == (None, 0)
        return edges[0][0]

    outer = output.grad_fn
    native = borrowed_input(outer)
    edges = native.next_functions
    assert len(edges) == 2 and all(edge[1] == 0 for edge in edges)
    accumulator = borrowed_input(edges[1][0])
    assert accumulator.name() == "torch::autograd::AccumulateGrad"
    assert accumulator.variable is parameter
    return [outer.name(), native.name()], [READ_LEASE, accumulator.name()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(model, optimizer, totals):
    """CPU-only clones; never retain the model's graph/Tensor examples."""
    return {
        "state": snapshot(model, optimizer),
        "gradients": {
            name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
        },
        "totals": {str(i): v.detach().cpu().clone() for i, v in enumerate(totals)},
    }


def clocks(optimizer, wanted):
    values = [int(state["step"].item()) for state in optimizer.state.values()]
    assert values and all(value == wanted for value in values), values


def sample_vjp(model, optimizer, inputs, targets):
    """Use one real full-model backward, then the identical native research VJP."""
    before = snapshot(model, optimizer)
    site, saved, handles = model[2], {}, []

    def keep(name, value):
        saved[name] = value.detach().cpu().clone()

    def pre_hook(module, args):
        keep("X", args[0])
        handles.append(args[0].register_hook(lambda g: keep("dX", g)))

    def forward_hook(module, args, output):
        keep("Y", output)
        names, parameter_names = native_read_lease_path(output, site.atoms.p)
        assert "OwnerWindowFloat" in names[-1], names
        saved["ordinary_grad_fn_path"] = names
        saved["ordinary_parameter_grad_fn_path"] = parameter_names
        handles.append(output.register_hook(lambda g: keep("dY", g)))

    keep("P", site.atoms.p)
    handles.extend(
        [
            site.register_forward_pre_hook(pre_hook),
            site.register_forward_hook(forward_hook),
            site.atoms.p.register_hook(lambda g: keep("dP", g)),
        ]
    )
    try:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    handles.clear()
    del logits, loss
    assert exact(before, snapshot(model, optimizer))["all_byte_equal"]
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        hierarchy_from_operator,
    )

    layout = hierarchy_from_operator(site.declaration())
    configs = layout.input.native_config(), layout.output.native_config()
    x = saved["X"].cuda().requires_grad_()
    p = saved["P"].cuda().requires_grad_()
    dy = saved["dY"].cuda()
    output = getattr(executor._extension(), ENTRY)(x, p, *configs, True)
    assert output.grad_fn.name() == saved["ordinary_grad_fn_path"][-1]
    dx, dp = torch.autograd.grad(output, (x, p), dy)
    direct = {
        name: value.detach().cpu().clone()
        for name, value in (("Y", output), ("dX", dx), ("dP", dp))
    }
    comparison = exact({key: saved[key] for key in direct}, direct)
    assert comparison["all_byte_equal"], comparison
    assert saved["P"].shape == (52428, 4)
    assert exact(before, snapshot(model, optimizer))["all_byte_equal"]
    del x, p, dy, output, dx, dp, direct
    gc.collect()
    return saved, comparison


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(9041)
    init_path = args.inputs / "initialization.pt"
    init = torch.load(init_path, map_location="cpu", weights_only=True)
    data, info = load_mnist(args.inputs / "data")
    tx = normalize_images(
        data["train_x"],
        "cuda",
        info["mean"],
        info["std"],
        data_normalization="original",
    )
    ty = data["train_y"].cuda()
    order_cpu = torch.randperm(
        len(tx), generator=torch.Generator().manual_seed(init["seed"] + 100)
    )
    order = order_cpu.cuda()
    model = make_model(
        init,
        args.kind,
        radius_execution="cached" if args.kind == "cst" else "global-response4",
        execution_plan=plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
        if args.kind == "cst"
        else None,
    ).cuda()
    checkpoint_path, checkpoint = None, None
    if args.state == "final":
        checkpoint_path = args.inputs / (
            "fp32_final.pt" if args.kind == "cst" else "dense_final.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    optimizer = make_optimizer(model, init, args.kind)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        wanted = snapshot(model, optimizer)
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
    for i in range(8):
        ordinary_step(model, optimizer, tx, ty, order[4 * i : 4 * i + 4], totals)
    if checkpoint is not None:
        restore(model, optimizer, wanted)
    clock_start = 8 if args.state == "initial" else 45000
    clocks(optimizer, clock_start)
    wanted = snapshot(model, optimizer)
    vjp, vjp_check = None, None
    if args.kind == "cst":
        vjp, vjp_check = sample_vjp(model, optimizer, tx[order[:4]], ty[order[:4]])
    renew_certificates(model, optimizer, tx[order[:4]], args.kind)
    assert exact(wanted, snapshot(model, optimizer))["all_byte_equal"]
    step_type = PreparedStep if args.guard == "qualified" else FlatPreparedStep
    program = step_type(model, optimizer, tx, ty, args.kind)
    assert program.eligible() and not program.invalidated, program.qualification_error
    extension = None
    if args.kind == "cst":
        assert program.linear_program.cached.algorithm.id == "cuda_radius_cell_mass"
        recipe = program.linear_program.cached.state.recipe
        assert recipe.native_entry and recipe.native_autograd
        assert recipe.schedule == SCHEDULE and program.fp32_entry == ENTRY
        assert program.extension is executor._extension()
        path = Path(program.extension.__file__)
        extension = {"path": str(path), "sha256": file_hash(path)}
    program.indices.copy_(order[:4])
    assert program.capture() is True
    assert program.graph is not None and program.eligible() and not program.invalidated
    assert exact(wanted, snapshot(model, optimizer))["all_byte_equal"]
    clocks(optimizer, clock_start)
    checkpoints = {"start": evidence(model, optimizer, program.totals)}
    assert len(checkpoints["start"]["gradients"]) == 5
    assert args.kind != "cst" or len(checkpoints["start"]["state"]) == 42
    for i in range(16):
        assert program.eligible() and not program.invalidated
        program.step(order[4 * i : 4 * i + 4])
        assert not program.invalidated, "unexpected eager fallback"
        if i in (0, 15):
            torch.cuda.synchronize()
            clocks(optimizer, clock_start + i + 1)
            checkpoints[f"step{i + 1}"] = evidence(model, optimizer, program.totals)
    torch.cuda.synchronize()
    peak = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoints": checkpoints, "vjp": vjp}, args.output.with_suffix(".pt"))
    row = {
        "purpose": "Stage-1 numerical qualification only; no performance claim",
        "kind": args.kind,
        "state": args.state,
        "guard": args.guard,
        "qualification_backend": str(program._qualification.native_backend),
        "flat_signature_active": bool(getattr(program, "_use_flat", False)),
        "actual_updates": 16,
        "clock_start": clock_start,
        "clock_end": clock_start + 16,
        "unexpected_fallbacks": 0,
        "capture_restored_state": True,
        "entry": ENTRY if extension else None,
        "plan_schedule": SCHEDULE if extension else None,
        "extension": extension,
        "vjp_check": vjp_check,
        "checkpoints": {
            name: {k: hashes(v) for k, v in values.items()}
            for name, values in checkpoints.items()
        },
        "order_sha256": hashlib.sha256(order_cpu.numpy().tobytes()).hexdigest(),
        "initialization_sha256": file_hash(init_path),
        "checkpoint_sha256": file_hash(checkpoint_path) if checkpoint_path else None,
        "capture_inclusive_peak_bytes_diagnostic_only": peak,
        "memory_scope": "one reset before optimizer/bootstrap/VJP/renewal/capture; not memory-v2.1 admission",
        "data": info,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "stream": torch.cuda.current_stream().cuda_stream,
    }
    args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--kind", choices=("cst", "dense"), required=True)
    parser.add_argument("--state", choices=("initial", "final"), required=True)
    parser.add_argument("--guard", choices=("qualified", "flat"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run(args)


if __name__ == "__main__":
    main()
