"""Whole-step replay contracts, separate from throughput measurements."""

import gc
import os
from pathlib import Path
from types import MethodType

import pytest
import torch

from benchmarks.training.cell_mass_mnist import make_model, make_optimizer
from benchmarks.training.full_step_graph import (
    PreparedStep,
    exact,
    ordinary_step,
    restore,
    snapshot,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import plans

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("CST_GRAPH_INPUTS"),
    reason="requires CUDA and the hash-verified frozen Graph experiment inputs",
)


@pytest.fixture
def cohort():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(9041)
        init = torch.load(
            Path(os.environ["CST_GRAPH_INPUTS"]) / "initialization.pt",
            map_location="cpu",
            weights_only=True,
        )
        tx, ty = (
            torch.randn(64, 784, device="cuda"),
            torch.arange(64, device="cuda") % 10,
        )
        model = make_model(
            init,
            "cst",
            execution_plan=(
                plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
                if os.environ.get("CST_GRAPH_FP32") == "1"
                else plans.GLOBAL_OWNER_WINDOW_RECOMPUTE
            ),
        ).cuda()
        opt = make_optimizer(model, init, "cst")
        indices = torch.arange(4, device="cuda")
        totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
        for _ in range(8):
            ordinary_step(model, opt, tx, ty, indices, totals)
        step_type = PreparedStep
        if os.environ.get("CST_GRAPH_GUARD") == "flat":
            from benchmarks.training.flat_step_graph import FlatPreparedStep

            step_type = FlatPreparedStep
        program = step_type(model, opt, tx, ty, "cst", observe=True)
        assert program.capture() is True
        yield model, opt, tx, ty, indices, program, init
        del program, opt, model
        gc.collect()
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "change",
    [
        "module_hook",
        "tensor_hook",
        "optimizer_pre_hook",
        "optimizer_post_hook",
        "saved_hooks",
        "lr",
        "setting",
        "storage",
        "retained_reader",
        "finite_checks",
        "grad_storage",
        "group_params",
        "optimizer_callback",
        "zero_grad_callback",
    ],
)
def test_replay_invalidates_before_changed_contract(cohort, change):
    model, opt, tx, _ty, indices, program, _ = cohort
    called = []
    handle = None
    context = None
    retained = None
    if change == "module_hook":
        handle = model[0].register_forward_hook(lambda *_: called.append("module"))
    elif change == "tensor_hook":
        handle = model[2].atoms.p.register_hook(
            lambda grad: called.append("grad") or grad
        )
    elif change == "optimizer_pre_hook":
        handle = opt.register_step_pre_hook(lambda *_: called.append("pre"))
    elif change == "optimizer_post_hook":
        handle = opt.register_step_post_hook(lambda *_: called.append("post"))
    elif change == "saved_hooks":

        def pack(value):
            called.append("pack")
            return value

        context = torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value)
        context.__enter__()
    elif change == "group_params":
        opt.param_groups[0]["params"].reverse()
    elif change == "optimizer_callback":
        previous = opt.step

        def replacement():
            called.append("step")
            return previous()

        opt.step = replacement
    elif change == "zero_grad_callback":
        previous = opt.zero_grad

        def replacement(*args, **kwargs):
            called.append("zero_grad")
            return previous(*args, **kwargs)

        opt.zero_grad = replacement
    elif change == "lr":
        opt.param_groups[0]["lr"] *= 0.5
    elif change == "setting":
        with torch.no_grad():
            model[2].kernel.w_c.mul_(1.01)
    elif change == "storage":
        with torch.no_grad():
            model[0].weight.set_(model[0].weight.detach().clone())
    elif change == "retained_reader":
        retained = model[2](tx.new_zeros((4, 1024), requires_grad=True))
    elif change == "finite_checks":
        opt.check_finite = True
    elif change == "grad_storage":
        model[0].weight.grad = None
    try:
        assert not program.eligible(), change
        clocks = [state["step"].clone() for state in opt.state.values()]
        program.step(indices)
        assert program.invalidated
        assert all(
            torch.equal(state["step"], clock + 1)
            for state, clock in zip(opt.state.values(), clocks)
        )
        if "hook" in change or "callback" in change:
            assert called, change
        if retained is not None:
            with pytest.raises(RuntimeError, match="modified.*inplace"):
                retained.sum().backward()
    finally:
        if handle is not None:
            handle.remove()
        if context is not None:
            context.__exit__(None, None, None)


def test_current_boundary_wide_atoms_gradients_and_version(cohort):
    model, opt, tx, ty, indices, program, init = cohort
    with torch.no_grad():
        model[2].atoms.p[:7].copy_(
            torch.tensor(
                [
                    [1, 0, -1.2, 1022.8],
                    [0.3, 0.9539392014169457, -1025.2, -1.2001],
                    [0.15, 0.9886859966642595, 1022.79, 1022.81],
                    [0.15, 1.33, 4096, -4096],
                    [0.15, 1.6, 7.3, 8.2],
                    [0, 2, 32, 64],
                    [0.3, 1.977371993328519, 128, 256],
                ],
                device="cuda",
            )
        )
    start = snapshot(model, opt)
    reference = make_model(
        init,
        "cst",
        execution_plan=(
            plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
            if os.environ.get("CST_GRAPH_FP32") == "1"
            else plans.GLOBAL_OWNER_WINDOW_RECOMPUTE
        ),
    ).cuda()
    ropt = make_optimizer(reference, init, "cst")
    totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
    for _ in range(8):
        ordinary_step(reference, ropt, tx, ty, indices, totals)
    restore(reference, ropt, start)
    observed = []

    def retain_middle(_module, args):
        args[0].retain_grad()
        observed.append(args[0])

    hook = reference[2].register_forward_pre_hook(retain_middle)
    try:
        for _ in range(3):
            y, loss = ordinary_step(reference, ropt, tx, ty, indices, totals)
            versions = [p._version for p in model.parameters()]
            gy, gloss = program.step(indices)
            assert not program.invalidated
            assert all(p._version > v for p, v in zip(model.parameters(), versions))
            a = {
                **snapshot(reference, ropt),
                "logits": y,
                "loss": loss,
                "dX": observed[-1].grad,
                **{"grad/" + n: p.grad for n, p in reference.named_parameters()},
            }
            b = {
                **snapshot(model, opt),
                "logits": gy,
                "loss": gloss,
                "dX": program.middle.grad,
                **{"grad/" + n: p.grad for n, p in model.named_parameters()},
            }
            assert exact(a, b)["all_byte_equal"]
    finally:
        hook.remove()


@pytest.mark.parametrize("when", ["cold", "live"])
def test_unknown_forward_fallback_matches_current_ordinary_update(cohort, when):
    """A rejected cohort must execute current callbacks and update exactly once."""
    model, opt, tx, ty, indices, captured, init = cohort
    start = snapshot(model, opt)
    reference = make_model(
        init,
        "cst",
        execution_plan=(
            plans.GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C
            if os.environ.get("CST_GRAPH_FP32") == "1"
            else plans.GLOBAL_OWNER_WINDOW_RECOMPUTE
        ),
    ).cuda()
    ropt = make_optimizer(reference, init, "cst")
    reference_totals = (
        tx.new_zeros(()),
        torch.zeros((), dtype=torch.int64, device="cuda"),
    )
    # Allocate the reference's ordinary optimizer state before restoring the
    # exact captured cohort state. No preparation callback is used as an anchor.
    for _ in range(8):
        ordinary_step(reference, ropt, tx, ty, indices, reference_totals)
    restore(reference, ropt, start)
    called, reference_called = [], []

    def install(module, calls):
        def changed_forward(self, value):
            calls.append(True)
            return torch.nn.functional.linear(value, self.weight, self.bias) * 0.875

        module.forward = MethodType(changed_forward, module)

    install(model[0], called)
    install(reference[0], reference_called)
    if when == "cold":
        program = type(captured)(model, opt, tx, ty, "cst")
        assert program.invalidated and not called
        assert exact(start, snapshot(model, opt))["all_byte_equal"]
    else:
        program = captured
    assert not program.eligible() and not called
    with torch.no_grad():
        for value, wanted in zip(reference_totals, program.totals):
            value.copy_(wanted)

    class NeverReplay:
        def replay(self):
            raise AssertionError("rejected cohort attempted Graph replay")

    # Keep the real Graph alive while the sentinel proves the rejected step
    # does not enqueue it. Eligibility has already failed for the live callback.
    original_graph = program.graph
    program.graph = NeverReplay()
    try:
        ordinary_step(reference, ropt, tx, ty, indices, reference_totals)
        program.step(indices)
        assert program.invalidated and called == reference_called == [True]

        def result(current_model, optimizer, totals):
            return {
                **snapshot(current_model, optimizer),
                **{
                    "grad/" + name: point.grad.detach().cpu().clone()
                    for name, point in current_model.named_parameters()
                },
                **{
                    "total/" + str(i): value.detach().cpu().clone()
                    for i, value in enumerate(totals)
                },
            }

        assert exact(
            result(reference, ropt, reference_totals),
            result(model, opt, program.totals),
        )["all_byte_equal"]
        assert all(int(state["step"].item()) == 9 for state in opt.state.values())
    finally:
        program.graph = original_graph


def test_public_radius_entrance_still_rejects_capture(cohort):
    model, _, tx, _, _, program, _ = cohort
    # Native eager validation runs before any work is enqueued. End capture
    # cleanly after catching its deliberate rejection.
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph, stream=torch.cuda.current_stream()),
        pytest.raises(RuntimeError, match="not CUDA Graphs"),
    ):
        program.extension.checked_autograd_linear_owner_window(
            tx.new_zeros((4, 1024)), model[2].atoms.p, *program.configs, 2
        )
