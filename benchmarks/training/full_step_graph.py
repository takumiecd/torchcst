"""Bounded N1024/B4 whole-step Graph experiment, not a public execution API.

Use unchanged owner-window arithmetic and PyTorch fused AdamW. All capture
permissions are confined to a separate native entry on this research branch.
Python hooks/settings/storage changes invalidate replay before any launch.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from torch import nn

from benchmarks.training.cell_mass_mnist import make_model, make_optimizer
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass import executor, plans
from torchcst._backends.cuda.algorithms.polar_update.algorithm import _period_tensors
from torchcst._backends.cuda.algorithms.polar_update.executor import fused_update_
from torchcst._backends.torch import finite


def tensors(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, path + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, path + "/" + str(index))


def live(model, optimizer):
    return dict(
        tensors({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
    )


def snapshot(model, optimizer):
    return {k: v.detach().cpu().clone() for k, v in live(model, optimizer).items()}


def restore(model, optimizer, saved):
    current = live(model, optimizer)
    assert current.keys() == saved.keys(), "state layout changed"
    with torch.no_grad():
        for key, value in current.items():
            target = saved[key].to(value.device)
            if not torch.equal(
                value.contiguous().reshape(-1).view(torch.uint8),
                target.contiguous().reshape(-1).view(torch.uint8),
            ):
                value.copy_(target)


def exact(a, b):
    assert a.keys() == b.keys(), "tensor keys changed"
    wrong = []
    for key in a:
        x, y = a[key].detach().cpu(), b[key].detach().cpu()
        assert torch.isfinite(x).all() and torch.isfinite(y).all(), key
        if (
            x.shape != y.shape
            or x.dtype != y.dtype
            or not torch.equal(
                x.contiguous().reshape(-1).view(torch.uint8),
                y.contiguous().reshape(-1).view(torch.uint8),
            )
        ):
            wrong.append(key)
    return {"all_byte_equal": not wrong, "fields": len(a), "different": wrong}


def ordinary_step(model, optimizer, tx, ty, indices, totals):
    inputs, target = tx[indices], ty[indices]
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
    totals[0].add_(loss.detach() * 4)
    totals[1].add_((logits.detach().argmax(1) == target).sum())
    return logits, loss


def metadata(t):
    return (
        id(t),
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        t.device,
        t.requires_grad,
    )


@torch.no_grad()
def dense_proposal(optimizer):
    """Same PyTorch fused AdamW call, without eager-only health-check logic."""
    from torch.optim.adamw import adamw

    for group in optimizer.param_groups:
        params, grads, means, squares, maxima, clocks = [], [], [], [], [], []
        has_complex = optimizer._init_group(
            group, params, grads, means, squares, maxima, clocks
        )
        beta1, beta2 = group["betas"]
        adamw(
            params,
            grads,
            means,
            squares,
            maxima,
            clocks,
            foreach=group["foreach"],
            capturable=group["capturable"],
            differentiable=group["differentiable"],
            fused=group["fused"],
            amsgrad=group["amsgrad"],
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group["maximize"],
            has_complex=has_complex,
        )


class PreparedStep:
    """Own an exclusive complete forward/backward/update invocation.

    This deliberately supports one canonical benchmark model. Unsupported
    hooks/settings cause the *whole* step to use the original eager path.
    No external autograd output is exposed by the captured invocation.
    """

    def __init__(self, model, optimizer, tx, ty, kind, *, observe=False):
        self.model, self.optimizer, self.tx, self.ty, self.kind = (
            model,
            optimizer,
            tx,
            ty,
            kind,
        )
        self.indices = torch.zeros(4, dtype=torch.long, device=tx.device)
        self.totals = (
            tx.new_zeros(()),
            torch.zeros((), dtype=torch.int64, device=tx.device),
        )
        self.observe = observe
        self.graph = None
        self.owner_stream = torch.cuda.current_stream().cuda_stream
        if self.owner_stream == 0:
            raise ValueError(
                "construct and first-use the cohort on an explicit non-default stream"
            )
        self.invalidated = False
        if type(model) is not nn.Sequential or [type(m) for m in model] != [
            nn.Linear,
            nn.ReLU,
            type(model[2]),
            nn.ReLU,
            nn.Linear,
        ]:
            raise ValueError("only the canonical five-element model is supported")
        self.params = tuple(model.parameters())
        self.site = model[2] if kind == "cst" else None
        self.middle = tx.new_empty((4, 1024), requires_grad=True)
        self.periods = None
        if self.site is not None:
            self.linear_program = self.site.__dict__.get("_execution_program")
            binding = optimizer._update_bindings[self.site]
            self.polar_program = binding.__dict__.get("_optimizer_program")
            if self.linear_program is None or self.polar_program is None:
                raise RuntimeError(
                    "canonical native execution certificates not prepared"
                )
            plan = self.linear_program.cached.state.layout
            self.fp32_body = (
                self.linear_program.cached.state.recipe.schedule
                == "global_owner_window_fp32"
            )
            self.configs = plan.input.native_config(), plan.output.native_config()
            self.previous = torch.empty_like(self.site.atoms.p)
            self.periods = _period_tensors(self.polar_program.state, self.site.atoms.p)
            self.atom_group = next(
                g
                for g in optimizer.param_groups
                if any(p is self.site.atoms.p for p in g["params"])
            )
            self.extension = executor._extension()
        self.signature = self._signature()
        if not self.eligible():
            raise RuntimeError("initial whole-step guard refused the cohort")

    def _signature(self):
        # Tensor values of learned parameters may change. Constant buffers and
        # all storage identities/layouts must remain unchanged.
        return (
            (
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision(),
            ),
            tuple(
                (fn, getattr(fn, "__code__", None))
                for fn in (
                    nn.functional.relu,
                    nn.functional.cross_entropy,
                    type(self.optimizer).step,
                    self.optimizer.step,
                    self.optimizer.zero_grad,
                    self.optimizer._init_group,
                    self.optimizer._adamw_proposal,
                )
                if fn is not None
            )
            if self.site is not None
            else tuple(
                (fn, getattr(fn, "__code__", None))
                for fn in (
                    nn.functional.relu,
                    nn.functional.cross_entropy,
                    type(self.optimizer).step,
                    self.optimizer.step,
                    self.optimizer.zero_grad,
                    self.optimizer._init_group,
                )
            ),
            tuple(metadata(p.grad) if p.grad is not None else None for p in self.params)
            if self.graph is not None
            else (),
            tuple((name, metadata(p)) for name, p in self.model.named_parameters()),
            tuple(
                (name, metadata(b), b._version)
                for name, b in self.model.named_buffers()
                if ".optimizer_state." not in name
            ),
            tuple(
                (
                    id(point),
                    tuple(
                        (k, metadata(v) if isinstance(v, torch.Tensor) else repr(v))
                        for k, v in state.items()
                    ),
                )
                for point, state in self.optimizer.state.items()
            ),
            tuple(
                (
                    tuple(id(p) for p in g["params"]),
                    tuple((k, repr(v)) for k, v in g.items() if k != "params"),
                )
                for g in self.optimizer.param_groups
            ),
            tuple(
                (id(m), type(m), m.training, m.__dict__.get("forward"), type(m).forward)
                for m in self.model.modules()
            ),
            (
                metadata(self.tx),
                metadata(self.ty),
                tuple((metadata(v), v._version) for v in self.periods),
            )
            if self.periods
            else (metadata(self.tx), metadata(self.ty)),
        )

    def eligible(self):
        if (
            self.invalidated
            or torch.cuda.current_stream().cuda_stream != self.owner_stream
            or torch.cuda.is_current_stream_capturing()
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
        ):
            return False
        if (
            torch._C._is_torch_function_mode_enabled()
            or torch._C._len_torch_dispatch_stack()
            or torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level != -1
            or torch._C._get_tracing_state() is not None
            or torch.is_autocast_enabled("cuda")
            or torch.are_deterministic_algorithms_enabled()
        ):
            return False
        saved_hooks = getattr(
            torch._C._autograd, "_top_saved_tensors_default_hooks", None
        )
        if saved_hooks is None or saved_hooks(False) is not None:
            return False
        from torch.nn.modules import module as module_globals
        from torch.optim import optimizer as optimizer_globals

        if any(
            getattr(module_globals, name, {})
            for name in (
                "_global_forward_pre_hooks",
                "_global_forward_hooks",
                "_global_backward_pre_hooks",
                "_global_backward_hooks",
            )
        ):
            return False
        if any(
            getattr(optimizer_globals, name, {})
            for name in ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks")
        ):
            return False
        if (
            self.optimizer._optimizer_step_pre_hooks
            or self.optimizer._optimizer_step_post_hooks
        ):
            return False
        if any(
            any(
                getattr(m, name, {})
                for name in (
                    "_forward_pre_hooks",
                    "_forward_hooks",
                    "_backward_pre_hooks",
                    "_backward_hooks",
                )
            )
            for m in self.model.modules()
        ):
            return False
        if any(
            p._backward_hooks or getattr(p, "_post_accumulate_grad_hooks", None)
            for p in self.params
        ):
            return False
        if self._signature() != self.signature:
            return False
        if self.site is not None:
            if self.optimizer.check_finite:
                return False
            try:
                self.site.atom_state._require_boundary()
            except RuntimeError:
                return False
            lp = self.linear_program
            roots = lp._checked_roots(lp.dispatcher, self.site, self.middle)
            if roots is None or not lp.guard.check(roots):
                return False
            pp = self.polar_program
            with finite.checking(False):
                if not pp.preflight(
                    self.optimizer._update_dispatcher,
                    self.optimizer.state_adapter,
                    self.site.atoms.p,
                    self.site.atoms.p.grad,
                    self.atom_group["amplitude_weight_decay"],
                ):
                    return False
                if pp.launch is not None and pp.launch.hooks_active():
                    return False
        return True

    def body(self):
        opt, model = self.optimizer, self.model
        opt.zero_grad(set_to_none=True)
        target = self.ty[self.indices]
        x = nn.functional.relu(model[0](self.tx[self.indices]))
        if self.site is not None:
            if self.observe:
                x.retain_grad()
            self.middle = x
            if self.fp32_body:
                x = self.extension.research_owner_window_fp32(
                    x, self.site.atoms.p, *self.configs, True
                )
            else:
                x = self.extension.research_graph_owner_window(
                    x, self.site.atoms.p, *self.configs
                )
        else:
            x = model[2](x)
        self.logits = model[4](nn.functional.relu(x))
        self.loss = nn.functional.cross_entropy(self.logits, target)
        self.loss.backward()
        if self.site is not None:
            with torch.no_grad(), finite.checking(False):
                self.previous.copy_(self.site.atoms.p)
                opt._adamw_proposal()
                fused_update_(
                    self.site.kernel,
                    self.previous,
                    self.site.atoms.p,
                    step_size=self.atom_group["lr"],
                    amplitude_weight_decay=self.atom_group["amplitude_weight_decay"],
                    periods=self.periods,
                )
        else:
            dense_proposal(opt)
        self.totals[0].add_(self.loss.detach() * 4)
        self.totals[1].add_((self.logits.detach().argmax(1) == target).sum())

    def capture(self):
        saved = snapshot(self.model, self.optimizer)
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.body()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.body()
        torch.cuda.synchronize()
        restore(self.model, self.optimizer, saved)
        for t in self.totals:
            t.zero_()
        self.signature = self._signature()
        assert self.eligible(), "capture invalidated replay guard"

    def step(self, indices):
        if not self.eligible():
            self.invalidated = True
            return ordinary_step(
                self.model, self.optimizer, self.tx, self.ty, indices, self.totals
            )
        self.indices.copy_(indices)
        if self.graph is None:
            self.body()
        else:
            self.graph.replay()
            # Host mutation notifications are deliberately outside capture.
            modified = [
                *self.params,
                *[p.grad for p in self.params if p.grad is not None],
            ]
            modified.extend(
                v
                for state in self.optimizer.state.values()
                for v in state.values()
                if isinstance(v, torch.Tensor)
            )
            torch.autograd.graph.increment_version(modified)
        return self.logits, self.loss


def run_probe(inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    init = torch.load(
        inputs / "initialization.pt", map_location="cpu", weights_only=True
    )
    torch.manual_seed(9041)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tx, ty = torch.randn(64, 784, device="cuda"), torch.arange(64, device="cuda") % 10
    reports = {}
    for kind in ("cst", "dense"):
        model = make_model(
            init,
            kind,
            radius_execution="cached" if kind == "cst" else "global-response4",
            execution_plan=plans.GLOBAL_OWNER_WINDOW_RECOMPUTE
            if kind == "cst"
            else None,
        ).cuda()
        opt = make_optimizer(model, init, kind)
        totals = (tx.new_zeros(()), torch.zeros((), dtype=torch.int64, device="cuda"))
        order = torch.arange(64, device="cuda")
        for i in range(8):
            ordinary_step(model, opt, tx, ty, order[i * 4 : i * 4 + 4], totals)
        start = snapshot(model, opt)
        references = []
        for i in range(4):
            y, loss = ordinary_step(
                model, opt, tx, ty, order[i * 4 : i * 4 + 4], totals
            )
            references.append(
                {
                    **snapshot(model, opt),
                    "/logits": y.detach().cpu().clone(),
                    "/loss": loss.detach().cpu().clone(),
                    **{
                        f"/grad/{n}": p.grad.detach().cpu().clone()
                        for n, p in model.named_parameters()
                    },
                }
            )
        del y, loss
        gc.collect()
        reports[kind] = {}
        for mode in ("prepared", "graph"):
            restore(model, opt, start)
            program = PreparedStep(model, opt, tx, ty, kind)
            if mode == "graph":
                program.capture()
            rows = []
            for i in range(4):
                y, loss = program.step(order[i * 4 : i * 4 + 4])
                assert not program.invalidated, "unexpected eager fallback"
                actual = {
                    **snapshot(model, opt),
                    "/logits": y.detach().cpu().clone(),
                    "/loss": loss.detach().cpu().clone(),
                    **{
                        f"/grad/{n}": p.grad.detach().cpu().clone()
                        for n, p in model.named_parameters()
                    },
                }
                rows.append(exact(references[i], actual))
            reports[kind][mode] = rows
            (output / "probe.json").write_text(json.dumps(reports, indent=2))
            assert all(r["all_byte_equal"] for r in rows), (kind, mode, rows)
            del program, y, loss
            gc.collect()
        print(kind, "capture and byte state PASS", flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run_probe(args.inputs, args.output)


if __name__ == "__main__":
    main()
