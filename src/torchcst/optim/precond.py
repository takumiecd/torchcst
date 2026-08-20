"""The coordinate optimizer: a step measured in how far it moves W."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .._validation import require_int
from ..storage import SynapseStore
from . import metric


class CoordPreconditioner:
    """LM-damped diagonal Gauss-Newton momentum for one site's ``s``/``t`` blocks.

    Coordinate gradients through a Gaussian kernel are exponentially
    nonuniform: an atom far from every neuron has a vanishing Jacobian, so a
    single learning rate either freezes the far tail or destabilizes the near
    atoms.  This preconditioner measures each atom's step by how much it moves
    the represented ``W`` -- the per-atom squared Jacobian norm ``J^2``, in
    closed form from kernel column sums (family-generic: the derivative
    comes from the kernel's ``profile_grad``, so any family implementing it
    is preconditioned correctly) -- so near atoms fine-tune
    (``step ~ 1/J``) and far atoms are suppressed only linearly
    (``step ~ J/lambda``) instead of exponentially.

    The update rule is the lm1d-frozen one (registered in the ``cst``
    repository, ``docs/research/lm1/lm1d_spec.md``; confirmed 3/3 seeds in
    lm1e): EMA momentum ``v`` with ``beta``, damping ``lambda`` at the live
    median of ``J^2``, one global step size calibrated on the first step so
    the live median step equals ``target_step * sigma``, and a per-atom trust
    cap of ``cap_sigma * sigma`` per update.  Gates and mass scales are
    deliberately not part of the metric.

    The owning optimizer must not also step ``s``/``t``: build its parameter
    groups without the site's coordinates (the coordinate group of
    :func:`parameter_groups` is what to drop or hand over), and call
    :meth:`zero_grad` alongside ``optimizer.zero_grad()``.

    Momentum and travel are physical-slot-indexed, so the instance subscribes
    to the store's :class:`~torchcst.storage.FollowerHub` and follows the
    same contract as :class:`OptimizerStateFollower`: capacity growth pads,
    birth and death zero the affected rows (a reborn atom starts with fresh
    momentum), remap moves surviving rows.  Statistics (``lambda``, the
    calibration median) are computed over live slots only.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        cap_sigma: float,
        beta: float = 0.9,
        target_step: float = 0.01,
        subscribe: bool = True,
        traffic_rows: int = 0,
        chunk_elements: int = 1 << 24,
    ) -> None:
        store = getattr(module, "synapses", None)
        kernel_in = getattr(module, "kernel_in", None)
        kernel_out = getattr(module, "kernel_out", None)
        if not isinstance(store, SynapseStore) or kernel_in is None:
            raise TypeError(
                "module must be a continuous CST map with .synapses and kernels"
            )
        if not isinstance(store.s, nn.Parameter) or not isinstance(
            store.t, nn.Parameter
        ):
            raise TypeError(
                "CoordPreconditioner requires learnable continuous coordinates"
            )
        if not (isinstance(cap_sigma, (int, float)) and cap_sigma > 0):
            raise ValueError("cap_sigma must be a positive number")
        if not (isinstance(beta, (int, float)) and 0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if not (isinstance(target_step, (int, float)) and target_step > 0):
            raise ValueError("target_step must be a positive number")
        require_int(traffic_rows, "traffic_rows", minimum=0)
        require_int(chunk_elements, "chunk_elements", minimum=1)
        self.module = module
        self.store = store
        self.kernel_in = kernel_in
        self.kernel_out = kernel_out
        self.cap_sigma = float(cap_sigma)
        self.beta = float(beta)
        self.target_step = float(target_step)
        # Traffic-weighted metric.  Zero keeps the Frobenius form this class has
        # always used; a positive count subsamples that many rows of the site's
        # own inputs and output gradients per step and weights the metric by
        # them.  Frobenius is the metric of a customer base that arrives equally
        # from every direction, which no layer has -- E-func measured the gap
        # directly, rescoring a census under the traffic and watching conv1's
        # error fall from 0.36 to 0.16.  The same identity sits in here.
        self.traffic_rows = int(traffic_rows)
        # How many elements one `[N, atoms]` temporary may reach before the
        # atom loop splits -- a budget to set, not a fixed policy, because the
        # split costs time and the memory it saves is only worth paying for
        # once there is a shortage.  The metric holds four such temporaries, so
        # the peak is about four times this in elements.  The default clears a
        # ten-thousand-atom site against a fifteen-hundred-neuron chart in one
        # pass, and starts splitting from roughly four times that.
        self.chunk_elements = int(chunk_elements)
        self._x: torch.Tensor | None = None
        self._g: torch.Tensor | None = None
        if self.traffic_rows:
            module.register_forward_pre_hook(self._observe_input)
            module.register_full_backward_hook(self._observe_gradient)
        self._capacity = store.capacity
        # Lazily materialized on the first step: the owning model may move
        # devices between construction and training (create state after .to).
        self.v_s: torch.Tensor | None = None
        self.v_t: torch.Tensor | None = None
        self.travel: torch.Tensor | None = None
        self.eta: float | None = None
        if subscribe:
            store.followers().subscribe(self)

    # ---- traffic capture (off unless traffic_rows) ------------------------

    @staticmethod
    def _sample(tensor: Tensor, count: int) -> Tensor:
        """The last axis kept, everything else flattened, then truncated.

        Named apart from this class's follower-contract _rows, which it
        would otherwise shadow -- silently, and only at call time.
        """
        flat = tensor.detach().reshape(-1, tensor.shape[-1])
        return flat[:count] if flat.shape[0] > count else flat

    def _observe_input(self, module, inputs) -> None:
        del module
        if inputs and isinstance(inputs[0], torch.Tensor):
            self._x = self._sample(inputs[0], self.traffic_rows)

    def _observe_gradient(self, module, grad_input, grad_output) -> None:
        del module, grad_input
        if grad_output and isinstance(grad_output[0], torch.Tensor):
            self._g = self._sample(grad_output[0], self.traffic_rows)


    def _jacobian_sq(self):
        store = self.store
        mu_in = self.module.in_neurons.mu.to(store.s)
        mu_out = self.module.out_neurons.mu.to(store.t)
        atoms = store.s.shape[0]
        width = max(mu_in.shape[0], mu_out.shape[0])
        step = max(1, min(atoms, self.chunk_elements // max(width, 1)))
        if step >= atoms:
            return self._jacobian_block(mu_in, mu_out, slice(0, atoms))
        j_s = store.s.new_zeros(atoms)
        j_t = store.t.new_zeros(atoms)
        for start in range(0, atoms, step):
            block = slice(start, min(start + step, atoms))
            j_s[block], j_t[block] = self._jacobian_block(mu_in, mu_out, block)
        return j_s, j_t

    @torch.no_grad()
    def _jacobian_block(self, mu_in, mu_out, block):
        store = self.store
        k_in, g_in = metric.columns(self.kernel_in, mu_in, store.s[block])
        k_out, g_out = metric.columns(self.kernel_out, mu_out, store.t[block])
        w_sq = store.w.detach()[block].square()
        return metric.jacobian_sq(
            k_in=k_in, g_in=g_in, k_out=k_out, g_out=g_out,
            mu_in=mu_in, mu_out=mu_out,
            source=store.s[block], target=store.t[block], w_sq=w_sq,
            traffic_in=self._x if self.traffic_rows else None,
            traffic_out=self._g if self.traffic_rows else None,
        )


    def _materialize(self) -> None:
        store = self.store
        self.v_s = torch.zeros_like(store.s)
        self.v_t = torch.zeros_like(store.t)
        self.travel = torch.zeros(
            store.s.shape[0], device=store.s.device, dtype=store.s.dtype
        )

    @torch.no_grad()
    def step(self, lr_scale: float = 1.0) -> None:
        """Apply one preconditioned momentum update to ``s`` and ``t``.

        ``lr_scale`` follows the outer schedule (the same factor the
        optimizer's learning rate is being multiplied by).  A call with no
        coordinate gradients is a no-op.
        """
        store = self.store
        if store.s.grad is None or store.t.grad is None:
            return
        if self.v_s is None:
            self._materialize()
        sigma = float(self.kernel_in.sigma.detach())
        live = store.live_slots().to(store.s.device)
        self.v_s.mul_(self.beta).add_(store.s.grad, alpha=1.0 - self.beta)
        self.v_t.mul_(self.beta).add_(store.t.grad, alpha=1.0 - self.beta)
        j_s, j_t = self._jacobian_sq()
        if live.numel() == 0:
            return
        lam_s = j_s.index_select(0, live).median().clamp(min=1e-30)
        lam_t = j_t.index_select(0, live).median().clamp(min=1e-30)
        raw_s = self.v_s / (j_s + lam_s)[:, None]
        raw_t = self.v_t / (j_t + lam_t)[:, None]
        if self.eta is None:
            med = (
                torch.cat(
                    [
                        raw_s.index_select(0, live).norm(dim=1),
                        raw_t.index_select(0, live).norm(dim=1),
                    ]
                )
                .median()
                .clamp(min=1e-30)
            )
            self.eta = self.target_step * sigma / float(med)
        cap = self.cap_sigma * sigma
        for param, raw in ((store.s, raw_s), (store.t, raw_t)):
            delta = raw * (self.eta * lr_scale)
            norms = delta.norm(dim=1, keepdim=True).clamp(min=1e-30)
            delta = delta * (norms.clamp(max=cap) / norms)
            mask = torch.zeros(
                param.shape[0], 1, device=param.device, dtype=param.dtype
            )
            mask[live] = 1.0
            delta = delta * mask
            param.sub_(delta)
            if param is store.s:
                self.travel += delta.norm(dim=1) / sigma

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear the coordinate gradients this preconditioner consumes.

        The coordinates are outside every optimizer group, so
        ``optimizer.zero_grad()`` never touches them; call this alongside it.
        """
        for param in (self.store.s, self.store.t):
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    def mobility(self, threshold: float = 0.5) -> float:
        """The live fraction of atoms whose cumulative travel exceeds
        ``threshold`` sigma."""
        if self.travel is None:
            return 0.0
        live = self.store.live_slots().to(self.travel.device)
        if live.numel() == 0:
            return 0.0
        return float((self.travel.index_select(0, live) > threshold).float().mean())

    # ---- follower contract ------------------------------------------------

    def _rows(self):
        return (tensor for tensor in (self.v_s, self.v_t, self.travel)
                if tensor is not None)

    def _zero_rows(self, slots: torch.Tensor) -> None:
        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots must be a Tensor")
        slots = slots.detach().to("cpu")
        if slots.numel() and bool(
            ((slots < 0) | (slots >= self._capacity)).any()
        ):
            raise IndexError("preconditioner follower slots are outside capacity")
        for tensor in self._rows():
            if slots.numel():
                tensor.index_fill_(0, slots.to(tensor.device), 0)

    def grow(self, new_capacity: int) -> None:
        require_int(new_capacity, "new_capacity")
        if new_capacity < self._capacity:
            raise ValueError("CoordPreconditioner cannot shrink")
        if new_capacity == self._capacity:
            return
        for name in ("v_s", "v_t", "travel"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            grown = tensor.new_zeros((new_capacity, *tensor.shape[1:]))
            grown[: self._capacity].copy_(tensor)
            setattr(self, name, grown)
        self._capacity = new_capacity

    def on_birth(self, slots: torch.Tensor, lineage: torch.Tensor) -> None:
        del lineage
        self._zero_rows(slots)

    def on_death(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_refit(self, slots: torch.Tensor) -> None:
        """Refit solves amplitudes in place; coordinate momentum survives."""

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        mapping = old_to_new.detach().to("cpu")
        if mapping.numel() != self._capacity:
            raise ValueError("old_to_new must align with preconditioner capacity")
        old = torch.nonzero(mapping >= 0, as_tuple=False).flatten()
        if old.numel() and bool((mapping[old] >= self._capacity).any()):
            raise IndexError("preconditioner remap targets outside capacity")
        for name in ("v_s", "v_t", "travel"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            remapped = torch.zeros_like(tensor)
            if old.numel():
                source = old.to(tensor.device)
                target = mapping[old].to(tensor.device)
                remapped.index_copy_(0, target, tensor.index_select(0, source))
            setattr(self, name, remapped)

    # ---- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        def _snapshot(tensor: torch.Tensor | None) -> torch.Tensor | None:
            # clone: on CPU ``.detach().cpu()`` would alias live state, and a
            # later in-place step would silently rewrite the snapshot.
            return None if tensor is None else tensor.detach().cpu().clone()

        return {
            "schema": "torchcst-coord-preconditioner-v1",
            "capacity": self._capacity,
            "eta": self.eta,
            "v_s": _snapshot(self.v_s),
            "v_t": _snapshot(self.v_t),
            "travel": _snapshot(self.travel),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-coord-preconditioner-v1"
        ):
            raise ValueError("unsupported CoordPreconditioner state schema")
        self._capacity = require_int(
            state.get("capacity"), "preconditioner capacity", minimum=0
        )
        eta = state.get("eta")
        if eta is not None and not isinstance(eta, (int, float)):
            raise TypeError("preconditioner eta must be a number or None")
        self.eta = None if eta is None else float(eta)
        device = self.store.s.device
        for name in ("v_s", "v_t", "travel"):
            value = state.get(name)
            if value is None:
                setattr(self, name, None)
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"preconditioner {name} must be a Tensor or None")
            if value.shape[0] != self._capacity:
                raise ValueError(f"preconditioner {name} does not match capacity")
            setattr(self, name, value.to(device).clone())
