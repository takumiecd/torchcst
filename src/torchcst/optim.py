"""Optimizer-state following, coordinate preconditioning, and parameter grouping."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from ._validation import require_int
from .storage import SynapseStore


class OptimizerStateFollower:
    """Keep slot-indexed optimizer tensors aligned with a mutable store.

    Optimizers key state by :class:`~torch.nn.Parameter` identity. Growing a
    parameter's storage therefore does not grow its optimizer moments. This
    follower treats every non-scalar state tensor whose leading dimension
    matches the store capacity as slot-indexed state. Those tensors are padded
    on growth and cleared whenever a slot dies or is reused.

    An engine-owned optimizer gets one follower subscribed automatically per
    store; :meth:`SynapseStore.reconcile_optimizer_state` is the manual
    counterpart for hand-driven stores.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[nn.Parameter],
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        params = tuple(parameters)
        if not params or not all(isinstance(param, nn.Parameter) for param in params):
            raise TypeError("parameters must contain at least one Parameter")
        capacities = {int(param.shape[0]) for param in params if param.ndim > 0}
        if len(capacities) != 1 or any(param.ndim == 0 for param in params):
            raise ValueError("parameters must share one non-scalar leading capacity")
        self._optimizer = optimizer
        self._parameters = params
        self._capacity = capacities.pop()

    @property
    def capacity(self) -> int:
        return self._capacity

    @staticmethod
    def _validated_slots(slots: torch.Tensor) -> torch.Tensor:
        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots must be a Tensor")
        if slots.ndim != 1 or slots.dtype != torch.int64:
            raise TypeError("slots must be a rank-1 int64 Tensor")
        return slots.detach().to(device="cpu")

    def _slot_tensors(self, parameter: nn.Parameter):
        """Yield the state tensors of ``parameter`` indexed by slot capacity."""
        state = self._optimizer.state.get(parameter)
        if not state:
            return
        for name, value in tuple(state.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == self._capacity
            ):
                yield state, name, value

    def _zero_rows(self, slots: torch.Tensor) -> None:
        slots = self._validated_slots(slots)
        if slots.numel() and bool(((slots < 0) | (slots >= self._capacity)).any()):
            raise IndexError("optimizer follower slots are outside capacity")
        for parameter in self._parameters:
            for _, _, value in self._slot_tensors(parameter):
                if slots.numel():
                    value.index_fill_(0, slots.to(value.device), 0)

    def grow(self, new_capacity: int) -> None:
        """Zero-pad all materialized slot tensors to ``new_capacity``."""
        require_int(new_capacity, "new_capacity")
        if new_capacity < self._capacity:
            raise ValueError("OptimizerStateFollower cannot shrink")
        if new_capacity == self._capacity:
            return
        for parameter in self._parameters:
            for state, name, value in self._slot_tensors(parameter):
                grown = value.new_zeros((new_capacity, *value.shape[1:]))
                grown[: self._capacity].copy_(value)
                state[name] = grown
        self._capacity = new_capacity

    def on_birth(self, slots: torch.Tensor, lineage: torch.Tensor) -> None:
        del lineage
        self._zero_rows(slots)

    def on_death(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_refit(self, slots: torch.Tensor) -> None:
        """Reset solved scalar amplitudes while preserving coordinate moments."""
        slots = self._validated_slots(slots)
        if slots.numel() and bool(((slots < 0) | (slots >= self._capacity)).any()):
            raise IndexError("optimizer follower slots are outside capacity")
        for parameter in self._parameters:
            if parameter.ndim != 1:
                continue
            for _, _, value in self._slot_tensors(parameter):
                if slots.numel():
                    value.index_fill_(0, slots.to(value.device), 0)

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        """Move surviving rows according to an old-slot to new-slot mapping."""
        mapping = self._validated_slots(old_to_new)
        if mapping.numel() != self._capacity:
            raise ValueError("old_to_new must align with follower capacity")
        old = torch.nonzero(mapping >= 0, as_tuple=False).flatten()
        if old.numel() and bool((mapping[old] >= self._capacity).any()):
            raise IndexError("optimizer follower remap targets outside capacity")
        for parameter in self._parameters:
            for state, name, value in self._slot_tensors(parameter):
                remapped = torch.zeros_like(value)
                if old.numel():
                    source = old.to(value.device)
                    target = mapping[old].to(value.device)
                    remapped.index_copy_(0, target, value.index_select(0, source))
                state[name] = remapped

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "torchcst-optimizer-state-follower-v1",
            "capacity": self._capacity,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-optimizer-state-follower-v1"
        ):
            raise ValueError("unsupported OptimizerStateFollower state schema")
        self._capacity = require_int(
            state.get("capacity"), "optimizer follower capacity", minimum=0
        )


class CoordPreconditioner:
    """LM-damped diagonal Gauss-Newton momentum for one site's ``s``/``t`` blocks.

    Coordinate gradients through a Gaussian kernel are exponentially
    nonuniform: an atom far from every neuron has a vanishing Jacobian, so a
    single learning rate either freezes the far tail or destabilizes the near
    atoms.  This preconditioner measures each atom's step by how much it moves
    the represented ``W`` -- the per-atom squared Jacobian norm ``J^2``, in
    closed form from kernel column sums -- so near atoms fine-tune
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
        self.module = module
        self.store = store
        self.kernel_in = kernel_in
        self.kernel_out = kernel_out
        self.cap_sigma = float(cap_sigma)
        self.beta = float(beta)
        self.target_step = float(target_step)
        self._capacity = store.capacity
        # Lazily materialized on the first step: the owning model may move
        # devices between construction and training (create state after .to).
        self.v_s: torch.Tensor | None = None
        self.v_t: torch.Tensor | None = None
        self.travel: torch.Tensor | None = None
        self.eta: float | None = None
        if subscribe:
            store.followers().subscribe(self)

    # ---- update rule (lm1d-frozen) ---------------------------------------

    @staticmethod
    @torch.no_grad()
    def _radial_moment(k: torch.Tensor, mu: torch.Tensor, x: torch.Tensor):
        """``sum_n k[n,i]^2 ||mu_n - x_i||^2`` without the ``[N, K, d]`` cube."""
        k_sq = k.square()
        mass = k_sq.sum(0)
        mu_sq = mu.square().sum(1)
        cross = k_sq.transpose(0, 1) @ mu  # [K, d]
        moment = (
            (k_sq * mu_sq[:, None]).sum(0)
            - 2.0 * (cross * x).sum(1)
            + mass * x.square().sum(1)
        )
        return moment, mass

    @torch.no_grad()
    def _jacobian_sq(self):
        store = self.store
        sigma = float(self.kernel_in.sigma.detach())
        mu_in = self.module.in_neurons.mu.to(store.s)
        mu_out = self.module.out_neurons.mu.to(store.t)
        k_in = self.kernel_in(mu_in, store.s)  # [N_in, K]
        k_out = self.kernel_out(mu_out, store.t)  # [N_out, K]
        w_sq = store.w.detach().square()
        r_in, _ = self._radial_moment(k_in, mu_in, store.s)
        r_out, _ = self._radial_moment(k_out, mu_out, store.t)
        j_s = w_sq * k_out.square().sum(0) * r_in / sigma**4
        j_t = w_sq * k_in.square().sum(0) * r_out / sigma**4
        return j_s, j_t

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


def parameter_groups(
    stores_or_model: nn.Module | Iterable[nn.Module],
    *,
    amplitude_lr: float,
    coordinate_lr: float,
    default_lr: float,
) -> list[dict[str, object]]:
    """Split parameters into amplitude / coordinate / default learning-rate groups.

    Every :class:`SynapseStore` amplitude ``w`` goes into the amplitude group
    and every learnable coordinate ``s``/``t`` into the coordinate group;
    remaining parameters take ``default_lr``. Coordinates with a buffer role
    (e.g. the entry family's ``IntegerGrid``) are not parameters and never
    appear in any group.
    """
    modules: list[nn.Module] = (
        [stores_or_model]
        if isinstance(stores_or_model, nn.Module)
        else list(stores_or_model)
    )
    if not modules:
        raise ValueError("stores_or_model must contain at least one module")

    amplitude_ids: set[int] = set()
    coordinate_ids: set[int] = set()
    for root in modules:
        for submodule in root.modules():
            if not isinstance(submodule, SynapseStore):
                continue
            weight = submodule.w
            if isinstance(weight, nn.Parameter):
                amplitude_ids.add(id(weight))
            for name in ("s", "t"):
                coordinate = getattr(submodule, name)
                if isinstance(coordinate, nn.Parameter):
                    coordinate_ids.add(id(coordinate))

    amplitude_params: list[nn.Parameter] = []
    coordinate_params: list[nn.Parameter] = []
    default_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for root in modules:
        for parameter in root.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if id(parameter) in amplitude_ids:
                amplitude_params.append(parameter)
            elif id(parameter) in coordinate_ids:
                coordinate_params.append(parameter)
            else:
                default_params.append(parameter)

    groups: list[dict[str, object]] = []
    if amplitude_params:
        groups.append({"params": amplitude_params, "lr": amplitude_lr})
    if coordinate_params:
        groups.append({"params": coordinate_params, "lr": coordinate_lr})
    if default_params:
        groups.append({"params": default_params, "lr": default_lr})
    return groups
