"""Parameter-coordinate AdamW with an optional finite-horizon cosine schedule.

CST moments have exactly the atom-table shape. Kernels may project gradients,
retract updates, and transport the vector first moment for constrained chart
geometries. No representation derivatives, visible moments, or Gram matrices
are constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from torchcst.nn import CSTModule

from .config import AdamWConfig, _validate_betas


@dataclass(frozen=True)
class ParameterAdamConfig:
    """Settings selected on a separate MNIST validation subset.

    ``decay_steps=None`` disables the cosine schedule. After ``decay_steps``
    updates the learning rate remains at ``lr * min_lr_ratio``.
    """

    lr: float = 0.03
    betas: tuple[float, float] = (0.5, 0.99)
    eps: float = 1e-8
    decay_steps: int | None = 128
    min_lr_ratio: float = 0.1

    def __post_init__(self):
        for name in ("lr", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "betas", _validate_betas(self.betas))
        if self.decay_steps is not None and (
            isinstance(self.decay_steps, bool)
            or not isinstance(self.decay_steps, int)
            or self.decay_steps < 2
        ):
            raise ValueError("decay_steps must be an integer >= 2 or None")
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be finite and in [0, 1]")


class CSTParameterAdam(torch.optim.AdamW):
    """Whole-model parameter Adam with two parameter-sized moment buffers.

    This is ordinary coordinate Adam, not an approximation of visible-space
    Adam. All CST sites must explicitly use the factored backend and frozen
    charts. Ordinary parameters require an explicit ``dense=AdamWConfig(...)``.
    The cosine schedule applies only to the CST group. ``param_groups[i]['lr']``
    remains the unscheduled base rate; the completed-update count is serialized
    in the group as ``schedule_step``. Missing gradients skip Adam state updates;
    the schedule advances when at least one parameter in that group has a grad.
    """

    def __init__(self, model: nn.Module, *, cst=None, dense=None):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        cst = ParameterAdamConfig() if cst is None else cst
        if not isinstance(cst, ParameterAdamConfig):
            raise TypeError("cst must be a ParameterAdamConfig")
        if dense is not None and not isinstance(dense, AdamWConfig):
            raise TypeError("dense must be an AdamWConfig or None")
        sites = [module for module in model.modules() if isinstance(module, CSTModule)]
        if not sites:
            raise ValueError("model must contain a CSTModule site")
        owners = set()
        for site in sites:
            if any(chart.trainable for chart in site.cst_charts()):
                raise ValueError("CSTParameterAdam requires frozen charts")
            if site.backend != "factored" or not site.kernel.supports_factorization:
                raise ValueError("CSTParameterAdam requires backend='factored'")
            if site.atoms.grad is not None:
                raise ValueError("CST site already has an attached AtomGrad program")
            if id(site.atoms.p) in owners:
                raise ValueError("CST parameters cannot be shared between sites")
            owners.add(id(site.atoms.p))
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        atom = [(name, p) for name, p in named if id(p) in owners]
        other = [(name, p) for name, p in named if id(p) not in owners]
        if not atom:
            raise ValueError("model must contain trainable CST parameters")
        if other and dense is None:
            raise ValueError("ordinary trainable parameters require dense=AdamWConfig")
        groups = [
            {
                "params": [p for _, p in atom],
                "param_names": [name for name, _ in atom],
                "param_shapes": [tuple(p.shape) for _, p in atom],
                "lr": cst.lr,
                "betas": cst.betas,
                "eps": cst.eps,
                "weight_decay": 0.0,
                "decay_steps": cst.decay_steps,
                "min_lr_ratio": cst.min_lr_ratio,
                "schedule_step": 0,
            }
        ]
        if other:
            groups.append(
                {
                    "params": [p for _, p in other],
                    "param_names": [name for name, _ in other],
                    "param_shapes": [tuple(p.shape) for _, p in other],
                    "lr": dense.lr,
                    "betas": dense.betas,
                    "eps": dense.eps,
                    "weight_decay": dense.weight_decay,
                    "decay_steps": None,
                    "min_lr_ratio": 1.0,
                    "schedule_step": 0,
                }
            )
        # One tensor at a time keeps optimizer temporaries proportional to the
        # parameter table and avoids foreach/compiled workspace caches.
        super().__init__(groups, foreach=False)
        self._cst_sites = tuple(sites)

    @staticmethod
    def _rate_scale(group):
        horizon = group["decay_steps"]
        if horizon is None:
            return 1.0
        phase = min(group["schedule_step"] / (horizon - 1), 1.0)
        floor = group["min_lr_ratio"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * phase))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        active = [
            any(p.grad is not None for p in g["params"]) for g in self.param_groups
        ]
        with torch.no_grad():
            for site in self._cst_sites:
                point = site.atoms.p
                if point.grad is None:
                    continue
                projected = site.kernel.project_parameter_gradient(
                    *site.cst_charts(),
                    point,
                    point.grad,
                )
                if projected.shape != point.shape:
                    raise ValueError("kernel projected gradient has the wrong shape")
                point.grad.copy_(projected)
        # Validate all gradients before Adam mutates any group. No dense visible
        # observation scope or derivative program is installed by this optimizer.
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is not None:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "CSTParameterAdam requires strided gradients"
                        )
                    if not bool(torch.isfinite(grad).all()):
                        raise FloatingPointError("non-finite parameter gradient")
        base_rates = [group["lr"] for group in self.param_groups]
        old_points = {site: site.atoms.p.detach().clone() for site in self._cst_sites}
        scheduled_cst_rate = base_rates[0] * self._rate_scale(self.param_groups[0])
        try:
            for group, rate in zip(self.param_groups, base_rates):
                group["lr"] = rate * self._rate_scale(group)
            super().step()
        finally:
            for group, rate in zip(self.param_groups, base_rates):
                group["lr"] = rate
        with torch.no_grad():
            for site in self._cst_sites:
                point = site.atoms.p
                if point.grad is None:
                    continue
                old = old_points[site]
                updated = site.kernel.apply_parameter_update(
                    *site.cst_charts(),
                    old,
                    point - old,
                    step_size=scheduled_cst_rate,
                )
                if not bool(torch.isfinite(updated).all()):
                    raise FloatingPointError("kernel parameter update must be finite")
                first_moment = self.state[point].get("exp_avg")
                if first_moment is not None:
                    transported = site.kernel.transport_parameter_state(
                        *site.cst_charts(),
                        old,
                        updated,
                        first_moment,
                    )
                    if transported.shape != first_moment.shape:
                        raise ValueError("kernel transported state has the wrong shape")
                    first_moment.copy_(transported)
                point.copy_(updated)
        for group, used in zip(self.param_groups, active):
            group["schedule_step"] += int(used)
        return loss

    def load_state_dict(self, state_dict):
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("optimizer parameter partition differs from checkpoint")
        for saved, current in zip(groups, self.param_groups):
            for key in ("param_names", "param_shapes"):
                if saved.get(key) != current[key]:
                    raise ValueError(f"optimizer {key} differ from checkpoint")
            count = saved.get("schedule_step")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid schedule_step in checkpoint")
            ParameterAdamConfig(
                lr=saved["lr"],
                betas=saved["betas"],
                eps=saved["eps"],
                decay_steps=saved["decay_steps"],
                min_lr_ratio=saved["min_lr_ratio"],
            )
            for key, shape in zip(saved["params"], saved["param_shapes"]):
                state = state_dict["state"].get(key, {})
                for moment in ("exp_avg", "exp_avg_sq"):
                    if moment in state and tuple(state[moment].shape) != tuple(shape):
                        raise ValueError(
                            "optimizer moment shape differs from parameter"
                        )
        return super().load_state_dict(state_dict)
