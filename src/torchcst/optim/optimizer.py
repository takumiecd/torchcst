"""Wrap a PyTorch optimizer with CST coordinate update policies."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim import LBFGS, Optimizer

from torchcst.geometry import EuclideanGeometry
from torchcst.nn import CSTModule

from .state import OptimizerStateAdapter, default_state_adapter


class CSTOptimizer(Optimizer):
    """Delegate proposals to ``optimizer`` and apply Kernel update policies.

    Parameters and state remain owned by the supplied optimizer. Ordinary
    model parameters, including Euclidean chart coordinates, pass through.
    Each managed atom table gets a parameter-sized old-point snapshot, never
    a dense weight matrix. Use this wrapper for step/zero_grad/checkpoint and
    scheduler/GradScaler integration; do not also step the base optimizer.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        model: nn.Module,
        state_adapter: OptimizerStateAdapter | None = None,
    ):
        if not isinstance(optimizer, Optimizer) or isinstance(optimizer, CSTOptimizer):
            raise TypeError("optimizer must be an unwrapped torch.optim.Optimizer")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if isinstance(optimizer, LBFGS):
            raise TypeError("LBFGS repeated proposals require a dedicated integration")
        if state_adapter is not None and not isinstance(
            state_adapter, OptimizerStateAdapter
        ):
            raise TypeError("state_adapter must be an OptimizerStateAdapter")
        self.model = model
        self._sites = tuple(
            module for module in model.modules() if isinstance(module, CSTModule)
        )
        self.state_adapter = state_adapter or (
            default_state_adapter(optimizer) if self._sites else OptimizerStateAdapter()
        )
        self._validate_groups(optimizer.param_groups)
        # Initialize PyTorch's scheduler and step-hook infrastructure, then
        # share the actual groups and state rather than copying their contents.
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.base_optimizer = optimizer
        self._sync_from_base()
        self._bound_parameters = {site: site.atoms.p for site in self._sites}

    def _sync_from_base(self):
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def _validate_groups(self, groups):
        names = {id(p): name for name, p in self.model.named_parameters()}
        seen = set()
        for group in groups:
            if group.get("differentiable", False):
                raise ValueError(
                    "CSTOptimizer does not support differentiable optimizer steps"
                )
            for p in group["params"]:
                if id(p) not in names:
                    raise ValueError("optimizer parameter is not owned by model")
                if id(p) in seen:
                    raise ValueError("optimizer parameters must be unique")
                seen.add(id(p))
        owners = set()
        for site in self._sites:
            if id(site.atoms.p) in owners:
                raise ValueError("an atom table cannot have multiple CST policy owners")
            owners.add(id(site.atoms.p))
            for chart in site.cst_charts():
                if chart.trainable and not isinstance(
                    chart.geometry, EuclideanGeometry
                ):
                    raise ValueError(
                        "trainable non-Euclidean charts need a chart update policy"
                    )

    def add_param_group(self, param_group):
        if not hasattr(self, "base_optimizer"):
            return super().add_param_group(param_group)
        group = dict(param_group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, torch.Tensor) else list(params)
        self._validate_groups([*self.param_groups, group])
        self.base_optimizer.add_param_group(group)
        self._sync_from_base()

    def zero_grad(self, set_to_none: bool = True):
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Evaluate once before projection/snapshot, at the current point.
            # Standard SGD/Adam-family steps only use the closure's result.
            with torch.enable_grad():
                loss = closure()
        self._validate_groups(self.param_groups)
        groups = {id(p): group for group in self.param_groups for p in group["params"]}
        active = []
        for site, point in self._bound_parameters.items():
            if site.atoms.p is not point:
                raise RuntimeError("atom Parameter was replaced; rebuild CSTOptimizer")
            group = groups.get(id(point))
            if group is None or point.grad is None:
                continue
            rate = group["lr"]
            if isinstance(rate, torch.Tensor) or not math.isfinite(rate) or rate < 0:
                raise ValueError(
                    "CST learning rates must be finite nonnegative Python scalars"
                )
            if group.get("capturable", False):
                raise ValueError(
                    "CST coordinate policies do not yet support CUDA Graph capture"
                )
            if point.grad.is_sparse:
                raise ValueError(
                    "CST coordinate policies require strided atom gradients"
                )
            active.append((site, point, rate))
        # Check before base optimizer mutates any parameter or state.
        for group in self.param_groups:
            for point in group["params"]:
                if point.grad is not None:
                    grad = point.grad
                    values = grad.coalesce().values() if grad.is_sparse else grad
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("non-finite parameter gradient")
        old_points = []
        for site, point, rate in active:
            projected = site.kernel.project_parameter_gradient(
                *site.cst_charts(), point, point.grad
            )
            if projected.shape != point.shape:
                raise ValueError("projected atom gradient has the wrong shape")
            if not bool(torch.isfinite(projected).all()):
                raise FloatingPointError("non-finite projected atom gradient")
            point.grad.copy_(projected)
            old_points.append((site, point, rate, point.detach().clone()))
        if closure is None:
            result = self.base_optimizer.step()
        else:
            result = self.base_optimizer.step(closure=lambda: loss)
        self._sync_from_base()
        for site, point, rate, old in old_points:
            if rate == 0:
                continue
            new = site.kernel.apply_parameter_update(
                *site.cst_charts(), old, point - old, step_size=rate
            )
            if new.shape != point.shape:
                raise ValueError("updated atom parameters have the wrong shape")
            if not bool(torch.isfinite(new).all()):
                raise FloatingPointError("non-finite CST parameter update")
            self.state_adapter.transport(site, old, new, self.state.get(point, {}))
            point.copy_(new)
        return loss if closure is not None else result

    def _manifest(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return {
            "version": 1,
            "optimizer_type": f"{type(self.base_optimizer).__module__}.{type(self.base_optimizer).__qualname__}",
            "vector_keys": self.state_adapter.vector_keys,
            "state_adapter_type": f"{type(self.state_adapter).__module__}.{type(self.state_adapter).__qualname__}",
            "groups": [
                [(names[id(p)], tuple(p.shape)) for p in group["params"]]
                for group in self.param_groups
            ],
            "sites": [
                (
                    names[id(site.atoms.p)],
                    f"{type(site.kernel).__module__}.{type(site.kernel).__qualname__}",
                )
                for site in self._sites
            ],
        }

    def state_dict(self):
        result = super().state_dict()
        result["cst"] = self._manifest()
        return result

    def load_state_dict(self, state_dict):
        if state_dict.get("cst") != self._manifest():
            raise ValueError("CST optimizer checkpoint contract differs")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, self.param_groups)
        ):
            raise ValueError("optimizer parameter groups differ from checkpoint")
        for saved, current in zip(saved_groups, self.param_groups):
            for index, point in zip(saved["params"], current["params"]):
                for key, value in state_dict["state"].get(index, {}).items():
                    if (
                        isinstance(value, torch.Tensor)
                        and key
                        in {
                            *self.state_adapter.vector_keys,
                            "exp_avg_sq",
                            "max_exp_avg_sq",
                            "exp_inf",
                            "square_avg",
                            "sum",
                            "acc_delta",
                            "step_size",
                        }
                        and value.shape != point.shape
                    ):
                        raise ValueError(f"optimizer state {key!r} has the wrong shape")
        payload = {key: value for key, value in state_dict.items() if key != "cst"}
        super().load_state_dict(payload)
        self.base_optimizer.state = self.state
        self.base_optimizer.param_groups = self.param_groups
