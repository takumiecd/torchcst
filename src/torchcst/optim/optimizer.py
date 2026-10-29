"""Wrap a PyTorch optimizer with CST coordinate update policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.optim import LBFGS, Optimizer

from torchcst.atoms import OptimizerFieldSpec
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.nn import CSTModule

from .binding import bind_atom_states, field_specs, validate_binding
from .state import OptimizerStateAdapter, default_state_adapter


@dataclass
class _PreparedAtomStep:
    site: CSTModule
    point: torch.Tensor
    previous: torch.Tensor
    step_size: float
    amplitude_weight_decay: float
    plan: object


class CSTOptimizer(Optimizer):
    """Delegate proposals to ``optimizer`` and apply Kernel update policies.

    Managed atom state is owned by AtomState and shared with the supplied optimizer. Ordinary
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
        state_specs: dict[str, OptimizerFieldSpec] | None = None,
        update_selector=None,
        check_finite: bool = True,
    ):
        if not isinstance(optimizer, Optimizer) or isinstance(optimizer, CSTOptimizer):
            raise TypeError("optimizer must be an unwrapped torch.optim.Optimizer")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if type(check_finite) is not bool:
            raise TypeError("check_finite must be a bool")
        self.check_finite = check_finite
        self._allows_amplitude_decay = type(optimizer) is torch.optim.AdamW
        if isinstance(optimizer, LBFGS):
            raise TypeError("LBFGS repeated proposals require a dedicated integration")
        if state_adapter is not None and not isinstance(
            state_adapter, OptimizerStateAdapter
        ):
            raise TypeError("state_adapter must be an OptimizerStateAdapter")
        from torchcst._backends.dispatch import Selector

        if update_selector is not None and not isinstance(update_selector, Selector):
            raise TypeError("update_selector must be a Selector")
        self.model = model
        self._sites = tuple(
            module for module in model.modules() if isinstance(module, CSTModule)
        )
        self.state_adapter = state_adapter or (
            default_state_adapter(optimizer) if self._sites else OptimizerStateAdapter()
        )
        self._validate_groups(optimizer.param_groups)
        self._state_specs = field_specs(
            optimizer, self.state_adapter.vector_keys, state_specs
        )
        bind_atom_states(optimizer, self._sites, self._state_specs)
        # Initialize PyTorch's scheduler and step-hook infrastructure, then
        # share the actual groups and state rather than copying their contents.
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.base_optimizer = optimizer
        self._sync_from_base()
        self._bound_parameters = {site: site.atoms.p for site in self._sites}
        # Every atom site uses one coordinate-update dispatcher. Kernel and
        # Geometry select the implementation; optimizer code has no Polar branch.
        from torchcst._backends.catalog import get_registry
        from torchcst._backends.dispatch import Dispatcher, FixedSelector
        from torchcst._backends.torch.algorithms.atom_update.plans import REFERENCE
        from torchcst.operators import Operator
        from torchcst.operators.atom_update import AtomUpdateBinding

        self._update_dispatcher = Dispatcher(
            selector=update_selector
            or FixedSelector(REFERENCE, registry=get_registry())
        )
        self._update_bindings = {
            site: AtomUpdateBinding(
                Operator(charts=site.cst_charts(), kernel=site.kernel, atoms=site.atoms)
            )
            for site in self._sites
        }

    def _sync_from_base(self):
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def _validate_groups(self, groups):
        names = {id(p): name for name, p in self.model.named_parameters()}
        atom_ids = {id(site.atoms.p) for site in self._sites}
        seen = set()
        for group in groups:
            if "amplitude_weight_decay" in group:
                decay = group["amplitude_weight_decay"]
                if not self._allows_amplitude_decay:
                    raise ValueError("amplitude_weight_decay requires exact AdamW")
                if (
                    type(decay) not in (int, float)
                    or not math.isfinite(decay)
                    or decay < 0
                ):
                    raise ValueError(
                        "amplitude_weight_decay must be a finite nonnegative Python scalar"
                    )
                if not any(id(p) in atom_ids for p in group["params"]):
                    raise ValueError("amplitude_weight_decay requires an atom group")
                raw_decay = group.get("weight_decay", 0)
                if type(raw_decay) not in (int, float) or raw_decay != 0:
                    raise ValueError(
                        "amplitude_weight_decay requires raw weight_decay=0"
                    )
                rate = group["lr"]
                if (
                    type(rate) not in (int, float)
                    or not math.isfinite(rate)
                    or rate < 0
                    or rate * decay > 1
                ):
                    raise ValueError(
                        "lr * amplitude_weight_decay must be in [0, 1] with finite Python lr"
                    )
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
                if (
                    chart.trainable
                    and type(chart.geometry.spec) is not EuclideanGeometrySpec
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
        validate_binding(
            self.base_optimizer,
            self._sites,
            self._state_specs,
            groups=[*self.param_groups, group],
        )
        self.base_optimizer.add_param_group(group)
        bind_atom_states(self.base_optimizer, self._sites, self._state_specs)
        self._sync_from_base()

    def zero_grad(self, set_to_none: bool = True):
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        from torchcst._backends.torch import finite

        loss = None
        if closure is not None:
            # Evaluate once before projection/snapshot, at the current point.
            # Standard SGD/Adam-family steps only use the closure's result.
            with torch.enable_grad():
                loss = closure()
        # A closure executes outside this scope: only optimizer value checks
        # are optional. Model/geometry calls made by the closure keep theirs.
        with finite.checking(self.check_finite):
            return self._step_impl(loss, has_closure=closure is not None)

    def _step_impl(self, loss, *, has_closure):
        from torchcst._backends.torch.kernels import execution as _kernel

        if any("amplitude_weight_decay" in group for group in self.param_groups):
            return self._step_with_amplitude_decay(loss, has_closure=has_closure)
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
        from torchcst._backends.torch import finite

        gradients = []
        for group in self.param_groups:
            for point in group["params"]:
                if point.grad is not None:
                    grad = point.grad
                    values = grad.coalesce().values() if grad.is_sparse else grad
                    gradients.append(values)
        if finite.first_nonfinite(gradients) is not None:
            raise FloatingPointError("non-finite parameter gradient")
        del gradients  # Release any temporary coalesced sparse values before Adam.
        old_points = []
        for site, point, rate in active:
            projected = _kernel.project_parameter_gradient(
                site.kernel, *site.cst_charts(), point, point.grad
            )
            if projected.shape != point.shape:
                raise ValueError("projected atom gradient has the wrong shape")
            if self.check_finite and not bool(torch.isfinite(projected).all()):
                raise FloatingPointError("non-finite projected atom gradient")
            old = point.detach().clone()
            plan = None
            if rate != 0:
                from torchcst.operators.atom_update import AtomUpdateInputs

                inputs = AtomUpdateInputs(old, rate)
                binding = self._update_bindings[site]
                plan = self._update_dispatcher.prepare(binding, inputs).plan
            if projected is not point.grad:
                point.grad.copy_(projected)
            old_points.append((site, point, rate, old, plan))
        if not has_closure:
            result = self.base_optimizer.step()
        else:
            result = self.base_optimizer.step(closure=lambda: loss)
        self._sync_from_base()
        for site, point, rate, old, plan in old_points:
            if rate == 0:
                continue
            from torchcst.operators.atom_update import AtomUpdateInputs

            new = self._update_dispatcher.run(
                self._update_bindings[site], AtomUpdateInputs(old, rate), plan=plan
            )
            if new.shape != point.shape:
                raise ValueError("updated atom parameters have the wrong shape")
            if self.check_finite and not bool(torch.isfinite(new).all()):
                raise FloatingPointError("non-finite CST parameter update")
            self.state_adapter.transport(site, old, new, self.state.get(point, {}))
        return loss if has_closure else result

    def _step_with_amplitude_decay(self, loss, *, has_closure):
        """Prepare atoms after user pre-hooks and decay their logical amplitude.

        A final, temporary base pre-hook observes the settings and gradients
        actually used by AdamW. It never replaces groups or reruns the user's
        closure. Ordinary parameters therefore keep standard AdamW behavior.
        """
        from torchcst._backends.torch import finite
        from torchcst._backends.torch.kernels import execution as _kernel
        from torchcst.operators.atom_update import AtomUpdateInputs

        self._validate_groups(self.param_groups)
        protected = {
            id(p)
            for group in self.param_groups
            if "amplitude_weight_decay" in group
            for p in group["params"]
        }
        prepared = []
        guard_called = False
        replay_closure = (lambda: loss) if has_closure else None

        def prepare_before_adam(base, args, kwargs):
            nonlocal guard_called
            if (
                len(args) != 1
                or args[0] is not base
                or kwargs.get("closure") is not replay_closure
            ):
                raise ValueError("amplitude decay hooks must not replace the closure")
            self._validate_groups(base.param_groups)
            groups = {id(p): g for g in base.param_groups for p in g["params"]}
            gradients = [
                p.grad.coalesce().values() if p.grad.is_sparse else p.grad
                for group in base.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if finite.first_nonfinite(gradients) is not None:
                raise FloatingPointError("non-finite parameter gradient")
            for site, point in self._bound_parameters.items():
                if site.atoms.p is not point:
                    raise RuntimeError(
                        "atom Parameter was replaced; rebuild CSTOptimizer"
                    )
                group = groups.get(id(point))
                if group is None:
                    continue
                raw_decay = group.get("weight_decay", 0)
                if id(point) in protected and (
                    type(raw_decay) not in (int, float) or raw_decay != 0
                ):
                    raise ValueError("amplitude decay requires raw weight_decay=0")
                if point.grad is None:
                    continue
                rate = group["lr"]
                if (
                    type(rate) not in (int, float)
                    or not math.isfinite(rate)
                    or rate < 0
                ):
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
                projected = _kernel.project_parameter_gradient(
                    site.kernel, *site.cst_charts(), point, point.grad
                )
                if projected.shape != point.shape:
                    raise ValueError("projected atom gradient has the wrong shape")
                if finite.first_nonfinite((projected,)) is not None:
                    raise FloatingPointError("non-finite projected atom gradient")
                previous = point.detach().clone()
                decay = group.get("amplitude_weight_decay", 0.0)
                plan = None
                if rate:
                    inputs = AtomUpdateInputs(previous, rate, decay)
                    plan = self._update_dispatcher.prepare(
                        self._update_bindings[site], inputs
                    ).plan
                if projected is not point.grad:
                    point.grad.copy_(projected)
                prepared.append(
                    _PreparedAtomStep(site, point, previous, rate, decay, plan)
                )
            guard_called = True

        # Registered last, after existing user pre-hooks; always remove it,
        # including on invalid settings, hook exceptions or AdamW failures.
        handle = self.base_optimizer.register_step_pre_hook(prepare_before_adam)
        try:
            if has_closure:
                result = self.base_optimizer.step(closure=replay_closure)
            else:
                result = self.base_optimizer.step()
        finally:
            handle.remove()
        if not guard_called:
            raise RuntimeError("AdamW must execute its standard step pre-hooks")
        self._sync_from_base()
        for step in prepared:
            if not step.step_size:
                continue
            inputs = AtomUpdateInputs(
                step.previous, step.step_size, step.amplitude_weight_decay
            )
            updated = self._update_dispatcher.run(
                self._update_bindings[step.site], inputs, plan=step.plan
            )
            if updated.shape != step.point.shape:
                raise ValueError("updated atom parameters have the wrong shape")
            if finite.first_nonfinite((updated,)) is not None:
                raise FloatingPointError("non-finite CST parameter update")
            self.state_adapter.transport(
                step.site, step.previous, updated, self.state.get(step.point, {})
            )
        return loss if has_closure else result

    def _manifest(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return {
            "version": 3,
            "optimizer_type": f"{type(self.base_optimizer).__module__}.{type(self.base_optimizer).__qualname__}",
            "vector_keys": self.state_adapter.vector_keys,
            "state_adapter_type": f"{type(self.state_adapter).__module__}.{type(self.state_adapter).__qualname__}",
            "state_specs": {k: v.atom_axis for k, v in self._state_specs.items()},
            "groups": [
                [(names[id(p)], tuple(p.shape)) for p in group["params"]]
                for group in self.param_groups
            ],
            "sites": [
                (
                    names[id(site.atoms.p)],
                    asdict(site.kernel.declaration()),
                    tuple(site.atom_state.row_to_id.cpu().tolist()),
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
        bind_atom_states(
            self.base_optimizer, self._sites, self._state_specs, restored=True
        )
        self._sync_from_base()
