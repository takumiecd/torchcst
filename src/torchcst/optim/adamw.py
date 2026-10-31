"""One PyTorch optimizer lifecycle for AdamW proposals and CST retractions."""

from __future__ import annotations

import math
import re
from dataclasses import asdict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim._functional import adamw

from torchcst.nn import CSTModule

from ._coordinate_step import finish_atom_steps, prepare_atom_steps
from .binding import bind_atom_states, field_specs, validate_binding
from .state import OptimizerStateAdapter


class CSTAdamW(AdamW):
    """AdamW followed by the Kernel's coordinate update, in a single step.

    This optimizer owns its groups, state and hooks. Pre-hooks run before both
    operations; post-hooks see the final retracted parameters. Managed atom
    groups use raw weight_decay=0 and logical amplitude_weight_decay instead.
    Ordinary groups retain AdamW defaults and may override weight_decay.
    Nonfinite propagation follows PyTorch unless check_finite=True is requested.
    """

    def __init__(
        self,
        params,
        *,
        model: nn.Module,
        amplitude_weight_decay: float = 0.0,
        state_adapter: OptimizerStateAdapter | None = None,
        state_specs=None,
        update_selector=None,
        check_finite: bool = False,
        **adamw_options,
    ):
        version = re.match(r"(\d+)\.(\d+)", torch.__version__)
        if version is None or tuple(map(int, version.groups())) < (2, 6):
            raise RuntimeError("CSTAdamW requires PyTorch 2.6 or newer")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if type(check_finite) is not bool:
            raise TypeError("check_finite must be a bool")
        if state_adapter is not None and not isinstance(
            state_adapter, OptimizerStateAdapter
        ):
            raise TypeError("state_adapter must be an OptimizerStateAdapter")
        from torchcst._backends.dispatch import Selector

        if update_selector is not None and not isinstance(update_selector, Selector):
            raise TypeError("update_selector must be a Selector")
        self._validate_decay(amplitude_weight_decay)
        self.model = model
        self.check_finite = check_finite
        self._default_amplitude_decay = amplitude_weight_decay
        self._sites = tuple(m for m in model.modules() if isinstance(m, CSTModule))
        self._bound_parameters = {site: site.atoms.p for site in self._sites}
        self.state_adapter = state_adapter or OptimizerStateAdapter(("exp_avg",))
        self._coordinate_ready = False
        self._default_atom_raw_decay = adamw_options.get("weight_decay", 0.0)
        if isinstance(params, torch.Tensor):
            raise TypeError("params must be an iterable of Tensors or parameter groups")
        parameter_list = list(params)
        if not parameter_list:
            raise ValueError("optimizer got an empty parameter list")
        groups = (
            parameter_list
            if parameter_list and isinstance(parameter_list[0], dict)
            else [{"params": parameter_list}]
        )
        for group in groups:
            values = group["params"]
            group["params"] = (
                [values] if isinstance(values, torch.Tensor) else list(values)
            )
            self._set_atom_defaults(group)
        super().__init__(groups, **adamw_options)
        # GradScaler must skip the entire composite step on overflow. Fused
        # AdamW's internal found_inf alone would not skip the Polar retraction.
        # Use GradScaler's standard unscale/skip route for this optimizer.
        self._step_supports_amp_scaling = False
        self._state_specs = field_specs(
            self, ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"), state_specs
        )
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
        for group in self.param_groups:
            self._set_atom_defaults(group)
        self._refresh_structure()
        self._coordinate_ready = True
        # Internal hooks are installed first. Standard user post-hooks therefore
        # observe the complete CST payload and rebound atom-owned dictionaries.
        self.register_state_dict_post_hook(self._save_cst_metadata)
        self.register_load_state_dict_pre_hook(self._validate_cst_checkpoint)
        self.register_load_state_dict_post_hook(self._restore_cst_state)

    @staticmethod
    def _validate_decay(decay):
        if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
            raise ValueError(
                "amplitude_weight_decay must be a finite nonnegative Python scalar"
            )

    def _set_atom_defaults(self, group):
        atom_ids = {id(p) for p in self._bound_parameters.values()}
        points = (p[1] if isinstance(p, tuple) else p for p in group["params"])
        if any(id(p) in atom_ids for p in points):
            group.setdefault("weight_decay", self._default_atom_raw_decay)
            group.setdefault("amplitude_weight_decay", self._default_amplitude_decay)

    def _groups_for_step(self, groups=None):
        """Read current group scalars; never walk the whole model per step."""
        groups = self.param_groups if groups is None else groups
        atom_ids = {id(p) for p in self._bound_parameters.values()}
        mapping = {}
        seen = set()
        for group in groups:
            if group.get("differentiable", self.defaults["differentiable"]):
                raise ValueError(
                    "CSTAdamW does not support differentiable optimizer steps"
                )
            has_atoms = False
            for point in group["params"]:
                key = id(point)
                if key not in self._model_parameter_ids:
                    raise ValueError("optimizer parameter is not owned by model")
                if key in seen:
                    raise ValueError("optimizer parameters must be unique")
                seen.add(key)
                if key in atom_ids:
                    has_atoms = True
                    mapping[key] = group
            if "amplitude_weight_decay" in group:
                if not has_atoms:
                    raise ValueError("amplitude_weight_decay requires an atom group")
                decay = group["amplitude_weight_decay"]
                self._validate_decay(decay)
                rate = group.get("lr", self.defaults["lr"])
                if (
                    type(rate) not in (int, float)
                    or not math.isfinite(rate)
                    or rate < 0
                    or rate * decay > 1
                ):
                    raise ValueError(
                        "lr * amplitude_weight_decay must be in [0, 1] with finite Python lr"
                    )
            if has_atoms:
                raw_decay = group.get("weight_decay", self.defaults["weight_decay"])
                if type(raw_decay) not in (int, float) or raw_decay != 0:
                    raise ValueError(
                        "managed atoms require raw weight_decay=0; use amplitude_weight_decay"
                    )
        return mapping

    def _refresh_structure(self, *, restored=False):
        """Topology validation belongs to construction/add-group/load boundaries."""
        from torchcst.geometry.spec import EuclideanGeometrySpec

        points = list(self._bound_parameters.values())
        if len({id(p) for p in points}) != len(points):
            raise ValueError("an atom table cannot have multiple CST policy owners")
        for site in self._sites:
            for chart in site.cst_charts():
                if (
                    chart.trainable
                    and type(chart.geometry.spec) is not EuclideanGeometrySpec
                ):
                    raise ValueError(
                        "trainable non-Euclidean charts need a chart update policy"
                    )
        self._model_parameter_ids = {id(p) for p in self.model.parameters()}
        self._groups_for_step()
        bind_atom_states(self, self._sites, self._state_specs, restored=restored)

    def add_param_group(self, param_group):
        if not getattr(self, "_coordinate_ready", False):
            return super().add_param_group(param_group)
        group = dict(param_group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, torch.Tensor) else list(params)
        self._set_atom_defaults(group)
        # Validate before mutating PyTorch's actual group table.
        self._model_parameter_ids = {id(p) for p in self.model.parameters()}
        self._groups_for_step([*self.param_groups, group])
        validate_binding(
            self, self._sites, self._state_specs, groups=[*self.param_groups, group]
        )
        super().add_param_group(group)
        self._refresh_structure()

    @torch.no_grad()
    def step(self, closure=None):
        """CST preparation -> PyTorch functional AdamW -> CST retraction."""
        from torchcst._backends.torch import finite

        health_check = getattr(self, "_accelerator_graph_capture_health_check", None)
        if health_check is None:
            health_check = self._cuda_graph_capture_health_check
        health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with finite.checking(self.check_finite):
            groups = self._groups_for_step()
            prepared = prepare_atom_steps(self, groups, param_groups=self.param_groups)
            self._adamw_proposal()
            finish_atom_steps(self, prepared)
        return loss

    def _adamw_proposal(self):
        """Keep PyTorch's lazy state creation and AdamW numerical implementation.

        _init_group is the small version-dependent seam inherited from AdamW.
        No wrapped parent step is called: that would run optimizer hooks twice.
        """
        for group in self.param_groups:
            params, grads, means, squares, maxima, clocks = [], [], [], [], [], []
            has_complex = self._init_group(
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

    def _manifest(self):
        names = {id(p): name for name, p in self.model.named_parameters()}
        return {
            "version": 3,
            "optimizer_type": f"{type(self).__module__}.{type(self).__qualname__}",
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

    def _save_cst_metadata(self, optimizer, state_dict):
        state_dict["cst"] = self._manifest()

    def _validate_cst_checkpoint(self, optimizer, state_dict):
        if state_dict.get("cst") != self._manifest():
            raise ValueError("CST optimizer checkpoint contract differs")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, self.param_groups)
        ):
            raise ValueError("optimizer parameter groups differ from checkpoint")
        prospective = []
        shape_keys = {
            *self.state_adapter.vector_keys,
            "exp_avg",
            "exp_avg_sq",
            "max_exp_avg_sq",
        }
        for saved, current in zip(saved_groups, self.param_groups):
            prospective.append(dict(saved, params=current["params"]))
            for index, point in zip(saved["params"], current["params"]):
                for key, value in state_dict["state"].get(index, {}).items():
                    if (
                        isinstance(value, torch.Tensor)
                        and key in shape_keys
                        and value.shape != point.shape
                    ):
                        raise ValueError(f"optimizer state {key!r} has the wrong shape")
        self._groups_for_step(prospective)

    def _restore_cst_state(self, optimizer):
        self._refresh_structure(restored=True)

    def load_legacy_state_dict(self, state_dict):
        """Explicitly migrate a matching CSTOptimizer(exact AdamW) checkpoint."""
        expected = self._manifest()
        legacy = dict(expected, optimizer_type="torch.optim.adamw.AdamW")
        if state_dict.get("cst") != legacy:
            raise ValueError("legacy AdamW checkpoint contract differs")
        self.load_state_dict(dict(state_dict, cst=expected))
