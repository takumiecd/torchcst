"""Model-level coordinator for CST and dense optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torchcst._runtime.validation import device_checks, require
from torchcst.nn import CSTLinear

from .atom_grad import ImplicitLinearAtomGrad
from .config import (
    AdamWConfig,
    DenseVisibleAdamConfig,
    FirstOrderAdamConfig,
    LocalAdamConfig,
    LocalVisibleAdamConfig,
    SecondOrderAdamConfig,
)
from .dense import DenseAdamWProposal, FunctionalAdamW
from .moments import (
    ExpandedMoments,
    MomentContext,
    MomentSystem,
    MomentSystemState,
)
from .solvers import QuarticSolveResult


@dataclass(frozen=True)
class CSTStepResult:
    """Diagnostics from the most recent optimizer step."""

    site_results: tuple[QuarticSolveResult, ...]
    device_valid: Tensor | None = None
    compression_results: tuple = ()


@dataclass
class _CSTSite:
    name: str
    module: CSTLinear
    moments: MomentSystem
    atom_grad: ImplicitLinearAtomGrad
    state: MomentSystemState


@dataclass(frozen=True)
class _CSTProposal:
    site: _CSTSite
    context: MomentContext
    expanded: ExpandedMoments
    solve: QuarticSolveResult


class _ModelOptimizer(Optimizer):
    """Own and update every trainable parameter in one coordinated step."""

    _STATE_VERSION = 2

    def __init__(
        self,
        model: nn.Module,
        *,
        cst: FirstOrderAdamConfig
        | DenseVisibleAdamConfig
        | LocalAdamConfig
        | LocalVisibleAdamConfig
        | SecondOrderAdamConfig
        | None = None,
        dense: AdamWConfig | None = None,
        strict: bool = True,
        **options,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if cst is not None and options:
            raise TypeError("pass either cst config or optimizer options, not both")
        if cst is None:
            cst = self.config_type(**options)
        if not isinstance(cst, self.config_type):
            raise TypeError(f"cst must be a {self.config_type.__name__}")
        if dense is not None and not isinstance(dense, AdamWConfig):
            raise TypeError("dense must be an AdamWConfig or None")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool")
        self.model = model
        self.cst_config = cst
        self.dense_config = dense
        self.strict = strict
        self._stepping = False
        self.last_step: CSTStepResult | None = None
        self._device_valid: Tensor | None = None

        site_modules = [
            (name or "<root>", module)
            for name, module in model.named_modules()
            if isinstance(module, CSTLinear)
        ]
        if not site_modules:
            raise ValueError("model does not contain a CSTLinear site")

        named_parameters = list(model.named_parameters())
        trainable = [
            (name, value) for name, value in named_parameters if value.requires_grad
        ]
        if cst.device_execution and len({value.device for _, value in trainable}) != 1:
            raise ValueError(
                "device_execution currently requires all parameters on one device"
            )
        aliases: dict[int, list[str]] = {}
        for name, value in model.named_parameters(remove_duplicate=False):
            if value.requires_grad:
                aliases.setdefault(id(value), []).append(name)
        if strict:
            shared = [names for names in aliases.values() if len(names) > 1]
            if shared:
                names = ", ".join("/".join(group) for group in shared)
                raise ValueError(f"strict ownership rejects shared parameters: {names}")

        names_by_id = {id(value): name for name, value in trainable}
        cst_owner_by_id: dict[int, str] = {}
        for site_name, site in site_modules:
            if site.input_chart.trainable or site.output_chart.trainable:
                raise ValueError("CST optimizer supports frozen charts only")
            for parameter in site.cst_parameters():
                if not parameter.requires_grad:
                    raise ValueError(f"CST parameter owned by {site_name} is frozen")
                identity = id(parameter)
                if identity in cst_owner_by_id:
                    raise ValueError(
                        "a trainable parameter is owned by multiple CST sites"
                    )
                if identity not in names_by_id:
                    raise ValueError("a CST-owned parameter is absent from the model")
                cst_owner_by_id[identity] = site_name

        dense_parameters = [
            value for _, value in trainable if id(value) not in cst_owner_by_id
        ]
        if dense is None and dense_parameters:
            dense_names = [names_by_id[id(value)] for value in dense_parameters]
            raise ValueError(
                "dense=None cannot own dense trainable parameters: "
                + ", ".join(dense_names)
            )
        for site_name, site in site_modules:
            if site.atoms.grad is not None:
                raise ValueError(f"CST site {site_name} already has an AtomGrad owner")

        super().__init__((value for _, value in trainable), defaults={})
        self._parameter_names = names_by_id
        self._trainable_parameters = tuple(value for _, value in trainable)
        self._dense_parameters = tuple(dense_parameters)
        self._dense_engine = FunctionalAdamW(dense) if dense is not None else None
        self._dense_states = {
            parameter: self._dense_engine.initialize(parameter)
            for parameter in self._dense_parameters
        }

        sites: list[_CSTSite] = []
        for site_name, site in site_modules:
            moments = self._make_moments()
            geometry = self._make_geometry(site)
            context = MomentContext(geometry, geometry.current_point())
            atom_grad = ImplicitLinearAtomGrad(
                mode=cst.atom_grad_mode,
                row_chunk_size=cst.row_chunk_size,
                request=moments.observation_request,
                factored=cst.factored_geometry,
            )
            site.atoms.set_grad(atom_grad)
            sites.append(
                _CSTSite(
                    name=site_name,
                    module=site,
                    moments=moments,
                    atom_grad=atom_grad,
                    state=moments.initialize(context),
                )
            )
        self._sites = tuple(sites)
        self._manifest = self._make_manifest(cst_owner_by_id)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients and begin the next CST observation scope."""

        if self._stepping:
            raise RuntimeError("CST optimizer zero_grad cannot run during step")
        self._abort_capture()
        super().zero_grad(set_to_none=set_to_none)
        self._begin_capture()

    def step(self) -> None:
        """Consume one backward pass and commit its model-wide update."""

        if self._stepping:
            raise RuntimeError("CST optimizer step cannot be called recursively")
        self._stepping = True
        try:
            if self.cst_config.device_execution:
                with device_checks() as checks:
                    self._step(checks)
            else:
                self._step()
        finally:
            self._stepping = False

    def _step(self, checks=None) -> None:
        try:
            self._complete_capture()
        except BaseException:
            self._abort_capture()
            raise

        try:
            cst_proposals = self._build_cst_proposals()
            dense_proposals = self._build_dense_proposals()
            next_cst_states = tuple(
                proposal.site.moments.compress(
                    proposal.expanded,
                    proposal.solve.displacement,
                    proposal.context,
                )
                for proposal in cst_proposals
            )
        except BaseException:
            self._abort_capture()
            raise

        valid = None
        if checks is not None:
            for proposal in dense_proposals.values():
                require(
                    torch.isfinite(proposal.displacement).all(),
                    "non-finite dense update",
                    FloatingPointError,
                )
            valid = torch.stack(checks).all()
            if self._device_valid is not None:
                valid = valid & self._device_valid
            self._device_valid = valid
            next_cst_states = tuple(
                _select_device_state(valid, new, proposal.site.state)
                for proposal, new in zip(cst_proposals, next_cst_states)
            )
        with torch.no_grad():
            for proposal in cst_proposals:
                delta = proposal.solve.displacement
                proposal.site.module.atoms.p.add_(
                    delta if valid is None else torch.where(valid, delta, 0)
                )
            for parameter, proposal in dense_proposals.items():
                delta = proposal.displacement
                parameter.add_(delta if valid is None else torch.where(valid, delta, 0))
        for proposal, state in zip(cst_proposals, next_cst_states):
            proposal.site.state = state
        for parameter, proposal in dense_proposals.items():
            self._dense_states[parameter] = (
                proposal.pending_state
                if valid is None
                else _select_device_state(
                    valid, proposal.pending_state, self._dense_states[parameter]
                )
            )

        self.last_step = CSTStepResult(
            site_results=tuple(proposal.solve for proposal in cst_proposals),
            device_valid=valid,
            compression_results=tuple(
                getattr(proposal.context.geometry, "compression_result", None)
                for proposal in cst_proposals
            ),
        )

    def check_errors(self) -> None:
        """Explicit synchronization boundary for deferred errors.

        Failure latches the optimizer: all subsequent parameter/tensor-state
        commits are suppressed. Restore a valid checkpoint into a new optimizer
        before resuming; this method never clears the latch.
        """
        if self._device_valid is not None and not bool(self._device_valid):
            raise FloatingPointError(
                "deferred CST validation failed; updates are disabled"
            )

    def _begin_capture(self) -> None:
        begun = []
        try:
            for site in self._sites:
                site.atom_grad.begin()
                begun.append(site.atom_grad)
        except BaseException:
            for atom_grad in begun:
                if atom_grad.active:
                    atom_grad.cancel()
            raise

    def _complete_capture(self) -> None:
        for site in self._sites:
            site.atom_grad.complete()

    def _abort_capture(self) -> None:
        for site in self._sites:
            if site.atom_grad.active:
                site.atom_grad.cancel()
            else:
                site.atom_grad.clear()

    @staticmethod
    def _pcg_options(config):
        from torchcst._derivatives._pcg import PCGOptions

        return PCGOptions(
            config.gram_iterations, config.gram_rtol, config.gram_block_size
        )

    def _build_cst_proposals(self) -> tuple[_CSTProposal, ...]:
        proposals = []
        for site in self._sites:
            geometry = self._make_geometry(site.module)
            context = MomentContext(geometry, geometry.current_point())
            expanded = site.moments.expand(
                site.state,
                site.atom_grad.snapshot(),
                context,
            )
            solve = self._solve(context, expanded)
            self._validate_solve(solve, context)
            proposals.append(_CSTProposal(site, context, expanded, solve))
        return tuple(proposals)

    def _validate_solve(
        self,
        solve: QuarticSolveResult,
        context: MomentContext,
    ) -> None:
        self._validate_displacement(solve, context)
        displacement = solve.displacement
        norm = torch.linalg.vector_norm(displacement)
        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        require(
            norm <= self.cst_config.trust_radius * (1.0 + tolerance),
            "CST displacement exceeds the trust radius",
        )

    def _validate_displacement(self, solve, context):
        if not isinstance(solve, QuarticSolveResult):
            raise TypeError("CST solver must return a QuarticSolveResult")
        displacement = solve.displacement
        if displacement.shape != context.geometry.point_shape:
            raise ValueError("CST displacement has the wrong shape")
        point = context.current_point
        if displacement.device != point.device or displacement.dtype != point.dtype:
            raise ValueError("CST displacement must match its CST point")
        require(
            torch.isfinite(displacement).all(),
            "CST displacement must be finite",
            FloatingPointError,
        )

    def _build_dense_proposals(self) -> dict[nn.Parameter, DenseAdamWProposal]:
        if self._dense_engine is None:
            return {}
        return {
            parameter: self._dense_engine.expand(
                parameter, self._dense_states[parameter]
            )
            for parameter in self._dense_parameters
        }

    def _make_manifest(
        self, cst_owner_by_id: dict[int, str]
    ) -> tuple[dict[str, Any], ...]:
        manifest = []
        for parameter in self._trainable_parameters:
            identity = id(parameter)
            site_name = cst_owner_by_id.get(identity)
            manifest.append(
                {
                    "name": self._parameter_names[identity],
                    "shape": tuple(parameter.shape),
                    "owner": f"cst:{site_name}" if site_name else "dense",
                }
            )
        return tuple(manifest)

    def _moment_contract(self):
        c = self.cst_config
        return {
            "betas": c.betas,
            "eps": c.eps,
            "second_moment": c.second_moment,
            "first_moment_damping": c.first_moment_damping,
            "tangent_rtol": getattr(c, "tangent_rtol", None),
        }

    @classmethod
    def _validate_state_shape(cls, value, template):
        if type(value) is not type(template):
            raise TypeError("optimizer state component has the wrong type")
        if isinstance(value, Tensor):
            if value.shape != template.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("optimizer state tensor shape or values are invalid")
        elif is_dataclass(value):
            for field in fields(value):
                cls._validate_state_shape(
                    getattr(value, field.name), getattr(template, field.name)
                )

    def state_dict(self) -> dict[str, Any]:
        """Return compact optimizer state plus its exact ownership manifest."""

        # Saving is an explicit host boundary: do not serialize a latched,
        # partially advanced metadata state as a resumable checkpoint.
        self.check_errors()
        return {
            "version": self._STATE_VERSION,
            "algorithm": type(self).__name__,
            "moment_contract": self._moment_contract(),
            "manifest": copy.deepcopy(self._manifest),
            "cst": {
                site.name: self._map_state(site.state, clone=True)
                for site in self._sites
            },
            "dense": {
                self._parameter_names[id(parameter)]: self._map_state(state, clone=True)
                for parameter, state in self._dense_states.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state only when parameter names, shapes, and owners all match."""

        if not isinstance(state_dict, dict):
            raise TypeError("optimizer state_dict must be a dictionary")
        if state_dict.get("version") != self._STATE_VERSION:
            raise ValueError("unsupported CST optimizer state version")
        if tuple(state_dict.get("manifest", ())) != self._manifest:
            raise ValueError("optimizer state manifest does not match the model")
        if state_dict.get("algorithm") != type(self).__name__:
            raise ValueError("optimizer algorithm does not match the checkpoint")
        if state_dict.get("moment_contract") != self._moment_contract():
            raise ValueError("optimizer moment contract does not match the checkpoint")
        cst_values = state_dict.get("cst")
        dense_values = state_dict.get("dense")
        if not isinstance(cst_values, dict) or not isinstance(dense_values, dict):
            raise TypeError("optimizer state CST and dense blocks must be dictionaries")
        if set(cst_values) != {site.name for site in self._sites}:
            raise ValueError("optimizer CST state sites do not match the model")
        expected_dense = {
            self._parameter_names[id(parameter)] for parameter in self._dense_parameters
        }
        if set(dense_values) != expected_dense:
            raise ValueError("optimizer dense state parameters do not match the model")

        next_cst = []
        for site in self._sites:
            value = self._map_state(
                cst_values[site.name],
                reference=site.module.atoms.p,
            )
            if not isinstance(value, type(site.state)):
                raise TypeError("loaded CST moment state has the wrong type")
            self._validate_state_shape(value, site.state)
            next_cst.append(value)
        next_dense = {}
        by_name = {
            self._parameter_names[id(parameter)]: parameter
            for parameter in self._dense_parameters
        }
        for name, parameter in by_name.items():
            value = self._map_state(dense_values[name], reference=parameter)
            FunctionalAdamW._validate_state(parameter, value)
            next_dense[parameter] = value

        for site, value in zip(self._sites, next_cst):
            site.state = value
        self._dense_states = next_dense

    @classmethod
    def _map_state(
        cls,
        value: Any,
        *,
        clone: bool = False,
        reference: Tensor | None = None,
    ) -> Any:
        if isinstance(value, Tensor):
            if reference is not None:
                return (
                    value.detach()
                    .to(
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                    .clone()
                )
            return value.detach().clone() if clone else value
        if is_dataclass(value) and not isinstance(value, type):
            return type(value)(
                **{
                    field.name: cls._map_state(
                        getattr(value, field.name),
                        clone=clone,
                        reference=reference,
                    )
                    for field in fields(value)
                }
            )
        if isinstance(value, dict):
            return {
                key: cls._map_state(item, clone=clone, reference=reference)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                cls._map_state(item, clone=clone, reference=reference) for item in value
            )
        if isinstance(value, list):
            return [
                cls._map_state(item, clone=clone, reference=reference) for item in value
            ]
        return copy.deepcopy(value) if clone else value


def _select_device_state(valid, new, previous):
    if isinstance(new, Tensor):
        return torch.where(valid, new, previous)
    if is_dataclass(new):
        return type(new)(
            **{
                f.name: _select_device_state(
                    valid, getattr(new, f.name), getattr(previous, f.name)
                )
                for f in fields(new)
            }
        )
    # Counters/configuration are host metadata. After a failure the latch keeps
    # every tensor frozen; the optimizer cannot resume without restoration.
    return new
