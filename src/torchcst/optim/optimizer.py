"""Model-level coordinator for transactional CST and dense optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torchcst.nn import CSTLinear

from .acceptance import AcceptancePolicy, AcceptanceResult, ExactLossAcceptance
from .atom_grad import ImplicitLinearAtomGrad
from .config import AdamWConfig, ImplicitAdamConfig
from .dense import DenseAdamWProposal, FunctionalAdamW
from .moments import (
    AcceptedFrameFirstMoment,
    ExpandedMoments,
    MomentContext,
    MomentSystem,
    MomentSystemState,
    SeparableDiagonalSecondMoment,
)
from .problem import QuarticProblem
from .solvers import QuarticSolveResult


@dataclass(frozen=True)
class CSTStepResult:
    """Diagnostics from the most recent joint optimizer transaction."""

    accepted: bool
    scale: float
    base_loss: Tensor
    loss: Tensor
    acceptance_trials: int
    site_results: tuple[QuarticSolveResult, ...]


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


class CSTOptimizer(Optimizer):
    """Own and update every trainable parameter in a model as one transaction."""

    _STATE_VERSION = 1

    def __init__(
        self,
        model: nn.Module,
        *,
        cst: ImplicitAdamConfig,
        dense: AdamWConfig | None = None,
        acceptance: AcceptancePolicy | None = None,
        strict: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(cst, ImplicitAdamConfig):
            raise TypeError("cst must be an ImplicitAdamConfig")
        if dense is not None and not isinstance(dense, AdamWConfig):
            raise TypeError("dense must be an AdamWConfig or None")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool")
        selected_acceptance = (
            ExactLossAcceptance() if acceptance is None else acceptance
        )
        if not isinstance(selected_acceptance, AcceptancePolicy):
            raise TypeError("acceptance must be an AcceptancePolicy")

        self.model = model
        self.cst_config = cst
        self.dense_config = dense
        self.acceptance = selected_acceptance
        self.strict = strict
        self._stepping = False
        self.last_step: CSTStepResult | None = None

        site_modules = [
            (name or "<root>", module)
            for name, module in model.named_modules()
            if isinstance(module, CSTLinear)
        ]
        if not site_modules:
            raise ValueError("model does not contain a CSTLinear site")

        named_parameters = list(model.named_parameters())
        trainable = [(name, value) for name, value in named_parameters if value.requires_grad]
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
                raise ValueError("CSTOptimizer supports frozen charts only")
            for parameter in site.cst_parameters():
                if not parameter.requires_grad:
                    raise ValueError(f"CST parameter owned by {site_name} is frozen")
                identity = id(parameter)
                if identity in cst_owner_by_id:
                    raise ValueError("a trainable parameter is owned by multiple CST sites")
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
            moments = MomentSystem(
                first=AcceptedFrameFirstMoment(
                    cst.betas[0], damping=cst.first_moment_damping
                ),
                second=SeparableDiagonalSecondMoment(
                    cst.betas[1], eps=cst.eps
                ),
            )
            geometry = site.cst_frame_geometry()
            context = MomentContext(geometry, geometry.current_point())
            atom_grad = ImplicitLinearAtomGrad(
                mode=cst.atom_grad_mode,
                row_chunk_size=cst.row_chunk_size,
                request=moments.observation_request,
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

    def step(self, closure: Any = None) -> Tensor:
        """Build, test, and atomically commit one model-wide proposal."""

        if closure is None or not callable(closure):
            raise TypeError("CSTOptimizer.step requires a callable closure")
        if self._stepping:
            raise RuntimeError("CSTOptimizer.step cannot be called recursively")
        self._stepping = True
        try:
            return self._step(closure)
        finally:
            self._stepping = False

    def _step(self, closure: Any) -> Tensor:
        self._begin_capture()
        try:
            with torch.enable_grad():
                base_loss = closure()
            self._complete_capture()
            base_loss = self._scalar_loss(base_loss, name="base loss")
            if not torch.isfinite(base_loss):
                raise FloatingPointError("base loss must be finite")
        except BaseException:
            self._abort_capture()
            raise

        cst_proposals = self._build_cst_proposals()
        dense_proposals = self._build_dense_proposals()
        base_values = {
            parameter: parameter.detach().clone()
            for parameter in self._trainable_parameters
        }

        def evaluate(scale: float) -> Tensor:
            if not 0.0 < scale <= 1.0:
                raise ValueError("acceptance candidate scale must satisfy 0 < scale <= 1")
            self._set_candidate(
                base_values,
                cst_proposals,
                dense_proposals,
                scale=scale,
            )
            with torch.enable_grad():
                return closure()

        try:
            acceptance = self.acceptance.select(base_loss, evaluate)
            self._validate_acceptance(acceptance)
            if acceptance.accepted:
                next_cst_states = tuple(
                    proposal.site.moments.compress(
                        proposal.expanded,
                        acceptance.scale * proposal.solve.displacement,
                        proposal.context,
                    )
                    for proposal in cst_proposals
                )
                self._set_candidate(
                    base_values,
                    cst_proposals,
                    dense_proposals,
                    scale=acceptance.scale,
                )
                for proposal, state in zip(cst_proposals, next_cst_states):
                    proposal.site.state = state
                for parameter, proposal in dense_proposals.items():
                    self._dense_states[parameter] = proposal.pending_state
            else:
                for proposal in cst_proposals:
                    proposal.site.moments.reject(
                        proposal.expanded, proposal.site.state
                    )
                self._restore(base_values)
                self.zero_grad(set_to_none=True)
        except BaseException:
            self._restore(base_values)
            self.zero_grad(set_to_none=True)
            raise

        self.last_step = CSTStepResult(
            accepted=acceptance.accepted,
            scale=acceptance.scale,
            base_loss=base_loss.clone(),
            loss=self._scalar_loss(acceptance.loss, name="accepted loss"),
            acceptance_trials=acceptance.trials,
            site_results=tuple(proposal.solve for proposal in cst_proposals),
        )
        return self.last_step.loss.clone()

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

    def _build_cst_proposals(self) -> tuple[_CSTProposal, ...]:
        proposals = []
        for site in self._sites:
            geometry = site.module.cst_frame_geometry()
            context = MomentContext(geometry, geometry.current_point())
            expanded = site.moments.expand(
                site.state,
                site.atom_grad.snapshot(),
                context,
            )
            problem = QuarticProblem(
                context,
                expanded,
                learning_rate=self.cst_config.lr,
            )
            solve = self.cst_config.quartic.solve(
                problem,
                trust_radius=self.cst_config.trust_radius,
            )
            self._validate_solve(solve, context)
            proposals.append(_CSTProposal(site, context, expanded, solve))
        return tuple(proposals)

    def _validate_solve(
        self,
        solve: QuarticSolveResult,
        context: MomentContext,
    ) -> None:
        if not isinstance(solve, QuarticSolveResult):
            raise TypeError("quartic solver must return a QuarticSolveResult")
        displacement = solve.displacement
        if displacement.shape != context.geometry.point_shape:
            raise ValueError("quartic displacement has the wrong shape")
        point = context.current_point
        if displacement.device != point.device or displacement.dtype != point.dtype:
            raise ValueError("quartic displacement must match its CST point")
        if not torch.isfinite(displacement).all():
            raise FloatingPointError("quartic displacement must be finite")
        norm = torch.linalg.vector_norm(displacement)
        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        if norm > self.cst_config.trust_radius * (1.0 + tolerance):
            raise ValueError("quartic displacement exceeds the trust radius")

    def _build_dense_proposals(self) -> dict[nn.Parameter, DenseAdamWProposal]:
        if self._dense_engine is None:
            return {}
        return {
            parameter: self._dense_engine.expand(
                parameter, self._dense_states[parameter]
            )
            for parameter in self._dense_parameters
        }

    @staticmethod
    def _restore(base_values: dict[nn.Parameter, Tensor]) -> None:
        with torch.no_grad():
            for parameter, value in base_values.items():
                parameter.copy_(value)

    @staticmethod
    def _scalar_loss(value: Tensor, *, name: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return value.detach().reshape(()).clone()

    @classmethod
    def _validate_acceptance(cls, result: AcceptanceResult) -> None:
        if not isinstance(result, AcceptanceResult):
            raise TypeError("acceptance policy must return an AcceptanceResult")
        if not isinstance(result.accepted, bool):
            raise TypeError("acceptance decision must be a bool")
        if isinstance(result.trials, bool) or not isinstance(result.trials, int):
            raise TypeError("acceptance trials must be an integer")
        if result.trials < 1:
            raise ValueError("acceptance trials must be positive")
        loss = cls._scalar_loss(result.loss, name="accepted loss")
        if not torch.isfinite(loss):
            raise FloatingPointError("accepted loss must be finite")
        if result.accepted:
            if not 0.0 < result.scale <= 1.0:
                raise ValueError("accepted scale must satisfy 0 < scale <= 1")
        elif result.scale != 0.0:
            raise ValueError("rejected acceptance result must have scale zero")

    def _set_candidate(
        self,
        base_values: dict[nn.Parameter, Tensor],
        cst_proposals: tuple[_CSTProposal, ...],
        dense_proposals: dict[nn.Parameter, DenseAdamWProposal],
        *,
        scale: float,
    ) -> None:
        self._restore(base_values)
        with torch.no_grad():
            for proposal in cst_proposals:
                parameter = proposal.site.module.atoms.p
                parameter.add_(proposal.solve.displacement, alpha=scale)
            for parameter, proposal in dense_proposals.items():
                parameter.add_(proposal.displacement, alpha=scale)

    def _make_manifest(self, cst_owner_by_id: dict[int, str]) -> tuple[dict[str, Any], ...]:
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

    def state_dict(self) -> dict[str, Any]:
        """Return compact optimizer state plus its exact ownership manifest."""

        return {
            "version": self._STATE_VERSION,
            "manifest": copy.deepcopy(self._manifest),
            "cst": {
                site.name: self._map_state(site.state, clone=True)
                for site in self._sites
            },
            "dense": {
                self._parameter_names[id(parameter)]: self._map_state(
                    state, clone=True
                )
                for parameter, state in self._dense_states.items()
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state only when parameter names, shapes, and owners all match."""

        if not isinstance(state_dict, dict):
            raise TypeError("optimizer state_dict must be a dictionary")
        if state_dict.get("version") != self._STATE_VERSION:
            raise ValueError("unsupported CSTOptimizer state version")
        if tuple(state_dict.get("manifest", ())) != self._manifest:
            raise ValueError("optimizer state manifest does not match the model")
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
                return value.detach().to(
                    device=reference.device,
                    dtype=reference.dtype,
                ).clone()
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
                cls._map_state(item, clone=clone, reference=reference)
                for item in value
            )
        if isinstance(value, list):
            return [
                cls._map_state(item, clone=clone, reference=reference)
                for item in value
            ]
        return copy.deepcopy(value) if clone else value
