"""One pullback Adam for every trainable CST representation parameter.

The optimizer owns the CST half of a model and deliberately leaves the
remainder to an ordinary PyTorch optimizer. Its two metric tiers never form
cross-atom terms: ``diag`` keeps only the diagonal of each same-atom Gram and
``block`` keeps the complete same-atom Gram. Gauges may declare exact zeros
inside that local Gram; absent such a declaration the complete block is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from ..representation import (
    GaussianFactor,
    L2NormalizedColumns,
    MaturityGaussianFactor,
)
from ..storage import NeuronStore, SynapseStore
from . import metric as metric_mod

MetricForm = Literal["diag", "block"]

__all__ = ["CSTPullbackAdam"]


def _gaussian_l2_side_terms(
    mu: Tensor, centers: Tensor, sigma: Tensor
) -> tuple[Tensor, Tensor]:
    """Gaussian coordinate Gram and sigma norm in one fused column pass."""
    displacement = centers[None, :, :] - mu[:, None, :]
    squared_distance = displacement.square().sum(-1)
    centered = squared_distance - squared_distance.amin(dim=0, keepdim=True)
    value = torch.exp(-centered / (2.0 * sigma.square()))
    unit = value / torch.linalg.vector_norm(value, dim=0, keepdim=True)
    derivative = (-unit / sigma.square())[:, :, None] * displacement
    radial = (unit[:, :, None] * derivative).sum(0, keepdim=True)
    tangent = derivative - unit[:, :, None] * radial
    gram = torch.einsum("nka,nkb->kab", tangent, tangent)
    gram.diagonal(dim1=-2, dim2=-1).clamp_min_(0)
    sigma_radial = (unit.square() * centered).sum(0, keepdim=True)
    sigma_tangent = unit * (centered - sigma_radial) / sigma.pow(3)
    sigma_sq = sigma_tangent.square().sum(0).clamp_min(0)
    return gram, sigma_sq


_compiled_gaussian_l2_side_terms = None


def _fast_gaussian_l2_side_terms(
    mu: Tensor, centers: Tensor, sigma: Tensor, *, compiled: bool
) -> tuple[Tensor, Tensor]:
    if compiled and centers.is_cuda:
        global _compiled_gaussian_l2_side_terms
        if _compiled_gaussian_l2_side_terms is None:
            _compiled_gaussian_l2_side_terms = torch.compile(
                _gaussian_l2_side_terms, fullgraph=True
            )
        return _compiled_gaussian_l2_side_terms(mu, centers, sigma)
    return _gaussian_l2_side_terms(mu, centers, sigma)


def _maturity_gaussian_l2_side_terms(
    mu: Tensor,
    centers: Tensor,
    sigma: Tensor,
    maturity: Tensor,
    min_scale: float,
    max_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Coordinate, sigma, and maturity terms for normalized Gaussians.

    The maturity column controls squared inverse width.  Returning its
    coordinate cross term is enough to assemble the joint same-atom Gram;
    normalization makes amplitude--maturity and opposite-side coordinate
    cross terms exactly zero.
    """
    displacement = centers[None, :, :] - mu[:, None, :]
    squared_distance = displacement.square().sum(-1)
    centered = squared_distance - squared_distance.amin(dim=0, keepdim=True)
    fraction = torch.sigmoid(maturity.reshape(1, -1))
    scale_range = max_scale**2 - min_scale**2
    scale_sq = min_scale**2 + scale_range * fraction
    value = torch.exp(-scale_sq * centered / (2.0 * sigma.square()))
    unit = value / torch.linalg.vector_norm(value, dim=0, keepdim=True)

    derivative = (
        -unit[:, :, None]
        * (scale_sq / sigma.square())[:, :, None]
        * displacement
    )
    radial = (unit[:, :, None] * derivative).sum(0, keepdim=True)
    tangent = derivative - unit[:, :, None] * radial
    gram = torch.einsum("nka,nkb->kab", tangent, tangent)
    gram.diagonal(dim1=-2, dim2=-1).clamp_min_(0)

    centered_mean = (unit.square() * centered).sum(0, keepdim=True)
    shape = centered - centered_mean
    sigma_tangent = unit * scale_sq * shape / sigma.pow(3)
    sigma_sq = sigma_tangent.square().sum(0).clamp_min(0)

    scale_slope = scale_range * fraction * (1.0 - fraction)
    maturity_tangent = (
        -unit * scale_slope * shape / (2.0 * sigma.square())
    )
    maturity_sq = maturity_tangent.square().sum(0).clamp_min(0)
    coordinate_maturity = torch.einsum(
        "nka,nk->ka", tangent, maturity_tangent
    )
    return gram, sigma_sq, maturity_sq, coordinate_maturity


_compiled_maturity_gaussian_l2_side_terms = None


def _fast_maturity_gaussian_l2_side_terms(
    mu: Tensor,
    centers: Tensor,
    sigma: Tensor,
    maturity: Tensor,
    min_scale: float,
    max_scale: float,
    *,
    compiled: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if compiled and centers.is_cuda:
        global _compiled_maturity_gaussian_l2_side_terms
        if _compiled_maturity_gaussian_l2_side_terms is None:
            _compiled_maturity_gaussian_l2_side_terms = torch.compile(
                _maturity_gaussian_l2_side_terms, fullgraph=True
            )
        return _compiled_maturity_gaussian_l2_side_terms(
            mu, centers, sigma, maturity, min_scale, max_scale
        )
    return _maturity_gaussian_l2_side_terms(
        mu, centers, sigma, maturity, min_scale, max_scale
    )


def _continuous_sites(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Discover continuous CST sites, rejecting shared-store aliases."""
    found: list[tuple[str, nn.Module]] = []
    claimed: dict[int, str] = {}
    for name, module in model.named_modules():
        store = getattr(module, "synapses", None)
        if not isinstance(store, SynapseStore):
            continue
        if getattr(module, "factor_in", None) is None or getattr(
            module, "factor_out", None
        ) is None:
            continue
        previous = claimed.get(id(store))
        if previous is not None:
            raise ValueError(
                f"synapse store {store.site!r} is reachable as both "
                f"{previous!r} and {name!r}; one store needs one owning site"
            )
        claimed[id(store)] = name
        found.append((name, module))
    return tuple(found)


def _inverse_sqrt(blocks: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(blocks)
    tiny = torch.finfo(blocks.dtype).tiny
    inverted = eigenvalues.clamp_min(tiny).rsqrt()
    return (eigenvectors * inverted[..., None, :]) @ eigenvectors.transpose(
        -1, -2
    )


def _adam(
    state: dict[str, Any], incoming: Tensor, betas: tuple[float, float], eps: float
) -> Tensor:
    beta1, beta2 = betas
    state["step"] += 1
    state["m"].mul_(beta1).add_(incoming, alpha=1.0 - beta1)
    state["v"].mul_(beta2).addcmul_(incoming, incoming, value=1.0 - beta2)
    if "mass1" in state:
        state["mass1"].mul_(beta1).add_(1.0 - beta1)
        state["mass2"].mul_(beta2).add_(1.0 - beta2)
        tiny = torch.finfo(incoming.dtype).tiny
        correction1 = state["mass1"].clamp_min(tiny)
        correction2 = state["mass2"].clamp_min(tiny)
    else:
        correction1 = 1.0 - beta1 ** state["step"]
        correction2 = 1.0 - beta2 ** state["step"]
    return (state["m"] / correction1) / (
        (state["v"] / correction2).sqrt() + eps
    )


@dataclass(frozen=True)
class _RowField:
    name: str
    parameter: nn.Parameter
    width: int
    start: int
    stop: int


@dataclass
class _AtomSite:
    name: str
    module: nn.Module
    store: SynapseStore
    fields: tuple[_RowField, ...]
    width: int
    state: dict[str, Any]
    follower: _RowsFollower | None = None


@dataclass
class _Chart:
    store: NeuronStore
    sites: list[nn.Module]
    state: dict[str, Any]
    follower: _RowsFollower | None = None


@dataclass
class _Sigma:
    parameter: nn.Parameter
    uses: list[tuple[nn.Module, str]]
    state: dict[str, Any]


class _RowsFollower:
    """Keep one optimizer state's slot rows aligned with a mutable store."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def grow(self, new_capacity: int) -> None:
        for name in ("m", "v", "mass1", "mass2"):
            old = self.state[name]
            if new_capacity < old.shape[0]:
                raise ValueError("optimizer state cannot shrink")
            if new_capacity == old.shape[0]:
                continue
            grown = old.new_zeros((new_capacity, *old.shape[1:]))
            grown[: old.shape[0]] = old
            self.state[name] = grown

    def _reset(self, slots: Tensor) -> None:
        device_slots = slots.to(self.state["m"].device)
        for name in ("m", "v", "mass1", "mass2"):
            self.state[name].index_fill_(0, device_slots, 0)

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None:
        del lineage
        self._reset(slots)

    def on_death(self, slots: Tensor) -> None:
        self._reset(slots)

    def on_refit(self, slots: Tensor) -> None:
        self._reset(slots)

    def on_remap(self, old_to_new: Tensor) -> None:
        for name in ("m", "v", "mass1", "mass2"):
            old = self.state[name]
            mapping = old_to_new.to(old.device)
            remapped = torch.zeros_like(old)
            source = torch.nonzero(mapping >= 0, as_tuple=False).flatten()
            remapped.index_copy_(0, mapping[source], old.index_select(0, source))
            self.state[name] = remapped


class CSTPullbackAdam(torch.optim.Optimizer):
    """Pullback Adam over every trainable CST parameter in ``model``.

    Both metric forms are block-Jacobi across atoms. ``diag`` retains only
    each same-atom Gram's diagonal; ``block`` retains its complete coupling.
    ``lr`` and ``target_map_step`` are mutually exclusive. The latter lazily
    calibrates the first median linearised map displacement. A non-``None``
    ``max_step_sigma`` additionally caps coordinate-like movement.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        metric: MetricForm = "block",
        lr: float | None = 1e-3,
        target_map_step: float | None = None,
        max_step_sigma: float | None = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        damping: float = 1e-2,
        metric_chunk_elements: int = 1 << 24,
        compile_metric: bool = False,
        subscribe: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if metric not in ("diag", "block"):
            raise ValueError("metric must be 'diag' or 'block'")
        if (lr is None) == (target_map_step is None):
            raise ValueError("set exactly one of lr or target_map_step")
        for value, name in ((lr, "lr"), (target_map_step, "target_map_step")):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive or None")
        if max_step_sigma is not None and (
            not isinstance(max_step_sigma, (int, float))
            or isinstance(max_step_sigma, bool)
            or max_step_sigma <= 0
        ):
            raise ValueError("max_step_sigma must be positive or None")
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or not all(isinstance(beta, (int, float)) for beta in betas)
            or not all(0 <= beta < 1 for beta in betas)
        ):
            raise ValueError("betas must be a pair in [0, 1)")
        for value, name in (
            (eps, "eps"),
            (damping, "damping"),
        ):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(subscribe, bool):
            raise TypeError("subscribe must be a bool")
        if (
            not isinstance(metric_chunk_elements, int)
            or isinstance(metric_chunk_elements, bool)
            or metric_chunk_elements < 1
        ):
            raise ValueError("metric_chunk_elements must be a positive integer")
        if not isinstance(compile_metric, bool):
            raise TypeError("compile_metric must be a bool")

        self.model = model
        self.metric = metric
        self.target_map_step = (
            None if target_map_step is None else float(target_map_step)
        )
        self.max_step_sigma = (
            None if max_step_sigma is None else float(max_step_sigma)
        )
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.damping = float(damping)
        self.metric_chunk_elements = int(metric_chunk_elements)
        self.compile_metric = compile_metric
        self._calibrated_scale: float | None = None

        names = {id(parameter): name for name, parameter in model.named_parameters()}
        self._names = names
        sites = _continuous_sites(model)
        if not sites:
            raise ValueError("model contains no continuous CST sites")
        owned: list[nn.Parameter] = []
        owned_ids: set[int] = set()

        def claim(parameter: object, description: str) -> None:
            if not isinstance(parameter, nn.Parameter) or not parameter.requires_grad:
                return
            if id(parameter) not in names:
                raise ValueError(f"{description} is not registered by model")
            if id(parameter) not in owned_ids:
                owned_ids.add(id(parameter))
                owned.append(parameter)

        atom_sites: list[_AtomSite] = []
        chart_users: dict[int, _Chart] = {}
        sigma_users: dict[int, _Sigma] = {}
        for site_name, module in sites:
            store = module.synapses
            if getattr(module, "input_offset_axes", 0):
                raise NotImplementedError(
                    f"site {site_name!r} has non-factor displacement axes; "
                    "their stencil Jacobian has no pullback block yet"
                )
            if parametrize.is_parametrized(store, "w"):
                raise ValueError(
                    f"{store.site!r}: remove the amplitude parametrization; "
                    "CSTPullbackAdam owns the delivered amplitude directly"
                )
            fields: list[_RowField] = []
            offset = 0
            for field_name in ("w", "s", "t", *store.atom_column_names):
                parameter = getattr(store, field_name)
                if not isinstance(parameter, nn.Parameter) or not parameter.requires_grad:
                    continue
                width = 1 if parameter.ndim == 1 else parameter.shape[1]
                fields.append(
                    _RowField(field_name, parameter, width, offset, offset + width)
                )
                offset += width
                claim(parameter, f"{store.site!r}.{field_name}")
            if not fields or fields[0].name != "w":
                raise TypeError(f"{store.site!r}: trainable amplitudes are required")
            state = {
                "step": 0,
                "m": torch.zeros(
                    store.capacity, offset, device=store.w.device, dtype=store.w.dtype
                ),
                "v": torch.zeros(
                    store.capacity, offset, device=store.w.device, dtype=store.w.dtype
                ),
                "mass1": torch.zeros(
                    store.capacity, 1, device=store.w.device, dtype=store.w.dtype
                ),
                "mass2": torch.zeros(
                    store.capacity, 1, device=store.w.device, dtype=store.w.dtype
                ),
            }
            atom_site = _AtomSite(site_name, module, store, tuple(fields), offset, state)
            atom_sites.append(atom_site)
            if subscribe:
                atom_site.follower = _RowsFollower(state)
                store.followers().subscribe(atom_site.follower)

            for chart_store in (module.in_neurons, module.out_neurons):
                mu = chart_store.mu
                if not isinstance(mu, nn.Parameter) or not mu.requires_grad:
                    continue
                claim(mu, f"chart {chart_store.site!r}.mu")
                chart = chart_users.get(id(chart_store))
                if chart is None:
                    chart_state = {
                        "step": 0,
                        "m": torch.zeros_like(mu),
                        "v": torch.zeros_like(mu),
                        "mass1": torch.zeros(mu.shape[0], 1).to(mu),
                        "mass2": torch.zeros(mu.shape[0], 1).to(mu),
                    }
                    chart = _Chart(chart_store, [], chart_state)
                    chart_users[id(chart_store)] = chart
                    if subscribe:
                        chart.follower = _RowsFollower(chart_state)
                        chart_store.followers().subscribe(chart.follower)
                if module not in chart.sites:
                    chart.sites.append(module)

            for side, factor in (
                ("in", module.factor_in),
                ("out", module.factor_out),
            ):
                sigma = factor.sigma
                if not isinstance(sigma, nn.Parameter) or not sigma.requires_grad:
                    continue
                claim(sigma, f"{site_name!r}.{side} factor sigma")
                block = sigma_users.get(id(sigma))
                if block is None:
                    scalar = torch.zeros_like(sigma)
                    block = _Sigma(
                        sigma,
                        [],
                        {
                            "step": 0,
                            "m": scalar.clone(),
                            "v": scalar.clone(),
                            "mass1": scalar.clone(),
                            "mass2": scalar.clone(),
                        },
                    )
                    sigma_users[id(sigma)] = block
                block.uses.append((module, side))

        factor_modules = {
            id(factor): factor
            for _, site in sites
            for factor in (site.factor_in, site.factor_out)
        }.values()
        for factor in factor_modules:
            for parameter_name, parameter in factor.named_parameters(recurse=False):
                if parameter.requires_grad and id(parameter) not in owned_ids:
                    raise NotImplementedError(
                        f"trainable factor parameter {parameter_name!r} on "
                        f"{type(factor).__name__} has no pullback block"
                    )

        self._atom_sites = atom_sites
        self._charts = list(chart_users.values())
        self._sigmas = list(sigma_users.values())
        self._owned_ids = frozenset(owned_ids)
        self._owned_parameters = tuple(owned)
        self._non_cst_parameters = tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in owned_ids
        )
        defaults = {"lr": 1.0 if lr is None else float(lr)}
        super().__init__(owned, defaults)
        for site in self._atom_sites:
            self.state[site.store.w] = site.state
        for chart in self._charts:
            self.state[chart.store.mu] = chart.state
        for sigma in self._sigmas:
            self.state[sigma.parameter] = sigma.state
        self.param_groups[0]["calibrated_scale"] = None

    # -- ownership -------------------------------------------------------

    def owned_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._owned_parameters

    def non_cst_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._non_cst_parameters

    def ownership(self) -> dict[str, str]:
        return {
            name: (
                "cst"
                if id(parameter) in self._owned_ids
                else "frozen"
                if not parameter.requires_grad
                else "non_cst"
            )
            for name, parameter in self.model.named_parameters()
        }

    def validate_dense_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        held = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        duplicate = held & self._owned_ids
        if duplicate:
            raise ValueError(
                "dense optimizer also owns CST parameter(s): "
                f"{[self._names[key] for key in duplicate]!r}"
            )
        expected = {id(parameter) for parameter in self._non_cst_parameters}
        missing = expected - held
        outside = held - expected
        if missing or outside:
            raise ValueError(
                "dense optimizer parameter partition mismatch: "
                f"missing={[self._names[key] for key in missing]!r}, "
                f"outside={[self._names.get(key, '<outside model>') for key in outside]!r}"
            )

    # -- atom metric ----------------------------------------------------

    @staticmethod
    def _field(site: _AtomSite, name: str) -> _RowField | None:
        return next((field for field in site.fields if field.name == name), None)

    @staticmethod
    def _pack_gradients(site: _AtomSite, slots: Tensor) -> Tensor:
        parts = []
        for field in site.fields:
            value = field.parameter.grad
            if value is None:
                part = field.parameter.new_zeros(
                    (field.parameter.shape[0], field.width)
                )
            else:
                part = value[:, None] if value.ndim == 1 else value
            parts.append(part.index_select(0, slots))
        return torch.cat(parts, dim=1)

    def _atom_gram(
        self,
        site: _AtomSite,
        slots: Tensor,
        sigma_metrics: dict[int, tuple[Tensor, int]] | None = None,
    ) -> Tensor:
        fast_maturity = (
            site.store.atom_column_names == ("maturity",)
            and type(site.module.factor_in) is MaturityGaussianFactor
            and type(site.module.factor_out) is MaturityGaussianFactor
            and isinstance(site.module.gauge, L2NormalizedColumns)
        )
        if not site.store.atom_column_names or fast_maturity:
            try:
                return self._radial_atom_gram(site, slots, sigma_metrics)
            except NotImplementedError:
                pass
        return self._generic_atom_gram(site, slots)

    def _radial_atom_gram(
        self,
        site: _AtomSite,
        slots: Tensor,
        sigma_metrics: dict[int, tuple[Tensor, int]] | None = None,
    ) -> Tensor:
        """Vectorised same-atom Gram for radial ``(w, s, t)`` families."""
        module, store = site.module, site.store
        source = store.s.detach().index_select(0, slots)
        target = store.t.detach().index_select(0, slots)
        mu_in = module.in_neurons.mu.detach().to(source)
        mu_out = module.out_neurons.mu.detach().to(target)

        if (
            isinstance(module.gauge, L2NormalizedColumns)
            and type(module.factor_in) is GaussianFactor
            and type(module.factor_out) is GaussianFactor
            and [field.name for field in site.fields] == ["w", "s", "t"]
        ):
            width = max(
                mu_in.shape[0] * source.shape[1],
                mu_out.shape[0] * target.shape[1],
                1,
            )
            block_size = max(1, self.metric_chunk_elements // width)
            blocks = []
            source_sigma_total = store.w.new_zeros(())
            target_sigma_total = store.w.new_zeros(())
            s_field, t_field = self._field(site, "s"), self._field(site, "t")
            assert s_field is not None and t_field is not None
            for start in range(0, slots.numel(), block_size):
                stop = min(start + block_size, slots.numel())
                block_source = source[start:stop]
                block_target = target[start:stop]
                weight_sq = (
                    store.w.detach().index_select(0, slots[start:stop]).square()
                )
                source_gram, source_sigma_sq = _fast_gaussian_l2_side_terms(
                    mu_in,
                    block_source,
                    module.factor_in.sigma.detach().to(block_source),
                    compiled=self.compile_metric,
                )
                target_gram, target_sigma_sq = _fast_gaussian_l2_side_terms(
                    mu_out,
                    block_target,
                    module.factor_out.sigma.detach().to(block_target),
                    compiled=self.compile_metric,
                )
                gram = store.w.new_zeros(
                    (stop - start, site.width, site.width)
                )
                gram[:, 0, 0] = 1.0
                gram[
                    :, s_field.start : s_field.stop, s_field.start : s_field.stop
                ] = weight_sq[:, None, None] * source_gram
                gram[
                    :, t_field.start : t_field.stop, t_field.start : t_field.stop
                ] = weight_sq[:, None, None] * target_gram
                blocks.append(gram)
                source_sigma_total += (weight_sq * source_sigma_sq).sum()
                target_sigma_total += (weight_sq * target_sigma_sq).sum()
            if sigma_metrics is not None:
                for factor, contribution in (
                    (module.factor_in, source_sigma_total),
                    (module.factor_out, target_sigma_total),
                ):
                    parameter = factor.sigma
                    if not isinstance(parameter, nn.Parameter) or parameter.grad is None:
                        continue
                    previous, count = sigma_metrics.get(
                        id(parameter), (parameter.new_zeros(()), 0)
                    )
                    sigma_metrics[id(parameter)] = (previous + contribution, count + 1)
            return torch.cat(blocks) if blocks else store.w.new_zeros(
                (0, site.width, site.width)
            )

        if (
            isinstance(module.gauge, L2NormalizedColumns)
            and type(module.factor_in) is MaturityGaussianFactor
            and type(module.factor_out) is MaturityGaussianFactor
            and [field.name for field in site.fields]
            == ["w", "s", "t", "maturity"]
        ):
            width = max(
                mu_in.shape[0] * source.shape[1],
                mu_out.shape[0] * target.shape[1],
                1,
            )
            block_size = max(1, self.metric_chunk_elements // width)
            blocks = []
            source_sigma_total = store.w.new_zeros(())
            target_sigma_total = store.w.new_zeros(())
            s_field = self._field(site, "s")
            t_field = self._field(site, "t")
            maturity_field = self._field(site, "maturity")
            assert s_field is not None and t_field is not None
            assert maturity_field is not None and maturity_field.width == 1
            maturity = store.maturity.detach().index_select(0, slots)
            for start in range(0, slots.numel(), block_size):
                stop = min(start + block_size, slots.numel())
                block_source = source[start:stop]
                block_target = target[start:stop]
                block_maturity = maturity[start:stop]
                weight_sq = (
                    store.w.detach().index_select(0, slots[start:stop]).square()
                )
                source_terms = _fast_maturity_gaussian_l2_side_terms(
                    mu_in,
                    block_source,
                    module.factor_in.sigma.detach().to(block_source),
                    block_maturity,
                    module.factor_in.min_scale,
                    module.factor_in.max_scale,
                    compiled=self.compile_metric,
                )
                target_terms = _fast_maturity_gaussian_l2_side_terms(
                    mu_out,
                    block_target,
                    module.factor_out.sigma.detach().to(block_target),
                    block_maturity,
                    module.factor_out.min_scale,
                    module.factor_out.max_scale,
                    compiled=self.compile_metric,
                )
                source_gram, source_sigma_sq, source_m_sq, source_cross = (
                    source_terms
                )
                target_gram, target_sigma_sq, target_m_sq, target_cross = (
                    target_terms
                )
                gram = store.w.new_zeros(
                    (stop - start, site.width, site.width)
                )
                gram[:, 0, 0] = 1.0
                gram[
                    :, s_field.start : s_field.stop, s_field.start : s_field.stop
                ] = weight_sq[:, None, None] * source_gram
                gram[
                    :, t_field.start : t_field.stop, t_field.start : t_field.stop
                ] = weight_sq[:, None, None] * target_gram
                m_index = maturity_field.start
                source_coupling = weight_sq[:, None] * source_cross
                target_coupling = weight_sq[:, None] * target_cross
                gram[:, s_field.start : s_field.stop, m_index] = source_coupling
                gram[:, m_index, s_field.start : s_field.stop] = source_coupling
                gram[:, t_field.start : t_field.stop, m_index] = target_coupling
                gram[:, m_index, t_field.start : t_field.stop] = target_coupling
                gram[:, m_index, m_index] = weight_sq * (
                    source_m_sq + target_m_sq
                )
                blocks.append(gram)
                source_sigma_total += (weight_sq * source_sigma_sq).sum()
                target_sigma_total += (weight_sq * target_sigma_sq).sum()
            if sigma_metrics is not None:
                for factor, contribution in (
                    (module.factor_in, source_sigma_total),
                    (module.factor_out, target_sigma_total),
                ):
                    parameter = factor.sigma
                    if not isinstance(parameter, nn.Parameter) or parameter.grad is None:
                        continue
                    previous, count = sigma_metrics.get(
                        id(parameter), (parameter.new_zeros(()), 0)
                    )
                    sigma_metrics[id(parameter)] = (
                        previous + contribution,
                        count + 1,
                    )
            return torch.cat(blocks) if blocks else store.w.new_zeros(
                (0, site.width, site.width)
            )

        def side(factor, mu: Tensor, centers: Tensor):
            if isinstance(module.gauge, L2NormalizedColumns):
                value, slope = metric_mod.normalized_columns(factor, mu, centers)
            else:
                value, slope = metric_mod.columns(factor, mu, centers)
            derivatives = []
            for axis in range(centers.shape[1]):
                derivative = slope * (
                    centers[:, axis][None, :] - mu[:, axis][:, None]
                )
                if isinstance(module.gauge, L2NormalizedColumns):
                    radial = (value * derivative).sum(0, keepdim=True)
                    derivative = derivative - value * radial
                derivatives.append(derivative)
            return value, torch.stack(derivatives, dim=-1)

        incoming, derivative_in = side(module.factor_in, mu_in, source)
        outgoing, derivative_out = side(module.factor_out, mu_out, target)
        count = slots.numel()
        q_width = site.width - 1
        jac_in = store.w.new_zeros((incoming.shape[0], count, q_width))
        jac_out = store.w.new_zeros((outgoing.shape[0], count, q_width))
        s_field, t_field = self._field(site, "s"), self._field(site, "t")
        assert s_field is not None and t_field is not None
        jac_in[:, :, s_field.start - 1 : s_field.stop - 1] = derivative_in
        jac_out[:, :, t_field.start - 1 : t_field.stop - 1] = derivative_out
        a_sq = incoming.square().sum(0)
        b_sq = outgoing.square().sum(0)
        a_radial = torch.einsum("nkp,nk->kp", jac_in, incoming)
        b_radial = torch.einsum("nkp,nk->kp", jac_out, outgoing)
        weight = store.w.detach().index_select(0, slots)
        gram = store.w.new_zeros((count, site.width, site.width))
        gram[:, 0, 0] = a_sq * b_sq
        coupling = weight[:, None] * (
            b_sq[:, None] * a_radial + a_sq[:, None] * b_radial
        )
        if module.gauge.pullback_structure.amplitude_tangent_orthogonal:
            coupling.zero_()
        gram[:, 0, 1:] = coupling
        gram[:, 1:, 0] = coupling
        gram[:, 1:, 1:] = weight.square()[:, None, None] * (
            b_sq[:, None, None] * torch.einsum("nkp,nkq->kpq", jac_in, jac_in)
            + a_sq[:, None, None]
            * torch.einsum("nkp,nkq->kpq", jac_out, jac_out)
            + a_radial[:, :, None] * b_radial[:, None, :]
            + b_radial[:, :, None] * a_radial[:, None, :]
        )
        gram.diagonal(dim1=-2, dim2=-1).clamp_min_(0)
        return gram

    def _generic_atom_gram(self, site: _AtomSite, slots: Tensor) -> Tensor:
        """Family-generic local autograd path for extra per-atom columns."""
        module, store = site.module, site.store
        w_field = self._field(site, "w")
        assert w_field is not None and w_field.start == 0 and w_field.stop == 1
        non_w = [field for field in site.fields if field.name != "w"]
        blocks: list[Tensor] = []
        structure = module.gauge.pullback_structure
        mu_in = module.in_neurons.mu.detach().to(store.w)
        mu_out = module.out_neurons.mu.detach().to(store.w)

        for slot in slots.tolist():
            q_parts = []
            slices: dict[str, slice] = {}
            cursor = 0
            for field in non_w:
                value = field.parameter.detach()[slot].reshape(-1)
                q_parts.append(value)
                slices[field.name] = slice(cursor, cursor + field.width)
                cursor += field.width
            q = torch.cat(q_parts).requires_grad_(True)

            def side_column(
                values: Tensor, side: str, local_slices=slices
            ) -> Tensor:
                center_name = "s" if side == "in" else "t"
                center = values[local_slices[center_name]].reshape(1, -1)
                extras = {
                    name: values[field_slice].reshape(1, -1)
                    for name, field_slice in local_slices.items()
                    if name not in ("s", "t")
                }
                factor = module.factor_in if side == "in" else module.factor_out
                query = mu_in if side == "in" else mu_out
                return module.gauge.columns(factor, query, center, extras)[:, 0]

            with torch.enable_grad():
                a = side_column(q, "in")
                b = side_column(q, "out")
                jac_a = torch.autograd.functional.jacobian(
                    lambda values: side_column(values, "in"), q, vectorize=True
                )
                jac_b = torch.autograd.functional.jacobian(
                    lambda values: side_column(values, "out"), q, vectorize=True
                )
            a, b, jac_a, jac_b = (
                value.detach() for value in (a, b, jac_a, jac_b)
            )
            weight = store.w.detach()[slot]
            a_sq, b_sq = a.square().sum(), b.square().sum()
            a_radial = jac_a.transpose(0, 1) @ a
            b_radial = jac_b.transpose(0, 1) @ b
            gram = store.w.new_zeros((site.width, site.width))
            gram[0, 0] = a_sq * b_sq
            coupling = weight * (b_sq * a_radial + a_sq * b_radial)
            if structure.amplitude_tangent_orthogonal:
                coupling.zero_()
            gram[0, 1:] = coupling
            gram[1:, 0] = coupling
            gram[1:, 1:] = weight.square() * (
                b_sq * (jac_a.transpose(0, 1) @ jac_a)
                + a_sq * (jac_b.transpose(0, 1) @ jac_b)
                + a_radial[:, None] * b_radial[None, :]
                + b_radial[:, None] * a_radial[None, :]
            )
            gram.diagonal().clamp_min_(0)
            blocks.append(gram)
        if not blocks:
            return store.w.new_zeros((0, site.width, site.width))
        return torch.stack(blocks)

    def _whitener(self, gram: Tensor):
        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        positive = diagonal[diagonal > 0]
        reference = (
            positive.median() if positive.numel() else diagonal.new_tensor(1.0)
        )
        floor = self.damping * reference
        if self.metric == "diag":
            scale = (
                diagonal + floor
            ).clamp_min(torch.finfo(gram.dtype).tiny).sqrt()
            return lambda value: value / scale
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        inverse = _inverse_sqrt(gram + floor * eye)
        return lambda value: torch.einsum("kij,kj->ki", inverse, value)

    def _atom_whitener(self, site: _AtomSite, gram: Tensor):
        """Exploit exact L2 gauge zeros instead of solving one padded block."""
        module = site.module
        if (
            self.metric != "block"
            or not isinstance(module.gauge, L2NormalizedColumns)
            or type(module.factor_in) is not GaussianFactor
            or type(module.factor_out) is not GaussianFactor
            or [field.name for field in site.fields] != ["w", "s", "t"]
        ):
            return self._whitener(gram)

        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        positive = diagonal[diagonal > 0]
        reference = (
            positive.median() if positive.numel() else diagonal.new_tensor(1.0)
        )
        floor = self.damping * reference
        s_field, t_field = self._field(site, "s"), self._field(site, "t")
        assert s_field is not None and t_field is not None

        def inverse(field: _RowField) -> Tensor:
            block = gram[
                :, field.start : field.stop, field.start : field.stop
            ]
            eye = torch.eye(
                field.width, device=gram.device, dtype=gram.dtype
            )
            return _inverse_sqrt(block + floor * eye)

        amplitude = (gram[:, 0, 0] + floor).clamp_min(
            torch.finfo(gram.dtype).tiny
        ).sqrt()
        inverse_s = inverse(s_field)
        inverse_t = inverse(t_field)

        def apply(value: Tensor) -> Tensor:
            result = torch.empty_like(value)
            result[:, 0] = value[:, 0] / amplitude
            result[:, s_field.start : s_field.stop] = torch.einsum(
                "kij,kj->ki",
                inverse_s,
                value[:, s_field.start : s_field.stop],
            )
            result[:, t_field.start : t_field.stop] = torch.einsum(
                "kij,kj->ki",
                inverse_t,
                value[:, t_field.start : t_field.stop],
            )
            return result

        return apply

    @staticmethod
    def _quadratic_norm(value: Tensor, gram: Tensor) -> Tensor:
        return torch.einsum("ki,kij,kj->k", value, gram, value).clamp_min(0).sqrt()

    def _apply_atom_delta(self, site: _AtomSite, slots: Tensor, delta: Tensor) -> None:
        for field in site.fields:
            part = delta[:, field.start : field.stop]
            if field.parameter.ndim == 1:
                part = part[:, 0]
            field.parameter.index_add_(0, slots, part)

    # -- shared metrics --------------------------------------------------

    def _chart_metric(self, chart: _Chart) -> Tensor:
        store = chart.store
        mu = store.mu.detach()
        blocks = torch.zeros(
            mu.shape[0], mu.shape[1], mu.shape[1], device=mu.device, dtype=mu.dtype
        )
        for module in chart.sites:
            for side, factor, anchor in (
                ("in", module.factor_in, module.in_neurons),
                ("out", module.factor_out, module.out_neurons),
            ):
                if anchor is not store:
                    continue
                synapses = module.synapses
                slots = synapses.live_slots().to(mu.device)
                centers = (
                    synapses.s if side == "in" else synapses.t
                ).detach().index_select(0, slots).to(mu)
                extras = {
                    name: getattr(synapses, name)
                    .detach()
                    .index_select(0, slots)
                    .to(mu)
                    for name in synapses.atom_column_names
                }
                weights_sq = (
                    synapses.w.detach().index_select(0, slots).to(mu).square()
                )
                other_factor = (
                    module.factor_out if side == "in" else module.factor_in
                )
                other_mu = (
                    module.out_neurons.mu if side == "in" else module.in_neurons.mu
                ).detach().to(mu)
                other_centers = (
                    synapses.t if side == "in" else synapses.s
                ).detach().index_select(0, slots).to(mu)
                other = module.gauge.columns(
                    other_factor, other_mu, other_centers, extras
                )
                other_sq = other.square().sum(0)
                if isinstance(module.gauge, L2NormalizedColumns):
                    try:
                        unit, slope = metric_mod.normalized_columns(
                            factor, mu, centers, extras
                        )
                    except NotImplementedError:
                        self._generic_normalized_chart_metric(
                            blocks, module, factor, mu, centers, extras,
                            weights_sq * other_sq,
                        )
                        continue
                    diff = centers[None, :, :] - mu[:, None, :]
                    derivative = slope[:, :, None] * diff
                    for axis in range(mu.shape[1]):
                        for other_axis in range(mu.shape[1]):
                            product = (
                                derivative[:, :, axis]
                                * derivative[:, :, other_axis]
                                * (1.0 - unit.square())
                            )
                            blocks[:, axis, other_axis] += (
                                product * (weights_sq * other_sq)[None, :]
                            ).sum(1)
                else:
                    for atom in range(centers.shape[0]):
                        center = centers[atom : atom + 1]
                        atom_extras = {
                            name: value[atom : atom + 1]
                            for name, value in extras.items()
                        }
                        for row in range(mu.shape[0]):
                            point = mu[row].detach().requires_grad_(True)
                            with torch.enable_grad():
                                scalar = module.gauge.columns(
                                    factor, point[None], center, atom_extras
                                )[0, 0]
                                (gradient,) = torch.autograd.grad(scalar, point)
                            blocks[row] += (
                                weights_sq[atom]
                                * other_sq[atom]
                                * gradient[:, None]
                                * gradient[None, :]
                            )
            if (
                not isinstance(module.gauge, L2NormalizedColumns)
                and module.in_neurons is store
                and module.out_neurons is store
            ):
                self._raw_cross_side_chart_metric(blocks, module, mu)
        blocks.diagonal(dim1=-2, dim2=-1).clamp_min_(0)
        return blocks

    @staticmethod
    def _raw_cross_side_chart_metric(
        blocks: Tensor, module: nn.Module, mu: Tensor
    ) -> None:
        """Add the input/output cross term when one raw chart serves both sides."""
        store = module.synapses
        slots = store.live_slots().to(mu.device)
        source = store.s.detach().index_select(0, slots).to(mu)
        target = store.t.detach().index_select(0, slots).to(mu)
        weights_sq = store.w.detach().index_select(0, slots).to(mu).square()
        extras = {
            name: getattr(store, name).detach().index_select(0, slots).to(mu)
            for name in store.atom_column_names
        }
        incoming = module.gauge.columns(module.factor_in, mu, source, extras)
        outgoing = module.gauge.columns(module.factor_out, mu, target, extras)
        for atom in range(slots.numel()):
            atom_extras = {
                name: value[atom : atom + 1] for name, value in extras.items()
            }
            for row in range(mu.shape[0]):
                point = mu[row].detach().requires_grad_(True)
                with torch.enable_grad():
                    value_in = module.gauge.columns(
                        module.factor_in,
                        point[None],
                        source[atom : atom + 1],
                        atom_extras,
                    )[0, 0]
                    value_out = module.gauge.columns(
                        module.factor_out,
                        point[None],
                        target[atom : atom + 1],
                        atom_extras,
                    )[0, 0]
                    gradient_in = torch.autograd.grad(
                        value_in, point, retain_graph=True
                    )[0]
                    gradient_out = torch.autograd.grad(value_out, point)[0]
                coefficient = (
                    weights_sq[atom] * incoming[row, atom] * outgoing[row, atom]
                )
                blocks[row] += coefficient * (
                    gradient_in[:, None] * gradient_out[None, :]
                    + gradient_out[:, None] * gradient_in[None, :]
                )

    @staticmethod
    def _generic_normalized_chart_metric(
        blocks: Tensor,
        module: nn.Module,
        factor: nn.Module,
        mu: Tensor,
        centers: Tensor,
        extras: dict[str, Tensor],
        atom_scale: Tensor,
    ) -> None:
        """Correct fallback when a family has no radial derivative contract."""
        for atom in range(centers.shape[0]):
            center = centers[atom : atom + 1]
            atom_extras = {
                name: value[atom : atom + 1] for name, value in extras.items()
            }
            query = mu.detach().requires_grad_(True)
            with torch.enable_grad():
                jacobian = torch.autograd.functional.jacobian(
                    lambda value, atom_center=center,
                    local_extras=atom_extras: module.gauge.columns(
                        factor, value, atom_center, local_extras
                    )[:, 0],
                    query,
                    vectorize=True,
                )
            # [output row, differentiated query row, chart axis].
            blocks += atom_scale[atom] * torch.einsum(
                "nid,nie->ide", jacobian.detach(), jacobian.detach()
            )

    def _sigma_metric(self, block: _Sigma) -> Tensor:
        if all(
            isinstance(module.gauge, L2NormalizedColumns)
            and type(module.factor_in if side == "in" else module.factor_out)
            is GaussianFactor
            for module, side in block.uses
        ):
            total = block.parameter.new_zeros(())
            by_site: dict[int, tuple[nn.Module, set[str]]] = {}
            for module, side in block.uses:
                entry = by_site.setdefault(id(module), (module, set()))
                entry[1].add(side)
            for module, sides in by_site.values():
                store = module.synapses
                slots = store.live_slots().to(store.w.device)
                weights_sq = store.w.detach().index_select(0, slots).square()
                source = store.s.detach().index_select(0, slots)
                target = store.t.detach().index_select(0, slots)
                mu_in = module.in_neurons.mu.detach().to(source)
                mu_out = module.out_neurons.mu.detach().to(target)
                width = max(mu_in.shape[0], mu_out.shape[0], 1)
                block_size = max(1, self.metric_chunk_elements // width)
                for start in range(0, slots.numel(), block_size):
                    stop = min(start + block_size, slots.numel())
                    if "in" in sides:
                        _, sigma_sq = _fast_gaussian_l2_side_terms(
                            mu_in,
                            source[start:stop],
                            block.parameter.detach().to(source),
                            compiled=self.compile_metric,
                        )
                        total += (
                            weights_sq[start:stop]
                            * sigma_sq
                        ).sum()
                    if "out" in sides:
                        _, sigma_sq = _fast_gaussian_l2_side_terms(
                            mu_out,
                            target[start:stop],
                            block.parameter.detach().to(target),
                            compiled=self.compile_metric,
                        )
                        total += (
                            weights_sq[start:stop]
                            * sigma_sq
                        ).sum()
            return total.clamp_min(0)

        total = block.parameter.new_zeros(())
        by_site: dict[int, tuple[nn.Module, set[str]]] = {}
        for module, side in block.uses:
            entry = by_site.setdefault(id(module), (module, set()))
            entry[1].add(side)
        for module, sides in by_site.values():
            store = module.synapses
            slots = store.live_slots().to(store.w.device)
            source = store.s.detach().index_select(0, slots)
            target = store.t.detach().index_select(0, slots)
            extras = {
                name: getattr(store, name).detach().index_select(0, slots)
                for name in store.atom_column_names
            }
            weights = store.w.detach().index_select(0, slots)
            mu_in = module.in_neurons.mu.detach().to(source)
            mu_out = module.out_neurons.mu.detach().to(target)

            def delivered(
                factor: nn.Module,
                query: Tensor,
                centers: Tensor,
                sigma: Tensor,
                local_extras=extras,
                local_module=module,
            ) -> Tensor:
                raw = torch.func.functional_call(
                    factor, {"sigma": sigma}, (query, centers, local_extras)
                )
                if isinstance(local_module.gauge, L2NormalizedColumns):
                    norm = torch.linalg.vector_norm(raw, dim=0, keepdim=True)
                    raw = raw / norm.clamp_min(torch.finfo(raw.dtype).tiny)
                return raw

            sigma = block.parameter.detach().requires_grad_(True)
            with torch.enable_grad():
                a = (
                    delivered(module.factor_in, mu_in, source, sigma)
                    if "in" in sides
                    else module.gauge.columns(
                        module.factor_in, mu_in, source, extras
                    ).detach()
                )
                b = (
                    delivered(module.factor_out, mu_out, target, sigma)
                    if "out" in sides
                    else module.gauge.columns(
                        module.factor_out, mu_out, target, extras
                    ).detach()
                )
                # Sigma is a scalar.  Forward-mode differentiates every
                # delivered entry in one tangent pass; reverse-mode Jacobian
                # construction would allocate an output-sized standard basis
                # (quadratic in neurons x atoms on CUDA).
                da = (
                    torch.func.jacfwd(
                        lambda value, factor=module.factor_in,
                        query=mu_in, centers=source: delivered(
                            factor, query, centers, value
                        )
                    )(sigma)
                    if "in" in sides
                    else torch.zeros_like(a)
                )
                db = (
                    torch.func.jacfwd(
                        lambda value, factor=module.factor_out,
                        query=mu_out, centers=target: delivered(
                            factor, query, centers, value
                        )
                    )(sigma)
                    if "out" in sides
                    else torch.zeros_like(b)
                )
            a, b, da, db = (value.detach() for value in (a, b, da, db))
            a_sq = a.square().sum(0)
            b_sq = b.square().sum(0)
            total += (
                weights.square()
                * (
                    b_sq * da.square().sum(0)
                    + a_sq * db.square().sum(0)
                    + 2.0 * (a * da).sum(0) * (b * db).sum(0)
                )
            ).sum()
        return total.clamp_min(0)

    # -- step ------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        direct_scale = self.param_groups[0]["lr"]
        pending: list[tuple[_AtomSite, Tensor, Tensor]] = []
        pending_charts: list[tuple[_Chart, Tensor]] = []
        pending_sigmas: list[tuple[_Sigma, Tensor]] = []
        map_norms: list[Tensor] = []
        sigma_metric_cache: dict[int, tuple[Tensor, int]] = {}
        need_map_norms = (
            self.target_map_step is not None and self._calibrated_scale is None
        )
        for site in self._atom_sites:
            slots = site.store.live_slots().to(site.store.w.device)
            if slots.numel() == 0:
                continue
            if not any(field.parameter.grad is not None for field in site.fields):
                continue
            gradient = self._pack_gradients(site, slots)
            gram = self._atom_gram(site, slots, sigma_metric_cache)
            whiten = self._atom_whitener(site, gram)
            state = site.state
            incoming = whiten(gradient)
            rows = slots.to(state["m"].device)
            local_state = {
                "step": state["step"],
                "m": state["m"].index_select(0, rows),
                "v": state["v"].index_select(0, rows),
                "mass1": state["mass1"].index_select(0, rows),
                "mass2": state["mass2"].index_select(0, rows),
            }
            direction = _adam(local_state, incoming, self.betas, self.eps)
            state["step"] = local_state["step"]
            state["m"].index_copy_(0, rows, local_state["m"])
            state["v"].index_copy_(0, rows, local_state["v"])
            state["mass1"].index_copy_(0, rows, local_state["mass1"])
            state["mass2"].index_copy_(0, rows, local_state["mass2"])
            raw = -whiten(direction)
            pending.append((site, slots, raw))
            if need_map_norms:
                map_norms.append(self._quadratic_norm(raw, gram))

        for chart in self._charts:
            parameter = chart.store.mu
            if parameter.grad is None:
                continue
            gram = self._chart_metric(chart)
            whiten = self._whitener(gram)
            direction = _adam(chart.state, whiten(parameter.grad), self.betas, self.eps)
            raw = -whiten(direction)
            pending_charts.append((chart, raw))
            if need_map_norms:
                map_norms.append(self._quadratic_norm(raw, gram))

        for sigma in self._sigmas:
            parameter = sigma.parameter
            if parameter.grad is None:
                continue
            cached = sigma_metric_cache.get(id(parameter))
            gram = (
                cached[0]
                if cached is not None and cached[1] == len(sigma.uses)
                else self._sigma_metric(sigma)
            )
            reference = gram.detach().clamp_min(torch.finfo(gram.dtype).tiny)
            gearing = (gram + self.damping * reference).sqrt()
            direction = _adam(
                sigma.state, parameter.grad / gearing, self.betas, self.eps
            )
            raw = -direction / gearing
            pending_sigmas.append((sigma, raw))
            if need_map_norms:
                map_norms.append((raw.abs() * gram.sqrt()).reshape(1))

        if self.target_map_step is not None and self._calibrated_scale is None:
            available = [value for value in map_norms if value.numel()]
            if not available:
                return loss
            values = torch.cat(available)
            median = values.median().clamp_min(1e-30)
            self._calibrated_scale = float(self.target_map_step / median)
            self.param_groups[0]["calibrated_scale"] = self._calibrated_scale
        if self.target_map_step is None:
            scale = direct_scale
        else:
            assert self._calibrated_scale is not None
            scale = direct_scale * self._calibrated_scale
        for site, slots, raw in pending:
            delta = raw * scale
            if self.max_step_sigma is not None:
                coordinate_parts = []
                for field in (self._field(site, "s"), self._field(site, "t")):
                    if field is not None:
                        coordinate_parts.append(delta[:, field.start : field.stop])
                if coordinate_parts:
                    norm = torch.cat(coordinate_parts, dim=1).norm(dim=1)
                    sigma = torch.minimum(
                        site.module.factor_in.sigma.detach().to(norm),
                        site.module.factor_out.sigma.detach().to(norm),
                    )
                    cap = self.max_step_sigma * sigma
                    row_scale = (cap / norm.clamp_min(1e-30)).clamp(max=1.0)
                    delta = delta * row_scale[:, None]
            self._apply_atom_delta(site, slots, delta)

        for chart, raw in pending_charts:
            parameter = chart.store.mu
            delta = scale * raw
            if self.max_step_sigma is not None:
                reference = min(
                    float(factor.sigma.detach())
                    for site in chart.sites
                    for factor in (site.factor_in, site.factor_out)
                    if site.in_neurons is chart.store or site.out_neurons is chart.store
                )
                norm = delta.norm(dim=1)
                row_scale = (
                    self.max_step_sigma * reference / norm.clamp_min(1e-30)
                ).clamp(max=1.0)
                delta = delta * row_scale[:, None]
            parameter.add_(delta)

        for sigma, raw in pending_sigmas:
            parameter = sigma.parameter
            delta = scale * raw
            if self.max_step_sigma is not None:
                cap = self.max_step_sigma * parameter.detach().abs()
                delta = delta.clamp(min=-cap, max=cap)
            parameter.add_(delta)
            parameter.clamp_min_(torch.finfo(parameter.dtype).tiny)
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._calibrated_scale = self.param_groups[0].get("calibrated_scale")
        for site in self._atom_sites:
            site.state = self.state[site.store.w]
            if site.follower is not None:
                site.follower.state = site.state
        for chart in self._charts:
            chart.state = self.state[chart.store.mu]
            if chart.follower is not None:
                chart.follower.state = chart.state
        for sigma in self._sigmas:
            sigma.state = self.state[sigma.parameter]
        return result

    def __repr__(self) -> str:
        mode = (
            f"lr={self.param_groups[0]['lr']}"
            if self.target_map_step is None
            else f"target_map_step={self.target_map_step}"
        )
        return (
            f"CSTPullbackAdam(sites={len(self._atom_sites)}, "
            f"charts={len(self._charts)}, sigmas={len(self._sigmas)}, "
            f"metric={self.metric!r}, {mode})"
        )
