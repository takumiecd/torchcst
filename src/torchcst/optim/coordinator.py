"""One owner per trainable parameter, for a whole multi-site model.

:class:`PullbackAdam` is the single-site primitive: it owns one site's
``s``/``t`` (and, under rent or decay, that site's ``w``) and nothing else.
A model with many CST sites -- six transformer layers times FFN up/down --
therefore left the caller to hand-exclude every site's coordinates from the
dense optimizer's parameter list and to hand-step a list of coordinate
optimizers.  Every parameter that fell through that manual seam was either
stepped twice or not at all, silently.

This module closes the seam without changing any update rule:

``PullbackConfig``
    the immutable *policy* -- exactly :class:`PullbackAdam`'s keywords minus
    the module it binds to, so one record can be reused across layers or
    varied per site.

``CSTOptimizer``
    the model-level *coordinator* -- discovers the continuous sites, builds
    one :class:`PullbackAdam` each, decides which optimizer owns which
    parameter, hands the remainder to the caller's dense optimizer, and
    rejects any parameter with two owners or none.

Separation of concerns is deliberate and matches the rest of the package.
The optimization target (which parameters), the update algorithm
(:class:`PullbackAdam` / the base optimizer), the geometry (``metric``), the
moment clock (``betas`` for the step clock, ``moment_distance`` for the
per-row geometric clock), step control (``target_step``/``cap_sigma``/
``lr_scale``), the structural forces (``rent``/``repulsion``), the slot
lifecycle (the store's follower hub) and this orchestration layer stay
distinct: the coordinator only decides *who owns what and in what order*,
never *how* an update is computed.

The engine's update order is unchanged -- the coordinator is the
``optimizer`` in it::

    engine.begin_update()
    loss.backward()
    engine.observe_microbatch(weight=...)
    engine.finalize_backward()
    optimizer.step()          # CSTOptimizer.step()
    engine.step()
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..representation import L2NormalizedColumns
from ..storage import SynapseStore
from .chart import ChartPullbackAdam
from .forces import PairRepulsion, SmoothRent
from .pullback import MetricForm, MomentSpace, PullbackAdam

__all__ = [
    "CSTOptimizer",
    "PullbackConfig",
    "continuous_sites",
    "is_continuous_site",
]


def is_continuous_site(module: object) -> bool:
    """Whether :class:`PullbackAdam` can bind to ``module``.

    Exactly :class:`PullbackAdam`'s own eligibility test, asked without
    raising: a continuous CST map with a :class:`SynapseStore`, both kernels,
    an :class:`~torchcst.representation.L2NormalizedColumns` gauge, and
    learnable coordinates.  Duck-typed on purpose, so ``optim`` keeps its
    one-way independence from ``compute``; a wrapper that merely *delegates*
    ``.synapses`` (``DepthwiseCSTConv2d``) is not itself a site, and the
    :class:`~torchcst.compute.CSTLinear` it holds is discovered on its own.
    """
    store = getattr(module, "synapses", None)
    if not isinstance(store, SynapseStore):
        return False
    if getattr(module, "kernel_in", None) is None:
        return False
    if getattr(module, "kernel_out", None) is None:
        return False
    if not isinstance(getattr(module, "gauge", None), L2NormalizedColumns):
        return False
    return isinstance(store.s, nn.Parameter) and isinstance(store.t, nn.Parameter)


def continuous_sites(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """``(qualified name, module)`` for every continuous site under ``model``.

    Ordered by ``model.named_modules()``, i.e. by registration order, so the
    result is deterministic for a given model.  The model itself can be a
    site, in which case its name is the empty string.

    Two names reaching one :class:`SynapseStore` is an error rather than a
    silent double-step: the two modules would each get their own
    :class:`PullbackAdam` over the same rows.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    found: list[tuple[str, nn.Module]] = []
    claimed: dict[int, str] = {}
    for name, submodule in model.named_modules():
        if not is_continuous_site(submodule):
            continue
        store = submodule.synapses
        previous = claimed.get(id(store))
        if previous is not None:
            raise ValueError(
                f"site store {store.site!r} is reachable as both {previous!r} "
                f"and {name!r}; one store must have one owning module"
            )
        claimed[id(store)] = name
        found.append((name, submodule))
    return tuple(found)


@dataclass(frozen=True, kw_only=True)
class PullbackConfig:
    """Immutable coordinate policy: :class:`PullbackAdam` minus its module.

    Every field is the identically named :class:`PullbackAdam` keyword and
    carries its meaning and its default verbatim; read that class for the
    semantics.  Being frozen, one config can be shared by many sites, and
    :func:`dataclasses.replace` derives a variant for one of them::

        base = PullbackConfig(moment_space="tangent", cap_sigma=0.1)
        mobile = replace(base, moment_distance=(0.25, 1.0))

    Two independent clocks stay independent here as well: ``betas`` is the
    step clock and ``moment_distance`` is the per-row geometric clock in
    sigma units.  ``moment_distance=None`` is the framework default and
    preserves ordinary step-clock Adam exactly.

    ``metric`` (the geometry ``G`` is measured in) and ``moment_space``
    (where the moments live) likewise remain independent choices.

    The structural forces (``rent``, ``repulsion``, ``decoupled_repulsion``)
    are stateless, so sharing one config across sites shares those objects
    safely.  ``seed`` is shared verbatim too; give a site its own ``seed``
    through an override when its repulsion sampling must decorrelate.

    Validation is :class:`PullbackAdam`'s, at :meth:`build` time -- a config
    is a record of intent, not a second copy of the rules.
    """

    moment_space: MomentSpace
    cap_sigma: float
    metric: MetricForm = "diag"
    betas: tuple[float, float] = (0.9, 0.99)
    amplitude_betas: tuple[float, float] | None = None
    eps: float = 1e-8
    damping: float = 1e-2
    target_step: float = 0.01
    rent: SmoothRent | None = None
    decay: float = 0.0
    lr_w: float | None = None
    repulsion: PairRepulsion | None = None
    decoupled_repulsion: PairRepulsion | None = None
    wall: bool = False
    seed: int = 0
    chunk_elements: int = 1 << 24
    compile_metric: bool = False
    moment_distance: tuple[float, float] | None = None

    @property
    def owns_amplitudes(self) -> bool:
        """Whether the built optimizer will also step the site's ``w``."""
        return self.rent is not None or self.decay > 0

    def build(self, module: nn.Module, *, subscribe: bool = True) -> PullbackAdam:
        """Bind this policy to one continuous site."""
        return PullbackAdam(
            module,
            moment_space=self.moment_space,
            metric=self.metric,
            cap_sigma=self.cap_sigma,
            betas=self.betas,
            amplitude_betas=self.amplitude_betas,
            eps=self.eps,
            damping=self.damping,
            target_step=self.target_step,
            rent=self.rent,
            decay=self.decay,
            lr_w=self.lr_w,
            repulsion=self.repulsion,
            decoupled_repulsion=self.decoupled_repulsion,
            wall=self.wall,
            seed=self.seed,
            subscribe=subscribe,
            chunk_elements=self.chunk_elements,
            compile_metric=self.compile_metric,
            moment_distance=self.moment_distance,
        )


BaseOptimizer = Callable[[list[nn.Parameter]], torch.optim.Optimizer]


class CSTOptimizer:
    """Own every trainable parameter of a multi-site CST model, exactly once.

    Discovery walks ``model.named_modules()``, so any number of sites in any
    number of layers is found by construction::

        optimizer = CSTOptimizer(
            model,
            coordinates=PullbackConfig(moment_space="tangent", cap_sigma=0.1),
            overrides={"layers.*.ffn_down": replace(shared, target_step=0.005)},
            base=lambda params: torch.optim.AdamW(params, lr=1e-3),
        )

    The partition, in one place instead of at every call site:

    - a site's ``s``/``t`` belong to that site's :class:`PullbackAdam`;
    - its ``w`` belongs to the same optimizer **only** when that site's
      config claims amplitudes through ``rent=``/``decay=`` (with ``lr_w=``);
      otherwise amplitudes are ordinary parameters of the base optimizer;
    - a learnable chart's ``mu`` belongs to a :class:`ChartPullbackAdam`
      passed in ``charts=``, and to the base optimizer when none is;
    - everything else -- dense weights, kernel bandwidths, per-atom family
      columns, norms -- belongs to the base optimizer.

    A parameter claimed twice raises, and a trainable parameter claimed by
    nobody raises unless ``allow_unowned=True``.  ``ownership()`` names the
    owner of every parameter, and :meth:`summary` prints the same picture.

    ``overrides`` maps a module-name glob to a :class:`PullbackConfig`, or to
    ``None`` to leave that site's coordinates to the base optimizer.  The
    first matching pattern in insertion order wins -- the rule the policy
    tree's ``overrides=`` already uses -- and a pattern matching no
    discovered site raises rather than passing silently.

    ``base`` is either a factory called with the parameters this coordinator
    determined the base optimizer owns, or an already-constructed
    ``torch.optim.Optimizer`` whose parameter list is then *validated*
    against that same partition, or ``None``.  With a factory and nothing
    left to own, no base optimizer is built and :attr:`base` stays ``None``.

    **This is a coordinator, not a** ``torch.optim.Optimizer``.  It holds no
    ``param_groups``: a coordinate optimizer has no ``lr`` to schedule -- its
    step size is the ``target_step``-calibrated ``eta`` and a ``cap_sigma``
    trust region -- so a ``torch.optim.lr_scheduler`` writing ``lr`` into
    group dicts would silently move only the dense half.  Attach schedulers
    to :attr:`base` and pass the same multiplicative factor to
    :meth:`step` as ``lr_scale``, which every coordinate optimizer applies::

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base, T)
        ...
        optimizer.step(lr_scale=scheduler.get_last_lr()[0] / initial_lr)
        scheduler.step()

    Ordering is deterministic *and numerically load-bearing*: the base
    optimizer, then sites in discovery order, then charts in the order given.
    Disjoint ownership does **not** make the order neutral -- a pullback
    metric is a function of the amplitudes, the chart coordinates, the
    incident atom coordinates and the kernel bandwidth, so a base step that
    moves ``w``, ``mu`` or ``sigma`` changes the very ``G`` a later site or
    chart optimizer whitens with, within the same :meth:`step` call.  This
    order is therefore a compatibility contract: it reproduces what the
    existing hand-wired runners do (``ordinary.step()`` then
    ``coordinates.step()``, see ``docs/pullback-adam.md``).  Changing it
    changes results, so it is fixed here rather than configurable.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        coordinates: PullbackConfig | None,
        overrides: Mapping[str, PullbackConfig | None] | None = None,
        base: BaseOptimizer | torch.optim.Optimizer | None = None,
        charts: Iterable[ChartPullbackAdam] = (),
        subscribe: bool = True,
        allow_unowned: bool = False,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if coordinates is not None and not isinstance(coordinates, PullbackConfig):
            raise TypeError("coordinates must be a PullbackConfig or None")
        if not isinstance(subscribe, bool):
            raise TypeError("subscribe must be a bool")
        if not isinstance(allow_unowned, bool):
            raise TypeError("allow_unowned must be a bool")
        patterns: dict[str, PullbackConfig | None] = (
            {} if overrides is None else dict(overrides)
        )
        for pattern, config in patterns.items():
            if not isinstance(pattern, str):
                raise TypeError("overrides keys must be module-name globs")
            if config is not None and not isinstance(config, PullbackConfig):
                raise TypeError(
                    "overrides values must be a PullbackConfig or None"
                )

        self.model = model
        # Built first: every later ownership claim is checked against it, so
        # nothing outside the model can be claimed -- and then stepped -- by
        # this coordinator.
        names: dict[int, str] = {}
        for name, parameter in model.named_parameters():
            names.setdefault(id(parameter), name)
        self._names = names
        discovered = continuous_sites(model)

        sites: dict[str, PullbackAdam] = {}
        delegated: list[str] = []
        matched: set[str] = set()
        for name, module in discovered:
            config = coordinates
            for pattern, override in patterns.items():
                if fnmatch.fnmatchcase(name, pattern):
                    config = override
                    matched.add(pattern)
                    break
            if config is None:
                delegated.append(name)
                continue
            # Built detached: nothing subscribes to a store until the whole
            # partition validates, so a rejected coordinator leaves no
            # half-registered follower behind.
            sites[name] = config.build(module, subscribe=False)
        unmatched = [pattern for pattern in patterns if pattern not in matched]
        if unmatched:
            raise ValueError(
                f"override pattern(s) {unmatched!r} match no discovered site "
                f"{[name for name, _ in discovered]!r}"
            )
        self.sites = sites
        #: Sites whose coordinates were deliberately left to the base optimizer.
        self.delegated_sites = tuple(delegated)

        charts = tuple(charts)
        chart_stores: dict[int, str] = {}
        chart_names: set[str] = set()
        for chart in charts:
            if not isinstance(chart, ChartPullbackAdam):
                raise TypeError("charts must be ChartPullbackAdam instances")
            store = chart.store
            if id(store) in chart_stores:
                raise ValueError(
                    f"two chart optimizers own chart {store.site!r}; a shared "
                    "chart has exactly one owner"
                )
            if store.site in chart_names:
                raise ValueError(
                    f"two charts are named {store.site!r}; chart names key the "
                    "state dict and must be unique"
                )
            chart_stores[id(store)] = store.site
            chart_names.add(store.site)
        self.charts = charts

        # -- the partition ---------------------------------------------------
        owners: dict[int, str] = {}

        def claim(parameter: nn.Parameter, owner: str, what: str) -> None:
            if id(parameter) not in names:
                # Stepping a parameter the model does not register would move
                # weights outside the model this coordinator reports on, and
                # ownership() could not even name it.
                raise ValueError(
                    f"{owner} owns {what}, which is not among "
                    "model.named_parameters(); every stepped parameter must "
                    "belong to the model handed to CSTOptimizer"
                )
            previous = owners.get(id(parameter))
            if previous is not None:
                raise ValueError(
                    f"{owner} and {previous} both claim the same parameter"
                )
            owners[id(parameter)] = owner

        for name, optimizer in sites.items():
            store = optimizer.store
            label = f"site:{name or '<model>'}"
            claim(store.s, label, f"{store.site!r} coordinates s")
            claim(store.t, label, f"{store.site!r} coordinates t")
            if optimizer.owns_amplitudes:
                claim(store.w, label, f"{store.site!r} amplitudes w")
        for chart in charts:
            site = chart.store.site
            claim(chart.store.mu, f"chart:{site}", f"chart {site!r} mu")

        base_parameters: list[nn.Parameter] = []
        for parameter in model.parameters():
            if not parameter.requires_grad or id(parameter) in owners:
                continue
            base_parameters.append(parameter)
        self._owners = owners
        self._expected_base_parameters = tuple(base_parameters)

        self.base, self._base_held = self._resolve_base(
            base, base_parameters, allow_unowned
        )

        # Commit: every check passed, so the site optimizers may now follow
        # their stores' births, deaths, growth and remaps.
        if subscribe:
            for optimizer in sites.values():
                optimizer.store.followers().subscribe(optimizer)

    # ---- construction helpers ---------------------------------------------

    def _describe(self, parameter: nn.Parameter) -> str:
        return self._names.get(
            id(parameter), f"<unregistered {tuple(parameter.shape)}>"
        )

    def _resolve_base(
        self,
        base: BaseOptimizer | torch.optim.Optimizer | None,
        base_parameters: list[nn.Parameter],
        allow_unowned: bool,
    ) -> tuple[torch.optim.Optimizer | None, frozenset[int]]:
        """Resolve ``base=`` into ``(optimizer, ids it actually holds)``."""
        if isinstance(base, torch.optim.Optimizer):
            return base, self._validate_base(base, base_parameters, allow_unowned)
        if base is None:
            if base_parameters and not allow_unowned:
                raise ValueError(
                    f"parameter(s) "
                    f"{[self._describe(p) for p in base_parameters]!r} have no "
                    "owner: pass base= a torch optimizer or a factory (or "
                    "allow_unowned=True to accept this)"
                )
            return None, frozenset()
        if not callable(base):
            raise TypeError(
                "base must be a torch.optim.Optimizer, a factory taking the "
                "owned parameters, or None"
            )
        if not base_parameters:
            return None, frozenset()
        built = base(list(base_parameters))
        if not isinstance(built, torch.optim.Optimizer):
            raise TypeError("the base factory must return a torch.optim.Optimizer")
        # A factory that ignores the list it was handed and reaches for
        # model.parameters() would double-step the coordinates; check it too.
        return built, self._validate_base(built, base_parameters, allow_unowned)

    def _validate_base(
        self,
        optimizer: torch.optim.Optimizer,
        base_parameters: list[nn.Parameter],
        allow_unowned: bool,
    ) -> frozenset[int]:
        """Reject a base optimizer that double-steps; return what it holds."""
        held = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {id(parameter) for parameter in base_parameters}
        duplicates = [
            f"{self._names.get(key, '<unregistered>')} (owned by {owner})"
            for key, owner in self._owners.items()
            if key in held
        ]
        if duplicates:
            raise ValueError(
                "the base optimizer also holds structurally owned "
                f"parameter(s): {duplicates!r}; drop them from its parameter "
                "list -- CSTOptimizer.expected_base_parameters() is exactly "
                "the list it should hold"
            )
        outside = [
            self._names.get(key, f"<parameter id={key} outside model>")
            for key in held - expected
        ]
        if outside:
            raise ValueError(
                "the base optimizer holds parameter(s) outside its assigned "
                f"partition: {sorted(outside)!r}; every stepped parameter "
                "must belong to the model handed to CSTOptimizer"
            )
        missing = [
            self._describe(parameter)
            for parameter in base_parameters
            if id(parameter) not in held
        ]
        if missing and not allow_unowned:
            raise ValueError(
                f"parameter(s) {missing!r} have no owner: neither the base "
                "optimizer nor a site or chart optimizer steps them "
                "(pass allow_unowned=True to accept this)"
            )
        return frozenset(held)

    # ---- diagnostics -------------------------------------------------------

    @property
    def site_names(self) -> tuple[str, ...]:
        """Discovered site module names, in step order."""
        return tuple(self.sites)

    def expected_base_parameters(self) -> tuple[nn.Parameter, ...]:
        """What the partition says the base optimizer *should* own.

        In ``model.named_parameters()`` order, and computed from the
        partition alone -- pass it to a factory, or use it to build an
        explicit optimizer.  It is the *expectation*, not a report: under
        ``allow_unowned=True`` an explicit base optimizer may hold less than
        this, and :meth:`ownership` is what tells you what it actually holds.
        """
        return self._expected_base_parameters

    def ownership(self) -> dict[str, str]:
        """Parameter name to *actual* owner, in ``named_parameters()`` order.

        Labels are ``"site:<module name>"``, ``"chart:<chart name>"``,
        ``"base"`` (the base optimizer really holds it), ``"frozen"``
        (``requires_grad=False``) and ``"unowned"`` -- nothing steps it,
        only reachable under ``allow_unowned=True``.
        """
        report: dict[str, str] = {}
        for name, parameter in self.model.named_parameters():
            owner = self._owners.get(id(parameter))
            if owner is not None:
                report[name] = owner
            elif not parameter.requires_grad:
                report[name] = "frozen"
            elif id(parameter) in self._base_held:
                report[name] = "base"
            else:
                report[name] = "unowned"
        return report

    def summary(self) -> str:
        """A one-screen picture of who owns and steps what."""
        unowned = sum(
            1 for label in self.ownership().values() if label == "unowned"
        )
        headline = (
            f"CSTOptimizer: {len(self.sites)} site(s), "
            f"{len(self.charts)} chart(s), "
            f"{len(self._base_held)} base parameter(s)"
            + (f", {unowned} UNOWNED" if unowned else "")
        )
        lines = [headline]
        for name, optimizer in self.sites.items():
            lines.append(
                f"  site {name or '<model>'!r} -> {optimizer.store.site!r} "
                f"moment_space={optimizer.moment_space} "
                f"metric={optimizer.metric} "
                f"moment_distance={optimizer.moment_distance} "
                f"amplitudes={'site' if optimizer.owns_amplitudes else 'base'}"
            )
        for name in self.delegated_sites:
            lines.append(f"  site {name!r} -> base (override None)")
        for chart in self.charts:
            lines.append(
                f"  chart {chart.store.site!r} "
                f"moment_space={chart.moment_space} metric={chart.metric}"
            )
        kind = "none" if self.base is None else type(self.base).__name__
        lines.append(f"  base: {kind}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CSTOptimizer(sites={len(self.sites)}, charts={len(self.charts)}, "
            f"base={type(self.base).__name__ if self.base is not None else None})"
        )

    # ---- the update --------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear every gradient this coordinator's optimizers consume."""
        if self.base is not None:
            self.base.zero_grad(set_to_none=set_to_none)
        for optimizer in self.sites.values():
            optimizer.zero_grad(set_to_none)
        for chart in self.charts:
            chart.zero_grad(set_to_none)

    def step(self, lr_scale: float = 1.0) -> None:
        """Step base, then sites, then charts -- once each, in that order.

        The order is part of the contract, not an implementation detail: an
        earlier step moves quantities a later optimizer's pullback metric
        reads (amplitudes, chart points, atom coordinates, bandwidth), so
        base -> sites -> charts is what reproduces the hand-wired runners.

        ``lr_scale`` is the coordinate optimizers' multiplicative schedule
        (see the class docstring); the base optimizer keeps its own ``lr``.
        """
        if not isinstance(lr_scale, (int, float)) or isinstance(lr_scale, bool):
            raise TypeError("lr_scale must be a number")
        if lr_scale < 0:
            raise ValueError("lr_scale must be a non-negative number")
        if self.base is not None:
            self.base.step()
        for optimizer in self.sites.values():
            optimizer.step(lr_scale)
        for chart in self.charts:
            chart.step(lr_scale)

    # ---- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Site, chart and base state under one schema.

        Each part keeps its own convention: the site optimizers snapshot
        (clone) their tensors, while the base optimizer follows
        ``torch.optim``'s and hands out *live references*.  Loading a
        state dict into a second coordinator in the same process therefore
        needs a ``copy.deepcopy`` (or a ``torch.save``/``torch.load`` round
        trip) first, or the two base optimizers alias one set of moments.
        """
        return {
            "schema": "torchcst-cst-optimizer-v1",
            "sites": {
                name: optimizer.state_dict()
                for name, optimizer in self.sites.items()
            },
            "charts": {
                chart.store.site: chart.state_dict() for chart in self.charts
            },
            "base": None if self.base is None else self.base.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-cst-optimizer-v1"
        ):
            raise ValueError("unsupported CSTOptimizer state schema")
        sites = state.get("sites")
        charts = state.get("charts")
        if not isinstance(sites, Mapping) or not isinstance(charts, Mapping):
            raise TypeError(
                "CSTOptimizer state must carry sites and charts mappings"
            )
        expected_sites = set(self.sites)
        if set(sites) != expected_sites:
            raise ValueError(
                f"CSTOptimizer state covers sites {sorted(sites)!r}, this "
                f"coordinator owns {sorted(expected_sites)!r}"
            )
        expected_charts = {chart.store.site for chart in self.charts}
        if set(charts) != expected_charts:
            raise ValueError(
                f"CSTOptimizer state covers charts {sorted(charts)!r}, this "
                f"coordinator owns {sorted(expected_charts)!r}"
            )
        base_state = state.get("base")
        if (base_state is None) != (self.base is None):
            raise ValueError(
                "CSTOptimizer state disagrees about the base optimizer"
            )
        for name, optimizer in self.sites.items():
            optimizer.load_state_dict(sites[name])
        for chart in self.charts:
            chart.load_state_dict(charts[chart.store.site])
        if self.base is not None:
            self.base.load_state_dict(base_state)
