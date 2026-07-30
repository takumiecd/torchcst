"""Loss-blind random proposal mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Mapping

import torch
from torch import Tensor

from torchcst._validation import require_int
from torchcst.instruments import CertificateSubspace
from torchcst.representation import CoordinateDomain, IntegerGrid, Sphere
from torchcst.storage import SynapseBirth, SynapseMerge, SynapseView

from .contract import InstrumentSpec
from .registry import RetiredCandidateRegistry


Bounds = int | tuple[int, ...]


def _bounds(value: Bounds | None, width: int, name: str) -> tuple[int, ...]:
    if value is None:
        raise ValueError(f"{name} are required for an entry proposal")
    values = (value,) if isinstance(value, int) else tuple(value)
    if len(values) != width or any(bound <= 0 for bound in values):
        raise ValueError(f"{name} must contain {width} positive bounds")
    return tuple(int(bound) for bound in values)


def _decode(value: int, bounds: tuple[int, ...]) -> tuple[int, ...]:
    coordinates = [0] * len(bounds)
    for index in range(len(bounds) - 1, -1, -1):
        coordinates[index] = value % bounds[index]
        value //= bounds[index]
    return tuple(coordinates)


def _view_domain(
    view: SynapseView, side: str, override: Bounds | None
) -> CoordinateDomain:
    coordinate = view.s if side == "in" else view.t
    domain = getattr(view, f"domain_{side}", None)
    if override is not None:
        width = coordinate.shape[1]
        return IntegerGrid(_bounds(override, width, f"bounds_{side}"))
    if domain is not None:
        return domain
    value = getattr(view, f"bounds_{side}", None)
    width = coordinate.shape[1]
    return IntegerGrid(_bounds(value, width, f"bounds_{side}"))


def _continuous_lineages(
    view: SynapseView,
    count: int,
    registry: RetiredCandidateRegistry,
    next_by_site: dict[str, int],
) -> Tensor:
    lineages = getattr(view, "lineages", ())
    if lineages is None:
        lineages = ()
    if isinstance(lineages, Tensor):
        lineages = lineages.detach().cpu().tolist()
    existing = [int(value) for value in lineages]
    retired = [key for site, key in registry.snapshot() if site == view.site]
    floor = max((*existing, *retired), default=-1) + 1
    start = max(next_by_site.get(view.site, 0), floor)
    next_by_site[view.site] = start + count
    return torch.arange(start, start + count, dtype=torch.int64)


def _retired_ids(view: SynapseView, side: str) -> set[int]:
    values = getattr(view, f"retired_{side}", ())
    if isinstance(values, Tensor):
        values = values.detach().cpu().tolist()
    return {int(value) for value in values}


def _mask_sphere_components(rows: Tensor, retired: set[int]) -> Tensor:
    """Remove retired neuron components and restore the sphere gauge."""
    if not retired or rows.numel() == 0:
        return rows
    valid = [index for index in retired if 0 <= index < rows.shape[1]]
    if valid:
        rows = rows.clone()
        rows[:, valid] = 0
    norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    if bool((norms <= torch.finfo(rows.dtype).eps).any()):
        raise RuntimeError("retired neurons leave no usable sphere direction")
    return rows / norms


@dataclass
class UniformBirth:
    """Uniform domain-driven births.

    Integer grids exclude occupied and retired coordinate lineages. Continuous
    domains have no coordinate identity and receive proposer-owned unique
    lineage keys instead.
    """

    bounds_in: Bounds | None = None
    bounds_out: Bounds | None = None
    initial_weight: float = 0.0
    requires: tuple[InstrumentSpec, ...] = field(
        default=(), init=False, repr=False
    )
    _next_lineage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        domain_in = _view_domain(view, "in", self.bounds_in)
        domain_out = _view_domain(view, "out", self.bounds_out)
        if isinstance(domain_in, IntegerGrid) and isinstance(domain_out, IntegerGrid):
            source, target, lineages = self._sample_integer(
                view, budget, registry, rng, domain_in, domain_out
            )
            count = lineages.numel()
        else:
            source = domain_in.sample(budget, rng).to(view.s)
            target = domain_out.sample(budget, rng).to(view.t)
            if domain_in.lineage_key(source) is not None or domain_out.lineage_key(target) is not None:
                raise ValueError("mixed discrete/continuous UniformBirth is unsupported")
            count = budget
            lineages = _continuous_lineages(
                view, count, registry, self._next_lineage
            )
            if isinstance(domain_in, Sphere):
                source = _mask_sphere_components(
                    source, _retired_ids(view, "in")
                )
            if isinstance(domain_out, Sphere):
                target = _mask_sphere_components(
                    target, _retired_ids(view, "out")
                )
        if count == 0:
            return ()
        weights = view.w.new_full((count,), self.initial_weight)
        return (
            SynapseBirth(view.site, source, target, weights, lineages),
        )

    @classmethod
    def _sample_integer(
        cls,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
        domain_in: IntegerGrid,
        domain_out: IntegerGrid,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert domain_in.bounds is not None and domain_out.bounds is not None
        available = cls._available_lineages(view, registry, domain_in, domain_out)
        count = min(budget, len(available))
        if count == 0:
            return (
                view.s.new_zeros((0, view.s.shape[1])),
                view.t.new_zeros((0, view.t.shape[1])),
                torch.zeros(0, dtype=torch.int64),
            )
        selected = cls._rejection_sample(
            available, count, rng, domain_in, domain_out
        )
        if len(selected) < count:
            cls._exhaustive_fill(
                selected, available, count, rng, domain_in, domain_out
            )
        lineages = torch.tensor(tuple(selected), dtype=torch.int64)
        source = torch.stack([pair[0] for pair in selected.values()]).to(view.s)
        target = torch.stack([pair[1] for pair in selected.values()]).to(view.t)
        return source, target, lineages

    @staticmethod
    def _available_lineages(
        view: SynapseView,
        registry: RetiredCandidateRegistry,
        domain_in: IntegerGrid,
        domain_out: IntegerGrid,
    ) -> set[int]:
        """Every grid lineage that is unoccupied, unretired, and endpoint-live."""
        out_size = prod(domain_out.bounds)
        source_keys = domain_in.lineage_key(view.s.to(dtype=torch.int64))
        target_keys = domain_out.lineage_key(view.t.to(dtype=torch.int64))
        occupied = {
            int(source) * out_size + int(target)
            for source, target in zip(source_keys.tolist(), target_keys.tolist())
        }
        retired_in = _retired_ids(view, "in")
        retired_out = _retired_ids(view, "out")

        def endpoint_live(lineage: int) -> bool:
            source_axis = _decode(lineage // out_size, domain_in.bounds)[0]
            target_axis = _decode(lineage % out_size, domain_out.bounds)[0]
            return source_axis not in retired_in and target_axis not in retired_out

        return {
            lineage
            for lineage in range(prod(domain_in.bounds) * out_size)
            if lineage not in occupied
            and not registry.is_retired(view.site, lineage)
            and endpoint_live(lineage)
        }

    @staticmethod
    def _rejection_sample(
        available: set[int],
        count: int,
        rng: torch.Generator,
        domain_in: IntegerGrid,
        domain_out: IntegerGrid,
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """Draw through each domain's own sampler until ``count`` distinct hits.

        Bounded by an attempt limit so a nearly full grid falls through to
        :meth:`_exhaustive_fill` instead of looping forever.
        """
        out_size = prod(domain_out.bounds)
        selected: dict[int, tuple[Tensor, Tensor]] = {}
        attempts = 0
        limit = max(32, 8 * prod(domain_in.bounds) * out_size)
        while len(selected) < count and attempts < limit:
            draw = min(max(4, 2 * (count - len(selected))), limit - attempts)
            sources = domain_in.sample(draw, rng)
            targets = domain_out.sample(draw, rng)
            s_keys = domain_in.lineage_key(sources)
            t_keys = domain_out.lineage_key(targets)
            for row, (s_key, t_key) in enumerate(zip(s_keys.tolist(), t_keys.tolist())):
                lineage = int(s_key) * out_size + int(t_key)
                if lineage in available and lineage not in selected:
                    selected[lineage] = (sources[row].clone(), targets[row].clone())
                    if len(selected) == count:
                        break
            attempts += draw
        return selected

    @staticmethod
    def _exhaustive_fill(
        selected: dict[int, tuple[Tensor, Tensor]],
        available: set[int],
        count: int,
        rng: torch.Generator,
        domain_in: IntegerGrid,
        domain_out: IntegerGrid,
    ) -> None:
        """Complete ``selected`` from the finite remainder, in random order."""
        out_size = prod(domain_out.bounds)
        remaining = sorted(available.difference(selected))
        generator_device = getattr(rng, "device", torch.device("cpu"))
        order = torch.randperm(len(remaining), generator=rng, device=generator_device)
        for position in order[: count - len(selected)].cpu().tolist():
            lineage = remaining[position]
            source_key, target_key = divmod(lineage, out_size)
            selected[lineage] = (
                torch.tensor(_decode(source_key, domain_in.bounds), dtype=torch.int64),
                torch.tensor(_decode(target_key, domain_out.bounds), dtype=torch.int64),
            )


# Compatibility name retained exactly as an alias, not a second implementation.
UniformEntryBirth = UniformBirth


@dataclass
class MergeProposer:
    """Propose the most redundant disjoint pairs from a rank-one view.

    Pair similarity is ``|<s_i,s_j> <t_i,t_j>|``.  ``budget`` counts pairs,
    self-pairs are never formed, unordered duplicates are represented once,
    and greedy selection skips any pair sharing an ID with a higher-ranked
    pair so the resulting :class:`SynapseMerge` is valid as one atomic batch.
    """

    similarity_threshold: float = 0.9
    quota_kind: str = field(default="synapse_merge", init=False)
    requires: tuple[InstrumentSpec, ...] = field(
        default=(), init=False, repr=False
    )

    def __post_init__(self) -> None:
        threshold = float(self.similarity_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.similarity_threshold = threshold

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseMerge, ...]:
        require_int(budget, "budget", minimum=0)
        if not isinstance(view.domain_in, Sphere) or not isinstance(
            view.domain_out, Sphere
        ):
            raise TypeError("MergeProposer requires a rank-one Sphere×Sphere view")
        if budget == 0 or view.ids.numel() < 2:
            return ()

        # Negated score puts the plain tuple sort in descending-similarity
        # order, with the sorted id pair as a deterministic tie-break.
        candidates: list[tuple[float, int, int]] = []
        source = view.s.detach()
        target = view.t.detach()
        ids = view.ids.detach().cpu()
        for left in range(ids.numel()):
            for right in range(left + 1, ids.numel()):
                score = abs(
                    float(torch.dot(source[left], source[right]))
                    * float(torch.dot(target[left], target[right]))
                )
                if score >= self.similarity_threshold:
                    first_id, second_id = sorted((int(ids[left]), int(ids[right])))
                    candidates.append((-score, first_id, second_id))
        candidates.sort()
        used: set[int] = set()
        selected: list[tuple[int, int]] = []
        for _, first_id, second_id in candidates:
            if first_id in used or second_id in used:
                continue
            selected.append((first_id, second_id))
            used.update((first_id, second_id))
            if len(selected) == budget:
                break
        if not selected:
            return ()
        return (
            SynapseMerge(
                view.site,
                torch.tensor(selected, dtype=torch.int64),
            ),
        )


@dataclass
class OrthogonalBirth:
    """B4 anti-gradient rank-one birth proposer.

    Per CONN-2's asymmetric rank-one construction, only the output factor
    ``t`` is orthogonalized against the certificate's top-left singular
    subspace.  The input factor ``s`` remains an independent sphere sample.
    Every coefficient is initialized to zero.
    """

    rank: int = 1
    requires: tuple[InstrumentSpec, ...] = field(init=False)
    _certificates: dict[str, CertificateSubspace] = field(
        default_factory=dict, init=False, repr=False
    )
    _next_lineage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.requires = (InstrumentSpec("certificate_subspace", rank=self.rank),)

    def bind_instruments(self, site: str, instruments: Mapping[str, Any]) -> None:
        try:
            certificate = instruments["certificate_subspace"]
        except KeyError as exc:
            raise ValueError("OrthogonalBirth requires CertificateSubspace") from exc
        if not isinstance(certificate, CertificateSubspace):
            raise TypeError("certificate_subspace has the wrong instrument type")
        self._certificates[site] = certificate

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        domain_in = _view_domain(view, "in", None)
        domain_out = _view_domain(view, "out", None)
        if not isinstance(domain_in, Sphere) or not isinstance(domain_out, Sphere):
            raise TypeError("OrthogonalBirth requires Sphere×Sphere coordinates")
        try:
            certificate = self._certificates[view.site]
        except KeyError as exc:
            raise RuntimeError(
                f"certificate is not bound for site {view.site!r}"
            ) from exc

        target = domain_out.sample(budget, rng).to(view.t)
        if certificate.has_signal:
            reading = certificate.snapshot()
            basis = reading.U_r.to(target)
            target = target - (target @ basis) @ basis.transpose(0, 1)
            norms = torch.linalg.vector_norm(target, dim=1, keepdim=True)
            if bool((norms <= torch.finfo(target.dtype).eps).any()):
                raise RuntimeError("certificate subspace leaves no sampled output direction")
            target = target / norms
        target = _mask_sphere_components(target, _retired_ids(view, "out"))
        source = domain_in.sample(budget, rng).to(view.s)
        source = _mask_sphere_components(source, _retired_ids(view, "in"))
        lineages = _continuous_lineages(
            view, budget, registry, self._next_lineage
        )
        weights = view.w.new_zeros((budget,))
        return (SynapseBirth(view.site, source, target, weights, lineages),)


@dataclass
class IncidentOutputBirth:
    """Entry birth proposer with ``t`` fixed to a scheduled neuron ID."""

    initial_weight: float = 0.0
    requires: tuple[InstrumentSpec, ...] = field(
        default=(), init=False, repr=False
    )

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        """Incident births need a composer-supplied target and are never free-standing."""
        if budget:
            raise RuntimeError("IncidentOutputBirth must be called through BundleComposer")
        return ()

    def propose_incident(
        self,
        view: SynapseView,
        budget: int,
        target_id: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        domain_in = _view_domain(view, "in", None)
        domain_out = _view_domain(view, "out", None)
        if not isinstance(domain_in, IntegerGrid) or not isinstance(
            domain_out, IntegerGrid
        ):
            raise TypeError("IncidentOutputBirth requires entry IntegerGrid domains")
        assert domain_in.bounds is not None and domain_out.bounds is not None
        if len(domain_out.bounds) != 1:
            raise ValueError("incident neuron target requires scalar output coordinates")
        if not 0 <= int(target_id) < domain_out.bounds[0]:
            raise ValueError("target neuron is outside the output domain")
        if int(target_id) in _retired_ids(view, "out"):
            raise ValueError("cannot propose an incident birth to a retired neuron")

        out_size = prod(domain_out.bounds)
        target_key = int(target_id)
        occupied: set[int] = set()
        source_keys = domain_in.lineage_key(view.s.to(dtype=torch.int64))
        target_keys = domain_out.lineage_key(view.t.to(dtype=torch.int64))
        occupied.update(
            int(source) * out_size + int(target)
            for source, target in zip(source_keys.tolist(), target_keys.tolist())
        )
        retired_in = _retired_ids(view, "in")
        available: list[tuple[int, tuple[int, ...]]] = []
        for source_key in range(prod(domain_in.bounds)):
            row = _decode(source_key, domain_in.bounds)
            lineage = source_key * out_size + target_key
            if (
                row[0] not in retired_in
                and lineage not in occupied
                and not registry.is_retired(view.site, lineage)
            ):
                available.append((lineage, row))
        count = min(budget, len(available))
        if count == 0:
            return ()
        generator_device = getattr(rng, "device", torch.device("cpu"))
        order = torch.randperm(
            len(available), generator=rng, device=generator_device
        )[:count].cpu()
        selected = [available[index] for index in order.tolist()]
        source = torch.tensor(
            [row for _, row in selected], dtype=torch.int64, device=view.s.device
        )
        target = torch.full(
            (count, 1), int(target_id), dtype=torch.int64, device=view.t.device
        )
        lineages = torch.tensor(
            [lineage for lineage, _ in selected], dtype=torch.int64
        )
        weights = view.w.new_full((count,), self.initial_weight)
        return (SynapseBirth(view.site, source, target, weights, lineages),)


@dataclass
class GradFieldTopKBirth:
    """Greedily buy the highest candidate-gradient entries (control arm only).

    This intentionally represents the defeated gradient-greedy/vertex-buying
    family.  It remains in the catalog as a deterministic comparison arm, not
    as the lifecycle recipe's recommended birth rule.
    """

    initial_weight: float = 0.0
    decay: float = 0.9
    pool_size: int = 4096
    requires: tuple[InstrumentSpec, ...] = field(init=False)
    _fields: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.requires = (
            InstrumentSpec(
                "candidate_field", decay=self.decay, pool_size=self.pool_size
            ),
        )

    def bind_instruments(self, site: str, instruments: Mapping[str, Any]) -> None:
        try:
            field = instruments["candidate_field"]
        except KeyError as exc:
            raise ValueError("GradFieldTopKBirth requires CandidateField") from exc
        self._fields[site] = field

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        try:
            candidate_field = self._fields[view.site]
        except KeyError as exc:
            raise RuntimeError(
                f"candidate field is not bound for site {view.site!r}"
            ) from exc
        coordinates, scores = candidate_field.snapshot()
        count = min(budget, scores.numel())
        if count == 0:
            return ()
        positions = torch.argsort(scores, descending=True, stable=True)[:count]
        coordinates = coordinates.index_select(
            0, positions.to(coordinates.device)
        )
        source = coordinates[:, : view.s.shape[1]].to(view.s)
        target = coordinates[:, view.s.shape[1] :].to(view.t)
        lineages = candidate_field.lineages.index_select(0, positions.cpu())
        weights = view.w.new_full((count,), self.initial_weight)
        return (
            SynapseBirth(view.site, source, target, weights, lineages),
        )
