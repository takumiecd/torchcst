"""representation familyを一意に記述する最小仕様。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .domains import CoordinateDomain, IntegerGrid, Sphere


@dataclass(frozen=True)
class RepresentationSpec:
    """familyのdomain・kernel・site・価格・退役意味論を束ねる。"""

    domain_in: CoordinateDomain = field(default_factory=IntegerGrid)
    domain_out: CoordinateDomain = field(default_factory=IntegerGrid)
    kernel_in: str = "delta"
    kernel_out: str = "delta"
    sites: str = "independent"
    atom_cost: int = 1
    retirement: str = "endpoint_cascade"

    def __post_init__(self) -> None:
        entry = self.kernel_in == self.kernel_out == "delta"
        rank_one = self.kernel_in == self.kernel_out == "dot"
        if not entry and not rank_one:
            raise ValueError("kernel pair must describe entry or rank-one family")
        if entry:
            if self.atom_cost != 1:
                raise ValueError("entry family atom_cost must be 1")
            if not isinstance(self.domain_in, IntegerGrid) or not isinstance(
                self.domain_out, IntegerGrid
            ):
                raise TypeError("entry family requires IntegerGrid domains")
        else:
            if not isinstance(self.domain_in, Sphere) or not isinstance(
                self.domain_out, Sphere
            ):
                raise TypeError("rank-one family requires Sphere domains")
            expected = self.domain_in.dim + self.domain_out.dim + 1
            if self.atom_cost != expected:
                raise ValueError("rank-one atom_cost must be d_in + d_out + 1")
        expected_retirement = "endpoint_cascade" if entry else "gate_only"
        if self.retirement != expected_retirement:
            raise ValueError(
                f"this implementation requires retirement={expected_retirement!r}"
            )

    @classmethod
    def entry(
        cls,
        *,
        bounds_in: int | tuple[int, ...] | None = None,
        bounds_out: int | tuple[int, ...] | None = None,
    ) -> RepresentationSpec:
        """IntegerGrid×deltaのentry family仕様を生成する。"""
        return cls(
            domain_in=IntegerGrid(bounds_in),
            domain_out=IntegerGrid(bounds_out),
        )

    @classmethod
    def rank_one(cls, d_in: int, d_out: int) -> RepresentationSpec:
        """Sphere×Sphere×scalarのrank-one atom family仕様を生成する。"""
        return cls(
            domain_in=Sphere(d_in),
            domain_out=Sphere(d_out),
            kernel_in="dot",
            kernel_out="dot",
            atom_cost=d_in + d_out + 1,
            retirement="gate_only",
        )

    def incident_synapse_ids(
        self, view: Any, neuron_ids: Tensor, *, side: str
    ) -> Tensor:
        """Expand neuron retirement according to v4 §2.4 semantics.

        Entry atoms have graph endpoints, so matching scalar ``s``/``t`` rows
        cascade to SynapseDeath.  Rank-one atoms have distributed components;
        step 6 therefore uses gate-only retirement and emits no synapse op.
        A component-projection operation is intentionally deferred to a future
        implementation of the full v4 §2.4 rank-one retirement semantics.
        """
        if side not in {"in", "out"}:
            raise ValueError("side must be 'in' or 'out'")
        if not isinstance(neuron_ids, Tensor):
            raise TypeError("neuron_ids must be a Tensor")
        if neuron_ids.ndim != 1 or neuron_ids.dtype != torch.int64:
            raise TypeError("neuron_ids must be a rank-1 int64 tensor")
        if self.retirement == "gate_only":
            return torch.zeros(0, dtype=torch.int64)
        coordinates = view.s if side == "in" else view.t
        if coordinates.ndim != 2 or coordinates.shape[1] != 1:
            raise ValueError("endpoint cascade requires scalar entry coordinates")
        values = coordinates[:, 0].detach().to(device="cpu", dtype=torch.int64)
        targets = neuron_ids.detach().to(device="cpu")
        incident = (values[:, None] == targets[None, :]).any(dim=1)
        return view.ids.detach().to(device="cpu")[incident]

    def functional_mass(self, w: Tensor, s: Tensor, t: Tensor) -> Tensor:
        """Return the isolated-atom Frobenius mass defined by this family."""
        if w.ndim != 1 or s.ndim != 2 or t.ndim != 2:
            raise ValueError("functional_mass expects rank-1 w and rank-2 s/t")
        if s.shape[0] != w.numel() or t.shape[0] != w.numel():
            raise ValueError("functional_mass tensors must share the atom count")
        if self.kernel_in == self.kernel_out == "delta":
            return w.abs()
        return (
            w.abs()
            * torch.linalg.vector_norm(s, dim=1)
            * torch.linalg.vector_norm(t, dim=1)
        )
