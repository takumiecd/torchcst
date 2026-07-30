"""The minimal specification that uniquely describes a representation family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .domains import Box, CoordinateDomain, IntegerGrid, Sphere
from .kernels import CONTINUOUS_KERNELS


@dataclass(frozen=True)
class RepresentationSpec:
    """Bundle a family's domains, kernels, sites, atom price, and retirement semantics."""

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
        continuous = (
            self.kernel_in == self.kernel_out and self.kernel_in in CONTINUOUS_KERNELS
        )
        if not entry and not rank_one and not continuous:
            raise ValueError(
                "kernel pair must describe entry, rank-one, or continuous family"
            )
        if entry:
            if self.atom_cost != 1:
                raise ValueError("entry family atom_cost must be 1")
            if not isinstance(self.domain_in, IntegerGrid) or not isinstance(
                self.domain_out, IntegerGrid
            ):
                raise TypeError("entry family requires IntegerGrid domains")
        else:
            family, domain_type = (
                ("rank-one", Sphere) if rank_one else ("continuous", Box)
            )
            if not isinstance(self.domain_in, domain_type) or not isinstance(
                self.domain_out, domain_type
            ):
                raise TypeError(
                    f"{family} family requires {domain_type.__name__} domains"
                )
            expected = self.domain_in.dim + self.domain_out.dim + 1
            if self.atom_cost != expected:
                raise ValueError(f"{family} atom_cost must be d_in + d_out + 1")
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
        """Create the IntegerGrid×delta entry-family spec."""
        return cls(
            domain_in=IntegerGrid(bounds_in),
            domain_out=IntegerGrid(bounds_out),
        )

    @classmethod
    def rank_one(cls, d_in: int, d_out: int) -> RepresentationSpec:
        """Create the Sphere×Sphere×scalar rank-one atom-family spec."""
        return cls(
            domain_in=Sphere(d_in),
            domain_out=Sphere(d_out),
            kernel_in="dot",
            kernel_out="dot",
            atom_cost=d_in + d_out + 1,
            retirement="gate_only",
        )

    @classmethod
    def continuous(
        cls,
        d_in: int,
        d_out: int,
        *,
        bounds: tuple[Any, Any] = (0.0, 1.0),
        bounds_out: tuple[Any, Any] | None = None,
        kernel: str = "gaussian",
    ) -> RepresentationSpec:
        """Create a Box×Box continuous-coordinate family.

        ``kernel`` selects the profile: ``"gaussian"`` (global support) or
        ``"triangular"`` (compact support, and therefore the family that
        reaches the entry family's delta kernel as sigma shrinks).  Continuous
        functional mass depends on sampled neuron locations, sigma, and
        boundary effects, and the compact profile changes all three.
        Consequently the frozen entry/rank-one rent constants must not be
        inherited automatically, and neither may a Gaussian calibration be
        reused for a compact kernel: each requires fresh calibration.

        ``bounds`` is ``(lo, hi)``; each side may be one scalar (a cube) or
        one value per axis, because an isotropic chart is generally not a
        cube (see :class:`torchcst.representation.Box`).  ``bounds_out``
        defaults to ``bounds``, which is only right when both endpoint charts
        happen to share an extent -- pass it explicitly whenever they do not.
        """
        for name, value in (("bounds", bounds), ("bounds_out", bounds_out)):
            if value is None:
                continue
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError(f"{name} must be a (lo, hi) tuple")
        if kernel not in CONTINUOUS_KERNELS:
            raise ValueError(f"kernel must be one of {sorted(CONTINUOUS_KERNELS)}")
        lo, hi = bounds
        lo_out, hi_out = bounds if bounds_out is None else bounds_out
        return cls(
            domain_in=Box(lo, hi, d_in),
            domain_out=Box(lo_out, hi_out, d_out),
            kernel_in=kernel,
            kernel_out=kernel,
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
        if self.kernel_in == self.kernel_out and self.kernel_in in CONTINUOUS_KERNELS:
            raise RuntimeError(
                "continuous functional mass requires SynapseStore.mass_scale"
            )
        return (
            w.abs()
            * torch.linalg.vector_norm(s, dim=1)
            * torch.linalg.vector_norm(t, dim=1)
        )

    def merge_atoms(
        self,
        s1: Tensor,
        t1: Tensor,
        w1: Tensor | float,
        s2: Tensor,
        t2: Tensor,
        w2: Tensor | float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project two atom effects back into the rank-one family (CST S-4).

        The merged atom is the best Frobenius-norm rank-one approximation of
        ``w1 * s1 t1^T + w2 * s2 t2^T``.  SVD signs are canonicalized by making
        the largest-magnitude coordinate of each returned factor non-negative;
        the product of those two sign changes is folded into ``w``, hence
        ``w`` is ``+sigma`` or ``-sigma``.  This fixes the otherwise arbitrary
        singular-vector sign while preserving the represented matrix.

        This is the implementation of theoretical rule S-4: project the
        two-atom effect into the one-atom family before deleting its parents.
        Entry atoms deliberately have no merge on their discrete lattice.
        """
        self._validate_merge_factors(s1, t1, s2, t2)
        reference = s1
        left = torch.stack((s1, s2)).to(reference)
        right = torch.stack((t1, t2)).to(reference)
        raw_weights = tuple(
            torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
            for value in (w1, w2)
        )
        if any(value.numel() != 1 for value in raw_weights):
            raise ValueError("merge weights must be scalar")
        weights = torch.stack(tuple(value.reshape(()) for value in raw_weights))
        result_dtype = reference.dtype
        if result_dtype in {torch.float16, torch.bfloat16}:
            left = left.float()
            right = right.float()
            weights = weights.float()
        matrix = torch.einsum("k,ki,kj->ij", weights, left, right)
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        source = u[:, 0]
        target = vh[0]
        sigma = singular[0]

        source_sign = self._canonical_sign(source)
        target_sign = self._canonical_sign(target)
        source = source * source_sign
        target = target * target_sign
        weight = sigma * source_sign * target_sign
        return (
            source.to(dtype=result_dtype),
            target.to(dtype=result_dtype),
            weight.to(dtype=result_dtype),
        )

    def _validate_merge_factors(
        self, s1: Tensor, t1: Tensor, s2: Tensor, t2: Tensor
    ) -> None:
        if self.kernel_in != self.kernel_out or self.kernel_in != "dot":
            raise ValueError("entry-family merge is undefined on the discrete lattice")
        for name, value in (("s1", s1), ("t1", t1), ("s2", s2), ("t2", t2)):
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a rank-1 Tensor")
        if s1.shape != s2.shape or t1.shape != t2.shape:
            raise ValueError("merge factor dimensions must match")
        if s1.numel() != self.domain_in.dim or t1.numel() != self.domain_out.dim:
            raise ValueError("merge factors do not match the representation domains")
        if not all(value.is_floating_point() for value in (s1, t1, s2, t2)):
            raise TypeError("rank-one merge factors must have floating dtypes")

    @staticmethod
    def _canonical_sign(vector: Tensor) -> Tensor:
        pivot = vector[torch.argmax(vector.abs())]
        return torch.where(pivot < 0, pivot.new_tensor(-1.0), pivot.new_tensor(1.0))
