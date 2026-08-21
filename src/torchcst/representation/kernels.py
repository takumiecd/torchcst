"""Continuous kernels for the general CST composition path.

Delta and dot products are intentionally absent here: they are the implicit
specialized kernels implemented by :class:`torchcst.compute.EntryLinear` and
:class:`torchcst.compute.RankOneLinear`, respectively.  Keeping those paths
specialized avoids materializing general kernel matrices for discrete entry
and factorized rank-one families.

The kernels defined here are *global-bandwidth isotropic* profiles of the
squared endpoint distance.  They differ only in that profile, so
:class:`ContinuousKernel` owns the bandwidth, the validation, and the distance
computation and each concrete family supplies one function.  The delta kernel
of the entry family is the ``sigma -> 0`` limit of a compact profile, so
:class:`TriangularKernel` is the only member of this module that reaches the
entry family continuously; :class:`GaussianKernel` cannot, because its support
is the whole domain at every positive bandwidth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from math import isfinite
from typing import ClassVar

import torch
from torch import Tensor, nn

from torchcst._validation import require_real

from .domains import ParameterRole

#: Kernel names that :class:`torchcst.representation.RepresentationSpec`
#: accepts as a continuous family.  Kept here so the spec and the kernel
#: modules cannot disagree about which families exist.
CONTINUOUS_KERNELS: frozenset[str] = frozenset({"gaussian", "triangular"})

#: Live family registry, keyed by :attr:`ContinuousKernel.family`.  Populated
#: by ``__init_subclass__`` so a kernel defined outside this module is a
#: first-class family: its name validates in a spec and its per-atom column
#: declaration reaches the store without any edit here.  ``CONTINUOUS_KERNELS``
#: stays the frozen built-in set for callers that import it.
_KERNEL_FAMILIES: dict[str, type["ContinuousKernel"]] = {}


def continuous_family_names() -> frozenset[str]:
    """Every continuous family name a spec will accept right now."""
    return frozenset(CONTINUOUS_KERNELS) | frozenset(_KERNEL_FAMILIES)


def family_atom_columns(name: str) -> tuple["AtomColumn", ...]:
    """Per-atom columns the named family requires beyond ``(s, t, w)``."""
    kernel = _KERNEL_FAMILIES.get(name)
    return () if kernel is None else tuple(kernel.atom_columns)


@dataclass(frozen=True)
class AtomColumn:
    """One per-atom column a kernel family requires beyond ``(s, t, w)``.

    A family that parameterises each atom with more than a position and an
    amplitude -- a per-atom bandwidth, a Gabor frequency and phase -- declares
    those columns here.  :class:`~torchcst.storage.SynapseStore` then installs
    each one as a capacity-shaped slot column that follows birth, death,
    remap, and capacity growth exactly like ``s`` and ``t`` do, and every view
    and birth op carries it under :attr:`name`.

    ``width`` is either a literal column width or a callable of the site's
    ``(d_in, d_out)``, so a frequency vector conjugate to both charts is
    declared once and sized per site.  ``init`` is the value a birth that does
    *not* mention the column is given, which is what lets an existing policy
    keep proposing plain ``(s, t, w)`` births at a site whose kernel has extra
    columns: the atom is simply born at the family's neutral value.
    """

    name: str
    width: int | Callable[[int, int], int] = 1
    role: ParameterRole = ParameterRole.PARAMETER
    init: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("AtomColumn.name must be a Python identifier")
        if self.name in {"s", "t", "w", "mass", "ids", "lineage"}:
            raise ValueError(
                f"AtomColumn.name {self.name!r} collides with a core column"
            )
        if not isinstance(self.role, ParameterRole):
            raise TypeError("AtomColumn.role must be a ParameterRole")

    def resolve_width(self, d_in: int, d_out: int) -> int:
        """The concrete column width at a site of these chart dimensions."""
        width = self.width(d_in, d_out) if callable(self.width) else self.width
        width = int(width)
        if width <= 0:
            raise ValueError(
                f"AtomColumn {self.name!r} resolved to a non-positive width"
            )
        return width


class ContinuousKernel(nn.Module):
    """Global-bandwidth isotropic kernel over a squared endpoint distance.

    The scalar ``sigma`` is an ``nn.Parameter`` when ``learnable=True`` and a
    buffer otherwise.  Per-atom bandwidths are outside step 9; a future v0.3
    implementation may revive the old ``SynapseStore.add_extra`` design for
    that purpose.

    ``forward`` validates that ``sigma`` is finite and positive on every
    call by default (``validate_sigma=True``, construction time is always
    validated regardless of this flag).  That guard costs two host syncs
    per call -- ``torch._assert_async`` would avoid them but corrupts the
    CUDA context on failure, too weak a contract for a general library, so
    the default stays a hard, sync-costing ``raise``.  Pass
    ``validate_sigma=False`` only when a caller can prove sigma is positive
    and finite by construction on every write after ``__init__`` too (e.g.
    it is always written as ``exp(x)`` for some finite real ``x``, and
    nothing else ever touches the buffer/parameter).

    Subclasses set :attr:`family` to the name their representation spec
    declares and implement :meth:`profile`.
    """

    #: Spec-level family name; concrete kernels must override it.
    family: ClassVar[str] = ""

    #: Per-atom columns this family needs beyond ``(s, t, w)``.  Empty for
    #: every family whose atom is fully described by a position and an
    #: amplitude, which is why declaring nothing leaves the store, the ops,
    #: and the views byte-for-byte as they were.
    atom_columns: ClassVar[tuple[AtomColumn, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        family = getattr(cls, "family", "")
        if family:
            _KERNEL_FAMILIES[family] = cls

    def __init__(
        self,
        sigma: float | Tensor,
        learnable: bool = True,
        *,
        validate_sigma: bool = True,
    ) -> None:
        super().__init__()
        if not self.family:
            raise TypeError("ContinuousKernel subclasses must define a family name")
        if not isinstance(learnable, bool):
            raise TypeError("learnable must be a bool")
        if not isinstance(validate_sigma, bool):
            raise TypeError("validate_sigma must be a bool")
        if isinstance(sigma, bool):
            raise TypeError("sigma must be a real scalar")
        value = torch.as_tensor(sigma)
        if value.numel() != 1 or value.is_complex():
            raise ValueError("sigma must be a real scalar")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        value = value.detach().reshape(()).clone()
        reading = float(value)
        if not isfinite(reading) or reading <= 0:
            raise ValueError("sigma must be finite and positive")
        if learnable:
            self.sigma = nn.Parameter(value)
        else:
            self.register_buffer("sigma", value)
        # Per-call sigma re-validation flag; the contract and the opt-out
        # conditions live in the class docstring.
        self._validate_sigma = validate_sigma

    @property
    def learnable(self) -> bool:
        return isinstance(self.sigma, nn.Parameter)

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """Map squared endpoint distances to kernel values."""
        raise NotImplementedError

    def profile_grad(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """``d profile / d squared_distance``.

        The family-dependent half of a coordinate Jacobian: for an atom at
        ``c`` read by a neuron at ``x``, ``d kappa / d c = profile_grad *
        2 (c - x)``, so a caller needing ``||d kappa / d c||^2`` forms
        ``4 * profile_grad**2 * squared_distance`` and never has to know
        which family it is holding.  :class:`~torchcst.optim.
        CoordPreconditioner` is the in-tree consumer.

        Families that cannot be differentiated in closed form leave this
        unimplemented; the consumer then raises instead of silently
        substituting a Gaussian derivative.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement profile_grad(); a "
            "consumer that needs coordinate derivatives cannot use this "
            "kernel family"
        )

    def scaled_profile_pair(
        self, squared_distance: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Profile and its distance derivative under one shared column scale.

        Each returned column may be multiplied by an arbitrary positive
        factor, but the value and derivative must carry the *same* factor.
        Consumers project the derivative off the column before dividing by
        its norm, so both that factor and its position derivative disappear.
        This is the derivative-side counterpart of :meth:`scaled_columns`.

        The generic form is sufficient for bounded profiles.  Exponential
        families override it to move their range-restoring shift inside the
        exponent before either value is formed.
        """
        return (
            self.profile(squared_distance, sigma),
            self.profile_grad(squared_distance, sigma),
        )

    def overlap(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """Normalised atom-atom overlap ``<kappa_j, kappa_k> / ||kappa||^2``.

        The geometry factor ``rho_jk`` that twin-control courts price with:
        ``1`` for coincident atoms, decaying to ``0`` as they separate.  It is
        the *continuous* inner product of two kernel bumps, not a sampled
        Gram -- courts that can afford the sampled version build a
        :class:`~torchcst.representation.gram.GramService` from delivered
        kernel columns instead, and are kernel-agnostic already.

        Only families with a closed-form self-correlation implement it.  The
        radial tent's is not elementary in general dimension, so
        :class:`TriangularKernel` deliberately inherits the raise: a court
        asked to price triangular twins must fail loudly rather than quote
        Gaussian numbers for a non-Gaussian site.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement overlap(); a court "
            "that prices geometric twin overlap cannot use this kernel family"
        )

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """:meth:`forward`'s columns, each up to its own positive scale.

        Same shape and same direction as :meth:`forward`, but every column is
        free to carry an arbitrary positive factor, because each caller
        divides that factor back out -- ``L2NormalizedColumns`` normalises to
        unit L2 norm.  Giving up the scale buys range: an atom far from every
        neuron casts a column whose values underflow, and a column that has
        already flattened to zero cannot be rescaled back.  In bf16/fp16 a
        Gaussian starts losing columns around four sigma, well inside a
        lawful chart, so this is not a corner case.

        The generic form below divides by each column's largest magnitude,
        which is exact for any family whose values survive being computed at
        all -- the bounded profiles need nothing more.  A family built on an
        exponential must override and fold the shift *inside* the exponent,
        the same move softmax makes with its maximum.

        A compactly supported family may return an honestly zero column, for
        an atom outside every neuron's support.  That is a fact about the
        geometry rather than a numerical failure, and callers clamp instead
        of dividing by it.
        """
        return peak_scaled(self(query, centers, columns))

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Kernel matrix between ``query`` rows and atom ``centers``.

        ``columns`` delivers the family's declared per-atom values
        (:attr:`atom_columns`), row-aligned with ``centers``: an atom's
        frequency or bandwidth cannot be recovered from its position, so a
        family that declares columns is handed them here rather than digging
        into the store.  The isotropic families ignore the argument, which is
        why every existing caller and subclass keeps working untouched -- a
        family that needs the values overrides this method.
        """
        sigma, centers = self._prepare(query, centers, columns)
        squared_distance = (query[:, None, :] - centers[None, :, :]).square().sum(-1)
        return self.profile(squared_distance, sigma)

    def _prepare(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Validate a kernel call; return ``sigma`` and device-matched centers.

        Split out of :meth:`forward` so a family that does not go through
        :meth:`profile` -- :class:`GaborKernel` -- still gets one shared
        validation path instead of a second, drifting copy.
        """
        if columns:
            for name, value in columns.items():
                if not isinstance(value, Tensor):
                    raise TypeError(f"column {name!r} must be a Tensor")
                if value.ndim != 2 or value.shape[0] != centers.shape[0]:
                    raise ValueError(
                        f"column {name!r} must have one row per center "
                        f"({centers.shape[0]}), got {tuple(value.shape)}"
                    )
        if not isinstance(query, Tensor) or not isinstance(centers, Tensor):
            raise TypeError("query and centers must be Tensors")
        if query.ndim != 2 or centers.ndim != 2:
            raise ValueError("query and centers must be rank-2 Tensors")
        if query.shape[1] != centers.shape[1]:
            raise ValueError("query and centers must share their coordinate dimension")
        if not query.is_floating_point() or not centers.is_floating_point():
            raise TypeError("continuous coordinates must have floating dtypes")
        sigma = self.sigma.to(device=query.device, dtype=query.dtype)
        if self._validate_sigma:
            # Two host syncs per call; see the class docstring for the
            # contract and who may disable this.
            if not bool(torch.isfinite(sigma)) or bool(sigma <= 0):
                raise ValueError("sigma must remain finite and positive")
        return sigma, centers.to(device=query.device, dtype=query.dtype)


class GaussianKernel(ContinuousKernel):
    """Global-bandwidth isotropic Gaussian kernel.

    ``kappa(query, centers) = exp(-||query-center||^2 / (2 sigma^2))``.  Its
    support is the whole domain for every positive ``sigma``: the tail never
    vanishes, which keeps a position gradient alive everywhere but also means
    the represented matrix is never structurally sparse.
    """

    family: ClassVar[str] = "gaussian"

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        return torch.exp(-squared_distance / (2.0 * sigma.square()))

    def profile_grad(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        return -self.profile(squared_distance, sigma) / (2.0 * sigma.square())

    def scaled_profile_pair(
        self, squared_distance: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor]:
        nearest = squared_distance.amin(dim=0, keepdim=True)
        value = torch.exp(
            -(squared_distance - nearest) / (2.0 * sigma.square())
        )
        return value, -value / (2.0 * sigma.square())

    def overlap(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        # <G_j, G_k> / ||G||^2 for two unit-height Gaussians of one
        # bandwidth: the correlation widens the variance to 2 sigma^2.
        return torch.exp(-squared_distance / (4.0 * sigma.square()))

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        # Subtracting each column's nearest squared distance inside the
        # exponent is exact -- it scales the column by exp(+d_min^2/2 sigma^2)
        # -- and it means the largest entry is always exactly one, so no
        # column ever underflows regardless of how far the atom has walked.
        sigma, centers = self._prepare(query, centers, columns)
        squared_distance = (query[:, None, :] - centers[None, :, :]).square().sum(-1)
        nearest = squared_distance.amin(dim=0, keepdim=True)
        return torch.exp(-(squared_distance - nearest) / (2.0 * sigma.square()))


class TriangularKernel(ContinuousKernel):
    """Global-bandwidth isotropic triangular kernel with compact support.

    ``kappa(query, centers) = relu(1 - ||query-center|| / sigma)``: exactly
    zero outside the radius-``sigma`` ball, so the represented matrix is
    structurally sparse and ``sigma -> 0`` approaches the entry family's delta
    kernel.  The profile is radial rather than a per-axis product, mirroring
    the Gaussian's isotropy; unlike the Gaussian it does *not* factor over
    coordinate axes.

    Two non-smooth points are handled rather than smoothed away.  The knee at
    ``r == sigma`` is a subgradient in the same sense as ``relu``.  At
    ``r == 0`` the Euclidean norm has no gradient, so the squared distance is
    floored at the smallest positive normal of its dtype before the square
    root: below that floor ``clamp_min`` passes zero gradient, which is the
    correct subgradient at a coincident endpoint, and above it the ratio
    ``|q - c| / r`` stays bounded by one.  The floor perturbs the returned
    value by ``sqrt(tiny) / sigma``, far below every supported dtype's
    resolution.
    """

    family: ClassVar[str] = "triangular"

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        floor = torch.finfo(squared_distance.dtype).tiny
        distance = squared_distance.clamp_min(floor).sqrt()
        return torch.relu(1.0 - distance / sigma)

    def profile_grad(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        # d/d(r^2) relu(1 - r/sigma) = -1/(2 sigma r) inside the support.
        # The 1/r pole at a coincident endpoint is the one ``profile``
        # already floors, and it cancels in the ``4 g^2 r^2`` form every
        # in-tree consumer uses.
        floor = torch.finfo(squared_distance.dtype).tiny
        distance = squared_distance.clamp_min(floor).sqrt()
        inside = (distance < sigma).to(squared_distance.dtype)
        return -inside / (2.0 * sigma * distance)

    # ``overlap`` is deliberately not implemented: the radial tent's
    # self-correlation is not elementary in general dimension.


class GaborKernel(ContinuousKernel):
    """Gaussian envelope times a per-atom plane wave: an oscillating atom.

    ``kappa(x, c) = exp(-||x-c||^2 / 2 sigma^2) * cos(omega . (x - c) + phi)``
    with ``omega`` and ``phi`` carried per atom.  At ``omega == 0`` the column
    is ``cos(phi)`` times the Gaussian -- the same shape, and exactly
    :class:`GaussianKernel` when the phase is zero too -- so a Gabor site
    starts as an ordinary CST site and can only gain from there.  It does not
    *start* at zero phase, though; see :attr:`atom_columns` for the critical
    point that forces a quarter turn.

    The point of the extra parameters is what a purely positive bump cannot
    write.  Structure finer than ``sigma`` is representable in the Gaussian
    dictionary only as the near-cancellation of two large opposite atoms --
    the twin pairs whose amplitudes grow like ``exp((sigma/l)^2 / 2)`` -- and
    that is a way of forging an oscillation the dictionary does not stock.
    Stocking it makes the same structure an ``O(1)`` single atom.  The band
    this opens runs from ``1/sigma`` up to the chart's Nyquist ``pi/spacing``:
    a frequency the neuron grid cannot resolve buys nothing.

    ``side`` selects which chart this instance reads, because the composition
    ``W = K_out diag(w) K_in^T`` is a product of two per-side factors and a
    jointly modulated 2-D Gabor does not factor through it.  Each side
    therefore carries its own frequency and phase, and the represented atom is
    the separable product of two 1-D Gabors -- oscillation along the input
    chart, the output chart, or both.

    Two inherited raises are deliberate.  There is no radial ``profile``: the
    value depends on the *direction* of ``x - c`` (and on the phase), which a
    squared distance has already discarded, so consumers that reduce a kernel
    to its radial profile -- ``CoordPreconditioner``, the closed-form
    backends -- refuse this family rather than silently using the envelope.
    And ``overlap`` stays unimplemented because two Gabor atoms at the same
    place with different frequencies are nearly orthogonal, not twins: an
    absorb court keyed on distance alone would merge distinct atoms, so it
    must fail loudly until it prices the full ``(mu, omega)`` address.
    """

    family: ClassVar[str] = "gabor"
    #: ``phi`` is born at a quarter turn, not at zero, and E-omega is why.
    #: At ``(omega, phi) == (0, 0)`` the derivative of the kernel column with
    #: respect to *both* new parameters vanishes identically -- ``sin(0)`` at
    #: every displacement -- so that point is not a stationary point of some
    #: loss but a critical point of the parameterisation itself: no target and
    #: no position can produce a gradient there, and an atom born there stays
    #: at zero frequency forever.  A quarter turn costs nothing (at
    #: ``omega == 0`` the column is ``cos(phi)`` times the Gaussian, the same
    #: shape with a constant the amplitude absorbs) and restores the gradient
    #: as soon as the atom is not exactly centred on an even feature.
    #: Measured: ``|dL/domega|`` goes 0 -> 1.9e-2 at a 0.2-sigma offset, and
    #: Adam then carries omega from 0 to within 5% of a target frequency it
    #: could not previously see.
    atom_columns: ClassVar[tuple[AtomColumn, ...]] = (
        AtomColumn("omega_s", width=lambda d_in, _d_out: d_in),
        AtomColumn("phi_s", width=1, init=math.pi / 4),
        AtomColumn("omega_t", width=lambda _d_in, d_out: d_out),
        AtomColumn("phi_t", width=1, init=math.pi / 4),
    )

    def __init__(
        self,
        sigma: float | Tensor,
        learnable: bool = True,
        *,
        side: str = "in",
        validate_sigma: bool = True,
    ) -> None:
        if side not in ("in", "out"):
            raise ValueError('side must be "in" or "out"')
        super().__init__(sigma, learnable, validate_sigma=validate_sigma)
        self.side = side

    @property
    def column_names(self) -> tuple[str, str]:
        """The ``(frequency, phase)`` column names this instance reads."""
        return ("omega_s", "phi_s") if self.side == "in" else ("omega_t", "phi_t")

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError(
            "GaborKernel has no radial profile: its value depends on the "
            "direction of the displacement and on the atom's phase, both of "
            "which a squared distance has discarded"
        )

    def envelope(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """The Gaussian envelope alone, without the modulation."""
        return torch.exp(-squared_distance / (2.0 * sigma.square()))

    def _components(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``(squared_distance, angle, sigma)`` -- what both column forms share."""
        frequency_name, phase_name = self.column_names
        if not columns or frequency_name not in columns or phase_name not in columns:
            raise ValueError(
                f"GaborKernel(side={self.side!r}) needs the {frequency_name!r} "
                f"and {phase_name!r} columns; the site's store must declare "
                "them (spec kernel 'gabor') and the caller must deliver them"
            )
        sigma, centers = self._prepare(query, centers, columns)
        # The [N, K, d] displacement cube is unavoidable here -- the phase is
        # a signed inner product, not a function of the distance -- so this
        # family costs d times the isotropic families' kernel memory.
        displacement = query[:, None, :] - centers[None, :, :]
        squared_distance = displacement.square().sum(-1)
        frequency = columns[frequency_name].to(displacement)
        phase = columns[phase_name].to(displacement)
        angle = (displacement * frequency[None, :, :]).sum(-1) + phase.reshape(1, -1)
        return squared_distance, angle, sigma

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        squared_distance, angle, sigma = self._components(query, centers, columns)
        return self.envelope(squared_distance, sigma) * torch.cos(angle)

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        # The envelope carries the whole dynamic range, so shifting it is
        # enough; the fringe is already O(1).  The resulting column's peak is
        # not one -- cos may be small where the envelope is largest -- which
        # the contract allows, because the scale is divided out downstream.
        squared_distance, angle, sigma = self._components(query, centers, columns)
        nearest = squared_distance.amin(dim=0, keepdim=True)
        return self.envelope(squared_distance - nearest, sigma) * torch.cos(angle)



def peak_scaled(values: Tensor) -> Tensor:
    """Divide each column by its largest magnitude, leaving zero columns zero."""
    peak = values.abs().amax(dim=0, keepdim=True)
    return values / peak.clamp_min(torch.finfo(values.dtype).tiny)


#: What a twin-control court may price geometric overlap with: the site's own
#: kernel (family-generic) or a bare Gaussian bandwidth (explicitly Gaussian).
OverlapScale = ContinuousKernel | float


def pairwise_overlap(scale: OverlapScale, squared_distance: Tensor) -> Tensor:
    """Geometry overlap ``rho`` for squared coordinate distances.

    ``scale`` is either the site's :class:`ContinuousKernel` -- the
    family-generic form, which delegates to :meth:`ContinuousKernel.overlap`
    and therefore *raises* rather than misprice a family with no closed-form
    self-correlation -- or a bare positive bandwidth, which selects the
    Gaussian ``exp(-d^2 / 4 sigma^2)`` explicitly.

    The float form is kept because these courts are configured with a scale
    rather than bound to a compute module (``propose`` sees only a view), and
    because it is the form every registered experiment used.  It is now an
    explicit request for Gaussian geometry, not an implicit assumption: a
    non-Gaussian site should hand the court its kernel instead.
    """
    if isinstance(scale, ContinuousKernel):
        sigma = scale.sigma.detach().to(squared_distance)
        return scale.overlap(squared_distance, sigma)
    return torch.exp(-squared_distance / (4.0 * float(scale) ** 2))


def require_overlap_scale(value: object, name: str) -> OverlapScale:
    """Validate a court's overlap scale: a kernel, or a positive real."""
    if isinstance(value, ContinuousKernel):
        return value
    return require_real(value, name, positive=True)
