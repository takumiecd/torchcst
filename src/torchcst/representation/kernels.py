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

from math import isfinite
from typing import ClassVar

import torch
from torch import Tensor, nn

from torchcst._validation import require_real

#: Kernel names that :class:`torchcst.representation.RepresentationSpec`
#: accepts as a continuous family.  Kept here so the spec and the kernel
#: modules cannot disagree about which families exist.
CONTINUOUS_KERNELS: frozenset[str] = frozenset({"gaussian", "triangular"})


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

    def forward(self, query: Tensor, centers: Tensor) -> Tensor:
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
        centers = centers.to(device=query.device, dtype=query.dtype)
        squared_distance = (query[:, None, :] - centers[None, :, :]).square().sum(-1)
        return self.profile(squared_distance, sigma)


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

    def overlap(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        # <G_j, G_k> / ||G||^2 for two unit-height Gaussians of one
        # bandwidth: the correlation widens the variance to 2 sigma^2.
        return torch.exp(-squared_distance / (4.0 * sigma.square()))


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
