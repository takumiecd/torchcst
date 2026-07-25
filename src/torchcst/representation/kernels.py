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

    Subclasses set :attr:`family` to the name their representation spec
    declares and implement :meth:`profile`.
    """

    #: Spec-level family name; concrete kernels must override it.
    family: ClassVar[str] = ""

    def __init__(self, sigma: float | Tensor, learnable: bool = True) -> None:
        super().__init__()
        if not self.family:
            raise TypeError("ContinuousKernel subclasses must define a family name")
        if not isinstance(learnable, bool):
            raise TypeError("learnable must be a bool")
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

    @property
    def learnable(self) -> bool:
        return isinstance(self.sigma, nn.Parameter)

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """Map squared endpoint distances to kernel values."""
        raise NotImplementedError

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
