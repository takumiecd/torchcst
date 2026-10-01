"""Declaration of the Gaussian radial shape."""

from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class GaussianSpec(ProfileSpec):
    """f(q) = exp(-q/2); q is squared distance / sigma^2."""

    id: str = field(default="gaussian", init=False)
