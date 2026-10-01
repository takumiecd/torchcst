"""Scalar function shapes, independent of coordinates and execution."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ProfileSpec:
    """Identity fixes the function of dimensionless squared distance q.

    Bandwidth, center coordinates and normalization belong to separate
    declarations. No tensor evaluation methods belong on this type.
    """

    id: str
    revision: int = 1

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("profile ID must be a nonempty string")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("profile revision must be a positive integer")
