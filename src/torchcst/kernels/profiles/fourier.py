"""Native finite periodic profiles; M changes the mathematical function."""

from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class ChebyshevRadiusSpec(ProfileSpec):
    """Chebyshev-square density with a direct physical support radius.

    The radius sets beta=sec(pi*R/P), without a sigma concentration floor.
    Both even and odd density degrees are defined by this contract.
    """

    modes: int = 8
    id: str = field(default="chebyshev_radius", init=False)

    def __post_init__(self):
        super().__post_init__()
        if type(self.modes) is not int or not 1 <= self.modes <= 512:
            raise ValueError("modes must be an integer in [1, 512]")
