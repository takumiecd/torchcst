"""Amplitude-dependent radius bounds and an independently learned ratio."""

from dataclasses import dataclass, field
from typing import ClassVar

from .base import ParameterizationSpec, positive


@dataclass(frozen=True, kw_only=True)
class RadiusEnvelopeSpec(ParameterizationSpec):
    """Rows [a, q, output_center, input_center], with q in [0, 1].

    U = radius_max, L = radius_min + (U-radius_min)*exp(-(a/tau)**2).
    The physical radius is exp((1-q)*log(L) + q*log(U)). Both a and q
    receive the complete task-loss chain rule. Centers are independent.
    """

    radius_min: float = 0.5
    radius_max: float = 50.0
    tau: float = 0.03
    amplitude_max: float = 1.0
    id: str = field(default="radius_envelope", init=False)
    optimizer_prefix: ClassVar = (("amplitude", 1), ("radius_ratio", 1))

    def __post_init__(self):
        super().__post_init__()
        for name in ("radius_min", "radius_max", "tau", "amplitude_max"):
            positive(getattr(self, name), name)
        if self.radius_min >= self.radius_max:
            raise ValueError("radius_min must be smaller than radius_max")
