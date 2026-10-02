"""Amplitude-dependent shared bandwidth declaration."""

from dataclasses import dataclass, field
from typing import Literal

from .base import ParameterizationSpec, positive


@dataclass(frozen=True, kw_only=True)
class AmpWidthSpec(ParameterizationSpec):
    """Signed amplitude followed by input/output center coordinates.

    couple_bandwidth=False preserves the forward width law but suppresses
    its task gradient, so it is part of the declared derivative contract.
    """

    sigma_min: float
    sigma_max: float
    tau: float = 5e-3
    temperature: float = 0.25
    gate_eps: float = 1e-12
    law: Literal["interpolating", "inverse"] = "interpolating"
    couple_bandwidth: bool = True
    id: str = field(default="amplitude_dependent_width", init=False)

    def __post_init__(self):
        super().__post_init__()
        for name in ("sigma_min", "sigma_max", "tau", "temperature", "gate_eps"):
            positive(getattr(self, name), name)
        if self.sigma_min > self.sigma_max:
            raise ValueError("sigma_max must not be below sigma_min")
        if self.law not in ("interpolating", "inverse"):
            raise ValueError("unknown amplitude-width law")
        if type(self.couple_bandwidth) is not bool:
            raise TypeError("couple_bandwidth must be a bool")
