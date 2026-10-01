"""Signed amplitude, clamped log bandwidth and center coordinates."""

from dataclasses import dataclass, field

from .base import ParameterizationSpec, positive


@dataclass(frozen=True, kw_only=True)
class LogWidthSpec(ParameterizationSpec):
    """Width is exp(clamp(log_sigma)); ordinary task gradients are retained."""

    sigma_min: float
    sigma_max: float
    id: str = field(default="signed_amplitude_log_width", init=False)

    def __post_init__(self):
        super().__post_init__()
        positive(self.sigma_min, "sigma_min")
        positive(self.sigma_max, "sigma_max")
        if self.sigma_min > self.sigma_max:
            raise ValueError("sigma_max must not be below sigma_min")
