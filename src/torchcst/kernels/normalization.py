"""The scope and mathematical rule of profile normalization."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class NormalizationSpec:
    """L2 normalization uses max(norm, floor) when a floor is specified.

    chart_sites normalizes a profile over all sites of its chart. In a
    separable kernel, input and output profiles are normalized separately.
    operator_sites instead normalizes each atom over the complete operator;
    it never means normalizing each execution window independently.

    continuous_mass divides a nonnegative density by its positive, finite
    integral over profile_support. It has no floor. Cell evaluation and
    boundary rules require a separate compatible profile/kernel contract;
    declaring this rule does not make existing point-sampled kernels support it.
    """

    kind: Literal["none", "discrete_l2", "continuous_mass"] = "none"
    domain: Literal["none", "chart_sites", "operator_sites", "profile_support"] = "none"
    floor: float | None = None

    def __post_init__(self):
        if self.kind == "none":
            if self.domain != "none" or self.floor is not None:
                raise ValueError("no normalization has no domain or floor")
        elif self.kind == "discrete_l2":
            if self.domain not in ("chart_sites", "operator_sites"):
                raise ValueError("L2 normalization needs an explicit domain")
            if self.floor is not None and (
                type(self.floor) not in (int, float)
                or not math.isfinite(self.floor)
                or self.floor <= 0
            ):
                raise ValueError("normalization floor must be finite and positive")
        elif self.kind == "continuous_mass":
            if self.domain != "profile_support":
                raise ValueError(
                    "continuous mass normalization requires profile_support"
                )

            if self.floor is not None:
                raise ValueError("continuous mass normalization does not allow a floor")
        else:
            raise ValueError("unknown normalization rule")
