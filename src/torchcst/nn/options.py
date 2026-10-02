"""Linear execution preferences, independent of chart/kernel mathematics."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LinearOptions:
    """Prefer full storage or bounded windows where an algorithm supports them."""

    memory: Literal["full", "window"] = "full"

    def __post_init__(self):
        if self.memory not in ("full", "window"):
            raise ValueError("memory must be 'full' or 'window'")
