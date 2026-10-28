"""Explicit execution choices for radius-native cell masses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RadiusCellMassRecipe:
    # Merge reuses DC and divisible frequencies; unmatched modes are direct.
    moment_method: str = "merge"
    save_response: bool = True
