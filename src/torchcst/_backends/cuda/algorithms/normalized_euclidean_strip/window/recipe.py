"""Validated launch settings; mathematical meaning stays in OperatorSpec."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowRecipe:
    id: str = "normalized_window.rows512.v1"
    window_rows: int = 512
    enable_fp_fusion: bool = True
