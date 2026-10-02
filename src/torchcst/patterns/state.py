"""Bounded Tensor ownership for immutable axis-pattern declarations."""

from dataclasses import replace

import torch
from torch import nn

from torchcst._coordinate_state import (
    _coordinates,
    _floating_dtype,
    _validate_loaded_state,
)

from .spec import GridPatternSpec, LinePatternSpec, PatternSpec, PointsPatternSpec


class PatternState(nn.Module):
    """Own one small site pattern, never a Cartesian product table."""

    def __init__(self, spec: PatternSpec, *, device=None, dtype=None):
        super().__init__()
        if (
            type(spec) not in (GridPatternSpec, LinePatternSpec, PointsPatternSpec)
            or spec.revision != 1
        ):
            raise ValueError("unsupported pattern declaration")
        dtype = _floating_dtype(dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.features, self.dim = spec.features, spec.dim
        dtype = dtype or torch.get_default_dtype()
        if isinstance(spec, GridPatternSpec):
            self.shape = spec.shape
            self.register_buffer(
                "start", torch.tensor(spec.start, device=device, dtype=dtype)
            )
            self.register_buffer(
                "spacing", torch.tensor(spec.spacing, device=device, dtype=dtype)
            )
        else:
            self.register_buffer(
                "coordinates",
                torch.tensor(spec.coordinates, device=device, dtype=dtype),
            )

    @property
    def reference(self):
        return (
            self.start if isinstance(self.spec, GridPatternSpec) else self.coordinates
        )

    def declaration(self) -> PatternSpec:
        if isinstance(self.spec, GridPatternSpec):
            return replace(
                self.spec,
                start=tuple(self.start.detach().cpu().tolist()),
                spacing=tuple(self.spacing.detach().cpu().tolist()),
            )
        return replace(self.spec, coordinates=_coordinates(self.coordinates))

    def get_extra_state(self):
        return {
            "format_version": 1,
            "pattern": type(self.spec).__name__,
            "revision": self.spec.revision,
            "features": self.features,
            "dim": self.dim,
            "shape": getattr(self, "shape", None),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("site pattern checkpoint contract differs")
