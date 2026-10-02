"""Explicit point-table declaration and its Tensor owner."""

import math
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from torchcst._coordinate_state import _coordinates, _floating_dtype
from torchcst._validation import _points

from .base import ChartSpec, ChartState


@dataclass(frozen=True, kw_only=True)
class ExplicitChartSpec(ChartSpec):
    coordinates: tuple[tuple[float, ...], ...]
    trainable: bool = False
    spacing: tuple[float, ...] | None = None
    kind: str = field(default="explicit", init=False)

    def __post_init__(self):
        super().__post_init__()
        _points(self.coordinates)
        if len(self.coordinates) != self.features:
            raise ValueError("coordinate table size must match chart shape")
        if len(self.coordinates[0]) != self.geometry.embedding_dim:
            raise ValueError("explicit chart needs an ambient point table")
        if type(self.trainable) is not bool:
            raise ValueError("trainable must be a bool")
        if self.spacing is not None and (
            not isinstance(self.spacing, tuple)
            or len(self.spacing) != self.geometry.embedding_dim
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in self.spacing
            )
        ):
            raise ValueError("chart spacing must be a finite nonnegative tuple")


def _validate_loaded_points(state, incompatible_keys):
    from torchcst._backends.torch.geometry.execution import validate_points

    try:
        validate_points(state.geometry, state.coordinates, name="coordinates")
    except (ValueError, TypeError) as error:
        raise RuntimeError("invalid coordinate checkpoint configuration") from error


class ExplicitChartState(ChartState):
    spec_type = ExplicitChartSpec

    def __init__(self, spec, *, device=None, dtype=None):
        super().__init__(spec, device=device, dtype=dtype)
        from torchcst._backends.torch.geometry.execution import validate_points

        dtype = _floating_dtype(dtype)
        coordinates = torch.tensor(spec.coordinates, device=device, dtype=dtype)
        validate_points(self.geometry, coordinates, name="coordinates")
        if spec.trainable:
            self.coordinates = nn.Parameter(coordinates)
        else:
            self.register_buffer("coordinates", coordinates)
        self.register_buffer(
            "spacing",
            None
            if spec.spacing is None
            else torch.tensor(spec.spacing, device=device, dtype=dtype),
        )
        self.register_load_state_dict_post_hook(_validate_loaded_points)

    @property
    def reference(self):
        return self.coordinates

    def declaration(self) -> ExplicitChartSpec:
        return replace(
            self.spec,
            geometry=self.geometry.declaration(),
            coordinates=_coordinates(self.coordinates),
            spacing=None
            if self.spacing is None
            else tuple(self.spacing.detach().cpu().tolist()),
        )

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "trainable": self.trainable,
            "spacing": self.spacing is not None,
        }
