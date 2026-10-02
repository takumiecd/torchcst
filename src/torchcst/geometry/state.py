"""Own scalar tensors of Geometry declarations; no geometric computations."""

from dataclasses import asdict, replace

import torch
from torch import nn

from torchcst._coordinate_state import _floating_dtype, _validate_loaded_state

from .spec import (
    EuclideanGeometrySpec,
    GeometrySpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)


class GeometryState(nn.Module):
    """Own fixed scalar tensors, while the immutable spec defines their meaning."""

    def __init__(self, spec: GeometrySpec, *, device=None, dtype=None):
        super().__init__()
        if (
            type(spec)
            not in (EuclideanGeometrySpec, SphereGeometrySpec, TorusGeometrySpec)
            or spec.revision != 1
        ):
            raise ValueError("unsupported geometry declaration or revision")
        dtype = _floating_dtype(dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.intrinsic_dim = spec.intrinsic_dim
        self.embedding_dim = spec.embedding_dim
        self.center_parameter_dim = spec.center_parameter_dim
        self.representation = getattr(spec, "representation", "ambient")
        self.circle_axis = getattr(spec, "circle_axis", None)
        self.max_arc_step = getattr(spec, "max_arc_step", None)
        for name in ("radius", "major_radius", "minor_radius", "chart_margin"):
            if hasattr(spec, name):
                self.register_buffer(
                    name,
                    torch.tensor(
                        getattr(spec, name),
                        device=device,
                        dtype=dtype or torch.get_default_dtype(),
                    ),
                )

    def declaration(self) -> GeometrySpec:
        """Snapshot current scalar buffers only at a configuration boundary."""
        return replace(
            self.spec,
            **{
                name: float(value.detach())
                for name, value in self.named_buffers(recurse=False)
            },
        )

    def get_extra_state(self):
        values = asdict(self.spec)
        for name in self._buffers:
            values.pop(name)
        return {"format_version": 1, "geometry": values}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("geometry checkpoint contract differs")
