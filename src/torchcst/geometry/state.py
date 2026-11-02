"""Own immutable Geometry metadata; no Tensor buffers or geometric computations."""

from dataclasses import asdict, replace

from torch import nn

from torchcst._builtin_callbacks import seal_functions as _seal_functions
from torchcst._coordinate_state import Placement, _validate_loaded_state

from .spec import (
    EuclideanGeometrySpec,
    FlatTorusGeometrySpec,
    GeometrySpec,
    SphereGeometrySpec,
)


class GeometryState(nn.Module):
    """Own immutable geometry metadata without fixed Tensor buffers."""

    def __init__(self, spec: GeometrySpec, *, device=None, dtype=None):
        super().__init__()
        if (
            type(spec)
            not in (
                EuclideanGeometrySpec,
                FlatTorusGeometrySpec,
                SphereGeometrySpec,
            )
            or spec.revision != 1
        ):
            raise ValueError("unsupported geometry declaration or revision")
        self._placement = Placement(device=device, dtype=dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.intrinsic_dim = spec.intrinsic_dim
        self.embedding_dim = spec.embedding_dim
        self.center_parameter_dim = spec.center_parameter_dim
        self.representation = getattr(spec, "representation", "ambient")

    @property
    def radius(self):
        return self.spec.radius

    @property
    def periods(self):
        return self.spec.periods

    @property
    def chart_margin(self):
        return self.spec.chart_margin

    @property
    def device(self):
        return self._placement.device

    @property
    def dtype(self):
        return self._placement.dtype

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._placement = self._placement.converted(fn)
        return result

    def declaration(self) -> GeometrySpec:
        """Validate and return fixed metadata at a configuration boundary."""
        return replace(self.spec)

    def get_extra_state(self):
        return {"format_version": 2, "geometry": asdict(self.spec)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("geometry checkpoint contract differs")


# Optional prepared updates may specialize only these original definitions.

_seal_functions(globals())
