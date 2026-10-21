"""Shape-only Fourier peak; periods and physical widths belong to bindings."""

from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class FourierPeakSpec(ProfileSpec):
    """Peak-one constrained cosine projection with dimensionless width q=I/L.

    The mother shape is (1-(r/I)**2)_+**4. I is the requested effective
    half-width; actual outside-I amplitude/energy must be qualified for q.
    No universal small-tail or nonnegativity guarantee is implied.

    M=4 imposes F(pi)=F''(pi)=0; M=8 also imposes F''''(pi)=0.
    The function itself is periodic in phase. This declaration has no tile,
    support mask, period-selection, normalization or CUDA launch settings.
    """

    id: str = field(default="fourier_peak", init=False)
    modes: int = 8

    def __post_init__(self):
        super().__post_init__()
        if type(self.modes) is not int or self.modes not in (4, 8):
            raise ValueError("modes must be 4 or 8")
