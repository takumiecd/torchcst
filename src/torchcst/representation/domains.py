"""The coordinate-domain contract and the standard domain implementations."""

from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import MutableMapping, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor


class ParameterRole(str, Enum):
    """Whether a coordinate tensor is a learnable parameter or a fixed buffer."""

    PARAMETER = "parameter"
    BUFFER = "buffer"


Role = ParameterRole


@runtime_checkable
class CoordinateDomain(Protocol):
    """Owns coordinate validation, gradient projection, retraction, and
    optimizer-state projection for one coordinate space."""

    def parameter_role(self) -> ParameterRole: ...

    def validate_birth(self, coords: Tensor) -> None: ...

    def sample(self, n: int, rng: torch.Generator) -> Tensor: ...

    def lineage_key(self, coords: Tensor) -> Tensor | None: ...

    def project_grad(self, coords: Tensor, grad: Tensor) -> Tensor: ...

    def retract(self, coords: Tensor) -> Tensor: ...

    def project_state(
        self, coords: Tensor, opt_state: MutableMapping[str, object]
    ) -> None: ...


def _as_axis_tuple(
    value: float | Sequence[float], dim: int, name: str
) -> tuple[float, ...]:
    """Normalize a scalar-or-per-axis bound to one float per axis."""
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return (float(value),) * dim
        if value.ndim != 1 or value.numel() != dim:
            raise ValueError(f"{name} must be a scalar or have dim entries")
        return tuple(float(item) for item in value.detach().cpu())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),) * dim
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != dim:
            raise ValueError(f"{name} must be a scalar or have dim entries")
        return tuple(float(item) for item in values)
    raise TypeError(f"{name} must be a real scalar or a sequence of them")


def _validate_float_coords(coords: Tensor, dim: int, name: str) -> None:
    """Shared rank/dtype/dimension checks for continuous coordinate rows."""
    if not isinstance(coords, Tensor):
        raise TypeError("coords must be a Tensor")
    if coords.ndim != 2:
        raise ValueError(f"{name} coords must be rank 2")
    if not coords.is_floating_point():
        raise TypeError(f"{name} coords must have a floating dtype")
    if coords.shape[1] != dim:
        raise ValueError(f"{name} coordinate dimension does not match dim")


def _validate_coord_grad_pair(
    coords: Tensor, grad: Tensor, dim: int, name: str
) -> None:
    """Shared checks for a (coords, grad) pair handed to projection methods."""
    if not isinstance(coords, Tensor) or not isinstance(grad, Tensor):
        raise TypeError("coords and grad must be Tensors")
    if coords.ndim != 2 or coords.shape[1] != dim:
        raise ValueError(f"{name} coords have the wrong shape")
    if coords.shape != grad.shape:
        raise ValueError("coords and grad must have equal shape")
    if not coords.is_floating_point() or not grad.is_floating_point():
        raise TypeError(f"{name} coords and grad must have floating dtypes")


class IntegerGrid:
    """A metric-free integer lattice treating delta entry coordinates as int64 buffers."""

    def __init__(self, bounds: int | tuple[int, ...] | None = None) -> None:
        if isinstance(bounds, int):
            bounds = (bounds,)
        if bounds is not None:
            if not bounds or any(bound <= 0 for bound in bounds):
                raise ValueError("IntegerGrid bounds must be positive")
            bounds = tuple(int(bound) for bound in bounds)
        self.bounds = bounds

    def parameter_role(self) -> ParameterRole:
        """Integer coordinates are fixed buffers."""
        return ParameterRole.BUFFER

    def validate_birth(self, coords: Tensor) -> None:
        """Check birth coordinates for rank, dtype, and optional bounds."""
        if not isinstance(coords, Tensor):
            raise TypeError("coords must be a Tensor")
        if coords.ndim != 2:
            raise ValueError("IntegerGrid coords must be rank 2")
        if coords.dtype != torch.int64:
            raise TypeError("IntegerGrid coords must have dtype int64")
        if self.bounds is None:
            return
        if coords.shape[1] != len(self.bounds):
            raise ValueError("IntegerGrid coordinate dimension does not match bounds")
        bounds = torch.tensor(self.bounds, dtype=torch.int64, device=coords.device)
        if bool(((coords < 0) | (coords >= bounds)).any()):
            raise ValueError("IntegerGrid coordinates are out of bounds")

    def sample(self, n: int, rng: torch.Generator) -> Tensor:
        """Draw integer rows uniformly within bounds (duplicate rows allowed)."""
        _validate_sample_args(n, rng)
        if self.bounds is None:
            raise ValueError("IntegerGrid.sample requires bounds")
        device = getattr(rng, "device", torch.device("cpu"))
        columns = [
            torch.randint(bound, (n,), generator=rng, device=device)
            for bound in self.bounds
        ]
        if not columns:
            return torch.zeros((n, 0), dtype=torch.int64, device=device)
        return torch.stack(columns, dim=1).to(dtype=torch.int64)

    def lineage_key(self, coords: Tensor) -> Tensor:
        """Assign each integer coordinate a deterministic mixed-radix key."""
        self.validate_birth(coords)
        if self.bounds is None:
            raise ValueError("IntegerGrid.lineage_key requires bounds")
        result = torch.zeros(coords.shape[0], dtype=torch.int64, device=coords.device)
        limit = torch.iinfo(torch.int64).max
        cardinality = 1
        for column, bound in enumerate(self.bounds):
            if cardinality > limit // bound:
                raise OverflowError("IntegerGrid lineage key exceeds int64")
            result = result * bound + coords[:, column]
            cardinality *= bound
        return result

    def project_grad(self, coords: Tensor, grad: Tensor) -> Tensor:
        """Project gradients on fixed buffer coordinates to zero."""
        if coords.shape != grad.shape:
            raise ValueError("coords and grad must have equal shape")
        return torch.zeros_like(grad)

    def retract(self, coords: Tensor) -> Tensor:
        """Validate lattice coordinates and return them unchanged."""
        self.validate_birth(coords)
        return coords

    def project_state(
        self, coords: Tensor, opt_state: MutableMapping[str, object]
    ) -> None:
        """Assert that fixed buffer coordinates own no optimizer state."""
        self.validate_birth(coords)
        if opt_state:
            raise ValueError("IntegerGrid buffer coordinates cannot own optimizer state")


class Sphere:
    """Unit-sphere coordinates with retraction and tangent-state projection.

    ``lineage_key`` intentionally returns ``None``: a continuous domain has no
    coordinate-based candidate identity.  The engine/proposer assigns an
    entity-unique lineage key instead.  Consequently the 5c no-rebirth rule,
    which is a contract over a discrete candidate universe, is vacuously
    satisfied for continuously sampled coordinates.
    """

    _SIGNED_MOMENT_NAMES = frozenset(
        {"momentum_buffer", "exp_avg", "grad_avg", "momentum", "first_moment"}
    )

    def __init__(self, dim: int, *, tol: float = 1e-6) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("Sphere dim must be an int")
        if dim <= 0:
            raise ValueError("Sphere dim must be positive")
        if float(tol) <= 0:
            raise ValueError("Sphere tol must be positive")
        self.dim = dim
        self.tol = float(tol)

    def parameter_role(self) -> ParameterRole:
        return ParameterRole.PARAMETER

    def validate_birth(self, coords: Tensor) -> None:
        _validate_float_coords(coords, self.dim, "Sphere")
        norms = torch.linalg.vector_norm(coords, dim=1)
        if not bool(torch.isfinite(norms).all()):
            raise ValueError("Sphere coordinates must be finite")
        if not torch.allclose(
            norms, torch.ones_like(norms), atol=self.tol, rtol=0.0
        ):
            raise ValueError("Sphere coordinate rows must have unit norm")

    def sample(self, n: int, rng: torch.Generator) -> Tensor:
        """Gaussian directions normalized row-wise give uniform sphere draws."""
        _validate_sample_args(n, rng)
        device = getattr(rng, "device", torch.device("cpu"))
        coords = torch.randn((n, self.dim), generator=rng, device=device)
        if n == 0:
            return coords
        norms = torch.linalg.vector_norm(coords, dim=1, keepdim=True)
        if bool((norms == 0).any()):
            raise RuntimeError("Sphere sampler produced a zero direction")
        return coords / norms

    def lineage_key(self, coords: Tensor) -> None:
        """Continuous coordinates do not define candidate identity; see class docstring."""
        if not isinstance(coords, Tensor) or coords.ndim != 2:
            raise ValueError("Sphere coords must be a rank-2 Tensor")
        return None

    def project_grad(self, coords: Tensor, grad: Tensor) -> Tensor:
        """Project a row-wise gradient onto the tangent space at ``coords``."""
        _validate_coord_grad_pair(coords, grad, self.dim, "Sphere")
        radial = (coords * grad).sum(dim=1, keepdim=True)
        return grad - radial * coords

    def retract(self, coords: Tensor) -> Tensor:
        """Normalize every row, rejecting undefined zero-row retractions."""
        _validate_float_coords(coords, self.dim, "Sphere")
        norms = torch.linalg.vector_norm(coords, dim=1, keepdim=True)
        if bool((norms == 0).any()):
            raise ValueError("Sphere cannot retract a zero row")
        if not bool(torch.isfinite(norms).all()):
            raise ValueError("Sphere coordinates must be finite")
        return coords / norms

    def project_state(
        self, coords: Tensor, opt_state: MutableMapping[str, object]
    ) -> None:
        """Remove radial components from known signed optimizer moments.

        A state entry is projected only when its tensor shape equals ``coords``
        and its name belongs to the known signed-moment set
        (``momentum_buffer``, ``exp_avg``, ``grad_avg``, ``momentum``, or
        ``first_moment``).  Non-negative directionless accumulators such as
        ``exp_avg_sq`` and ``max_exp_avg_sq`` are deliberately left unchanged.
        """
        if not isinstance(coords, Tensor) or coords.ndim != 2:
            raise ValueError("Sphere coords must be a rank-2 Tensor")
        for name, value in opt_state.items():
            if (
                name in self._SIGNED_MOMENT_NAMES
                and isinstance(value, Tensor)
                and value.shape == coords.shape
            ):
                with torch.no_grad():
                    value.copy_(self.project_grad(coords, value))


class Box:
    """Axis-aligned continuous box with ordinary Euclidean coordinates.

    ``lo``/``hi`` accept either one scalar (a cube, the original contract) or
    one value per axis.  The per-axis form exists because an *isotropic
    chart* -- one whose neuron lattice has the same spacing on every axis, so
    that a single isotropic kernel bandwidth resolves every axis equally --
    is generally **not** a cube: an axis carrying ``C`` lattice points and an
    axis carrying ``k`` lattice points at equal spacing span
    ``(C-1)h`` and ``(k-1)h``.  Forcing such a chart into a cube is exactly
    the failure ``torchcst.compute.conv2d_neuron_coordinates``'s default
    produces (see its docstring): the short axes get stretched until one
    bandwidth can no longer reach across them, and every atom drawn from the
    cube lands where its kernel column is numerically zero.  Sampling,
    validation and retraction therefore have to know the real per-axis
    extent, or the geometry fix is defeated by the coordinate *law* the
    atoms are drawn from.

    ``lo``/``hi`` stay plain floats (and ``bounds`` a plain ``(lo, hi)``
    pair) whenever every axis agrees, so a cube behaves exactly as before,
    element for element.  When the axes differ they become per-axis tuples
    and callers that need a single number should use :attr:`width` -- the
    geometric-mean edge length, i.e. ``volume ** (1/dim)`` -- which is the
    quantity every uniform-pool spacing law actually depends on and which
    reduces to ``hi - lo`` on a cube.

    ``project_state`` is intentionally a no-op.  Unlike :class:`Sphere`, a
    box has no directional gauge or tangent-moment invariant: optimizer state
    remains meaningful after the coordinate itself is clamped to the box.
    ``lineage_key`` returns ``None`` by the common continuous-domain rule.

    ``retract_in_training`` (default ``True``) controls only whether
    :meth:`SynapseStore.retract_coordinates` -- called on every structural
    event -- clamps *live* coordinates back into the box during training.
    Measurement shows that projection is worth little there: letting
    coordinates drift outside the box between structural events scored
    slightly better on most seeds, and the box is really an initial-span
    choice, not a constraint learning must obey.  Sampling
    (:meth:`sample`/:meth:`SynapseStore` candidate pools) and structural
    write validation (:meth:`validate_birth`) always enforce the box
    regardless of this flag -- a candidate must still land inside the domain
    it was drawn to represent; only the *ordinary training-time* clamp is
    optional.  Set ``False`` to disable it; this is a per-instance flag, not
    a global switch, so different sites in the same run may choose
    differently.
    """

    def __init__(
        self,
        lo: float | Sequence[float],
        hi: float | Sequence[float],
        dim: int,
        *,
        retract_in_training: bool = True,
    ) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("Box dim must be an int")
        if dim <= 0:
            raise ValueError("Box dim must be positive")
        if not isinstance(retract_in_training, bool):
            raise TypeError("Box retract_in_training must be a bool")
        lower = _as_axis_tuple(lo, dim, "Box lo")
        upper = _as_axis_tuple(hi, dim, "Box hi")
        if any(not isfinite(value) for value in lower + upper):
            raise ValueError("Box bounds must be finite")
        if any(low >= high for low, high in zip(lower, upper)):
            raise ValueError("Box lo must be less than hi")
        self.dim = dim
        self.retract_in_training = retract_in_training
        self.lo_per_axis = lower
        self.hi_per_axis = upper
        self.widths = tuple(high - low for low, high in zip(lower, upper))
        self.uniform = len(set(lower)) == 1 and len(set(upper)) == 1
        # A cube keeps the original scalar attributes, so `hi - lo` in
        # existing callers is untouched; an anisotropic box exposes tuples,
        # which makes such an expression fail loudly instead of silently
        # meaning the wrong thing.
        self.lo: float | tuple[float, ...] = lower[0] if self.uniform else lower
        self.hi: float | tuple[float, ...] = upper[0] if self.uniform else upper
        self.bounds = (self.lo, self.hi)
        self.width = self._geometric_mean_width()
        self._lo_row = torch.tensor(lower, dtype=torch.float64).reshape(1, dim)
        self._hi_row = torch.tensor(upper, dtype=torch.float64).reshape(1, dim)
        # `retract` runs once per layer per training step, so the per-axis
        # bound rows are memoized per (device, dtype) rather than re-copied
        # from the host on every call. A cube never reaches this path at all.
        self._row_cache: dict[tuple[torch.device, torch.dtype], tuple[Tensor, Tensor]] = {}

    def _geometric_mean_width(self) -> float:
        """``volume ** (1/dim)``: the edge of the cube with this box's volume."""
        if self.uniform:
            return float(self.widths[0])
        return float(
            torch.tensor(self.widths, dtype=torch.float64).log().mean().exp()
        )

    def _rows(self, like: Tensor) -> tuple[Tensor, Tensor]:
        key = (like.device, like.dtype)
        rows = self._row_cache.get(key)
        if rows is None:
            rows = (
                self._lo_row.to(device=like.device, dtype=like.dtype),
                self._hi_row.to(device=like.device, dtype=like.dtype),
            )
            self._row_cache[key] = rows
        return rows

    def parameter_role(self) -> ParameterRole:
        return ParameterRole.PARAMETER

    def validate_birth(self, coords: Tensor) -> None:
        _validate_float_coords(coords, self.dim, "Box")
        if not bool(torch.isfinite(coords).all()):
            raise ValueError("Box coordinates must be finite")
        if self.uniform:
            outside = (coords < self.lo_per_axis[0]) | (coords > self.hi_per_axis[0])
        else:
            lo, hi = self._rows(coords)
            outside = (coords < lo) | (coords > hi)
        if bool(outside.any()):
            raise ValueError("Box coordinates are out of bounds")

    def sample(self, n: int, rng: torch.Generator) -> Tensor:
        """Draw rows uniformly from the closed box (up to RNG endpoint rules)."""
        _validate_sample_args(n, rng)
        device = getattr(rng, "device", torch.device("cpu"))
        unit = torch.rand((n, self.dim), generator=rng, device=device)
        if self.uniform:
            return unit.mul(self.hi_per_axis[0] - self.lo_per_axis[0]).add(
                self.lo_per_axis[0]
            )
        lo, hi = self._rows(unit)
        return unit.mul(hi - lo).add(lo)

    def lineage_key(self, coords: Tensor) -> None:
        self.validate_birth(coords)
        return None

    def project_grad(self, coords: Tensor, grad: Tensor) -> Tensor:
        """Return the Euclidean gradient unchanged."""
        _validate_coord_grad_pair(coords, grad, self.dim, "Box")
        return grad

    def retract(self, coords: Tensor) -> Tensor:
        """Retract by coordinate-wise clamping to the closed box."""
        _validate_float_coords(coords, self.dim, "Box")
        if not bool(torch.isfinite(coords).all()):
            raise ValueError("Box coordinates must be finite")
        if self.uniform:
            return coords.clamp(self.lo_per_axis[0], self.hi_per_axis[0])
        lo, hi = self._rows(coords)
        return coords.clamp(min=lo, max=hi)

    def project_state(
        self, coords: Tensor, opt_state: MutableMapping[str, object]
    ) -> None:
        """Leave optimizer state unchanged because a box has no direction gauge."""
        if not isinstance(coords, Tensor) or coords.ndim != 2:
            raise ValueError("Box coords must be a rank-2 Tensor")
        if coords.shape[1] != self.dim:
            raise ValueError("Box coordinate dimension does not match dim")


def _validate_sample_args(n: int, rng: torch.Generator) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator")
