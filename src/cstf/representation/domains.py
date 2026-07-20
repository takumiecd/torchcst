"""座標domainの動的契約と標準domain実装。"""

from __future__ import annotations

from enum import Enum
from typing import MutableMapping, Protocol, runtime_checkable

import torch
from torch import Tensor


class ParameterRole(str, Enum):
    """座標tensorが学習parameterか固定bufferかを表す。"""

    PARAMETER = "parameter"
    BUFFER = "buffer"


Role = ParameterRole


@runtime_checkable
class CoordinateDomain(Protocol):
    """座標の検証・勾配射影・retraction・optimizer state射影を所有する契約。"""

    def parameter_role(self) -> ParameterRole: ...

    def validate_birth(self, coords: Tensor) -> None: ...

    def sample(self, n: int, rng: torch.Generator) -> Tensor: ...

    def lineage_key(self, coords: Tensor) -> Tensor | None: ...

    def project_grad(self, coords: Tensor, grad: Tensor) -> Tensor: ...

    def retract(self, coords: Tensor) -> Tensor: ...

    def project_state(
        self, coords: Tensor, opt_state: MutableMapping[str, object]
    ) -> None: ...


class IntegerGrid:
    """delta entry座標をint64 bufferとして扱う距離なし整数格子。"""

    def __init__(self, bounds: int | tuple[int, ...] | None = None) -> None:
        if isinstance(bounds, int):
            bounds = (bounds,)
        if bounds is not None:
            if not bounds or any(bound <= 0 for bound in bounds):
                raise ValueError("IntegerGrid bounds must be positive")
            bounds = tuple(int(bound) for bound in bounds)
        self.bounds = bounds

    def parameter_role(self) -> ParameterRole:
        """整数座標をbufferとして宣言する。"""
        return ParameterRole.BUFFER

    def validate_birth(self, coords: Tensor) -> None:
        """birth座標のrank・dtype・任意境界を検証する。"""
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
        """境界内から一様に整数行を標本化する（行の重複を許す）。"""
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
        """mixed-radix encodeにより各整数座標へ決定的なkeyを与える。"""
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
        """固定buffer座標への勾配を零へ射影する。"""
        if coords.shape != grad.shape:
            raise ValueError("coords and grad must have equal shape")
        return torch.zeros_like(grad)

    def retract(self, coords: Tensor) -> Tensor:
        """整数格子上の座標を検証してそのまま返す。"""
        self.validate_birth(coords)
        return coords

    def project_state(
        self, coords: Tensor, opt_state: MutableMapping[str, object]
    ) -> None:
        """固定buffer座標にはoptimizer stateが無いことを検証する。"""
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
        if not isinstance(coords, Tensor):
            raise TypeError("coords must be a Tensor")
        if coords.ndim != 2:
            raise ValueError("Sphere coords must be rank 2")
        if not coords.is_floating_point():
            raise TypeError("Sphere coords must have a floating dtype")
        if coords.shape[1] != self.dim:
            raise ValueError("Sphere coordinate dimension does not match dim")
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
        self._validate_pair(coords, grad)
        radial = (coords * grad).sum(dim=1, keepdim=True)
        return grad - radial * coords

    def retract(self, coords: Tensor) -> Tensor:
        """Normalize every row, rejecting undefined zero-row retractions."""
        if not isinstance(coords, Tensor) or coords.ndim != 2:
            raise ValueError("Sphere coords must be a rank-2 Tensor")
        if not coords.is_floating_point():
            raise TypeError("Sphere coords must have a floating dtype")
        if coords.shape[1] != self.dim:
            raise ValueError("Sphere coordinate dimension does not match dim")
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

    def _validate_pair(self, coords: Tensor, other: Tensor) -> None:
        if not isinstance(coords, Tensor) or not isinstance(other, Tensor):
            raise TypeError("coords and grad must be Tensors")
        if coords.ndim != 2 or coords.shape[1] != self.dim:
            raise ValueError("Sphere coords have the wrong shape")
        if coords.shape != other.shape:
            raise ValueError("coords and grad must have equal shape")
        if not coords.is_floating_point() or not other.is_floating_point():
            raise TypeError("Sphere coords and grad must have floating dtypes")


def _validate_sample_args(n: int, rng: torch.Generator) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator")
