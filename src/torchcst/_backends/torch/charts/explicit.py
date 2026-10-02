"""Explicit point-table evaluation."""

from __future__ import annotations

from torch import Tensor

from torchcst._backends.torch.geometry import execution as _geometry


def positions(self, indices: Tensor) -> Tensor:
    """Return coordinates for requested flattened site indices."""

    return self.coordinates.index_select(0, indices)


def squared_distance(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    """Pairwise site-center squared distance in this chart's geometry."""

    sites = self.coordinates if selection is None else self.coordinates[selection]
    return _geometry.squared_distance(self.geometry, sites, centers)


def center_offsets(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    """Site-center offsets in each center's tangent space."""

    sites = self.coordinates if selection is None else self.coordinates[selection]
    return _geometry.center_offsets(self.geometry, sites, centers)


def initialize_centers(self, atoms: int, *, mode: str) -> Tensor:
    """Initialize atom centers in the chart's geometry."""

    return _geometry.initialize_centers(
        self.geometry, self.coordinates, atoms, mode=mode
    )
