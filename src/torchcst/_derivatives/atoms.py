"""Atom-structured local derivatives of a kernel sum."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import hessian as functional_hessian
from torch.func import jvp, vjp, vmap

from torchcst.atoms import Atoms

MaterializeAtoms = Callable[[Tensor], Tensor]


class AtomDerivatives:
    """Differentiate ``sum(kernel(p[a]))`` in opaque ``[K, P]`` coordinates."""

    def __init__(self, atoms: Atoms, materialize_atoms: MaterializeAtoms) -> None:
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an Atoms instance")
        if not callable(materialize_atoms):
            raise TypeError("materialize_atoms must be callable")
        self.atoms = atoms
        self._materialize_atoms = materialize_atoms

    @property
    def parameters(self) -> tuple[nn.Parameter]:
        """The opaque tensor owned by this atom site."""

        return (self.atoms.p,)

    @property
    def point_shape(self) -> tuple[int, int]:
        return self.atoms.count, self.atoms.parameter_dim

    def current_point(self) -> Tensor:
        """Return a detached ``[K, P]`` snapshot of the current atom state."""

        return self.atoms.p.detach().clone()

    def represented_atoms(self, parameter_point: Tensor | None = None) -> Tensor:
        """Return complete contributions with shape ``[K, *visible_shape]``."""

        parameter_point = self._point_or_current(parameter_point)
        operators = self._materialize_atoms(parameter_point)
        if operators.ndim < 2 or operators.shape[0] != self.atoms.count:
            raise ValueError("materialize_atoms must preserve the atom dimension")
        return operators

    def represented(self, parameter_point: Tensor | None = None) -> Tensor:
        """Return the canonical kernel sum."""

        return self.represented_atoms(parameter_point).sum(dim=0)

    def jvp(
        self, direction: Tensor, *, parameter_point: Tensor | None = None
    ) -> Tensor:
        """Evaluate ``J direction`` without materializing ``J``."""

        parameter_point = self._point_or_current(parameter_point)
        self._validate_local(direction, name="direction")
        return jvp(self.represented, (parameter_point,), (direction,))[1]

    def second(
        self,
        left: Tensor,
        right: Tensor,
        *,
        parameter_point: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the bilinear contraction ``H[left, right]``."""

        parameter_point = self._point_or_current(parameter_point)
        self._validate_local(left, name="left direction")
        self._validate_local(right, name="right direction")

        def right_jvp(candidate: Tensor) -> Tensor:
            return jvp(self.represented, (candidate,), (right,))[1]

        return jvp(right_jvp, (parameter_point,), (left,))[1]

    def displacement(
        self, direction: Tensor, *, parameter_point: Tensor | None = None
    ) -> Tensor:
        """Evaluate ``J d + 1/2 H[d, d]``."""

        parameter_point = self._point_or_current(parameter_point)
        return self.jvp(direction, parameter_point=parameter_point) + 0.5 * self.second(
            direction,
            direction,
            parameter_point=parameter_point,
        )

    def pushforward(
        self,
        vector: Tensor,
        *,
        at: Tensor | None = None,
        parameter_point: Tensor | None = None,
    ) -> Tensor:
        """Evaluate ``V(at) vector = J vector + H[at, vector]``."""

        parameter_point = self._point_or_current(parameter_point)
        self._validate_local(vector, name="vector")
        at = self._zero_or_valid(at, parameter_point, name="linearization direction")
        return self.jvp(vector, parameter_point=parameter_point) + self.second(
            at, vector, parameter_point=parameter_point
        )

    def pullback(
        self,
        cotangent: Tensor,
        *,
        at: Tensor | None = None,
        parameter_point: Tensor | None = None,
    ) -> Tensor:
        """Evaluate ``V(at).T cotangent`` with result shape ``[K, Q]``."""

        parameter_point = self._point_or_current(parameter_point)
        self._validate_cotangent(cotangent, parameter_point)
        at = self._zero_or_valid(at, parameter_point, name="linearization direction")

        def local_displacement(direction: Tensor) -> Tensor:
            return self.displacement(direction, parameter_point=parameter_point)

        _, apply_pullback = vjp(local_displacement, at)
        return apply_pullback(cotangent)[0]

    def contracted_hessian(
        self, cotangent: Tensor, *, parameter_point: Tensor | None = None
    ) -> Tensor:
        """Return only the nonzero ``cotangent ⌟ H`` blocks as ``[K, Q, Q]``."""

        parameter_point = self._point_or_current(parameter_point)
        self._validate_cotangent(cotangent, parameter_point)

        def contracted_atom(atom_point: Tensor) -> Tensor:
            operators = self._materialize_atoms(atom_point.unsqueeze(0))
            if operators.shape[0] != 1:
                raise ValueError("materialize_atoms must preserve the atom dimension")
            return (operators[0] * cotangent).sum()

        return vmap(functional_hessian(contracted_atom))(parameter_point)

    def _point_or_current(self, parameter_point: Tensor | None) -> Tensor:
        if parameter_point is None:
            return self.current_point()
        self._validate_local(parameter_point, name="parameter point")
        return parameter_point

    def _zero_or_valid(
        self, value: Tensor | None, like: Tensor, *, name: str
    ) -> Tensor:
        if value is None:
            return torch.zeros_like(like)
        self._validate_local(value, name=name)
        return value

    def _validate_local(self, value: Tensor, *, name: str) -> None:
        if value.shape != self.point_shape:
            raise ValueError(f"{name} must have shape {list(self.point_shape)}")
        if value.device != self.atoms.p.device or value.dtype != self.atoms.p.dtype:
            raise ValueError(f"{name} must match the atom device and dtype")

    def _validate_cotangent(self, cotangent: Tensor, parameter_point: Tensor) -> None:
        expected_shape = self.represented(parameter_point).shape
        if cotangent.shape != expected_shape:
            raise ValueError(f"cotangent must have shape {list(expected_shape)}")
        if (
            cotangent.device != parameter_point.device
            or cotangent.dtype != parameter_point.dtype
        ):
            raise ValueError("cotangent must match the atom device and dtype")
