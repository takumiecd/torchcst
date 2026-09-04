"""Optimizer-provided, transient gradients for atom tables."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .atoms import Atoms

AtomGradMode = Literal["auto", "custom", "hooks"]


class AtomGrad(ABC):
    """Mutable, step-local information produced while autograd runs.

    An optimizer supplies a concrete instance and attaches it to ``Atoms``.
    The object is deliberately not a module: it owns transient observations,
    not model parameters or checkpoint state.
    """

    def __init__(self, *, mode: AtomGradMode = "auto") -> None:
        if mode not in ("auto", "custom", "hooks"):
            raise ValueError("mode must be 'auto', 'custom', or 'hooks'")
        self.mode: AtomGradMode = mode
        self._atoms_ref: weakref.ReferenceType[Atoms] | None = None
        self._active = False
        self._complete = False
        self._generation = 0

    @property
    def atoms(self) -> Atoms:
        """The atom table to which this object is attached."""

        atoms = self._atoms_ref() if self._atoms_ref is not None else None
        if atoms is None:
            raise RuntimeError("AtomGrad is not attached to an Atoms instance")
        return atoms

    @property
    def active(self) -> bool:
        """Whether subsequent forwards belong to the current capture scope."""

        return self._active

    @property
    def completed(self) -> bool:
        """Whether the current observations are ready for optimizer use."""

        return self._complete

    @property
    def generation(self) -> int:
        """Identify the current scope so stale backward callbacks are ignored."""

        return self._generation

    def begin(self) -> None:
        """Start observing the next base-point backward pass."""

        _ = self.atoms
        if self._active:
            raise RuntimeError("AtomGrad is already active")
        self._generation += 1
        self.clear()
        self._active = True
        self._complete = False
        self._begin()

    def complete(self) -> None:
        """Finalize accumulated observations and stop recording callbacks."""

        if not self._active:
            raise RuntimeError("AtomGrad is not active")
        self._complete_values()
        self._active = False
        self._complete = True
        self._generation += 1

    def cancel(self, *, clear: bool = True) -> None:
        """Abandon the active scope and invalidate its pending callbacks."""

        if not self._active:
            raise RuntimeError("AtomGrad is not active")
        self._active = False
        self._complete = False
        self._generation += 1
        if clear:
            self.clear()

    def clear(self) -> None:
        """Discard all transient values without detaching from the atom table."""

        self._clear_values()
        self._complete = False

    def accepts(self, *, generation: int) -> bool:
        """Return whether a backward callback belongs to the active scope."""

        return self._active and generation == self._generation

    def require_complete(self) -> None:
        """Reject optimizer reads before the backward scope is finalized."""

        if not self._complete:
            raise RuntimeError("AtomGrad has not been completed")

    def _attach(self, atoms: Atoms) -> None:
        current = self._atoms_ref() if self._atoms_ref is not None else None
        if current is not None and current is not atoms:
            raise ValueError("an AtomGrad instance cannot be shared by atom tables")
        self._atoms_ref = weakref.ref(atoms)

    def _detach(self, atoms: Atoms) -> None:
        current = self._atoms_ref() if self._atoms_ref is not None else None
        if current is atoms:
            if self._active:
                raise RuntimeError("cannot detach an active AtomGrad")
            self._atoms_ref = None

    def _begin(self) -> None:
        """Optional operation-specific setup after a scope starts."""

    @abstractmethod
    def _clear_values(self) -> None:
        """Clear concrete optimizer observations."""

    @abstractmethod
    def _complete_values(self) -> None:
        """Finalize concrete optimizer observations."""
