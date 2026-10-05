"""Shared CST site contract for Linear, Conv, and future families."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from torchcst.atoms import Atoms, AtomState
from torchcst.charts import ChartState

RepulsionKind = Literal["cosine", "raw", "abs"]


class CSTModule(nn.Module):
    """A fixed-shape atom site discovered by CSTOptimizer.

    Subclasses own their atom table and charts; ordinary backend autograd
    supplies gradients. The Kernel owns the coordinate update policy.
    """

    atom_state: AtomState

    @property
    def atoms(self) -> Atoms:
        return self.atom_state.atoms

    def _bind_atoms(self, atoms: Atoms) -> None:
        self.atom_state = AtomState.for_atoms(atoms)

    def __setattr__(self, name, value):
        if name == "atoms":
            if hasattr(self, "atom_state"):
                self.atom_state._require_boundary()
            self._bind_atoms(value)
            return
        super().__setattr__(name, value)

    def algorithm_state(self, algorithm, **configuration):
        """Own transient state for an algorithm/configuration, not in checkpoints."""
        from torchcst._backends.algorithm import Algorithm

        if not isinstance(algorithm, Algorithm):
            raise TypeError("expected Algorithm")
        states = self.__dict__.setdefault("_algorithm_states", [])
        for state in states:
            if (
                state.algorithm is algorithm
                and state.atom_state is self.atom_state
                and state.configuration == configuration
            ):
                return state
        state = algorithm.create_state(self.atom_state, **configuration)
        states.append(state)
        return state

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_algorithm_states", None)
        return state

    def get_extra_state(self) -> dict[str, object]:
        """Record the CST site family and its operator layout."""

        return {
            "format_version": 2,
            "site_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "layout": self._checkpoint_layout(),
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("CST site checkpoint contract differs from this site")

    def _checkpoint_layout(self) -> dict[str, object]:
        """Family-specific non-tensor settings that change the operator."""

        return {}

    @property
    def atom_parameter_dof(self) -> int:
        """Intrinsic degrees of freedom in one stored atom row."""
        from torchcst._backends.torch.kernels import execution as _kernel

        return _kernel.parameter_dof(self.kernel, *self.cst_charts())

    @property
    def cst_degrees_of_freedom(self) -> int:
        """Intrinsic degrees of freedom across this site's atom table."""

        return self.atoms.count * self.atom_parameter_dof

    def cst_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the fixed-shape parameters owned by this CST site."""

        raise NotImplementedError

    def cst_charts(self) -> tuple[ChartState, ...]:
        """Return the charts whose coordinates this site observes."""

        raise NotImplementedError

    def repulsion_terms(
        self, *, kind: RepulsionKind = "cosine"
    ) -> tuple[Tensor, Tensor]:
        """Return ``(S, κ)`` for the atom-operator repulsion identity.

        ``S`` has the realized operator shape of this family. ``κ`` is a
        scalar. The energy is ``||S||_F^2 - κ``.
        """

        raise NotImplementedError

    def repulsion_energy(self, *, kind: RepulsionKind = "cosine") -> Tensor:
        """Return the scalar pair-repulsion energy ``||S||_F^2 - κ``."""

        summed, kappa = self.repulsion_terms(kind=kind)
        return summed.square().sum() - kappa
