"""Shared CST site contract for Linear, Conv, and future families."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from torchcst.atoms import Atoms
from torchcst.geometry import Chart

RepulsionKind = Literal["cosine", "raw", "abs"]


class CSTModule(nn.Module):
    """A fixed-shape atom site.

    Family-specific backward stays on ``LinearAtomGrad`` or a future
    ``ConvAtomGrad``. This type does not unify those programs. Subclasses own
    one ``Atoms`` table, frozen charts, and the ``(S, κ)`` repulsion terms.
    ``CSTParameterAdam`` discovers every site through this contract. The N/D
    optimizer family supports ``CSTLinear`` through ``LinearAtomGrad`` and
    rejects other site families until they have observation programs.
    """

    atoms: Atoms

    def get_extra_state(self) -> dict[str, object]:
        """Record the CST site family and its operator layout."""

        return {
            "format_version": 1,
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

        return self.kernel.parameter_dof(*self.cst_charts())

    @property
    def cst_degrees_of_freedom(self) -> int:
        """Intrinsic degrees of freedom across this site's atom table."""

        return self.atoms.count * self.atom_parameter_dof

    def cst_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the fixed-shape parameters owned by this CST site."""

        raise NotImplementedError

    def cst_charts(self) -> tuple[Chart, ...]:
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
