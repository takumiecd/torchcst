"""Shared CST site contract for Linear, Conv, and future families."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from torchcst.atoms import Atoms
from torchcst.geometry import Chart

RepulsionKind = Literal["cosine", "raw"]


class CSTModule(nn.Module):
    """A fixed-shape atom site discovered by model-level CST optimizers.

    Family-specific backward stays on ``LinearAtomGrad`` or a future
    ``ConvAtomGrad``. This type does not unify those programs. Subclasses own
    one ``Atoms`` table, frozen charts, and the ``(S, κ)`` repulsion terms.
    """

    atoms: Atoms

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
