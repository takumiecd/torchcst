"""Normalized or Quadratic N/D Adam plus decoupled atom-operator repulsion.

``CSTAdamR`` uses Adam numerator and denominator moments. ``update_rule``
selects whether the task solver replaces or accumulates its displacement.
``R`` is applied to CST sites after the task solve, outside those moments.
Dense parameters still use the shared ``AdamWConfig`` block.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from torchcst.nn import CSTModule

from .config import AdamRConfig
from .nd_optimizer import _NDModelOptimizer
from .optimizer import _CSTProposal


class CSTAdamR(_NDModelOptimizer):
    """Normalized or Quadratic Adam plus decoupled atom-operator repulsion.

    Task moments and the selected solver see only the CST AtomGrad from the
    training loss. ``step()`` then adds ``-lr λ ∇_p R`` to the CST
    displacement and reprojects it. ``R`` is repulsion of realized atom operators, not
    parameter-coordinate decay, and not dense AdamW.
    """

    config_type = AdamRConfig
    _use_numerator_moment = True
    _use_denominator_moment = True

    def repulsion_energy(self) -> Tensor:
        """Return the sum of site energies ``||S||_F^2 - κ``."""

        kind = self.cst_config.kind
        total: Tensor | None = None
        for site in self._sites:
            energy = site.module.repulsion_energy(kind=kind)
            total = energy if total is None else total + energy
        if total is None:
            raise RuntimeError("CSTAdamR has no CST sites")
        return total

    def _build_cst_proposals(self) -> tuple[_CSTProposal, ...]:
        proposals = super()._build_cst_proposals()
        scale = self.cst_config.lr * self.cst_config.repulsion
        if scale == 0.0:
            return proposals
        return tuple(
            self._with_decoupled_repulsion(proposal, scale) for proposal in proposals
        )

    def _with_decoupled_repulsion(
        self, proposal: _CSTProposal, scale: float
    ) -> _CSTProposal:
        module = proposal.site.module
        if not isinstance(module, CSTModule):
            raise TypeError("CSTAdamR repulsion requires a CSTModule site")
        grad = self._repulsion_grad(module)
        solver = self.cst_config.solver
        displacement = solver.project_displacement(
            proposal.solve.displacement - scale * grad,
            trust_radius=self.cst_config.trust_radius,
        )
        solve = replace(
            proposal.solve,
            displacement=displacement,
            on_boundary=solver.displacement_is_on_boundary(
                displacement,
                trust_radius=self.cst_config.trust_radius,
            ),
        )
        self._validate_solve(solve, proposal.context)
        return _CSTProposal(
            proposal.site,
            proposal.context,
            proposal.expanded,
            solve,
        )

    def _repulsion_grad(self, module: CSTModule) -> Tensor:
        parameter = module.atoms.p
        with torch.enable_grad():
            energy = module.repulsion_energy(kind=self.cst_config.kind)
            (grad,) = torch.autograd.grad(energy, parameter)
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("non-finite repulsion gradient")
        return grad.detach()

    def _moment_contract(self):
        contract = super()._moment_contract()
        contract["update_rule"] = self.cst_config.update_rule
        contract["repulsion"] = self.cst_config.repulsion
        contract["kind"] = self.cst_config.kind
        return contract
