"""Coordinate Adam plus atom-operator repulsion.

R is repulsion of realized atom operators, not parameter-coordinate decay.
The coupled path adds ``optimizer.repulsion_loss()`` to the task loss before
backward. Existing CST Adam and SGD classes are left unchanged.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from torchcst.nn import CSTLinear

from .config import AdamWConfig, _validate_betas

RepulsionKind = Literal["cosine", "raw"]


class CSTAdamR(torch.optim.AdamW):
    """Whole-model Adam with a coupled atom-operator repulsion term.

    Discover each ``CSTLinear`` site, read ``(S, κ)`` from that site, and
    expose ``λ (‖S‖_F² - κ)`` as ``repulsion_loss()``. The Adam update itself
    is ordinary coordinate AdamW with CST ``weight_decay=0`` by default.
    Future parameter decay stays an argument on this class; it is not a
    separate ``CSTAdamRW`` optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        repulsion: float = 0.0,
        kind: RepulsionKind = "cosine",
        weight_decay: float = 0.0,
        dense: AdamWConfig | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        betas = _validate_betas(betas)
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if (
            isinstance(repulsion, bool)
            or not math.isfinite(repulsion)
            or repulsion < 0
        ):
            raise ValueError("repulsion must be finite and nonnegative")
        if kind not in ("cosine", "raw"):
            raise ValueError("kind must be 'cosine' or 'raw'")
        if (
            isinstance(weight_decay, bool)
            or not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and nonnegative")
        if dense is not None and not isinstance(dense, AdamWConfig):
            raise TypeError("dense must be an AdamWConfig or None")

        sites = [module for module in model.modules() if isinstance(module, CSTLinear)]
        if not sites:
            raise ValueError("model must contain a CSTLinear")
        owners = set()
        for site in sites:
            if site.input_chart.trainable or site.output_chart.trainable:
                raise ValueError("CSTAdamR requires frozen charts")
            if site.atoms.grad is not None:
                raise ValueError("CST site already has an attached AtomGrad program")
            if id(site.atoms.p) in owners:
                raise ValueError("CST parameters cannot be shared between sites")
            owners.add(id(site.atoms.p))

        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        atom = [(name, p) for name, p in named if id(p) in owners]
        other = [(name, p) for name, p in named if id(p) not in owners]
        if not atom:
            raise ValueError("model must contain trainable CST parameters")
        if other and dense is None:
            raise ValueError("ordinary trainable parameters require dense=AdamWConfig")

        groups = [
            {
                "params": [p for _, p in atom],
                "param_names": [name for name, _ in atom],
                "param_shapes": [tuple(p.shape) for _, p in atom],
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        ]
        if other:
            groups.append(
                {
                    "params": [p for _, p in other],
                    "param_names": [name for name, _ in other],
                    "param_shapes": [tuple(p.shape) for _, p in other],
                    "lr": dense.lr,
                    "betas": dense.betas,
                    "eps": dense.eps,
                    "weight_decay": dense.weight_decay,
                }
            )
        super().__init__(groups, foreach=False)
        self.model = model
        self._sites = sites
        self.repulsion = float(repulsion)
        self.kind: RepulsionKind = kind

    def repulsion_energy(self) -> Tensor:
        """Return the sum of site energies ``‖S‖_F² - κ``."""

        total: Tensor | None = None
        for site in self._sites:
            energy = site.repulsion_energy(kind=self.kind)
            total = energy if total is None else total + energy
        if total is None:
            raise RuntimeError("CSTAdamR has no CSTLinear sites")
        return total

    def repulsion_loss(self) -> Tensor:
        """Return ``λ (‖S‖_F² - κ)`` for coupled autograd with the task loss."""

        return self.repulsion * self.repulsion_energy()
