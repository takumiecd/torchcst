"""Coordinate Adam plus atom-operator repulsion.

R is repulsion of realized atom operators, not parameter-coordinate decay.
The default path is decoupled: Adam moments see the task gradient only, and
``step()`` then applies ``-lr λ ∇L``. ``coupled=True`` is the CE+λL oracle
that folds the repulsion gradient into those moments. Existing CST Adam and
SGD classes are left unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from torchcst.nn import CSTModule, RepulsionKind

from .config import AdamWConfig, _validate_betas


class CSTAdamR(torch.optim.AdamW):
    """Whole-model Adam with atom-operator repulsion applied in ``step()``.

    Discover each ``CSTModule`` site and read ``(S, κ)`` from that site.
    ``R`` is not mixed into the task loss. Coordinate ``weight_decay`` stays
    a separate argument; it is not a ``CSTAdamRW`` optimizer.
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
        coupled: bool = False,
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
        if not isinstance(coupled, bool):
            raise TypeError("coupled must be a bool")
        if dense is not None and not isinstance(dense, AdamWConfig):
            raise TypeError("dense must be an AdamWConfig or None")

        sites = [module for module in model.modules() if isinstance(module, CSTModule)]
        if not sites:
            raise ValueError("model must contain a CSTModule")
        owners = set()
        for site in sites:
            if any(chart.trainable for chart in site.cst_charts()):
                raise ValueError("CSTAdamR requires frozen charts")
            if site.atoms.grad is not None:
                raise ValueError("CST site already has an attached AtomGrad program")
            for parameter in site.cst_parameters():
                if id(parameter) in owners:
                    raise ValueError("CST parameters cannot be shared between sites")
                owners.add(id(parameter))

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
        self.coupled = coupled

    def repulsion_energy(self) -> Tensor:
        """Return the sum of site energies ``‖S‖_F² - κ``."""

        total: Tensor | None = None
        for site in self._sites:
            energy = site.repulsion_energy(kind=self.kind)
            total = energy if total is None else total + energy
        if total is None:
            raise RuntimeError("CSTAdamR has no CSTModule sites")
        return total

    def repulsion_loss(self) -> Tensor:
        """Return ``λ (‖S‖_F² - κ)`` for diagnostics and the coupled oracle."""

        return self.repulsion * self.repulsion_energy()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        repulsion_grads = None
        if self.repulsion > 0.0:
            repulsion_grads = self._atom_repulsion_grads()
            if self.coupled:
                self._accumulate_grads(self._atom_parameters(), repulsion_grads)

        self._validate_gradients()
        super().step()

        if repulsion_grads is not None and not self.coupled:
            lr = self.param_groups[0]["lr"]
            scale = -lr * self.repulsion
            with torch.no_grad():
                for parameter, grad in zip(self._atom_parameters(), repulsion_grads):
                    if grad is not None:
                        parameter.add_(grad, alpha=scale)
        return loss

    def _atom_parameters(self) -> list[nn.Parameter]:
        return self.param_groups[0]["params"]

    def _atom_repulsion_grads(self) -> tuple[Tensor | None, ...]:
        parameters = self._atom_parameters()
        with torch.enable_grad():
            grads = torch.autograd.grad(
                self.repulsion_energy(),
                parameters,
                allow_unused=True,
            )
        for grad in grads:
            if grad is not None and not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("non-finite repulsion gradient")
        return grads

    def _accumulate_grads(
        self, parameters: Iterable[nn.Parameter], grads: Iterable[Tensor | None]
    ) -> None:
        scale = self.repulsion
        for parameter, grad in zip(parameters, grads):
            if grad is None:
                continue
            update = grad * scale
            if parameter.grad is None:
                parameter.grad = update
            else:
                parameter.grad.add_(update)

    def _validate_gradients(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("CSTAdamR requires strided gradients")
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError("non-finite parameter gradient")
