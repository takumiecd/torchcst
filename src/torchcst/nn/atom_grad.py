"""Linear-operation integration for optimizer-provided atom gradients."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from torchcst.atoms import AtomGrad

if TYPE_CHECKING:
    from .linear import CSTLinear

LinearAtomGradRoute = Literal["custom", "hooks"]


class LinearAtomGrad(AtomGrad):
    """Operation-specific contract between ``CSTLinear`` and an optimizer."""

    def __init__(self, *, mode: Literal["auto", "custom", "hooks"] = "auto") -> None:
        super().__init__(mode=mode)
        self._last_route: LinearAtomGradRoute | None = None

    @property
    def supports_custom_autograd(self) -> bool:
        """Whether this program implements the custom backward route."""

        return False

    @property
    def last_route(self) -> LinearAtomGradRoute | None:
        """The route selected by the most recent observed forward."""

        return self._last_route

    def apply(self, site: CSTLinear, inputs: Tensor) -> Tensor:
        """Run the selected custom-autograd or hook integration."""

        if not self.active:
            raise RuntimeError("LinearAtomGrad must be active before use")
        if self.atoms is not site.atoms:
            raise ValueError("LinearAtomGrad is attached to a different atom table")

        route = self._resolve_route()
        self._last_route = route
        generation = self.generation
        backend = site._resolved_backend()
        if route == "custom":
            return _CSTLinearAutograd.apply(
                inputs,
                site.atoms.p,
                site,
                self,
                generation,
                backend,
            )

        outputs = site._forward_from_p(inputs, site.atoms.p, backend=backend)
        return self._install_hook(site, inputs, outputs, generation=generation)

    def _resolve_route(self) -> LinearAtomGradRoute:
        if self.mode == "hooks":
            return "hooks"
        if self.supports_custom_autograd:
            return "custom"
        if self.mode == "custom":
            raise RuntimeError(
                "custom AtomGrad mode was requested, but this LinearAtomGrad "
                "does not implement custom autograd"
            )
        return "hooks"

    def _install_hook(
        self,
        site: CSTLinear,
        inputs: Tensor,
        outputs: Tensor,
        *,
        generation: int,
    ) -> Tensor:
        if not outputs.requires_grad:
            return outputs
        saved_inputs = inputs.detach().clone()

        def collect(output_gradient: Tensor) -> None:
            self._accumulate_linear(
                site,
                saved_inputs,
                output_gradient.detach(),
                parameter_gradient=None,
                generation=generation,
            )

        outputs.register_hook(collect)
        return outputs

    def _begin(self) -> None:
        self._last_route = None

    @abstractmethod
    def _accumulate_linear(
        self,
        site: CSTLinear,
        inputs: Tensor,
        output_gradient: Tensor,
        *,
        parameter_gradient: Tensor | None,
        generation: int,
    ) -> None:
        """Set or accumulate concrete values during a Linear backward."""


class _CSTLinearAutograd(torch.autograd.Function):
    """Custom reverse-mode bridge that lets ``LinearAtomGrad`` observe backward."""

    @staticmethod
    def forward(
        ctx: object,
        inputs: Tensor,
        p: Tensor,
        site: CSTLinear,
        atom_grad: LinearAtomGrad,
        generation: int,
        backend: Literal["factored", "materialized"],
    ) -> Tensor:
        ctx.save_for_backward(inputs, p)
        ctx.site = site
        ctx.atom_grad = atom_grad
        ctx.generation = generation
        ctx.backend = backend
        return site._forward_from_p(inputs, p, backend=backend)

    @staticmethod
    @once_differentiable
    def backward(ctx: object, output_gradient: Tensor) -> tuple[object, ...]:
        inputs, p = ctx.saved_tensors
        with torch.enable_grad():
            differentiable_inputs = inputs.detach().requires_grad_(True)
            differentiable_p = p.detach().requires_grad_(True)
            outputs = ctx.site._forward_from_p(
                differentiable_inputs,
                differentiable_p,
                backend=ctx.backend,
            )
            input_gradient, parameter_gradient = torch.autograd.grad(
                outputs,
                (differentiable_inputs, differentiable_p),
                output_gradient,
                create_graph=False,
            )

        ctx.atom_grad._accumulate_linear(
            ctx.site,
            inputs.detach(),
            output_gradient.detach(),
            parameter_gradient=parameter_gradient.detach(),
            generation=ctx.generation,
        )
        return input_gradient, parameter_gradient, None, None, None, None
