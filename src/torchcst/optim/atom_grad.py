"""Atom-gradient observations used by the retained N/D optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import grad as functional_grad
from torch.func import vmap

from torchcst.atoms import AtomGradMode
from torchcst.nn import CSTLinear, LinearAtomGrad


@dataclass(frozen=True)
class AtomGradRequest:
    """Unionable observations requested by N/D moments."""

    jg: bool = False
    gh: bool = False

    def __or__(self, other: AtomGradRequest) -> AtomGradRequest:
        if not isinstance(other, AtomGradRequest):
            return NotImplemented
        return AtomGradRequest(jg=self.jg or other.jg, gh=self.gh or other.gh)

    @property
    def any(self) -> bool:
        return self.jg or self.gh


@dataclass(frozen=True)
class AtomGradientObservation:
    """Detached snapshot of the local gradient and optional curvature."""

    jg: Tensor | None = None
    gh: Tensor | None = None
    contributions: int = 0

    def require(self, request: AtomGradRequest) -> None:
        missing = []
        if request.jg and self.jg is None:
            missing.append("jg")
        if request.gh and self.gh is None:
            missing.append("gh")
        if missing:
            raise ValueError(f"observation is missing: {', '.join(missing)}")


CurvatureBlockMode = Literal["full", "no_m", "m_only", "none"]


@dataclass(frozen=True)
class CurvatureBlockMask:
    """Select blocks of an atom-local curvature observation.

    ``split`` partitions the final two axes into ``[:split]`` and
    ``[split:]``. ``no_m`` retains the two diagonal blocks, ``m_only`` retains
    the off-diagonal mixed blocks, and ``none`` removes all curvature. The
    mask is applied before numerator or denominator moments are expanded.
    """

    split: int
    mode: CurvatureBlockMode = "full"

    def __post_init__(self) -> None:
        if isinstance(self.split, bool) or not isinstance(self.split, int):
            raise TypeError("curvature block split must be an integer")
        if self.split < 1:
            raise ValueError("curvature block split must be positive")
        if self.mode not in ("full", "no_m", "m_only", "none"):
            raise ValueError(
                "curvature block mode must be 'full', 'no_m', 'm_only', or 'none'"
            )

    def apply(self, observation: AtomGradientObservation) -> AtomGradientObservation:
        """Return an observation whose ``gh`` has the selected block mask."""

        if not isinstance(observation, AtomGradientObservation):
            raise TypeError("observation must be an AtomGradientObservation")
        hessian = observation.gh
        if hessian is None or self.mode == "full":
            return observation
        if hessian.ndim != 3 or hessian.shape[-1] != hessian.shape[-2]:
            raise ValueError("gh must have shape [K, P, P]")
        parameters = hessian.shape[-1]
        if self.split >= parameters:
            raise ValueError("curvature block split must be smaller than P")

        masked = torch.zeros_like(hessian)
        if self.mode == "no_m":
            masked[:, : self.split, : self.split] = hessian[
                :, : self.split, : self.split
            ]
            masked[:, self.split :, self.split :] = hessian[
                :, self.split :, self.split :
            ]
        elif self.mode == "m_only":
            masked[:, : self.split, self.split :] = hessian[
                :, : self.split, self.split :
            ]
            masked[:, self.split :, : self.split] = hessian[
                :, self.split :, : self.split
            ]
        return AtomGradientObservation(
            jg=observation.jg,
            gh=masked,
            contributions=observation.contributions,
        )


class _LinearNDAtomGrad(LinearAtomGrad):
    """Shared backward collector for JG and JGH observations."""

    include_curvature = False

    def __init__(
        self,
        *,
        mode: AtomGradMode = "auto",
        factored: bool = False,
    ) -> None:
        super().__init__(mode=mode)
        if not isinstance(factored, bool):
            raise TypeError("factored must be a bool")
        self.factored = factored
        self._jg: Tensor | None = None
        self._gh: Tensor | None = None
        self._contributions = 0

    @property
    def supports_custom_autograd(self) -> bool:
        return True

    @property
    def observation_request(self) -> AtomGradRequest:
        return AtomGradRequest(jg=True, gh=self.include_curvature)

    @property
    def contributions(self) -> int:
        return self._contributions

    def snapshot(self) -> AtomGradientObservation:
        self.require_complete()
        return AtomGradientObservation(
            jg=self._jg.clone() if self._jg is not None else None,
            gh=self._gh.clone() if self._gh is not None else None,
            contributions=self._contributions,
        )

    def _clear_values(self) -> None:
        self._jg = None
        self._gh = None
        self._contributions = 0

    def _complete_values(self) -> None:
        if self._contributions == 0:
            raise RuntimeError("no Linear backward contribution was captured")
        if self._jg is None or (self.include_curvature and self._gh is None):
            raise RuntimeError("requested N/D observations were not captured")

    def _accumulate_linear(
        self,
        site: CSTLinear,
        inputs: Tensor,
        output_gradient: Tensor,
        *,
        parameter_gradient: Tensor | None,
        generation: int,
    ) -> None:
        if not self.accepts(generation=generation):
            return
        if any(chart.trainable for chart in site.cst_charts()):
            raise ValueError("N/D atom gradients require frozen charts")
        self._validate_linear_tensors(site, inputs, output_gradient)

        flat_inputs = inputs.detach().reshape(-1, site.in_features)
        flat_output_gradient = output_gradient.detach().reshape(-1, site.out_features)
        point = site.atoms.p.detach()
        gh = None

        if self.factored:
            from torchcst._derivatives._captured import call

            if not site.kernel.supports_factorization:
                raise ValueError("factored observations require factor-capable kernels")
            operation = (
                "factor_jgh_observation"
                if self.include_curvature
                else "factor_jg_observation"
            )
            observed = call(
                operation,
                site._factor_atoms,
                point,
                flat_inputs,
                flat_output_gradient,
            )
            observed_jg, gh = observed if self.include_curvature else (observed, None)
            if parameter_gradient is None:
                parameter_gradient = observed_jg
        else:
            from torchcst._derivatives._hessian import hessian_from_gradient

            with torch.enable_grad():

                def contracted_atom(atom_point: Tensor) -> Tensor:
                    atom = site._materialize_atoms(atom_point.unsqueeze(0))[0]
                    return (F.linear(flat_inputs, atom) * flat_output_gradient).sum()

                gradient = functional_grad(contracted_atom)
                if parameter_gradient is None:
                    parameter_gradient = vmap(gradient)(point)
                if self.include_curvature:
                    directions = torch.eye(
                        point.shape[-1], device=point.device, dtype=point.dtype
                    )
                    gh = vmap(
                        lambda parameter: hessian_from_gradient(
                            gradient,
                            parameter,
                            directions=directions,
                        )
                    )(point)

        expected = (site.atom_count, site.atoms.parameter_dim)
        if parameter_gradient is None or parameter_gradient.shape != expected:
            raise ValueError(f"parameter gradient must have shape {list(expected)}")
        self._jg = self._add(self._jg, parameter_gradient)
        if self.include_curvature:
            expected_hessian = (*expected, expected[-1])
            if gh is None or gh.shape != expected_hessian:
                raise ValueError(
                    f"contracted Hessian must have shape {list(expected_hessian)}"
                )
            self._gh = self._add(self._gh, gh)
        self._contributions += 1

    @staticmethod
    def _add(current: Tensor | None, contribution: Tensor) -> Tensor:
        contribution = contribution.detach()
        if current is None:
            return contribution.clone()
        if current.shape != contribution.shape:
            raise ValueError("atom-gradient observations have unstable shapes")
        if current.device != contribution.device or current.dtype != contribution.dtype:
            raise ValueError("atom-gradient observations must share device and dtype")
        current.add_(contribution)
        return current

    @staticmethod
    def _validate_linear_tensors(
        site: CSTLinear, inputs: Tensor, output_gradient: Tensor
    ) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != site.in_features:
            raise ValueError("inputs do not match the CSTLinear input shape")
        expected_output = (*inputs.shape[:-1], site.out_features)
        if output_gradient.shape != expected_output:
            raise ValueError(f"output gradient must have shape {list(expected_output)}")
        tensors = (inputs, output_gradient)
        if any(value.device != site.atoms.p.device for value in tensors) or any(
            value.dtype != site.atoms.p.dtype for value in tensors
        ):
            raise ValueError("Linear backward tensors must match atom device and dtype")


class LinearJGAtomGrad(_LinearNDAtomGrad):
    """Collect only the local atom gradient ``jg``."""


class LinearJGHAtomGrad(_LinearNDAtomGrad):
    """Collect the local atom gradient ``jg`` and Hessian blocks ``gh``."""

    include_curvature = True
