"""Atom-gradient programs used by implicit CST optimizers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import grad as functional_grad
from torch.func import hessian as functional_hessian
from torch.func import vmap

from torchcst.atoms import AtomGradMode
from torchcst.nn import CSTLinear, LinearAtomGrad


class ImplicitLinearAtomGrad(LinearAtomGrad):
    r"""Collect the local signals required by the implicit Linear optimizer.

    A completed scope exposes

    - ``jg`` with shape ``[K, P]``;
    - ``gh`` with shape ``[K, P, P]``;
    - ``r`` with shape ``[out_features]``;
    - ``c`` with shape ``[in_features]``.

    The first two quantities are accumulated during backward. Row and column
    square means are finalized afterward so repeated uses include their exact
    cross terms.
    """

    def __init__(
        self,
        *,
        mode: AtomGradMode = "auto",
        row_chunk_size: int = 64,
    ) -> None:
        super().__init__(mode=mode)
        if isinstance(row_chunk_size, bool) or not isinstance(row_chunk_size, int):
            raise TypeError("row_chunk_size must be an integer")
        if row_chunk_size < 1:
            raise ValueError("row_chunk_size must be positive")
        self.row_chunk_size = row_chunk_size
        self._jg: Tensor | None = None
        self._gh: Tensor | None = None
        self._r: Tensor | None = None
        self._c: Tensor | None = None
        self._terms: list[tuple[Tensor, Tensor]] = []
        self._contributions = 0

    @property
    def supports_custom_autograd(self) -> bool:
        return True

    @property
    def jg(self) -> Tensor:
        """Return the accumulated parameter pullback ``J.T g``."""

        self.require_complete()
        assert self._jg is not None
        return self._jg.clone()

    @property
    def gh(self) -> Tensor:
        """Return the atom blocks of ``g contracted with H``."""

        self.require_complete()
        assert self._gh is not None
        return self._gh.clone()

    @property
    def r(self) -> Tensor:
        """Return output-row means of the squared aggregate represented gradient."""

        self.require_complete()
        assert self._r is not None
        return self._r.clone()

    @property
    def c(self) -> Tensor:
        """Return input-column means of the squared aggregate represented gradient."""

        self.require_complete()
        assert self._c is not None
        return self._c.clone()

    @property
    def contributions(self) -> int:
        """Number of Linear backward callbacks accumulated in this scope."""

        return self._contributions

    def _clear_values(self) -> None:
        self._jg = None
        self._gh = None
        self._r = None
        self._c = None
        self._terms.clear()
        self._contributions = 0

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
        if site.input_chart.trainable or site.output_chart.trainable:
            raise ValueError("ImplicitLinearAtomGrad supports frozen charts only")
        self._validate_linear_tensors(site, inputs, output_gradient)

        flat_inputs = inputs.detach().reshape(-1, site.in_features)
        flat_output_gradient = output_gradient.detach().reshape(
            -1, site.out_features
        )
        parameter_point = site.atoms.p.detach()

        def contracted_atom(atom_point: Tensor) -> Tensor:
            atom = site._materialize_atoms(atom_point.unsqueeze(0))[0]
            return (
                F.linear(flat_inputs, atom) * flat_output_gradient
            ).sum()

        with torch.enable_grad():
            contracted_hessian = vmap(functional_hessian(contracted_atom))(
                parameter_point
            )
            if parameter_gradient is None:
                parameter_gradient = vmap(functional_grad(contracted_atom))(
                    parameter_point
                )

        expected_parameter_shape = (site.atom_count, site.atoms.parameter_dim)
        if parameter_gradient.shape != expected_parameter_shape:
            raise ValueError(
                f"parameter gradient must have shape {list(expected_parameter_shape)}"
            )
        self._jg = self._add(self._jg, parameter_gradient)
        self._gh = self._add(self._gh, contracted_hessian)
        self._terms.append((flat_inputs.clone(), flat_output_gradient.clone()))
        self._contributions += 1

    def _complete_values(self) -> None:
        if self._jg is None or self._gh is None or not self._terms:
            raise RuntimeError("no Linear backward contribution was captured")

        in_features = self._terms[0][0].shape[1]
        out_features = self._terms[0][1].shape[1]
        row_square_mean = self._jg.new_empty(out_features)
        column_square_sum = self._jg.new_zeros(in_features)

        with torch.no_grad():
            for start in range(0, out_features, self.row_chunk_size):
                stop = min(start + self.row_chunk_size, out_features)
                block = self._jg.new_zeros(stop - start, in_features)
                for inputs, output_gradient in self._terms:
                    block.add_(output_gradient[:, start:stop].T @ inputs)
                squared = block.square()
                row_square_mean[start:stop] = squared.mean(dim=1)
                column_square_sum.add_(squared.sum(dim=0))

        self._r = row_square_mean
        self._c = column_square_sum / out_features
        self._terms.clear()

    @staticmethod
    def _add(current: Tensor | None, contribution: Tensor) -> Tensor:
        contribution = contribution.detach()
        if current is None:
            return contribution.clone()
        if current.shape != contribution.shape:
            raise ValueError("AtomGrad contributions must have stable shapes")
        if current.device != contribution.device or current.dtype != contribution.dtype:
            raise ValueError("AtomGrad contributions must share one device and dtype")
        current.add_(contribution)
        return current

    @staticmethod
    def _validate_linear_tensors(
        site: CSTLinear, inputs: Tensor, output_gradient: Tensor
    ) -> None:
        if inputs.ndim < 1 or inputs.shape[-1] != site.in_features:
            raise ValueError("inputs do not match the CSTLinear input shape")
        expected_output_shape = (*inputs.shape[:-1], site.out_features)
        if output_gradient.shape != expected_output_shape:
            raise ValueError(
                "output gradient must have shape "
                f"{list(expected_output_shape)}"
            )
        if (
            inputs.device != site.atoms.p.device
            or output_gradient.device != site.atoms.p.device
            or inputs.dtype != site.atoms.p.dtype
            or output_gradient.dtype != site.atoms.p.dtype
        ):
            raise ValueError("Linear backward tensors must match atom device and dtype")
