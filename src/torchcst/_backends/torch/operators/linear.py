"""Reference linear algorithms consuming a common live Operator binding."""

import torch.nn.functional as F
from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst.profiling import cst_span


def materialize_atoms(operator, p):
    represented = _kernel.materialize_atoms(operator.kernel, *operator.charts, p)
    expected = (p.shape[0], *operator.shape)
    if represented.shape != expected:
        raise ValueError(f"kernel.materialize_atoms must return shape {list(expected)}")
    return represented


def weight(operator, p):
    if len(operator.charts) == 1:
        result = _kernel.weight(operator.kernel, operator.charts[0], p)
        if result.shape != operator.shape:
            raise ValueError("kernel.weight must match the operator shape")
        return result
    return materialize_atoms(operator, p).sum(dim=0)


def factors(operator, p):
    if not _kernel.supports_factorization(operator.kernel):
        raise ValueError("this operator does not provide input/output factors")
    phi_input, phi_output = _kernel.factors(operator.kernel, *operator.charts, p)
    if phi_input.shape != (operator.in_features, p.shape[0]) or phi_output.shape != (
        operator.out_features,
        p.shape[0],
    ):
        raise ValueError("kernel factors must match chart sizes and atom count")
    return phi_input, phi_output


def apply(operator, inputs, p, *, algorithm):
    if (
        not isinstance(inputs, Tensor)
        or inputs.ndim < 1
        or inputs.shape[-1] != operator.in_features
    ):
        raise ValueError(f"expected input shape [..., {operator.in_features}]")
    if algorithm == "materialized":
        with cst_span("cst.linear.materialize"):
            represented = weight(operator, p)
        with cst_span("cst.linear.dense_matmul"):
            return F.linear(inputs, represented)
    if algorithm == "factored":
        with cst_span("cst.linear.factors"):
            phi_input, phi_output = factors(operator, p)
        with cst_span("cst.linear.input_matmul"):
            atom_values = inputs @ phi_input
        with cst_span("cst.linear.output_matmul"):
            return atom_values @ phi_output.transpose(-2, -1)
    raise ValueError("unknown Torch linear algorithm")
