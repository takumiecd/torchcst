"""Linear invocation inputs and a non-Module binding for direct Plan execution."""

from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst.operators.context import context_from_tensors
from torchcst.operators.spec import OperatorSpec


@dataclass(frozen=True)
class LinearInputs:
    x: Tensor


def validate_linear_inputs(binding, inputs):
    if type(inputs) is not LinearInputs or not isinstance(inputs.x, Tensor):
        raise TypeError("LinearInputs requires a Tensor x")
    x, p = inputs.x, binding.execution_parameters()
    op = binding.execution_declaration()
    if x.ndim != 2 or x.shape[-1] != op.in_features:
        raise ValueError("expected a flattened [M, in_features] input")
    if not isinstance(p, Tensor) or p.ndim != 2:
        raise ValueError("parameters must be an [atoms, D] Tensor")
    if p.shape[1] <= 0:
        raise ValueError("parameter dimension must be positive")
    if x.device != p.device or x.dtype != p.dtype:
        raise ValueError("input and parameters must have the same device and dtype")


class LinearBinding:
    """Bind metadata and a live parameter reference once, without copying it.

    This direct binding is for declared operators. Module bindings additionally
    expose live Chart/Kernel state for the general Torch and fused algorithms.
    No input tensors or invocation-specific autograd saves are cached here.
    """

    input_type = LinearInputs

    def __init__(self, operator, parameters=None, *, workspace_limit_bytes=None):
        from torchcst.operators.binding import Operator

        if isinstance(operator, Operator):
            if parameters is not None and parameters is not operator.p:
                raise ValueError("parameters differ from the live Operator owner")
            parameters = operator.p
        elif not isinstance(operator, OperatorSpec) or not isinstance(
            parameters, Tensor
        ):
            raise TypeError(
                "LinearBinding requires a live Operator or OperatorSpec and Tensor parameters"
            )
        self.operator = operator
        self.parameters = parameters
        self.workspace_limit_bytes = workspace_limit_bytes
        self._algorithm_states = []
        self.execution_declaration()  # Warm configuration metadata outside capture.

    @property
    def atom_state(self):
        from torchcst.atoms import AtomState
        from torchcst.operators.binding import Operator

        if isinstance(self.operator, Operator):
            return AtomState.for_atoms(self.operator.atoms)
        owner = self.parameters.__dict__.get("_atom_state_owner")
        return owner if owner is not None and owner.atoms.p is self.parameters else None

    @property
    def atoms(self):
        return self.operator.atoms

    @property
    def kernel(self):
        return self.operator.kernel

    @property
    def chart(self):
        return self.operator.charts[0]

    @property
    def in_features(self):
        return self.operator.in_features

    @property
    def out_features(self):
        return self.operator.out_features

    def cst_charts(self):
        return self.operator.charts

    def declaration(self):
        return self.operator.declaration()

    def execution_declaration(self):
        if isinstance(self.operator, OperatorSpec):
            return self.operator
        return execution_declaration(self)

    def execution_parameters(self):
        return (
            self.operator.p
            if not isinstance(self.operator, OperatorSpec)
            else self.parameters
        )

    def validate_inputs(self, inputs):
        validate_linear_inputs(self, inputs)

    def build_context(self, inputs):
        return context_from_tensors(
            self.execution_declaration(),
            inputs.x,
            self.execution_parameters(),
            workspace_limit_bytes=self.workspace_limit_bytes,
        )

    def state_signature(self):
        from torchcst._backends.state import tensor_signature

        return tensor_signature(
            self.execution_parameters()
        ), self.execution_declaration()

    def algorithm_state(self, algorithm, *, recipe):
        declaration = self.execution_declaration()
        for state in self._algorithm_states:
            if (
                state.algorithm is algorithm
                and state.recipe == recipe
                and state.configuration.get("operator") == declaration
            ):
                return state
        state = algorithm.create_state(self, recipe=recipe, operator=declaration)
        self._algorithm_states.append(state)
        return state


def linear_execution(state, inputs):
    """Borrow current values at execution; metadata alone cannot certify them."""
    binding = state.binding
    owner = state.atom_state
    p = (
        binding.execution_parameters()
        if owner is None
        else owner.parameters_for_execution()
    )
    return inputs.x, p.contiguous(), binding.execution_declaration(), binding


def execution_declaration(binding):
    """Cache immutable live-operator metadata; refresh configuration outside capture."""
    roots = (*binding.cst_charts(), binding.kernel)
    signature = tuple(
        (id(m), getattr(m, "spec", None), getattr(m, "binding", None))
        for root in roots
        for m in root.modules()
    )
    buffers = tuple(t for root in roots for t in root.buffers())
    reusable = not any(t.is_inference() for t in buffers)
    if reusable:
        signature += tuple((id(t), t._version, t.dtype, t.device) for t in buffers)
    previous = binding.__dict__.get("_execution_declaration")
    if reusable and previous is not None and previous[0] == signature:
        return previous[1]
    if (
        binding.execution_parameters().is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "chart/kernel metadata must be refreshed outside CUDA graph capture"
        )
    declaration = binding.declaration()
    binding.__dict__["_execution_declaration"] = (signature, declaration)
    return declaration


def validate_live_operator(state, inputs):
    from torchcst.operators.binding import Operator

    if not isinstance(state.binding.operator, Operator):
        raise TypeError("this Algorithm requires a live Operator binding")
