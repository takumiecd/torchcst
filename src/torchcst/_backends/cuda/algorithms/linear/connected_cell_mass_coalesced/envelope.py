"""Opt-in revision-10 envelope composition with qualified connected physical M16.

The differentiable envelope map and update rule are unchanged. Only immutable
constants and the physical declaration are prepared; trainable rows are decoded
on every call. No production registration or default selection is added.
"""

from dataclasses import dataclass, replace
from functools import cache
from typing import ClassVar

from torchcst._backends.schema import ExecutionPlan, SupportResult
from torchcst.kernels.parameterizations import RadiusEnvelopeSpec
from torchcst.operators.context import LinearContext
from torchcst.operators.execution import LinearBinding, linear_execution

from .algorithm import (
    CoalescedRecipe,
    ConnectedCoalescedAlgorithm,
    ConnectedOrderedAlgorithm,
)


def physical_operator(operator):
    from torchcst import kernels

    spec = operator.kernel.parameterization
    if operator.kernel.revision != 10 or type(spec) is not RadiusEnvelopeSpec:
        raise ValueError("requires connected radius-envelope revision 10")
    physical = kernels.product(
        *(replace(p, connection=None) for p in operator.kernel.profiles),
        coordinates=kernels.physical(
            radius_init=(spec.radius_min, spec.radius_max), radius_floor=spec.radius_min
        ),
    )
    return replace(
        operator,
        kernel=replace(physical, revision=9, profiles=operator.kernel.profiles),
    )


@cache
def _dispatcher():
    from torchcst._backends.dispatch import Dispatcher
    from torchcst._backends.registry import Registry

    registry = Registry()
    registry.register(ConnectedCoalescedAlgorithm())
    registry.register(ConnectedOrderedAlgorithm())
    return Dispatcher(registry=registry)


@dataclass(frozen=True)
class ConnectedEnvelopeAlgorithm(ConnectedCoalescedAlgorithm):
    id: str = "cuda_connected_m16_coalesced_envelope"
    physical_algorithm: ClassVar[type] = ConnectedCoalescedAlgorithm

    def supports(self, context, recipe):
        if not isinstance(context, LinearContext):
            return SupportResult(("requires LinearContext",))
        try:
            self.validate_recipe(recipe)
            physical = physical_operator(context.operator)
            spec = context.operator.kernel.parameterization
            if spec.radius_min < 0.5 or spec.radius_max > 50 or spec.amplitude_max > 32:
                raise ValueError("requires radius[.5,50], amplitude_max<=32")
        except (ValueError, TypeError) as error:
            return SupportResult((str(error),))
        return ConnectedCoalescedAlgorithm().supports(
            replace(context, operator=physical), recipe
        )

    def prepare(self, state):
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepare_constants,
        )

        state.invalidate()
        p = state.binding.atom_state.atoms.p
        op = state.context.operator
        spec = op.kernel.parameterization
        state.buffers["constants"] = prepare_constants(
            p,
            **{
                k: getattr(spec, k)
                for k in ("radius_min", "radius_max", "tau", "amplitude_max")
            },
        )
        state.layout = physical_operator(op)
        state.mark_current()

    def execute(self, state, inputs):
        from torchcst._backends.torch.parameterizations.radius_envelope import (
            prepared_physical_parameters,
        )

        _, p, _, _ = linear_execution(state, inputs)
        physical = prepared_physical_parameters(p, state.buffers["constants"])
        plan = ExecutionPlan(self.physical_algorithm().id, "v1", state.recipe)
        return _dispatcher().run(
            LinearBinding(state.layout, physical), inputs, plan=plan
        )


CONNECTED_ENVELOPE = ExecutionPlan(
    ConnectedEnvelopeAlgorithm().id, "v1", CoalescedRecipe()
)


@dataclass(frozen=True)
class ConnectedOrderedEnvelopeAlgorithm(ConnectedEnvelopeAlgorithm):
    id: str = "cuda_connected_m16_ordered_envelope"
    physical_algorithm: ClassVar[type] = ConnectedOrderedAlgorithm


CONNECTED_ORDERED_ENVELOPE = ExecutionPlan(
    ConnectedOrderedEnvelopeAlgorithm().id, "v1", CoalescedRecipe()
)
