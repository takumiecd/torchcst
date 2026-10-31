"""Shared coordinate math for native and wrapped optimizer lifecycles.

These routines neither invoke an optimizer.step() nor install hooks. They project
and snapshot active atoms before a proposal, then retract and transport afterward.
"""

import math
from dataclasses import dataclass

import torch

from torchcst.nn import CSTModule


@dataclass
class _PreparedAtomStep:
    site: CSTModule
    point: torch.Tensor
    previous: torch.Tensor
    step_size: float
    amplitude_weight_decay: float
    plan: object
    program: object = None
    inputs: object = None


def prepare_atom_steps(optimizer, groups, *, param_groups, protected=()):
    from torchcst._backends.torch import finite
    from torchcst._backends.torch.kernels import execution as _kernel
    from torchcst.operators.atom_update import AtomUpdateInputs

    prepared = []
    # The finite checker returns immediately in unchecked mode. Avoid
    # preparing its input list (and coalescing sparse gradients) too.
    if finite.checks_enabled():
        gradients = [
            p.grad.coalesce().values() if p.grad.is_sparse else p.grad
            for group in param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if finite.first_nonfinite(gradients) is not None:
            raise FloatingPointError("non-finite parameter gradient")
        del gradients
    for site, point in optimizer._bound_parameters.items():
        if site.atoms.p is not point:
            raise RuntimeError("atom Parameter was replaced; rebuild CSTOptimizer")
        group = groups.get(id(point))
        if group is None:
            continue
        raw_decay = group.get("weight_decay", 0)
        if id(point) in protected and (
            type(raw_decay) not in (int, float) or raw_decay != 0
        ):
            raise ValueError("amplitude decay requires raw weight_decay=0")
        if point.grad is None:
            continue
        rate = group["lr"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0:
            raise ValueError(
                "CST learning rates must be finite nonnegative Python scalars"
            )
        if group.get("capturable", False):
            raise ValueError(
                "CST coordinate policies do not yet support CUDA Graph capture"
            )
        if point.grad.is_sparse:
            raise ValueError("CST coordinate policies require strided atom gradients")
        binding = optimizer._update_bindings[site]
        decay = group.get("amplitude_weight_decay", 0.0)
        program = binding.__dict__.get("_optimizer_program")
        if (
            rate
            and program is not None
            and program.preflight(
                optimizer._update_dispatcher,
                optimizer.state_adapter,
                point,
                point.grad,
                decay,
            )
        ):
            previous = point.detach().clone()
            inputs = AtomUpdateInputs(previous, rate, decay)
            validate = getattr(program, "validate_inputs", binding.validate_inputs)
            use_program = validate(inputs) is not False
            prepared.append(
                _PreparedAtomStep(
                    site,
                    point,
                    previous,
                    rate,
                    decay,
                    program.plan,
                    program if use_program else None,
                    inputs,
                )
            )
            continue
        projected = _kernel.project_parameter_gradient(
            site.kernel, *site.cst_charts(), point, point.grad
        )
        if projected.shape != point.shape:
            raise ValueError("projected atom gradient has the wrong shape")
        if finite.first_nonfinite((projected,)) is not None:
            raise FloatingPointError("non-finite projected atom gradient")
        previous = point.detach().clone()
        decay = group.get("amplitude_weight_decay", 0.0)
        plan = None
        if rate:
            inputs = AtomUpdateInputs(previous, rate, decay)
            plan = optimizer._update_dispatcher.prepare(
                optimizer._update_bindings[site], inputs
            ).plan
        if projected is not point.grad:
            point.grad.copy_(projected)
        prepared.append(_PreparedAtomStep(site, point, previous, rate, decay, plan))
    return prepared


def finish_atom_steps(optimizer, prepared):
    from torchcst._backends.torch import finite
    from torchcst.operators.atom_update import AtomUpdateInputs

    for step in prepared:
        if not step.step_size:
            continue
        # Only the certified path reuses its private, frozen inputs. Generic
        # callbacks keep fresh input construction after Adam and user hooks.
        if step.program is not None:
            inputs = step.inputs
            updated = step.program.run(
                optimizer._update_dispatcher,
                optimizer.state_adapter,
                inputs,
                optimizer.state.get(step.point, {}),
            )
            if updated is not None:
                if step.program.transport_required:
                    if updated.shape != step.point.shape:
                        raise ValueError("updated atom parameters have the wrong shape")
                    if finite.first_nonfinite((updated,)) is not None:
                        raise FloatingPointError("non-finite CST parameter update")
                    state = optimizer.state.get(step.point, {})
                    if not step.program.transport_is_identity(
                        optimizer._update_dispatcher,
                        optimizer.state_adapter,
                        step.point,
                        inputs,
                        state,
                    ):
                        optimizer.state_adapter.transport(
                            step.site, step.previous, updated, state
                        )
                continue
        inputs = AtomUpdateInputs(
            step.previous, step.step_size, step.amplitude_weight_decay
        )
        updated = optimizer._update_dispatcher.run(
            optimizer._update_bindings[step.site], inputs, plan=step.plan
        )
        if updated.shape != step.point.shape:
            raise ValueError("updated atom parameters have the wrong shape")
        if finite.first_nonfinite((updated,)) is not None:
            raise FloatingPointError("non-finite CST parameter update")
        optimizer.state_adapter.transport(
            step.site, step.previous, updated, optimizer.state.get(step.point, {})
        )
        binding = optimizer._update_bindings[step.site]
        if not optimizer.check_finite:
            binding.__dict__["_optimizer_program"] = (
                optimizer._update_dispatcher.prepare_optimizer_update(
                    binding, optimizer.state_adapter, step.site
                )
            )
        else:
            binding.__dict__.pop("_optimizer_program", None)
