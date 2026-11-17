"""Apply the Atom adjoint once after the fixed microbatch accumulation."""

from contextlib import ExitStack
from unittest.mock import patch

import torch
from cst_research.experiments.lm_connected import set_rate

from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.lm_width import (
    physical_operator,
)
from torchcst._backends.torch.parameterizations.radius_envelope import (
    prepare_constants,
    prepared_physical_parameters,
)


def shared_step(native, sites, model, optimizer, batches, clock):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    set_rate(optimizer, clock)
    loss_sum = 0.0
    differentiable, proxies = [], []
    with ExitStack() as scoped:
        for layer in sites.values():
            operator = layer.execution_declaration()
            specification = operator.kernel.parameterization
            layout = hierarchy_from_operator(physical_operator(operator))
            constants = prepare_constants(
                layer.atoms.p,
                radius_min=specification.radius_min,
                radius_max=specification.radius_max,
                tau=specification.tau,
                amplitude_max=specification.amplitude_max,
            )
            physical = prepared_physical_parameters(layer.atoms.p, constants)
            weight = native.packed_dense_fourier_weight_from_atoms(
                physical, layout.input.native_config(), layout.output.native_config()
            )
            # Each model backward can release its activations. Only a dense
            # weight cotangent for this optimizer update stays on the leaf.
            proxy = weight.detach().requires_grad_(True)
            differentiable.append(weight)
            proxies.append(proxy)

            def forward(x, matrix=proxy):
                return torch.nn.functional.linear(x, matrix)

            scoped.enter_context(patch.object(layer, "forward", forward))

        for x, target in batches:
            _, loss = model(x, target)
            (loss / len(batches)).backward()
            loss_sum += float(loss.detach())

    torch.autograd.backward(differentiable, [proxy.grad for proxy in proxies])
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return loss_sum / len(batches), {
        "gradient_norm_before_clip": float(norm),
        "topology": None,
    }
