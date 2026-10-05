"""Connect the existing AdamW proposal to an independent Polar update Plan.

This is complete-step runner glue, not a standalone benchmark or an optimizer
implementation. Timing events preserve the existing snapshot/proposal/update
phase boundaries. The model constructs its live binding before Graph capture.
"""

import torch

from torchcst import AtomUpdateInputs, Dispatcher
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import FUSED
from torchcst._backends.torch.algorithms.polar_update.plans import TORCH


def optimizer_step(binding, optimizer, *, step_size, polar_update="torch", events=None):
    if polar_update not in ("torch", "fused"):
        raise ValueError("unknown polar update implementation")
    plan = FUSED if polar_update == "fused" else TORCH
    dispatcher = Dispatcher(registry=get_registry())
    if events is not None:
        events[0].record()
    previous = binding.execution_parameters().detach().clone()
    inputs = AtomUpdateInputs(previous, step_size)
    # Reject invalid inputs/support before AdamW mutates parameters or moments.
    binding.validate_inputs(inputs)
    dispatcher.select(binding.build_context(inputs), plan=plan)
    if events is not None:
        events[1].record()
    optimizer.step()
    if events is not None:
        events[2].record()
    with torch.no_grad():
        dispatcher.run(binding, inputs, plan=plan)
    if events is not None:
        events[3].record()
