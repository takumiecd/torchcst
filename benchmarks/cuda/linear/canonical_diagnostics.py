"""Instrument the actual canonical route on a fixed post24 snapshot."""

import copy
from unittest.mock import patch

import torch

from benchmarks.cuda.linear.scaling_comparison import capture


def canonical_stages(step):
    from torchcst._backends.cuda.algorithms.linear.regular_grid_h import (
        canonical_executor,
        input_owner,
        kernels,
        output_owner,
    )

    model = copy.deepcopy(step.model)
    x = step.x.detach().clone().requires_grad_()
    p = model.atoms.p
    source = p.detach().clone()
    y0 = model(x)
    loss0 = (y0 - step.target.detach()).square().mean()
    dy = torch.autograd.grad(loss0, y0)[0].detach().clone()
    del y0, loss0
    names = (
        "sort_prepare_bounds",
        "forward_h_and_owner",
        "backward_g_h_dx_physical",
        "restore_source_vjp",
    )
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(8)]

    def instrument(function, index):
        def call(*args, **kwargs):
            events[2 * index].record()
            result = function(*args, **kwargs)
            events[2 * index + 1].record()
            return result

        return call

    class InstrumentedKernel:
        def __getitem__(self, grid):
            return instrument(kernels_reduce[grid], 3)

    kernels_reduce = kernels.reduce_parameters

    def call():
        y = model(x)
        dx, dp = torch.autograd.grad(y, (x, p), dy)
        return y, dx, dp

    reference = tuple(t.detach().clone() for t in call())
    with (
        patch.object(
            canonical_executor,
            "prepare_canonical",
            instrument(canonical_executor.prepare_canonical, 0),
        ),
        patch.object(
            output_owner,
            "forward_output_owned",
            instrument(output_owner.forward_output_owned, 1),
        ),
        patch.object(
            input_owner,
            "input_owned_backward",
            instrument(input_owner.input_owned_backward, 2),
        ),
        patch.object(kernels, "reduce_parameters", InstrumentedKernel()),
    ):
        graph, outputs = capture(call)
    samples = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(
            {
                name: events[2 * i].elapsed_time(events[2 * i + 1])
                for i, name in enumerate(names)
            }
        )
    for actual, expected in zip(outputs, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if not torch.equal(p.detach(), source):
        raise AssertionError("stage diagnostics changed post24 parameters")
    return {
        "scope": "actual instrumented canonical route; fixed separate post24 copy; nonadditive",
        "same_uninstrumented_forward_backward": True,
        "no_optimizer_updates": True,
        "sort_and_restoration_in_primary_complete_step": True,
        "samples_ms": samples,
    }
