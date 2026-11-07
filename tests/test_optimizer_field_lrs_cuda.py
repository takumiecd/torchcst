"""Explicit radius native field proposals and serialized continuation on CUDA."""

import copy
import io

import pytest
import torch
from test_native_cst_adamw import model
from test_optimizer_field_lrs import make

from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def native(m, wrapped, fields):
    opt = make(m, wrapped, fields)
    # Select the existing radius candidate explicitly, independently of Linear.
    opt._update_dispatcher.selector = FixedSelector(
        RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
    )
    return opt


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("amplitude", [0.0024, 0.0048, 0.0096, 0.0192])
def test_native_field_proposal_state_and_resume_match_cpu(wrapped, amplitude):
    cpu = model()
    gpu = copy.deepcopy(cpu).float().cuda()
    fields = {"amplitude": amplitude, "output_center": 0.0096, "input_center": 0.0012}
    reference = make(cpu, wrapped, fields)
    actual = native(gpu, wrapped, fields)
    for step, factor in enumerate((0.5, 1.0, 0.25, 0.0, 0.8)):
        for opt in (reference, actual):
            for group, peak in zip(opt.param_groups, (0.0048, 0.0003)):
                group["lr"] = peak * factor
        generator = torch.Generator().manual_seed(880 + step)
        for a, b in zip(cpu.parameters(), gpu.parameters()):
            a.grad = torch.randn(a.shape, dtype=a.dtype, generator=generator) * 0.001
            b.grad = a.grad.float().cuda()
        reference.step()
        actual.step()
        for a, b in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(b.double().cpu(), a, atol=3e-6, rtol=3e-5)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual.state[b][key].double().cpu(),
                    reference.state[a][key].double(),
                    atol=3e-8,
                    rtol=3e-5,
                )
    storage = io.BytesIO()
    torch.save({"model": gpu.state_dict(), "optimizer": actual.state_dict()}, storage)
    storage.seek(0)
    checkpoint = torch.load(storage, weights_only=False)
    restored = model().float().cuda()
    restored.load_state_dict(checkpoint["model"])
    resumed = native(restored, wrapped, None)
    resumed.load_state_dict(checkpoint["optimizer"])
    assert actual.get_field_lrs() == resumed.get_field_lrs()
    for a, b in zip(gpu.parameters(), restored.parameters()):
        a.grad = torch.full_like(a, 0.001)
        b.grad = a.grad.clone()
    actual.step()
    resumed.step()
    for a, b in zip(gpu.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual.state[a][key], resumed.state[b][key], atol=3e-7, rtol=3e-5
            )
