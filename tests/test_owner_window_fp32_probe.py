"""Independent math and restoration contracts for the FP32 research probe."""

from types import SimpleNamespace

import torch

from benchmarks.cuda.linear.radius_cell_mass import fixture
from benchmarks.training.full_step_graph import restore
from benchmarks.training.owner_window_fp32 import (
    adam_direction,
    errors,
    vjp_reference,
)


def test_quadrature_and_cpu_fourier_are_independent_consistent_references():
    model, x, _, dy = fixture(SimpleNamespace(size=1024, atoms=7, rows=4, seed=41))
    a = vjp_reference(model, x, dy, independent=False)
    b = vjp_reference(model, x, dy, independent=True)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, atol=2e-11, rtol=2e-10)


def test_near_epsilon_direction_gate_detects_small_gradient_sign_changes():
    reference = torch.tensor([[1e-8, -1e-8, 0.0, 0.0]])
    candidate = -reference
    assert errors(candidate, reference)["pass"]
    assert (adam_direction(candidate) - adam_direction(reference)).abs().max() > 0.01


def test_restore_reinstates_negative_zero_and_skips_identical_bytes():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    with torch.no_grad():
        model.weight.fill_(0.0)
    key = "/model/weight"
    saved = {key: torch.tensor([[-0.0]])}
    assert not torch.signbit(model.weight).item()
    restore(model, optimizer, saved)
    assert torch.signbit(model.weight).item()
    version = model.weight._version
    restore(model, optimizer, saved)
    assert version == model.weight._version
