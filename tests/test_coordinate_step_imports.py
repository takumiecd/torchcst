"""Cold optimizer module imports retain live coordinate callback lookups."""

import os
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from torchcst._backends.torch import finite
from torchcst._backends.torch.kernels import execution as kernel
from torchcst.operators import atom_update
from torchcst.optim import _coordinate_step as coordinate
from torchcst.optim.adamw import CSTAdamW


@pytest.mark.parametrize(
    "modules",
    [
        ("torchcst.optim", "torchcst.operators.atom_update"),
        ("torchcst.operators.atom_update", "torchcst.optim.adamw"),
        ("torchcst._backends.torch.kernels.execution", "torchcst.optim"),
        ("torchcst._backends.torch.finite", "torchcst.optim._coordinate_step"),
        ("torchcst.optim._coordinate_step", "torchcst._backends.torch.finite"),
    ],
)
def test_fresh_import_orders_do_not_cycle_or_initialize_cuda(modules):
    code = f"""
import importlib
import sys
execution_requested = False
for name in {modules!r}:
    execution_requested |= name.startswith('torchcst._backends.torch.')
    importlib.import_module(name)
    if not execution_requested:
        assert not any(name.startswith('torchcst._backends.torch.') for name in sys.modules)
import torch
from torchcst.optim import _coordinate_step as coordinate
from torchcst.optim.adamw import CSTAdamW
from torchcst.operators import atom_update
assert callable(coordinate.prepare_atom_steps)
assert callable(CSTAdamW.step)
assert not torch.cuda.is_initialized()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("prepared_program", [False, True])
def test_coordinate_callbacks_and_input_type_stay_live(monkeypatch, prepared_program):
    point = torch.nn.Parameter(torch.ones(3, 4))
    point.grad = torch.full_like(point, 0.03)
    calls = []

    class Site:
        atoms = SimpleNamespace(p=point)
        kernel = object()

        def cst_charts(self):
            return ()

    class LiveInputs:
        def __init__(self, previous, step_size, amplitude_weight_decay):
            calls.append("inputs")
            self.previous = previous
            self.step_size = step_size
            self.amplitude_weight_decay = amplitude_weight_decay

    def checks_enabled():
        calls.append("checks_enabled")
        return True

    def first_nonfinite(values):
        calls.append("finite")

    def project(state, value, gradient):
        calls.append("project")
        assert value is point and gradient is point.grad
        return gradient

    def validate(inputs):
        calls.append("validate")
        assert type(inputs) is LiveInputs

    def dispatch(*args, **kwargs):
        calls.append("dispatch")
        assert type(args[1]) is LiveInputs
        return point.detach().clone()

    monkeypatch.setattr(atom_update, "AtomUpdateInputs", LiveInputs)
    monkeypatch.setattr(finite, "checks_enabled", checks_enabled)
    monkeypatch.setattr(finite, "first_nonfinite", first_nonfinite)
    monkeypatch.setattr(kernel, "project_parameter_gradient", project)
    site = Site()
    binding = SimpleNamespace(validate_inputs=validate)
    if prepared_program:
        binding._optimizer_program = SimpleNamespace(
            preflight=lambda *args: True,
            validate_inputs=validate,
            plan="prepared",
            run=lambda *args: point.detach().clone(),
            transport_required=True,
            transport_is_identity=lambda *args: True,
        )
    optimizer = SimpleNamespace(
        _bound_parameters={site: point},
        _update_bindings={site: binding},
        _update_dispatcher=SimpleNamespace(
            prepare=lambda *args: SimpleNamespace(plan="generic"),
            run=dispatch,
            prepare_optimizer_update=lambda *args: None,
        ),
        state_adapter=SimpleNamespace(
            transport=lambda *args: calls.append("transport")
        ),
        state={},
        check_finite=False,
    )
    group = {"params": [point], "lr": 0.02, "amplitude_weight_decay": 0.01}
    prepared = coordinate.prepare_atom_steps(
        optimizer, {id(point): group}, param_groups=[group]
    )
    assert len(prepared) == 1
    coordinate.finish_atom_steps(optimizer, prepared)
    if prepared_program:
        assert calls == ["checks_enabled", "finite", "inputs", "validate", "finite"]
    else:
        assert calls == [
            "checks_enabled",
            "finite",
            "project",
            "finite",
            "inputs",
            "inputs",
            "dispatch",
            "finite",
            "transport",
        ]


def test_native_step_uses_live_finite_context(monkeypatch):
    calls = []
    original = finite.checking

    @contextmanager
    def checking(enabled):
        calls.append(enabled)
        with original(enabled):
            yield

    monkeypatch.setattr(finite, "checking", checking)
    model = torch.nn.Linear(2, 1)
    optimizer = CSTAdamW(model.parameters(), model=model, foreach=False)
    for point in model.parameters():
        point.grad = torch.ones_like(point)
    optimizer.step()
    assert calls == [False]
