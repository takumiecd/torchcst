"""Cache fixed configuration, never atom values or failed validation results."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from test_cell_mass_polar import layer

from torchcst import CSTOptimizer
from torchcst._backends.torch.kernels import execution


def test_validated_layout_is_reused_across_parameter_and_optimizer_updates():
    model = layer(count=7)
    state = model.kernel
    state.__dict__.pop("_cell_mass_layout_signature", None)
    charts = model.cst_charts()
    with patch.object(state, "declaration", wraps=state.declaration) as snapshot:
        assert execution.parameter_dim(state, *charts) == 4
        for _ in range(3):
            with torch.no_grad():
                model.atoms.p.add_(0.01)
            assert execution.parameter_dim(state, *charts) == 4
        assert snapshot.call_count == 1
        state.scalar("w_c").mul_(0.9)
        assert execution.parameter_dim(state, *charts) == 4
        assert snapshot.call_count == 2
        assert execution.parameter_dim(state, *charts) == 4
        assert snapshot.call_count == 2
    assert not any("signature" in key for key in model.state_dict())


@pytest.mark.parametrize(
    "mutation", ["sigma", "replacement", "checkpoint", "spacing", "period", "profile"]
)
def test_changed_invalid_configuration_is_rejected_before_adam_mutation(mutation):
    model = layer(count=3)
    state, charts = model.kernel, model.cst_charts()
    execution.parameter_dim(state, *charts)
    optimizer = CSTOptimizer(
        torch.optim.AdamW(model.parameters(), lr=0.01), model=model
    )
    if mutation == "sigma":
        state.scalar("sigma_max_input").fill_(50)
    elif mutation == "replacement":
        state.sigma_max_input = state.scalar("sigma_max_input").new_tensor(50.0)
    elif mutation == "checkpoint":
        saved = copy.deepcopy(state.state_dict())
        saved["sigma_max_input"].fill_(50)
        state.load_state_dict(saved)
    elif mutation == "spacing":
        charts[0].spacing.mul_(2)
    elif mutation == "period":
        charts[0].geometry.periods.mul_(2)
    else:
        state.profiles[0].binding = replace(
            state.profiles[0].binding, min_tile_sites=256
        )
    before = model.atoms.p.detach().clone()
    model.atoms.p.grad = torch.ones_like(before)
    for _ in range(2):
        with pytest.raises(ValueError):
            optimizer.step()
        torch.testing.assert_close(model.atoms.p, before, rtol=0, atol=0)
        assert not any(optimizer.state.values())


def test_valid_scalar_change_dtype_and_new_chart_refresh_validation():
    model = layer(count=3, dtype=torch.float32)
    state, charts = model.kernel, model.cst_charts()
    execution.parameter_dim(state, *charts)
    with patch.object(state, "declaration", wraps=state.declaration) as snapshot:
        state.scalar("sigma_max_input").fill_(30)
        execution.parameter_dim(state, *charts)
        assert snapshot.call_count == 1
        model.double()
        execution.parameter_dim(state, *charts)
        assert snapshot.call_count == 2
        replacement = copy.deepcopy(charts[0])
        execution.parameter_dim(state, replacement, charts[1])
        assert snapshot.call_count == 3
    # A deep copy must not inherit a validation certificate for the original.
    cloned = copy.deepcopy(model)
    cloned.kernel.scalar("sigma_max_input").fill_(50)
    with pytest.raises(ValueError, match="period/3"):
        execution.parameter_dim(cloned.kernel, *cloned.cst_charts())


def test_inference_buffers_are_revalidated_without_version_counters():
    with torch.inference_mode():
        model = layer(count=3)
        state, charts = model.kernel, model.cst_charts()
        with patch.object(state, "declaration", wraps=state.declaration) as snapshot:
            execution.parameter_dim(state, *charts)
            execution.parameter_dim(state, *charts)
            assert snapshot.call_count == 2
            state.scalar("sigma_max_input").fill_(50)
            with pytest.raises(ValueError, match="period/3"):
                execution.parameter_dim(state, *charts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_warm_layout_has_no_scalar_reads_and_dirty_capture_is_rejected():
    from torch.utils._python_dispatch import TorchDispatchMode

    model = layer(count=3, dtype=torch.float32, device="cuda")
    state, charts = model.kernel, model.cst_charts()
    execution.parameter_dim(state, *charts)

    class NoScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            assert function is not torch.ops.aten._local_scalar_dense.default
            return function(*args, **(kwargs or {}))

    with NoScalarReads():
        execution.parameter_dim(state, *charts)
        with torch.no_grad():
            model.atoms.p.add_(0.01)
        execution.parameter_dim(state, *charts)
    state.scalar("sigma_max_input").fill_(30)
    with (
        pytest.raises(RuntimeError, match="validated outside CUDA graph capture"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        execution.parameter_dim(state, *charts)
    # Outside capture the valid change can be validated and used normally.
    execution.parameter_dim(state, *charts)
