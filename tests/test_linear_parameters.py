"""Parameter reuse and common dispatch contracts, independent of old entry points."""

import copy
from dataclasses import replace

import pytest
import torch
from test_normalized_strip_public import chart, mixed, oracle
from torch import nn

from torchcst import (
    Atoms,
    CSTLinear,
    CSTOptimizer,
    LinearOptions,
    geometry_presets,
    presets,
)


def layer(atoms, **kwargs):
    return CSTLinear(
        chart=chart(), atoms=atoms, kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT, **kwargs
    )


@pytest.mark.parametrize("container", ["parameter", "atoms"])
def test_reuses_parameter_and_existing_adamw_state(container):
    p = nn.Parameter(mixed())
    optimizer = torch.optim.AdamW([p], lr=0.001)
    p.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    before = copy.deepcopy(optimizer.state[p])
    owner = Atoms(p) if container == "atoms" else p
    model = layer(owner)
    assert model.atoms.p is p
    if container == "atoms":
        assert model.atoms is owner
    assert list(model.parameters()) == [p]
    assert optimizer.param_groups[0]["params"][0] is model.atoms.p
    wrapped = CSTOptimizer(optimizer, model=model)
    for key, value in before.items():
        torch.testing.assert_close(wrapped.state[p][key], value, atol=0, rtol=0)
    x = torch.randn(2, 16, dtype=p.dtype)
    model(x).square().sum().backward()
    previous = p.detach().clone()
    wrapped.step()
    assert not torch.equal(previous, p)
    assert optimizer.state[p]["step"] == before["step"] + 1


def test_tensor_is_copied_and_can_be_converted_explicitly():
    p = mixed().requires_grad_()
    model = layer(p, dtype=torch.float32)
    assert model.atoms.p.dtype == torch.float32
    assert model.atoms.p.data_ptr() != p.data_ptr()
    model(torch.ones(1, 16)).sum().backward()
    assert p.grad is None
    assert model.atoms.p.grad is not None


@pytest.mark.parametrize("container", [nn.Parameter, Atoms])
def test_shared_parameters_cannot_be_silently_converted(container):
    p = nn.Parameter(mixed())
    with pytest.raises(ValueError, match="device and dtype"):
        layer(container(p), dtype=torch.float32)
    assert p.dtype == torch.float64


def test_reuses_frozen_parameter_without_enabling_gradients():
    p = nn.Parameter(mixed(), requires_grad=False)
    model = layer(p)
    assert model.atoms.p is p and not model.atoms.p.requires_grad
    x = torch.randn(2, 16, dtype=p.dtype, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and p.grad is None


def test_existing_atoms_can_be_used_by_ordinary_chart_pair_kernel():
    kernel = presets.separable(
        input_profile=presets.fixed_profile(presets.GaussianSpec(), 1.0),
        output_profile=presets.fixed_profile(presets.GaussianSpec(), 1.0),
    )
    source = CSTLinear(
        geometry_presets.linspace(4, spacing=1.0),
        geometry_presets.linspace(3, spacing=1.0),
        atoms=2,
        kernel=kernel,
    )
    target = CSTLinear(
        geometry_presets.linspace(4, spacing=1.0),
        geometry_presets.linspace(3, spacing=1.0),
        atoms=source.atoms,
        kernel=kernel,
    )
    assert target.atoms is source.atoms
    x = torch.randn(2, 4)
    torch.testing.assert_close(target(x), source(x), atol=0, rtol=0)


def test_atom_width_is_checked_against_selected_kernel():
    with pytest.raises(ValueError, match="shape"):
        layer(Atoms(torch.ones(2, 4)))
    with pytest.raises(ValueError, match="provided_atoms"):
        layer(2)


def test_product_and_strip_share_the_same_mathematical_operation():
    state = chart()
    spec = state.declaration()
    product = replace(spec, kind="product", tile_shape=None, tile_pitch=None, axis=None)
    # Storage tiling is not a separate Linear operation.
    model = CSTLinear(
        chart=product, atoms=mixed(), kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT
    )
    assert model._resolved_backend() == "normalized"
    x = torch.randn(2, 16, dtype=torch.float64)
    torch.testing.assert_close(model(x), layer(mixed())(x), atol=0, rtol=0)


def test_warm_dispatch_does_not_snapshot_or_read_parameter_values(monkeypatch):
    model = layer(nn.Parameter(mixed()))
    x = torch.randn(2, 16, dtype=torch.float64)
    model(x)

    def forbidden(*args, **kwargs):
        raise AssertionError("declaration snapshot in execution")

    monkeypatch.setattr(model, "declaration", forbidden)
    with torch.no_grad():
        model.atoms.p[0, 2].add_(0.4)
    torch.testing.assert_close(
        model(x), x @ oracle(model.atoms.p).T, atol=1e-9, rtol=1e-9
    )


def test_checkpoint_load_refreshes_live_metadata_and_preserves_optimizer_reference():
    source = layer(mixed())
    with torch.no_grad():
        source.chart.axes[0].start.add_(1)
        source.atoms.p[0, 0].add_(0.1)
    target = layer(nn.Parameter(mixed()))
    p = target.atoms.p
    optimizer = torch.optim.AdamW([p])
    x = torch.randn(2, 16, dtype=torch.float64)
    target(x)
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert target.atoms.p is p and optimizer.param_groups[0]["params"][0] is p
    torch.testing.assert_close(target(x), source(x), atol=0, rtol=0)


def test_non_regular_layout_uses_general_reference():
    model = CSTLinear(
        chart=geometry_presets.product(
            (4, 4),
            axes=(
                geometry_presets.line_pattern(4, spacing=1.0),
                geometry_presets.line_pattern(4, spacing=1.0),
            ),
            geometry=geometry_presets.euclidean(2),
        ),
        atoms=torch.tensor([[0.5, 0.0, 0.2, 0.3]]),
        kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
    )
    assert model._resolved_backend() == "materialized"
    x = torch.randn(2, 4)
    torch.testing.assert_close(model(x), model.operator.apply(x), atol=0, rtol=0)


def test_execution_preference_is_separate_from_declaration():
    full = layer(mixed())
    window = layer(mixed(), linear_options=LinearOptions(memory="window"))
    assert full.declaration() == window.declaration()
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        window.load_state_dict(full.state_dict())


def test_old_public_entry_point_is_removed():
    import importlib.util

    import torchcst
    import torchcst.nn

    assert not hasattr(torchcst, "NormalizedStripLinear")
    assert not hasattr(torchcst.nn, "NormalizedStripLinear")
    assert importlib.util.find_spec("torchcst.nn.normalized_strip") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strided_shared_parameter_keeps_identity_and_correct_cuda_gradients():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        p = nn.Parameter(mixed(torch.float32, "cuda").T.contiguous().T)
        model = CSTLinear(
            chart=chart(dtype=torch.float32, device="cuda"),
            atoms=p,
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
        )
        assert model.atoms.p is p and not p.is_contiguous()
        x = torch.randn(2, 16, device="cuda", requires_grad=True)
        xx = x.detach().double().requires_grad_()
        pp = p.detach().double().requires_grad_()
        actual = model(x)
        truth = xx @ oracle(pp, stored_dtype=torch.float32).T
        a = torch.autograd.grad(actual.sum(), (x, p))
        b = torch.autograd.grad(truth.sum(), (xx, pp))
        for left, right in [(actual, truth), *zip(a, b)]:
            torch.testing.assert_close(left.double(), right, atol=3e-4, rtol=3e-4)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_explicit_empty_atoms_and_empty_batch_keep_autograd():
    model = layer(nn.Parameter(torch.empty(0, 5, dtype=torch.float64)))
    x = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert model.atoms.p.grad.shape == (0, 5)
    model = layer(mixed())
    empty = torch.empty(0, 16, dtype=torch.float64, requires_grad=True)
    model(empty).sum().backward()
    assert empty.grad.shape == (0, 16)
    assert torch.equal(model.atoms.p.grad, torch.zeros_like(model.atoms.p))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_metadata_changes_require_rebinding_outside_graph_capture():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = CSTLinear(
            chart=chart(dtype=torch.float32, device="cuda"),
            atoms=mixed(torch.float32, "cuda"),
            kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
        )
        x = torch.ones(2, 16, device="cuda")
        # No CUDA work is necessary to reject a stale metadata binding.
        with torch.no_grad():
            model.chart.axes[0].start.add_(1.0)
        from unittest.mock import patch

        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            pytest.raises(RuntimeError, match="outside CUDA graph capture"),
        ):
            model._resolved_backend()
        model(x)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            assert model._resolved_backend() == "normalized"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype,spacing", [(torch.float64, 0.5), (torch.float32, 0.75)])
def test_cuda_without_matching_algorithm_uses_general_reference(dtype, spacing):
    state = chart(dtype=dtype, device="cuda")
    with torch.no_grad():
        state.axes[1].spacing.fill_(spacing)
    model = CSTLinear(
        chart=state,
        atoms=mixed(dtype, "cuda"),
        kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
    )
    assert model._resolved_backend() == "materialized"
    x = torch.randn(2, 16, device="cuda", dtype=dtype, requires_grad=True)
    actual = model(x)
    reference = model.operator.apply(x)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    actual.sum().backward()
    assert torch.isfinite(model.atoms.p.grad).all() and torch.isfinite(x.grad).all()
