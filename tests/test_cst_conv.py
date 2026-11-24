from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from kernel_cases import amplitude_state, gaussian_state, separable_state

from torchcst import CSTConv2d, CSTOptimizer, GaussianSpec, chart_presets, kernels
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.kernels import execution as _kernel


def make_kernel() -> amplitude_state:
    return amplitude_state(
        separable_state(
            input_profile=gaussian_state(0.4), output_profile=gaussian_state(0.7)
        )
    )


def make_model(
    *,
    backend: str = "factored",
    in_channels: int = 2,
    out_channels: int = 4,
    kernel_size: tuple[int, int] = (2, 3),
    stride: tuple[int, int] = (2, 1),
    padding: tuple[int, int] = (1, 0),
    dilation: tuple[int, int] = (1, 2),
    groups: int = 1,
) -> CSTConv2d:
    patch_features = in_channels // groups * math.prod(kernel_size)
    return CSTConv2d(
        _construction.linspace(out_channels, low=-1.0, high=1.0),
        _construction.linspace(patch_features, low=-1.0, high=1.0),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        atoms=3,
        kernel=make_kernel().declaration(),
        backend=backend,
        dtype=torch.float64,
    )


def test_atoms_are_flattened_locally_and_exposed_as_conv_weights() -> None:
    model = make_model()
    flat = _kernel.materialize_atoms(
        model.kernel, model.output_chart, model.input_chart, model.atoms.p
    )
    assert flat.shape == (3, 4, 12)
    assert model.materialized_atoms().shape == (3, 4, 2, 2, 3)
    assert model.dense_weight().shape == (4, 2, 2, 3)
    torch.testing.assert_close(model.materialized_atoms().flatten(start_dim=2), flat)
    torch.testing.assert_close(
        model.dense_weight().flatten(start_dim=1), flat.sum(dim=0)
    )
    assert model.cst_derivatives().represented().shape == (4, 12)
    torch.testing.assert_close(model.cst_derivatives().represented(), flat.sum(dim=0))


def test_materialized_forward_matches_unfolded_local_linear_map() -> None:
    torch.manual_seed(7)
    model = make_model(backend="materialized")
    inputs = torch.randn(2, 2, 7, 9, dtype=torch.float64)
    outputs = model(inputs)
    patches = F.unfold(
        inputs,
        kernel_size=model.kernel_size,
        dilation=model.dilation,
        padding=model.padding,
        stride=model.stride,
    ).transpose(1, 2)
    expected = F.linear(patches, model.dense_weight().flatten(start_dim=1))
    expected = expected.transpose(1, 2).reshape_as(outputs)
    torch.testing.assert_close(outputs, expected)


def test_factored_and_materialized_backends_match_outputs_and_gradients() -> None:
    torch.manual_seed(11)
    factored = make_model(backend="factored")
    materialized = make_model(backend="materialized")
    materialized.load_state_dict(copy.deepcopy(factored.state_dict()))
    factored_inputs = torch.randn(
        2, factored.in_channels, 7, 9, dtype=torch.float64, requires_grad=True
    )
    materialized_inputs = factored_inputs.detach().clone().requires_grad_(True)
    factored_outputs = factored(factored_inputs)
    materialized_outputs = materialized(materialized_inputs)
    factored_outputs.square().mean().backward()
    materialized_outputs.square().mean().backward()
    torch.testing.assert_close(factored_outputs, materialized_outputs)
    torch.testing.assert_close(factored_inputs.grad, materialized_inputs.grad)
    torch.testing.assert_close(factored.atoms.p.grad, materialized.atoms.p.grad)


@pytest.mark.parametrize("kernel_size,groups", [((1, 1), 1), ((3, 3), 1), ((3, 3), 2)])
def test_composed_matches_independent_periodic_gaussians(kernel_size, groups) -> None:
    in_channels, out_channels = 4, 6
    patch_features = in_channels // groups * math.prod(kernel_size)
    sigma = 0.75
    model = CSTConv2d(
        chart_presets.regular_grid(out_channels, geometry="flat_torus"),
        chart_presets.regular_grid(patch_features, geometry="flat_torus"),
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=(2, 1),
        padding=tuple(size // 2 for size in kernel_size),
        dilation=(1, 2),
        groups=groups,
        atoms=3,
        kernel=kernels.amplitude(kernels.product(kernels.l2(GaussianSpec(), sigma))),
        backend="composed",
        dtype=torch.float64,
    )
    # Centers on both sides of the periodic seam and interior off-grid centers.
    with torch.no_grad():
        model.atoms.p.copy_(
            model.atoms.p.new_tensor(
                [
                    [0.7, 0.21, patch_features - 0.32],
                    [-0.4, 5.73, 0.28],
                    [0.2, 3.11, 1.25],
                ]
            )
        )
    generator = torch.Generator().manual_seed(41)
    x = (
        torch.randn(2, in_channels, 7, 9, dtype=torch.float64, generator=generator)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    output = model(x)
    dy = torch.randn(output.shape, dtype=torch.float64, generator=generator)
    dx, dp = torch.autograd.grad(output, (x, model.atoms.p), dy)

    p_ref = model.atoms.p.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()

    def gaussian(period, centers):
        sites = torch.arange(period, dtype=torch.float64)[:, None]
        offset = (sites - centers + period / 2).remainder(period) - period / 2
        values = torch.exp(-0.5 * (offset / sigma).square())
        return values / torch.linalg.vector_norm(values, dim=0)

    amplitude, out_center, in_center = p_ref.unbind(dim=1)
    terms = (
        amplitude[:, None, None]
        * gaussian(out_channels, out_center).T[:, :, None]
        * gaussian(patch_features, in_center).T[:, None, :]
    )
    weight = terms.sum(dim=0).reshape(out_channels, in_channels // groups, *kernel_size)
    expected = F.conv2d(
        x_ref,
        weight,
        stride=model.stride,
        padding=model.padding,
        dilation=model.dilation,
        groups=groups,
    )
    dx_ref, dp_ref = torch.autograd.grad(expected, (x_ref, p_ref), dy)
    for actual, reference in ((output, expected), (dx, dx_ref), (dp, dp_ref)):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-11)


def test_composed_accumulates_weight_before_autocast(monkeypatch) -> None:
    model = make_model(backend="materialized").float()
    x = torch.randn(2, model.in_channels, 7, 9)
    expected_weight = model.dense_weight().detach()
    conv = F.conv2d
    observed = []

    def capture(inputs, weight, **settings):
        observed.append(weight.detach())
        return conv(inputs, weight, **settings)

    def reject_materialization(_):
        pytest.fail("composed execution materialized individual atom weights")

    model.backend = "composed"
    monkeypatch.setattr(model, "_materialize_atoms", reject_materialization)
    monkeypatch.setattr(F, "conv2d", capture)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(x)
    assert output.dtype == torch.bfloat16
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected_weight)


def test_composed_rejects_a_nonseparable_kernel() -> None:
    with pytest.raises(ValueError, match="does not support factorized execution"):
        CSTConv2d(
            chart_presets.regular_grid(4),
            chart_presets.regular_grid(2),
            in_channels=2,
            out_channels=4,
            kernel_size=1,
            atoms=3,
            kernel=kernels.radial(kernels.point(GaussianSpec(), 0.75)),
            backend="composed",
        )


def test_grouped_materialized_convolution_matches_torch() -> None:
    model = make_model(
        backend="materialized",
        in_channels=4,
        out_channels=6,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=2,
    )
    inputs = torch.randn(2, 4, 8, 7, dtype=torch.float64)
    expected = F.conv2d(inputs, model.dense_weight(), padding=1, groups=2)
    torch.testing.assert_close(model(inputs), expected)


def test_auto_uses_materialized_execution_for_groups() -> None:
    model = make_model(
        backend="auto",
        in_channels=4,
        out_channels=4,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=2,
    )
    assert model._resolved_backend() == "materialized"
    with pytest.raises(ValueError, match="groups=1"):
        make_model(
            backend="factored",
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=2,
        )


def test_constructor_and_forward_validate_flat_patch_shape() -> None:
    with pytest.raises(ValueError, match="flattened local patch"):
        CSTConv2d(
            _construction.linspace(4, low=-1.0, high=1.0),
            _construction.linspace(17, low=-1.0, high=1.0),
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            atoms=2,
            kernel=make_kernel().declaration(),
        )
    model = make_model()
    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.randn(2, 3, 7, 9, dtype=torch.float64))


@pytest.mark.parametrize("backend", ["factored", "composed"])
def test_parameter_adam_owns_conv_atom_coordinates(backend) -> None:
    model = make_model(backend=backend)
    optimizer = CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )
    before = model.atoms.p.detach().clone()
    optimizer.zero_grad()
    model(torch.randn(2, 2, 7, 9, dtype=torch.float64)).square().mean().backward()
    optimizer.step()
    assert model.atoms.p.grad is not None
    assert not torch.equal(model.atoms.p, before)
    assert optimizer.state[model.atoms.p]["step"] == 1


def test_parameter_adam_accepts_auto_when_conv_resolves_to_factored() -> None:
    model = make_model(backend="auto")
    assert model._resolved_backend() == "factored"
    CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=0.03,
            betas=(0.5, 0.99),
            eps=1e-08,
            weight_decay=0.0,
            foreach=False,
        ),
        model=model,
    )


def test_repulsion_energy_is_shape_invariant() -> None:
    model = make_model()
    atoms = model.materialized_atoms()
    normalized = atoms / torch.linalg.vector_norm(
        atoms, dim=(1, 2, 3, 4), keepdim=True
    ).clamp_min(torch.finfo(atoms.dtype).tiny)
    expected = normalized.sum(dim=0).square().sum() - normalized.square().sum()
    torch.testing.assert_close(model.repulsion_energy(), expected)


def test_charts_keyword_accepts_output_input_pair() -> None:
    out_chart = _construction.linspace(4, low=-1.0, high=1.0)
    in_chart = _construction.linspace(12, low=-1.0, high=1.0)
    model = CSTConv2d(
        charts=(out_chart, in_chart),
        in_channels=2,
        out_channels=4,
        kernel_size=(2, 3),
        atoms=3,
        kernel=make_kernel().declaration(),
    )
    assert model.charts == (model.output_chart, model.input_chart)
    assert (model.output_chart.ncs, model.input_chart.ncs) == (4, 12)
    assert model.dense_weight().shape == (4, 2, 2, 3)
