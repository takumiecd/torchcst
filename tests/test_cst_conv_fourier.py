"""Exercise the channel-tap mathematical layout through the public Conv2d."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from dispatch_cases import default_conv2d_plan, fixed_plan

from torchcst import ChebyshevSpec, CSTConv2d, chart_presets, kernels
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.conv2d.channel_algorithm import (
    CHANNEL_FOURIER,
    CHANNEL_FOURIER_ACTIVE,
)
from torchcst._backends.dispatch import Dispatcher, FixedSelector, UnsupportedPlan
from torchcst._backends.schema import DeviceInfo, PrecisionPolicy
from torchcst._backends.torch.algorithms.conv2d import (
    TORCH_CHANNEL_CONV2D,
    TORCH_MATERIALIZED_CONV2D,
)
from torchcst.operators.conv2d import Conv2dInputs

LAYERS = (
    (1, 16, 32, 32, 1),
    (3, 16, 32, 32, 1),
    (3, 32, 32, 32, 1),
    (1, 32, 64, 32, 2),
    (3, 32, 64, 32, 2),
    (3, 64, 64, 16, 1),
    (1, 64, 128, 16, 2),
    (3, 64, 128, 16, 2),
    (3, 128, 128, 8, 1),
)


def layer(k=3, ci=16, co=32, stride=1, *, dtype=torch.float32, **options):
    grid = chart_presets.regular_grid
    config = dict(
        charts=(grid(co, geometry="flat_torus"), grid(ci, geometry="flat_torus")),
        in_channels=ci,
        out_channels=co,
        kernel_size=k,
        stride=stride,
        padding=k // 2 if stride == 1 else 0,
        atoms=4 * k**2,
        kernel=kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.rational_radius(radius_max=min(ci, co) // 2),
        ),
        atom_layout="channel_taps",
        dtype=dtype,
    )
    config.update(options)
    return CSTConv2d(**config)


def initialize_rows(model):
    rows = model.atoms.p
    with torch.no_grad():
        rows[:, 0].copy_(torch.linspace(-0.35, 0.5, len(rows), device=rows.device))
        rows[:, 1].copy_(rows.new_tensor([0, 0.23, 0.73, 1]).repeat(len(rows) // 4))
        rows[:, 2].add_(model.out_channels - 0.21)
        rows[:, 3].sub_(model.in_channels + 0.31)


def test_channel_tap_layout_preserves_ownership_and_gradients_across_relayout():
    m = layer(dtype=torch.float64)
    initialize_rows(m)
    x = torch.randn(1, 16, 6, 7, dtype=torch.float64)
    assert default_conv2d_plan(m, x) == TORCH_CHANNEL_CONV2D
    count = m.atom_count
    weights = m.materialized_atoms()
    assert m.patch_features == 16 * 9
    assert weights.shape == (count, 32, 16, 3, 3)
    for tap in range(9):
        block = weights[4 * tap : 4 * (tap + 1)].flatten(-2)
        assert block[..., :tap].count_nonzero() == 0
        assert block[..., tap + 1 :].count_nonzero() == 0
        assert block[..., tap].count_nonzero() > 0
    expected = F.conv2d(x, weights.sum(dim=0), padding=1)
    torch.testing.assert_close(m(x), expected)
    torch.testing.assert_close(
        m.cst_derivatives().represented(), m.dense_weight().flatten(1)
    )
    (grad,) = torch.autograd.grad(m(x).square().sum(), m.atoms.p)
    ids = torch.arange(count).roll(5)
    m.atom_state.relayout(ids)
    torch.testing.assert_close(m(x), expected)
    (moved_grad,) = torch.autograd.grad(m(x).square().sum(), m.atoms.p)
    torch.testing.assert_close(moved_grad, grad[ids])
    restored = layer(
        dtype=torch.float64, selector=fixed_plan(TORCH_MATERIALIZED_CONV2D)
    )
    restored.load_state_dict(m.state_dict())
    torch.testing.assert_close(restored(x), expected)


def test_conv_fourier_selection_and_rejection_use_live_context():
    registry = get_registry()
    plan = registry.loads_plan(registry.dumps_plan(CHANNEL_FOURIER))
    active = registry.loads_plan(registry.dumps_plan(CHANNEL_FOURIER_ACTIVE))
    m = layer(selector=fixed_plan(CHANNEL_FOURIER))
    inp = torch.randn(1, 16, 6, 7)
    with pytest.raises(UnsupportedPlan, match="CUDA FP32"):
        m(inp)
    dispatch = Dispatcher(registry=registry)
    context = replace(
        m.build_context(Conv2dInputs(inp)),
        device=DeviceInfo("cuda", 0, "RTX 3090", (8, 6)),
    )
    assert dispatch.select(context, plan=plan).plan == CHANNEL_FOURIER
    assert dispatch.select(context, plan=active).plan == CHANNEL_FOURIER_ACTIVE
    graph = replace(context, execution_mode="cuda_graph")
    assert dispatch.select(graph, plan=plan).plan == CHANNEL_FOURIER
    for bad, candidate, message in (
        (graph, active, "requires eager"),
        (replace(context, atom_layout="patch"), plan, "channel_taps"),
        (replace(context, dtype=torch.float64), plan, "CUDA FP32"),
        (replace(context, precision=PrecisionPolicy(autocast=True)), plan, "autocast"),
        (replace(context, precision=PrecisionPolicy(allow_tf32=True)), plan, "TF32"),
        (replace(context, groups=2), plan, "groups=1"),
    ):
        with pytest.raises(UnsupportedPlan, match=message):
            dispatch.select(bad, plan=candidate)
    m.selector = FixedSelector(
        plan, registry=registry, fallback_plan=TORCH_CHANNEL_CONV2D
    )
    torch.testing.assert_close(m(inp), F.conv2d(inp, m.dense_weight(), padding=1))


def test_channel_tap_layout_is_a_checkpoint_contract():
    with pytest.raises(ValueError, match="equal number"):
        layer(atoms=10)
    m = layer(k=1, atom_layout="patch")
    assert "atom_layout" not in m.get_extra_state()["layout"]
    with pytest.raises(RuntimeError, match="CST site checkpoint contract"):
        layer(k=1).load_state_dict(m.state_dict())


@pytest.fixture
def ieee_cuda():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("requires CUDA compute capability 8.0+")
    matmul = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.allow_tf32 = cudnn


@pytest.mark.parametrize("k,ci,co,size,stride", LAYERS)
def test_public_fourier_wideresnet_shapes(ieee_cuda, k, ci, co, size, stride):
    atom_layout = "patch" if k == 1 else "channel_taps"
    ref = layer(
        k,
        ci,
        co,
        stride,
        dtype=torch.float64,
        selector=fixed_plan(TORCH_MATERIALIZED_CONV2D),
        atom_layout=atom_layout,
    )
    model = layer(
        k, ci, co, stride, selector=fixed_plan(CHANNEL_FOURIER), atom_layout=atom_layout
    )
    initialize_rows(model)
    ref.load_state_dict(model.state_dict())
    model.cuda()
    x = torch.randn(1, ci, size, size, dtype=torch.float64).requires_grad_()

    def apply(m, x):
        if k == 3 and stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
        return m(x)

    expected = apply(ref, x)
    dy = torch.randn_like(expected)
    ref_grads = torch.autograd.grad(expected, (x, ref.atoms.p), dy)
    gpu_x = x.detach().to(device="cuda", dtype=torch.float32)
    gpu_x = gpu_x.contiguous(memory_format=torch.channels_last).requires_grad_()
    for plan in (CHANNEL_FOURIER, CHANNEL_FOURIER_ACTIVE):
        model.selector = fixed_plan(plan)
        actual = apply(model, gpu_x)
        grads = torch.autograd.grad(actual, (gpu_x, model.atoms.p), dy.to(actual))
        assert model._prepared_dispatch.decision.plan == plan
        for have, want in zip((actual, *grads), (expected, *ref_grads), strict=True):
            torch.testing.assert_close(
                have.cpu().double(), want, atol=1.2e-4, rtol=5e-4
            )


def test_public_fourier_graph_replay_tracks_live_atom_rows(ieee_cuda):
    m = layer(selector=fixed_plan(CHANNEL_FOURIER)).cuda()
    initialize_rows(m)
    x = torch.randn(1, 16, 7, 7, device="cuda", requires_grad=True)

    def forward_backward():
        y = m(x)
        return y, *torch.autograd.grad(y.square().mean(), (x, m.atoms.p))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward_backward()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward_backward()
    with torch.no_grad():
        m.atoms.p[:, 1].fill_(0.8)
        m.atoms.p[:, 3].add_(0.31)
        x.add_(0.1)
    graph.replay()
    for actual, expected in zip(captured, forward_backward(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=5e-4)
