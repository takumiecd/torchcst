from dataclasses import replace

import pytest
import torch

from torchcst import (
    CSTLinear,
    FixedSelector,
    chart_presets,
    kernels,
    make_algorithm_registry,
)
from torchcst._backends.cuda.algorithms.linear.connected_cell_mass_coalesced.envelope import (
    CONNECTED_ENVELOPE,
    CONNECTED_ORDERED_ENVELOPE,
    ConnectedEnvelopeAlgorithm,
    ConnectedOrderedEnvelopeAlgorithm,
)
from torchcst._backends.schema import DeviceInfo
from torchcst._backends.torch.parameterizations.radius_envelope import (
    physical_parameters,
)
from torchcst.operators.execution import LinearInputs


def layer(ni=128, no=256):
    from benchmarks.cuda.linear.connected_cell_mass import kernel

    k = kernel(16, True)
    k = kernels.product(
        *(replace(p, connection=None) for p in k.profiles),
        coordinates=kernels.radius_envelope(
            radius_min=0.5,
            radius_max=50,
            tau=0.003,
            amplitude_max=32,
        ),
    )
    from torchcst.kernels.spec import RadiusConnectionSpec

    k = replace(
        k,
        revision=10,
        profiles=tuple(
            replace(p, connection=RadiusConnectionSpec()) for p in k.profiles
        ),
    )
    return CSTLinear(
        chart_presets.regular_grid(no, spacing=1.0, geometry="flat_torus"),
        chart_presets.regular_grid(ni, spacing=1.0, geometry="flat_torus"),
        atoms=10,
        kernel=k,
        dtype=torch.float32,
        backend="factored",
    )


def test_envelope_admission_is_metadata_only():
    m = layer()
    c = m.build_context(LinearInputs(torch.zeros(7, 128)))
    c = replace(c, device=DeviceInfo("cuda", 0, "metadata-only"))
    a = ConnectedEnvelopeAlgorithm()
    r = CONNECTED_ENVELOPE.recipe
    assert a.supports(c, r).supported
    for change in (
        {"dtype": torch.float64},
        {"execution_mode": "cuda_graph"},
        {"deterministic": True},
        {"input_shape": (257, 128)},
    ):
        assert not a.supports(replace(c, **change), r).supported
    assert m.kernel.spec.revision == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("rows", [1, 7, 65, 256])
@pytest.mark.parametrize("ordered", [False, True])
def test_rectangular_envelope_full_chain_rule(rows, ordered):
    from benchmarks.cuda.linear.connected_cell_mass import error, scalar_axis

    m = layer().cuda()
    registry = make_algorithm_registry()
    registry.register(ConnectedEnvelopeAlgorithm())
    registry.register(ConnectedOrderedEnvelopeAlgorithm())
    m.selector = FixedSelector(
        CONNECTED_ORDERED_ENVELOPE if ordered else CONNECTED_ENVELOPE, registry=registry
    )
    p = torch.tensor([[0.3, 0, 7.5, 13.5]] * 10)
    radius = torch.tensor([0.5, 0.72, 0.8, 0.9, 1.44, 1.8, 7, 16, 31.9, 50])
    p[:, 1] = torch.log(radius / 0.5) / torch.log(torch.tensor(100.0))
    p[-1, 1] = 1
    p[1, :2] = torch.tensor([0.003, 0])  # live amplitude -> radius chain
    with torch.no_grad():
        m.atoms.p.copy_(p.cuda())
    g = torch.Generator().manual_seed(87)
    x = torch.randn(rows, 256, generator=g)[:, ::2].cuda().requires_grad_()
    dy = torch.randn(rows, 256, generator=g).cuda()
    y = m(x)
    dx, dp = torch.autograd.grad(y, (x, m.atoms.p), dy)
    rp = p.requires_grad_()
    rx = x.detach().cpu().double().requires_grad_()
    phys = physical_parameters(
        rp, radius_min=0.5, radius_max=50.0, tau=0.003, amplitude_max=32.0
    ).double()
    vi = torch.stack([scalar_axis(row, 128, 16, True, 3) for row in phys]).T
    vo = torch.stack([scalar_axis(row, 256, 16, True, 2) for row in phys]).T
    ry = (rx @ vi * phys[:, 0]) @ vo.T
    ex, ep = torch.autograd.grad(ry, (rx, rp), dy.cpu().double())
    for actual, expected in ((y.cpu(), ry), (dx.cpu(), ex), (dp.cpu(), ep)):
        err = error(actual, expected)
        assert err["max_abs"] < 3e-4 and err["relative_l2"] < 3e-4, err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
def test_ordered_repeats_are_bitwise_for_direct_scale_mixture():
    from benchmarks.cuda.linear.connected_cell_mass import parameters

    m = layer().cuda()
    registry = make_algorithm_registry()
    registry.register(ConnectedOrderedEnvelopeAlgorithm())
    m.selector = FixedSelector(CONNECTED_ORDERED_ENVELOPE, registry=registry)
    raw = parameters(128, 10, 3)
    raw[:, 1] = torch.linspace(0, 1, 10)
    with torch.no_grad():
        m.atoms.p.copy_(raw.cuda())
    x = torch.randn(256, 128, device="cuda", requires_grad=True)
    dy = torch.randn(256, 256, device="cuda")
    results = []
    for _ in range(3):
        y = m(x)
        dx, dp = torch.autograd.grad(y, (x, m.atoms.p), dy)
        results.append((y, dx, dp))
    for a, b in zip(results[0], results[1]):
        assert torch.equal(a, b)
    for a, b in zip(results[0], results[2]):
        assert torch.equal(a, b)


def test_ordered_plan_catalog_roundtrip():
    from benchmarks.cuda.linear.manifest import REGISTRY, load_run

    run = load_run(
        "benchmarks/cuda/linear/cases/connected-radius-8192-r256-c8-ordered.json",
        "benchmarks/cuda/linear/plans-connected-radius-ordered.json",
    )
    for entry in run.plans:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan
