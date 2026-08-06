"""Contract tests for OffsetCSTConv2d — the FC-9 unfactored conv form.

Each test pins one property the FC-9 validation relied on: the represented
map (channel kernels × data-side bilinear displacement, checked against an
independent per-atom gather reference), the lawful product domain and
capacity rule, static support under displacement drift, gradient flow to
every coordinate including Δ, structural mutation semantics, the signed
capture capabilities that scored birth consumes, and the engine lifecycle
with candidates that carry their own displacement.
"""

import math

import pytest
import torch

from torchcst.compute import OffsetCSTConv2d
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ConstantQuota,
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    StructuralQuota,
    cSET,
)
from torchcst.representation import survey_chart
from torchcst.storage import SynapseDeath, SynapseStore

SIGMA = 0.1


def _block(**kwargs) -> OffsetCSTConv2d:
    return OffsetCSTConv2d.propose(
        "mix", 8, 6, 3, SIGMA,
        generator=torch.Generator().manual_seed(0), **kwargs,
    )


def _gather_reference(block: OffsetCSTConv2d, x: torch.Tensor) -> torch.Tensor:
    """Independent per-atom evaluation of the represented map.

    y[b, o, p] = sum_a w_a K(mu_out[o], t_a) sum_c K(mu_in[c], s_a) x[b, c](p + Δ_a)
    with x read by bilinear interpolation over zero padding — written with
    explicit python loops so it shares no code with the fast path.
    """
    view = block.synapses.view()
    mu_in = block.in_neurons.mu.to(x)
    mu_out = block.out_neurons.mu.to(x)
    chart_d = block.chart_d_in
    batch, _, height, width = x.shape
    out = torch.zeros(batch, block.out_channels, height, width, dtype=x.dtype)
    pad = 3
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    for a in range(view.s.shape[0]):
        s_chart = view.s[a, :chart_d].to(x)
        dy, dx = float(view.s[a, chart_d]), float(view.s[a, chart_d + 1])
        t = view.t[a].to(x)
        w = float(view.w[a])
        k_in = block.kernel_in(mu_in, s_chart[None, :])[:, 0]
        k_out = block.kernel_out(mu_out, t[None, :])[:, 0]
        u = torch.einsum("bchw,c->bhw", xp, k_in)
        iy, ix = math.floor(dy), math.floor(dx)
        ay, ax = dy - iy, dx - ix
        shifted = (
            u[:, pad + iy : pad + iy + height, pad + ix : pad + ix + width]
            * (1 - ay) * (1 - ax)
            + u[:, pad + iy : pad + iy + height, pad + ix + 1 : pad + ix + 1 + width]
            * (1 - ay) * ax
            + u[:, pad + iy + 1 : pad + iy + 1 + height, pad + ix : pad + ix + width]
            * ay * (1 - ax)
            + u[:, pad + iy + 1 : pad + iy + 1 + height,
                pad + ix + 1 : pad + ix + 1 + width] * ay * ax
        )
        out += w * torch.einsum("o,bhw->bohw", k_out, shifted)
    return out


def test_forward_matches_independent_gather_reference():
    torch.manual_seed(0)
    block = _block().double()
    x = torch.randn(2, 8, 7, 9, dtype=torch.float64)
    fast = block(x)
    reference = _gather_reference(block, x)
    assert fast.shape == (2, 6, 7, 9)
    assert torch.allclose(fast, reference, atol=1e-12)


def test_propose_builds_product_domain_with_capacity_rule():
    block = _block()
    store = block.synapses
    assert survey_chart(block.in_neurons.mu, SIGMA).notes == ()
    assert store.d_in == block.in_neurons.mu.shape[1] + 2
    assert store.spec.atom_cost == store.d_in + store.d_out + 1
    lo, hi = store.spec.domain_in.bounds
    assert tuple(lo[-2:]) == (-1.5, -1.5) and tuple(hi[-2:]) == (1.5, 1.5)
    base = OffsetCSTConv2d.displacement_store(
        "ref", block.in_neurons, block.out_neurons, SIGMA, 3
    )
    chart_base = SynapseStore.between(
        "ref2", block.in_neurons, block.out_neurons, SIGMA
    )
    assert base.capacity == chart_base.capacity  # Δ axes add no cells
    assert block.synapses.capacity == 2 * base.capacity
    assert block.synapses.k_live == block.synapses.capacity
    view = store.view()
    assert float(view.s[:, -2:].abs().max()) <= 1.5


def test_propose_dials_and_validation():
    lean = _block(capacity_scale=0.5)
    fat = _block(capacity_scale=2.0)
    assert lean.synapses.capacity < fat.synapses.capacity
    assert _block(seed_atoms=False).synapses.k_live == 0
    with pytest.raises(ValueError):
        _block(kernel="delta")
    with pytest.raises(ValueError):
        _block(capacity_scale=0.0)


def test_gradients_reach_chart_displacement_amplitude_and_bandwidth():
    block = _block()
    x = torch.randn(2, 8, 6, 6, generator=torch.Generator().manual_seed(2))
    block(x).square().mean().backward()
    store = block.synapses
    chart_d = block.chart_d_in
    assert store.w.grad is not None and store.w.grad.abs().sum() > 0
    assert store.s.grad is not None
    assert store.s.grad[:, :chart_d].abs().sum() > 0
    assert store.s.grad[:, chart_d:].abs().sum() > 0  # Δ gets data-side grads
    assert store.t.grad is not None and store.t.grad.abs().sum() > 0
    assert block.kernel_in.sigma.grad is not None


def test_support_is_static_under_displacement_drift():
    block = _block()
    span = 2 * block._r_int + 1
    assert block.dense_weight().shape == (6, 8, span, span)
    with torch.no_grad():
        block.synapses.s[:, -2:] += 100.0  # drift far outside the box
    x = torch.randn(1, 8, 6, 6)
    out = block(x)
    assert block.dense_weight().shape == (6, 8, span, span)
    with torch.no_grad():
        block.synapses.s[:, -2:].clamp_(-1.5, 1.5)
    assert torch.allclose(out, block(x))  # forward clamped exactly to the box


def test_structural_mutation_reflects_in_forward():
    block = _block()
    x = torch.randn(1, 8, 5, 5, generator=torch.Generator().manual_seed(3))
    before = block(x)
    victim = block.synapses.live_ids()[:4]
    block.synapses.apply([SynapseDeath("mix", victim)])
    after = block(x)
    assert block.synapses.k_live == block.synapses.view().ids.numel()
    assert not torch.allclose(before, after)


def test_atom_grads_matches_autograd_signed():
    block = _block().double()
    x = torch.randn(2, 8, 5, 5, dtype=torch.float64)
    proj = torch.randn(2, 6, 5, 5, dtype=torch.float64)
    out = block(x)
    (out * proj).sum().backward()
    reported = block.atom_grads(x, proj)
    slots = block._cached_slots
    expected = block.synapses.w.grad.index_select(0, slots)
    assert torch.allclose(reported, expected, atol=1e-10)


def test_candidate_weight_grads_equals_zero_weight_atom_grad():
    block = _block(seed_atoms=False).double()
    rng = torch.Generator().manual_seed(4)
    store = block.synapses
    source = store.spec.domain_in.sample(3, rng).double()
    target = store.spec.domain_out.sample(3, rng).double()
    x = torch.randn(2, 8, 5, 5, dtype=torch.float64)
    proj = torch.randn(2, 6, 5, 5, dtype=torch.float64)
    scores = block.candidate_weight_grads(x, proj, source, target)
    chunked = block.candidate_weight_grads(
        x, proj, source, target, chunk_size=1
    )
    assert torch.allclose(scores, chunked)
    # Place candidate 0 as a real zero-weight atom: its autograd w-grad must
    # equal the reported candidate score.
    from torchcst.storage import SynapseBirth

    store.apply([
        SynapseBirth(
            "mix", source[:1].float(), target[:1].float(),
            torch.zeros(1), torch.tensor([0]),
        )
    ])
    out = block(x)
    (out * proj).sum().backward()
    live_grad = store.w.grad.index_select(0, block._cached_slots)
    assert torch.allclose(live_grad[0], scores[0].to(live_grad), atol=1e-6)


def test_row_capabilities_raise():
    block = _block()
    with pytest.raises(NotImplementedError):
        block.pre_gate_rows(torch.randn(2, 8))
    with pytest.raises(NotImplementedError):
        block.input_row_energy()


def test_engine_lifecycle_grows_atoms_with_displacements():
    block = OffsetCSTConv2d.propose(
        "mix", 8, 6, 3, SIGMA, seed_atoms=False,
        generator=torch.Generator().manual_seed(0),
    )
    root = QuotaRegime(
        budget=8,
        method=cSET(initial_weight=1e-2),
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_birth=8)),
        distributor=EvenBudgetDistributor(),
    )
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
    engine = StructuralEngine(
        block.stores(), root, modules={block.capture_site: block},
        optimizer=optimizer, seed=0,
    )
    rng = torch.Generator().manual_seed(5)
    for _ in range(3):
        x = torch.randn(2, 8, 6, 6, generator=rng)
        engine.begin_update()
        optimizer.zero_grad()
        block(x).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        optimizer.step()
        engine.step()
    assert block.synapses.k_live > 0
    view = block.synapses.view()
    # Born candidates carried their own displacement, inside the box.
    assert float(view.s[:, -2:].abs().max()) <= 1.5


def test_integer_extent_box_boundary_atom_stays_in_grid():
    # A ±1.0 displacement box (kernel_size 2) clamps drifted atoms to exactly
    # ±1.0; the stencil base cell must cap at r-1 instead of leaving the R=3
    # grid (the S1 radius-gate crash).
    gen = torch.Generator().manual_seed(0)
    block = OffsetCSTConv2d.propose(
        "mix", 8, 6, 2, SIGMA, generator=gen,
    ).double()
    with torch.no_grad():
        block.synapses.s[:, -2:] = 5.0  # everything clamps to +1.0 exactly
    x = torch.randn(2, 8, 6, 6, dtype=torch.float64)
    out = block(x)
    assert torch.isfinite(out).all()
    assert block.dense_weight().shape[-1] == 3
    # the boundary atom reads exactly the +1 integer shift: compare against
    # explicitly setting the displacement inside the box epsilon-close
    with torch.no_grad():
        block.synapses.s[:, -2:] = 1.0 - 1e-9
    assert torch.allclose(out, block(x), atol=1e-6)


def test_compute_dtype_materialization_close_to_full_precision():
    gen = torch.Generator().manual_seed(0)
    block = OffsetCSTConv2d.propose("mix", 8, 6, 3, SIGMA, generator=gen)
    x = torch.randn(2, 8, 6, 6, generator=torch.Generator().manual_seed(1))
    full = block(x)
    block.compute_dtype = torch.bfloat16
    reduced = block(x)
    rel = (full - reduced).abs().max() / full.abs().max()
    assert float(rel) < 5e-2  # bf16 mantissa; contraction accumulates fp32
    block(x).square().mean().backward()
    assert block.synapses.s.grad is not None  # grads flow through the cast
    with pytest.raises(TypeError):
        OffsetCSTConv2d(
            block.in_neurons, block.out_neurons, block.synapses,
            block.kernel_in, compute_dtype=torch.int32,
        )


def _grads_of(block, x, proj):
    block.zero_grad()
    (block(x) * proj).sum().backward()
    s = block.synapses
    return {
        "s": s.s.grad.clone(), "t": s.t.grad.clone(), "w": s.w.grad.clone(),
        "sigma": block.kernel_in.sigma.grad.clone(),
    }


def test_lean_materialize_matches_default_path_values_and_grads():
    gen = torch.Generator().manual_seed(0)
    ref = OffsetCSTConv2d.propose("mix", 8, 6, 3, SIGMA, generator=gen).double()
    lean = OffsetCSTConv2d(
        ref.in_neurons, ref.out_neurons, ref.synapses, ref.kernel_in,
        track_mass=False, lean_materialize=True,
    ).double()
    # drift one atom onto the clamp boundary so the masked-gradient branch
    # is exercised too
    with torch.no_grad():
        ref.synapses.s[0, -2:] = 3.0
    x = torch.randn(2, 8, 6, 6, dtype=torch.float64)
    proj = torch.randn(2, 6, 6, 6, dtype=torch.float64)
    assert torch.allclose(ref.dense_weight(), lean.dense_weight(), atol=1e-12)
    g_ref = _grads_of(ref, x, proj)
    g_lean = _grads_of(lean, x, proj)
    for key in g_ref:
        assert torch.allclose(g_ref[key], g_lean[key], atol=1e-9), key


def test_lean_materialize_validation():
    gen = torch.Generator().manual_seed(0)
    block = OffsetCSTConv2d.propose("mix", 8, 6, 3, SIGMA, generator=gen)
    with pytest.raises(ValueError):
        OffsetCSTConv2d(
            block.in_neurons, block.out_neurons, block.synapses,
            block.kernel_in, lean_materialize=True,  # track_mass defaults True
        )
