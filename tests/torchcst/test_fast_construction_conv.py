"""cSFW tangent statistics reach ``CSTConv2d`` sites (FC-6b).

``CSTConv2d.forward`` routes its gated ``F.unfold`` patches through the very
same ``_ContinuousCSTMap._forward_rows`` capture hook that ``CSTLinear`` uses
(``src/torchcst/compute/cst_map.py``), and exposes the same ``in_features``/
``out_features``/``kernel_columns`` surface that ``TangentStatistics`` and
``TangentBirth``/``TangentRefit`` already read generically. Concretely:

* linear's ``x`` (``[batch, in_features]``)      <-> conv's gated unfolded
  patch rows (``[batch*locations, in_features]``), i.e. ``unfold(x)``.
* linear's ``g_out`` (``[batch, out_features]``)  <-> conv's per-patch-location
  gradient of the pre-out-gate map output, i.e. the row-flattened ``dL/dY``
  before it is folded back into image shape.
* ``TangentSnapshot.cross`` (``-g_out.T @ x``)     <-> ``-dL/dW`` for the
  hypothetical unconstrained dense patch weight ``W`` (``out_channels x
  in_channels*kh*kw``) that :meth:`CSTConv2d.dense_weight` would materialize.
* ``TangentSnapshot.covariance`` (``x.T @ x / N``) <-> ``Sigma_patch =
  E[unfold(x) unfold(x)^T]``, the input patch second moment.

No change to ``instruments/tangent.py``, ``policy/fast_construction.py``, or
``engine.py`` was needed for any of this: ``ComputeLinear`` already includes
``CSTConv2d`` (``src/torchcst/compute/__init__.py``), and
``TangentStatisticsRequest``/``TangentBirth``/``TangentRefit`` never
special-case the compute module type. These tests exercise that
correspondence directly; see docs/roadmap/FAST_CONSTRUCTION.md Appendix L
item 2 (FC-6b) in the sibling ``cst`` repository for the registered plan.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchcst.compute import CSTConv2d, conv2d_neuron_coordinates
from torchcst.engine import StructuralEngine
from torchcst.instruments import TangentStatisticsRequest
from torchcst.policy import EvenBudgetDistributor, PeriodicCadence, QuotaRegime, cSFW
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _conv_parts(
    *,
    in_channels: int = 2,
    out_channels: int = 3,
    kernel_size: tuple[int, int] = (3, 3),
    sigma: float = 0.3,
    capacity: int = 32,
) -> tuple[SynapseStore, NeuronStore, NeuronStore, CSTConv2d]:
    input_mu, output_mu = conv2d_neuron_coordinates(
        in_channels, out_channels, kernel_size, dtype=torch.float64
    )
    store = SynapseStore(
        "conv",
        3,
        1,
        capacity,
        max_capacity=capacity,
        spec=RepresentationSpec.continuous(3, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "conv-in",
        input_mu.shape[0],
        mu=input_mu,
        initial_live=input_mu.shape[0],
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "conv-out",
        output_mu.shape[0],
        mu=output_mu,
        initial_live=output_mu.shape[0],
        dtype=torch.float64,
    )
    kernel = GaussianKernel(sigma, learnable=True).double()
    conv = CSTConv2d(
        inputs, outputs, store, kernel, in_channels, kernel_size, bias=False
    )
    return store, inputs, outputs, conv


def _engine(
    store: SynapseStore,
    inputs: NeuronStore,
    outputs: NeuronStore,
    conv: CSTConv2d,
    *,
    method,
    budget: int,
    seed: int = 3,
    capture_mode: str = "deferred",
) -> StructuralEngine:
    policy = QuotaRegime(
        budget=budget,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    return StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        policy,
        modules={store.site: conv},
        seed=seed,
        capture_mode=capture_mode,
    )


def _tangent_snapshot(engine: StructuralEngine, site: str = "conv"):
    request = next(
        req for req in engine._tree.requires if isinstance(req, TangentStatisticsRequest)
    )
    return engine.instrument(site, request.name).snapshot()


def _teacher_problem(
    *, seed: int = 0, batch: int = 6, in_channels: int = 2, out_channels: int = 3,
    kernel_size: tuple[int, int] = (3, 3), height: int = 9, width: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Draw every random tensor from one locally seeded generator: nn.Conv2d's
    # default weight init otherwise reads the *global* RNG, which makes this
    # "teacher" depend on how many random draws earlier tests in the suite
    # already consumed -- exactly the flakiness this test must not have.
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, in_channels, height, width, dtype=torch.float64, generator=generator)
    teacher = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=False).double()
    with torch.no_grad():
        teacher.weight.copy_(
            torch.randn(teacher.weight.shape, dtype=torch.float64, generator=generator)
        )
        y = teacher(x)
    return x, y


# --- (a) correspondence: cross == -dL/dW_patch, covariance == Sigma_patch ---


def test_conv_tangent_cross_is_patch_weight_gradient_and_covariance_is_sigma_patch() -> None:
    store, inputs, outputs, conv = _conv_parts()
    # budget=0: only observe, never mutate structure, so conv stays at its
    # initial zero-synapse state (conv(x) is identically zero).
    engine = _engine(store, inputs, outputs, conv, method=cSFW(backfit=None), budget=0)
    x, y = _teacher_problem()

    engine.begin_update()
    output = conv(x)
    assert torch.equal(output, torch.zeros_like(output))
    loss = (output - y).square().mean()
    loss.backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    snapshot = _tangent_snapshot(engine)
    engine.step()

    # Independent reference: a bare F.conv2d with an unconstrained dense
    # weight, never touching CSTConv2d's own map at all.
    weight = torch.zeros(
        conv.out_channels, conv.in_channels, *conv.kernel_size,
        dtype=torch.float64, requires_grad=True,
    )
    reference_output = F.conv2d(x, weight, None, conv.stride, conv.padding, conv.dilation)
    torch.testing.assert_close(reference_output, output.detach())
    (reference_output - y).square().mean().backward()
    expected_cross = -weight.grad.reshape(conv.out_channels, conv.in_features)

    patches = F.unfold(
        x, conv.kernel_size, dilation=conv.dilation, padding=conv.padding, stride=conv.stride
    )
    patch_rows = patches.transpose(1, 2).reshape(-1, conv.in_features)
    expected_covariance = patch_rows.t() @ patch_rows / patch_rows.shape[0]

    torch.testing.assert_close(snapshot.cross, expected_cross)
    torch.testing.assert_close(snapshot.covariance, expected_covariance)


# --- (b) growth: cSFW births atoms and loss trends down over several events ---


def test_csfw_grows_k_and_reduces_loss_on_a_small_conv_teacher_task() -> None:
    store, inputs, outputs, conv = _conv_parts()
    engine = _engine(
        store, inputs, outputs, conv,
        method=cSFW(polish_iters=0, backfit=None, pool_size=1024, multistart=4),
        budget=2,
        seed=11,
    )
    x, y = _teacher_problem(seed=1)

    def _loss() -> float:
        with torch.no_grad():
            return float((conv(x) - y).square().mean())

    losses = [_loss()]
    for _ in range(10):
        conv.zero_grad(set_to_none=True)
        engine.begin_update()
        (conv(x) - y).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        engine.step()
        losses.append(_loss())

    k_after = int(store.view().ids.numel())
    assert k_after >= 15  # 10 events * up to 2 births, minus any rent-rejected ones
    assert losses[-1] < losses[0] * 0.4
    # Matching-pursuit-style sequential deflation should not make things
    # worse at any single event.
    assert all(later <= earlier + 1.0e-9 for earlier, later in zip(losses, losses[1:]))


# --- (c) dual-timing equivalence: backward_inline vs after_backward ---


@pytest.mark.parametrize("capture_mode", ["deferred", "inline_reduced"])
def test_conv_backward_inline_and_after_backward_timing_match(capture_mode: str) -> None:
    x, y = _teacher_problem(seed=2)

    def _run(timing: str):
        store, inputs, outputs, conv = _conv_parts()
        engine = _engine(
            store, inputs, outputs, conv,
            method=cSFW(polish_iters=0, backfit=None, pool_size=512, multistart=4, timing=timing),
            budget=2,
            seed=9,
            capture_mode=capture_mode,
        )
        engine.begin_update()
        (conv(x) - y).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        snapshot = _tangent_snapshot(engine)
        operations = engine.step()
        births = [op for op in operations if isinstance(op, SynapseBirth)]
        assert len(births) == 1
        return snapshot.cross, snapshot.covariance, births[0]

    inline_cross, inline_cov, inline_birth = _run("backward_inline")
    after_cross, after_cov, after_birth = _run("after_backward")

    torch.testing.assert_close(inline_cross, after_cross)
    torch.testing.assert_close(inline_cov, after_cov)
    torch.testing.assert_close(inline_birth.s, after_birth.s)
    torch.testing.assert_close(inline_birth.t, after_birth.t)
    torch.testing.assert_close(inline_birth.w, after_birth.w)


# --- (d) gradient accumulation: two half-batches match one full batch ---


def test_conv_tangent_statistics_are_additive_across_microbatches() -> None:
    x, y = _teacher_problem(seed=4, batch=8)

    def _full():
        store, inputs, outputs, conv = _conv_parts()
        engine = _engine(
            store, inputs, outputs, conv,
            method=cSFW(polish_iters=0, backfit=None, pool_size=512, multistart=4),
            budget=2, seed=13,
        )
        engine.begin_update()
        (conv(x) - y).square().mean().backward()
        engine.observe_microbatch()
        engine.finalize_backward()
        snapshot = _tangent_snapshot(engine)
        births = [op for op in engine.step() if isinstance(op, SynapseBirth)]
        return snapshot, births[0]

    def _split():
        store, inputs, outputs, conv = _conv_parts()
        engine = _engine(
            store, inputs, outputs, conv,
            method=cSFW(polish_iters=0, backfit=None, pool_size=512, multistart=4),
            budget=2, seed=13,
        )
        half = x.shape[0] // 2
        engine.begin_update()
        (conv(x[:half]) - y[:half]).square().mean().backward()
        engine.observe_microbatch(weight=0.5)
        (conv(x[half:]) - y[half:]).square().mean().backward()
        engine.observe_microbatch(weight=0.5)
        engine.finalize_backward()
        snapshot = _tangent_snapshot(engine)
        births = [op for op in engine.step() if isinstance(op, SynapseBirth)]
        return snapshot, births[0]

    full_snapshot, full_birth = _full()
    split_snapshot, split_birth = _split()

    torch.testing.assert_close(full_snapshot.cross, split_snapshot.cross)
    torch.testing.assert_close(full_snapshot.covariance, split_snapshot.covariance)
    torch.testing.assert_close(full_birth.s, split_birth.s)
    torch.testing.assert_close(full_birth.t, split_birth.t)
    torch.testing.assert_close(full_birth.w, split_birth.w)


# --- (e) curvature validity: correct a helps, too-small a is dangerous ---


def test_conv_birth_curvature_scale_is_correct_direction_sensitive() -> None:
    """The solved amplitude ``g/a`` must sit at the quadratic-in-weight
    minimum of the patch-level loss for the birth's chosen ``(source,
    target)``. Since the loss along that one line is an exact convex
    parabola (linear map, fixed coordinates), this pins down not just the
    sign but the *scale* of the captured curvature ``a``:

    * inflating ``a`` (under-stepping toward the true optimum) can only ever
      land strictly between the current point and the optimum -- always a
      loss decrease.
    * deflating ``a`` (over-stepping past the true optimum) can overshoot
      arbitrarily far past it -- and *does* make loss worse once the step
      exceeds twice the true optimum.
    """
    store, inputs, outputs, conv = _conv_parts()
    engine = _engine(
        store, inputs, outputs, conv,
        method=cSFW(polish_iters=0, backfit=None, pool_size=2048, multistart=4),
        budget=1,
        seed=21,
    )
    x, y = _teacher_problem(seed=5)

    with torch.no_grad():
        baseline_loss = float((conv(x) - y).square().mean())
        assert baseline_loss > 0.0

    engine.begin_update()
    (conv(x) - y).square().mean().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    snapshot = _tangent_snapshot(engine)
    operations = engine.step()
    births = [op for op in operations if isinstance(op, SynapseBirth)]
    assert len(births) == 1
    birth = births[0]
    source, target, weight_true = birth.s[0], birth.t[0], float(birth.w[0])
    assert abs(weight_true) > 1.0e-6

    k_in, k_out = conv.kernel_columns(source[None, :], target[None, :])

    def _loss_at(weight: float) -> float:
        delta = weight * torch.outer(k_out[:, 0], k_in[:, 0])
        filt = delta.reshape(conv.out_channels, conv.in_channels, *conv.kernel_size)
        with torch.no_grad():
            predicted = F.conv2d(x, filt, None, conv.stride, conv.padding, conv.dilation)
            return float((predicted - y).square().mean())

    # Sanity: the birth's own solved weight lands at (or very near) this
    # line's true minimum -- reproducing TangentBirth's internal g/a formula
    # independently through kernel_columns and the finalized snapshot.
    rhs = torch.einsum("ok,oi,ik->k", k_out, snapshot.cross.to(k_out), k_in)
    input_curvature = (k_in * (snapshot.covariance.to(k_in) @ k_in)).sum(dim=0)
    output_curvature = k_out.square().sum(dim=0)
    a_true = float((input_curvature * output_curvature)[0])
    g_true = float(rhs[0])
    torch.testing.assert_close(torch.tensor(weight_true), torch.tensor(g_true / a_true))

    true_step_loss = _loss_at(weight_true)
    assert true_step_loss < baseline_loss

    # Overestimating curvature by 20x -> understeps to weight_true/20, which
    # lies strictly between 0 and 2*weight_true: always safe.
    safe_weight = weight_true / 20.0
    assert _loss_at(safe_weight) < baseline_loss

    # Underestimating curvature by 20x -> oversteps to 20*weight_true, well
    # past 2*weight_true: guaranteed worse than doing nothing at all.
    dangerous_weight = weight_true * 20.0
    assert _loss_at(dangerous_weight) > baseline_loss
