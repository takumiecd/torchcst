"""Dense-oracle checks for the dense-free atom-local P2 primitives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from torchcst.compute import BackwardContext, CSTLinear, Factored
from torchcst.optim.p2 import (
    ContractedP2Model,
    CSTP2TrustRegion,
    P2ModelEMA,
    P2StepEMA,
    contracted_cst_linear_p2_model,
    contracted_p2_model,
    cst_linear_p2_reducer,
    solve_block_trust_region,
)
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _problem():
    generator = torch.Generator().manual_seed(31)
    dtype = torch.float64
    theta = torch.tensor([[0.2, 0.31, 0.63], [-0.08, 0.68, 0.42]], dtype=dtype)
    inputs = torch.randn(7, 5, generator=generator, dtype=dtype)
    targets = torch.randn(7, 4, generator=generator, dtype=dtype)
    mu_in = torch.linspace(0.0, 1.0, 5, dtype=dtype)
    mu_out = torch.linspace(0.0, 1.0, 4, dtype=dtype)
    sigma = 0.27

    def columns(atom):
        k_in = torch.exp(-0.5 * ((mu_in - atom[1]) / sigma).square())
        k_out = torch.exp(-0.5 * ((mu_out - atom[2]) / sigma).square())
        return k_in, k_out

    def factored_output(point):
        k_in = torch.stack([columns(atom)[0] for atom in point], dim=1)
        k_out = torch.stack([columns(atom)[1] for atom in point], dim=1)
        return ((inputs @ k_in) * point[:, 0]) @ k_out.T

    return theta, inputs, targets, columns, factored_output


def _factored_layer(theta, inputs, targets, site="p2-train"):
    synapses = SynapseStore(
        site,
        1,
        1,
        theta.shape[0],
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    synapses.apply(
        [
            SynapseBirth(
                synapses.site,
                theta[:, 1:2],
                theta[:, 2:3],
                theta[:, 0],
                torch.arange(theta.shape[0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            f"{site}-in",
            inputs.shape[1],
            mu=torch.linspace(0, 1, inputs.shape[1], dtype=torch.float64)[:, None],
            initial_live=inputs.shape[1],
            dtype=torch.float64,
        ),
        NeuronStore(
            f"{site}-out",
            targets.shape[1],
            mu=torch.linspace(0, 1, targets.shape[1], dtype=torch.float64)[:, None],
            initial_live=targets.shape[1],
            dtype=torch.float64,
        ),
        synapses,
        GaussianFactor(0.27, learnable=False).double(),
        track_mass=False,
        gauge=L2NormalizedColumns(),
        backend=Factored(),
    )
    return layer, synapses


def test_dense_free_contractions_equal_dense_weight_oracle():
    theta, inputs, targets, columns, factored_output = _problem()
    point = theta.detach().requires_grad_(True)
    output = factored_output(point)
    loss = F.mse_loss(output, targets)
    (upstream,) = torch.autograd.grad(loss, output, retain_graph=True)

    def atom_score(atom):
        k_in, k_out = columns(atom)
        activation = inputs @ k_in
        output_projection = upstream.detach() @ k_out
        return atom[0] * (activation * output_projection).sum()

    dense_free = contracted_p2_model(atom_score, point)

    def dense_weight(candidate):
        k_in = torch.stack([columns(atom)[0] for atom in candidate], dim=1)
        k_out = torch.stack([columns(atom)[1] for atom in candidate], dim=1)
        return (k_out * candidate[:, 0]) @ k_in.T

    weight = dense_weight(theta).detach().requires_grad_(True)
    weight_loss = F.mse_loss(F.linear(inputs, weight), targets)
    (weight_gradient,) = torch.autograd.grad(weight_loss, weight)

    def frozen_dense_score(candidate):
        return (weight_gradient.detach() * dense_weight(candidate)).sum()

    oracle_linear = torch.func.grad(frozen_dense_score)(theta)
    oracle_hessian = torch.func.hessian(frozen_dense_score)(theta)
    oracle_blocks = torch.stack(
        [oracle_hessian[k, :, k, :] for k in range(theta.shape[0])]
    )

    torch.testing.assert_close(dense_free.linear, oracle_linear)
    torch.testing.assert_close(dense_free.curvature, oracle_blocks)
    assert dense_free.linear.shape == theta.shape
    assert dense_free.curvature.shape == (theta.shape[0], 3, 3)


def test_cst_linear_p2_path_rejects_dense_execution_and_matches_parameter_grads():
    theta, inputs, targets, _, _ = _problem()
    synapses = SynapseStore(
        "p2",
        1,
        1,
        theta.shape[0],
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    synapses.apply(
        [
            SynapseBirth(
                synapses.site,
                theta[:, 1:2],
                theta[:, 2:3],
                theta[:, 0],
                torch.arange(theta.shape[0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            "p2-in",
            inputs.shape[1],
            mu=torch.linspace(0, 1, inputs.shape[1], dtype=torch.float64)[:, None],
            initial_live=inputs.shape[1],
            dtype=torch.float64,
        ),
        NeuronStore(
            "p2-out",
            targets.shape[1],
            mu=torch.linspace(0, 1, targets.shape[1], dtype=torch.float64)[:, None],
            initial_live=targets.shape[1],
            dtype=torch.float64,
        ),
        synapses,
        GaussianFactor(0.27, learnable=False).double(),
        track_mass=False,
        gauge=L2NormalizedColumns(),
        backend=Factored(),
    )

    # If either the training forward or P2 contraction asks for W, fail here.
    layer.dense_weight = lambda: (_ for _ in ()).throw(
        AssertionError("dense_weight must not be called")
    )
    output = layer(inputs)
    loss = F.mse_loss(output, targets)
    (grad_output,) = torch.autograd.grad(loss, output, retain_graph=True)
    model = contracted_cst_linear_p2_model(layer, inputs, grad_output)
    gradients = torch.autograd.grad(
        loss, (synapses.w, synapses.s, synapses.t), allow_unused=False
    )
    slots = synapses.live_slots().to(synapses.w.device)
    expected = torch.cat(
        (
            gradients[0].index_select(0, slots)[:, None],
            gradients[1].index_select(0, slots),
            gradients[2].index_select(0, slots),
        ),
        dim=1,
    )

    torch.testing.assert_close(model.linear, expected)

    # Dense materialization is allowed only here, as a small correctness oracle.
    def oracle_score(candidate):
        source = candidate[:, 1:2]
        target = candidate[:, 2:3]
        weights = candidate[:, 0]
        k_in = layer.gauge.columns(layer.factor_in, layer.in_neurons.mu, source, {})
        k_out = layer.gauge.columns(layer.factor_out, layer.out_neurons.mu, target, {})
        dense_weight = (k_out * weights) @ k_in.T
        return (grad_output.detach() * F.linear(inputs, dense_weight)).sum()

    oracle_hessian = torch.func.hessian(oracle_score)(theta)
    oracle_blocks = torch.stack(
        [oracle_hessian[k, :, k, :] for k in range(theta.shape[0])]
    )
    torch.testing.assert_close(model.curvature, oracle_blocks)


def test_cst_linear_p2_requires_explicit_factored_backend():
    theta, inputs, targets, _, _ = _problem()
    synapses = SynapseStore(
        "p2-backend",
        1,
        1,
        theta.shape[0],
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    synapses.apply(
        [
            SynapseBirth(
                synapses.site,
                theta[:, 1:2],
                theta[:, 2:3],
                theta[:, 0],
                torch.arange(theta.shape[0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            "p2-backend-in",
            inputs.shape[1],
            mu=torch.linspace(0, 1, inputs.shape[1], dtype=torch.float64)[:, None],
            initial_live=inputs.shape[1],
            dtype=torch.float64,
        ),
        NeuronStore(
            "p2-backend-out",
            targets.shape[1],
            mu=torch.linspace(0, 1, targets.shape[1], dtype=torch.float64)[:, None],
            initial_live=targets.shape[1],
            dtype=torch.float64,
        ),
        synapses,
        GaussianFactor(0.27, learnable=False).double(),
        track_mass=False,
        backend=Factored(),
    )
    output = layer(inputs)
    grad_output = torch.autograd.grad(F.mse_loss(output, targets), output)[0]
    layer.backend = "auto"
    try:
        contracted_cst_linear_p2_model(layer, inputs, grad_output)
    except ValueError as error:
        assert "explicit Factored" in str(error)
    else:
        raise AssertionError("materialized CSTLinear must be rejected")


def test_hook_time_reducer_retains_only_atom_sized_statistics():
    theta, inputs, targets, _, _ = _problem()
    synapses = SynapseStore(
        "p2-reducer",
        1,
        1,
        theta.shape[0],
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    synapses.apply(
        [
            SynapseBirth(
                synapses.site,
                theta[:, 1:2],
                theta[:, 2:3],
                theta[:, 0],
                torch.arange(theta.shape[0], dtype=torch.int64),
            )
        ]
    )
    layer = CSTLinear(
        NeuronStore(
            "p2-reducer-in",
            inputs.shape[1],
            mu=torch.linspace(0, 1, inputs.shape[1], dtype=torch.float64)[:, None],
            initial_live=inputs.shape[1],
            dtype=torch.float64,
        ),
        NeuronStore(
            "p2-reducer-out",
            targets.shape[1],
            mu=torch.linspace(0, 1, targets.shape[1], dtype=torch.float64)[:, None],
            initial_live=targets.shape[1],
            dtype=torch.float64,
        ),
        synapses,
        GaussianFactor(0.27, learnable=False).double(),
        track_mass=False,
        gauge=L2NormalizedColumns(),
        backend=Factored(),
    )
    context = BackwardContext(
        0,
        reducers={layer.capture_site: cst_linear_p2_reducer(layer)},
        raw_sites=set(),
    )
    layer.set_backward_context(context)
    F.mse_loss(layer(inputs), targets).backward()
    context.observe_microbatch()
    captured = context.finalize_capture()

    assert captured.observations == ()
    assert len(captured.reduced) == 1
    values = captured.reduced[0].values
    assert values["linear"].shape == theta.shape
    assert values["curvature"].shape == (theta.shape[0], 3, 3)
    retained = sum(value.numel() for value in values.values())
    assert retained == theta.numel() + theta.shape[0] * theta.shape[1] ** 2


def test_p2_optimizer_reduces_loss_without_dense_weight_or_dense_state():
    theta, inputs, targets, _, _ = _problem()
    layer, _ = _factored_layer(theta, inputs, targets)
    layer.dense_weight = lambda: (_ for _ in ()).throw(
        AssertionError("dense_weight must not be called")
    )
    optimizer = CSTP2TrustRegion(
        layer,
        radius=0.1,
        beta=0.5,
        amplitude_scale=0.2,
    )
    with torch.no_grad():
        initial_loss = F.mse_loss(layer(inputs), targets)

    for update_id in range(50):
        context = optimizer.capture_context(update_id)
        layer.set_backward_context(context)
        optimizer.zero_grad()
        loss = F.mse_loss(layer(inputs), targets)
        loss.backward()
        context.observe_microbatch()
        capture = context.finalize_capture()
        layer.set_backward_context(None)
        result = optimizer.step(capture)
        assert result.predicted_change <= 0

    with torch.no_grad():
        final_loss = F.mse_loss(layer(inputs), targets)
    assert final_loss < 0.9 * initial_loss
    assert optimizer.model_ema.linear is not None
    assert optimizer.model_ema.curvature is not None
    retained = (
        optimizer.model_ema.linear.numel() + optimizer.model_ema.curvature.numel()
    )
    fields = theta.shape[1]
    assert retained == theta.shape[0] * (fields + fields**2)


def test_p2_optimizer_averages_microbatches_before_one_ema_update():
    theta, inputs, targets, _, _ = _problem()
    layer, _ = _factored_layer(theta, inputs, targets, site="p2-micro")
    optimizer = CSTP2TrustRegion(layer, radius=0.01, beta=0.9)
    context = optimizer.capture_context(0)
    layer.set_backward_context(context)

    split = 3
    for local_inputs, local_targets, weight in (
        (inputs[:split], targets[:split], 0.25),
        (inputs[split:], targets[split:], 0.75),
    ):
        F.mse_loss(layer(local_inputs), local_targets).backward()
        context.observe_microbatch(weight)
    capture = context.finalize_capture()
    layer.set_backward_context(None)
    batch_model = optimizer._batch_model(capture)
    expected_linear = sum(
        item.values["linear"] * item.micro_weight for item in capture.reduced
    )
    expected_curvature = sum(
        item.values["curvature"] * item.micro_weight for item in capture.reduced
    )

    torch.testing.assert_close(batch_model.linear, expected_linear)
    torch.testing.assert_close(batch_model.curvature, expected_curvature)


def test_p2_optimizer_state_round_trip_preserves_model_ema():
    theta, inputs, targets, _, _ = _problem()
    layer, _ = _factored_layer(theta, inputs, targets, site="p2-state")
    optimizer = CSTP2TrustRegion(
        layer,
        radius=0.03,
        beta=0.7,
        amplitude_scale=0.4,
        curvature_scale=0.6,
        step_beta=0.8,
        root_iterations=37,
    )
    context = optimizer.capture_context(0)
    layer.set_backward_context(context)
    F.mse_loss(layer(inputs), targets).backward()
    context.observe_microbatch()
    capture = context.finalize_capture()
    layer.set_backward_context(None)
    optimizer.step(capture)

    restored = CSTP2TrustRegion(layer, radius=1.0, beta=0.0)
    restored.load_state_dict(optimizer.state_dict())

    assert restored.radius == optimizer.radius
    assert restored.amplitude_scale == optimizer.amplitude_scale
    assert restored.curvature_scale == optimizer.curvature_scale
    assert restored.root_iterations == optimizer.root_iterations
    assert restored.model_ema.beta == optimizer.model_ema.beta
    assert restored.model_ema.mass == optimizer.model_ema.mass
    torch.testing.assert_close(restored.model_ema.linear, optimizer.model_ema.linear)
    torch.testing.assert_close(
        restored.model_ema.curvature, optimizer.model_ema.curvature
    )
    assert restored.step_ema is not None
    assert optimizer.step_ema is not None
    torch.testing.assert_close(
        restored.step_ema.dimensionless_step,
        optimizer.step_ema.dimensionless_step,
    )


def test_model_ema_is_the_bias_corrected_average_of_quadratic_models():
    first = ContractedP2Model(
        linear=torch.tensor([[1.0, -2.0]]),
        curvature=torch.tensor([[[3.0, 0.5], [0.5, 4.0]]]),
    )
    second = ContractedP2Model(
        linear=torch.tensor([[-3.0, 6.0]]),
        curvature=torch.tensor([[[1.0, -0.5], [-0.5, 2.0]]]),
    )
    ema = P2ModelEMA(beta=0.5)
    ema.update(first)
    result = ema.update(second)

    # Corrected weights after two updates are 1/3 and 2/3.
    torch.testing.assert_close(result.linear, (first.linear + 2 * second.linear) / 3)
    torch.testing.assert_close(
        result.curvature, (first.curvature + 2 * second.curvature) / 3
    )


def test_step_ema_averages_dimensionless_steps_and_enforces_radius():
    ema = P2StepEMA(beta=0.5)
    scales = torch.tensor([[2.0, 0.5]], dtype=torch.float64)
    first = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    second = torch.tensor([[0.0, 0.5]], dtype=torch.float64)

    torch.testing.assert_close(ema.update(first, scales, radius=2.0), first)
    result = ema.update(second, scales, radius=2.0)
    # Corrected dimensionless weights are 1/3 and 2/3.
    torch.testing.assert_close(
        result, scales * torch.tensor([[1 / 3, 2 / 3]], dtype=torch.float64)
    )
    projected = ema.update(second, scales, radius=0.1)
    torch.testing.assert_close(
        torch.linalg.vector_norm(projected / scales),
        torch.tensor(0.1, dtype=torch.float64),
    )


def test_curvature_scale_zero_is_a_p1_boundary_step():
    theta, inputs, targets, _, _ = _problem()
    layer, _ = _factored_layer(theta, inputs, targets, site="p2-p1")
    optimizer = CSTP2TrustRegion(
        layer,
        radius=0.02,
        beta=0.0,
        curvature_scale=0.0,
    )
    context = optimizer.capture_context(0)
    layer.set_backward_context(context)
    F.mse_loss(layer(inputs), targets).backward()
    context.observe_microbatch()
    capture = context.finalize_capture()
    layer.set_backward_context(None)
    result = optimizer.step(capture)

    assert optimizer.model_ema.curvature is not None
    torch.testing.assert_close(
        optimizer.model_ema.curvature,
        torch.zeros_like(optimizer.model_ema.curvature),
    )
    scales = optimizer._scales(result.step)
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.step / scales),
        torch.tensor(optimizer.radius, dtype=result.step.dtype),
    )


def test_adaptive_radius_rejects_ascent_and_restores_atom_state():
    theta, inputs, targets, _, _ = _problem()
    layer, store = _factored_layer(theta, inputs, targets, site="p2-reject")
    optimizer = CSTP2TrustRegion(
        layer,
        radius=0.1,
        beta=0.0,
        adaptive_radius=True,
    )
    fields = 1 + store.d_in + store.d_out
    model = ContractedP2Model(
        linear=torch.ones(theta.shape[0], fields, dtype=theta.dtype),
        curvature=torch.zeros(theta.shape[0], fields, fields, dtype=theta.dtype),
    )
    optimizer._batch_model = lambda capture: model
    before = (store.w.clone(), store.s.clone(), store.t.clone())

    def ascent_loss():
        return sum(
            (current - old).square().sum()
            for current, old in zip((store.w, store.s, store.t), before)
        )

    result = optimizer.step(object(), closure=ascent_loss, current_loss=0.0)

    assert not result.accepted
    assert result.actual_change > 0
    assert result.radius_before == 0.1
    assert result.radius_after == 0.05
    assert optimizer.rejected_steps == 1
    torch.testing.assert_close(store.w, before[0])
    torch.testing.assert_close(store.s, before[1])
    torch.testing.assert_close(store.t, before[2])


def test_adaptive_radius_accepts_good_boundary_step_and_grows_radius():
    theta, inputs, targets, _, _ = _problem()
    layer, store = _factored_layer(theta, inputs, targets, site="p2-accept")
    optimizer = CSTP2TrustRegion(
        layer,
        radius=0.01,
        beta=0.0,
        adaptive_radius=True,
        max_radius=0.1,
    )
    fields = 1 + store.d_in + store.d_out
    linear = torch.zeros(theta.shape[0], fields, dtype=theta.dtype)
    linear[:, 0] = 1.0
    model = ContractedP2Model(
        linear=linear,
        curvature=torch.zeros(theta.shape[0], fields, fields, dtype=theta.dtype),
    )
    optimizer._batch_model = lambda capture: model
    target_w = store.w.detach().clone() - 1.0

    def improving_loss():
        return (store.w - target_w).square().sum()

    old_loss = improving_loss().detach()
    result = optimizer.step(object(), closure=improving_loss, current_loss=old_loss)

    assert result.accepted
    assert result.actual_change < 0
    assert result.reduction_ratio > 0.75
    assert result.radius_before == 0.01
    assert result.radius_after == 0.02
    assert optimizer.accepted_steps == 1


def test_trust_region_returns_the_unconstrained_positive_definite_solution():
    model = ContractedP2Model(
        linear=torch.tensor([[2.0, -1.0]], dtype=torch.float64),
        curvature=torch.tensor([[[4.0, 1.0], [1.0, 3.0]]], dtype=torch.float64),
    )
    expected = -torch.linalg.solve(model.curvature[0], model.linear[0])
    result = solve_block_trust_region(model, radius=10.0)

    torch.testing.assert_close(result.step[0], expected)
    assert not result.hit_boundary
    assert float(result.lagrange_multiplier) == 0.0


def test_trust_region_boundary_solution_satisfies_kkt_conditions():
    model = ContractedP2Model(
        linear=torch.tensor([[2.0, -1.0]], dtype=torch.float64),
        curvature=torch.tensor([[[0.8, 0.2], [0.2, 0.5]]], dtype=torch.float64),
    )
    result = solve_block_trust_region(model, radius=0.1)
    multiplier = result.lagrange_multiplier
    residual = model.linear[0] + model.curvature[0] @ result.step[0]
    residual = residual + multiplier * result.step[0]

    torch.testing.assert_close(
        torch.linalg.vector_norm(result.step), torch.tensor(0.1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=1e-10, rtol=1e-10
    )
    assert result.hit_boundary


def test_trust_region_hard_case_uses_negative_curvature_boundary():
    model = ContractedP2Model(
        linear=torch.zeros((1, 2), dtype=torch.float64),
        curvature=torch.tensor([[[-2.0, 0.0], [0.0, 1.0]]], dtype=torch.float64),
    )
    result = solve_block_trust_region(model, radius=0.4)

    torch.testing.assert_close(
        torch.linalg.vector_norm(result.step), torch.tensor(0.4, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.step[0, 1], torch.tensor(0.0, dtype=torch.float64)
    )
    assert result.predicted_change < 0
    assert result.hit_boundary
