"""batched_conv_dense_weights: bit-identical to per-layer dense_weight(),
matching gradients, and validated preconditions."""

from __future__ import annotations

import pytest
import torch

from torchcst.compute import CSTConv2d, batched_conv_dense_weights, conv2d_neuron_coordinates
from torchcst.compute.capture import BackwardContext
from torchcst.representation import GaussianKernel, RepresentationSpec, TriangularKernel
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _make_layer(
    *, seed: int, in_channels: int = 2, out_channels: int = 3, kernel_size=(2, 2),
    k: int = 4, sigma: float = 0.4, track_mass: bool = False, kernel_cls=GaussianKernel,
) -> CSTConv2d:
    generator = torch.Generator().manual_seed(seed)
    input_mu, output_mu = conv2d_neuron_coordinates(in_channels, out_channels, kernel_size, dtype=torch.float64)
    kernel_name = "triangular" if kernel_cls is TriangularKernel else "gaussian"
    store = SynapseStore(
        f"conv{seed}", 3, 1, k, spec=RepresentationSpec.continuous(3, 1, kernel=kernel_name), dtype=torch.float64,
    )
    s0 = torch.rand(k, 3, generator=generator, dtype=torch.float64)
    t0 = torch.rand(k, 1, generator=generator, dtype=torch.float64)
    w0 = torch.randn(k, generator=generator, dtype=torch.float64) * 0.3
    store.apply([SynapseBirth(store.site, s0, t0, w0, torch.arange(k, dtype=torch.int64))])
    in_neurons = NeuronStore(
        f"conv{seed}-in", input_mu.shape[0], mu=input_mu, initial_live=input_mu.shape[0], dtype=torch.float64,
    )
    out_neurons = NeuronStore(
        f"conv{seed}-out", output_mu.shape[0], mu=output_mu, initial_live=output_mu.shape[0], dtype=torch.float64,
    )
    kernel = kernel_cls(sigma, learnable=True).double()
    return CSTConv2d(
        in_neurons, out_neurons, store, kernel, in_channels, kernel_size,
        bias=False, implementation="materialized", track_mass=track_mass,
    )


def test_batched_matches_per_layer_dense_weight_bitwise() -> None:
    layers = [_make_layer(seed=i, sigma=0.3 + 0.05 * i) for i in range(4)]
    # Perturb gates so the gate-multiply stage is also exercised non-trivially.
    with torch.no_grad():
        layers[1].in_neurons.gate[0] = 0.4
        layers[2].out_neurons.gate[1] = 0.25

    expected = [layer.dense_weight() for layer in layers]
    actual = batched_conv_dense_weights(layers)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.shape == e.shape
        assert torch.equal(a, e)


def test_batched_matches_per_layer_gradients() -> None:
    """s/t/w and sigma gradients must be *close* to the per-layer
    computation's -- see the module docstring's determinism note in
    batched_conv.py: each of these broadcasts over a large axis to build
    the distance/kernel tensors, so its own gradient is a reduction over
    that axis, and a single fused reduction over the whole batched tensor
    is not guaranteed bit-identical to per-layer reductions (a real
    property of how PyTorch's broadcast-backward and GPU/CPU reduction
    kernels work, not a bug here -- measured at ~1e-6 absolute for sigma on
    real CUDA/float32 bench shapes, smaller still for s/t, both far below
    this repo's own documented 0.5% reproducibility floor). assert_close's
    default tolerance is comfortably tighter than that bound, so this is
    still a real regression check, not a rubber stamp."""

    layers = [_make_layer(seed=10 + i) for i in range(3)]

    actual = batched_conv_dense_weights(layers)
    grad_seeds = [torch.randn_like(w) for w in actual]
    # actual[i] shares graph nodes (the stacked intermediates) with every
    # other actual[j] -- a real caller sums all layer losses into one
    # scalar before backward()ing once, so this does the equivalent: one
    # combined backward call over every output at once, not one call per
    # output (which would free shared buffers after the first and error on
    # the second).
    torch.autograd.backward(actual, grad_seeds)
    actual_grads = {
        (layer.synapses.site, name): getattr(layer.synapses, name).grad.clone()
        for layer in layers for name in ("s", "t", "w")
    }
    actual_sigma_grads = [layer.kernel_in.sigma.grad.clone() for layer in layers]
    for layer in layers:
        layer.synapses.s.grad = None
        layer.synapses.t.grad = None
        layer.synapses.w.grad = None
        layer.kernel_in.sigma.grad = None

    expected = [layer.dense_weight() for layer in layers]
    for w, g in zip(expected, grad_seeds):
        w.backward(g)
    for layer in layers:
        for name in ("s", "t", "w"):
            torch.testing.assert_close(
                getattr(layer.synapses, name).grad, actual_grads[(layer.synapses.site, name)],
            )
    for layer, actual_sigma_grad in zip(layers, actual_sigma_grads):
        torch.testing.assert_close(layer.kernel_in.sigma.grad, actual_sigma_grad)


def test_empty_layers_returns_empty_list() -> None:
    assert batched_conv_dense_weights([]) == []


def test_rejects_mismatched_feature_shape() -> None:
    a = _make_layer(seed=1, in_channels=2, out_channels=3)
    b = _make_layer(seed=2, in_channels=3, out_channels=3)
    with pytest.raises(ValueError, match="in_features and out_features"):
        batched_conv_dense_weights([a, b])


def test_rejects_mismatched_kernel_family() -> None:
    a = _make_layer(seed=1, kernel_cls=GaussianKernel)
    b = _make_layer(seed=2, kernel_cls=TriangularKernel)
    with pytest.raises(ValueError, match="kernel_in/kernel_out family"):
        batched_conv_dense_weights([a, b])


def test_rejects_mismatched_live_atom_count() -> None:
    a = _make_layer(seed=1, k=4)
    b = _make_layer(seed=2, k=5)
    with pytest.raises(ValueError, match="live atom count"):
        batched_conv_dense_weights([a, b])


def test_rejects_track_mass_layer() -> None:
    a = _make_layer(seed=1, track_mass=True)
    b = _make_layer(seed=2, track_mass=True)
    with pytest.raises(ValueError, match="track_mass"):
        batched_conv_dense_weights([a, b])


def test_rejects_capture_enabled_layer() -> None:
    a = _make_layer(seed=1)
    b = _make_layer(seed=2)
    a.set_backward_context(BackwardContext(0))
    with pytest.raises(ValueError, match="capture context"):
        batched_conv_dense_weights([a, b])
