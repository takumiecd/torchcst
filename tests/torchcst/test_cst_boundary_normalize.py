"""``CSTBoundary(..., normalize=...)``: an owned pre-activation normalizer.

Centering the pre-activation is needed to stack CST boundaries at all (see
``README.md``, "Stacking layers, and who owns a neuron's gate"): a CST layer
has no bias and no normalization of its own, so nothing keeps the sum of its
atoms centered, and a narrow domain saturates the activation and freezes
training at chance. The pre-existing workaround -- folding a ``LayerNorm``
into the ``activation`` closure -- leaves that module's parameters owned by
nobody, so a caller who forgets to add them to the optimizer gets a
normalizer that silently never trains.

``normalize`` must:
  * default to ``None`` (identity), leaving today's behavior bit-identical;
  * apply to the *pre-activation*, before ``activation``;
  * show up in ``boundary.parameters()`` once assigned, because it is a real
    submodule rather than a closure;
  * be fed the exact same tensor by :meth:`CSTBoundary.forward` and by
    :meth:`CSTBoundary.activated_rows` (the reconstruction
    :meth:`CSTBoundary.gate_tangent` uses for the dormant-gate field) -- this
    is the part most likely to go subtly wrong, so it is checked directly via
    the algebraic identity ``d(gate*activated)/dgate == activated``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchcst.compute import CSTBoundary, CSTLinear
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

D_IN, H_MAX, H_LIVE = 6, 10, 4


def _world(normalize: nn.Module | None):
    store = SynapseStore(
        "layer", 1, 1, 32,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    grid = torch.linspace(0.1, 0.9, 8, dtype=torch.float64)[:, None]
    store.apply(
        [
            SynapseBirth(
                "layer", grid.clone(), grid.flip(0).clone(),
                torch.full((8,), 0.1, dtype=torch.float64), torch.arange(8),
            )
        ]
    )
    mu = lambda n: torch.linspace(0.0, 1.0, n, dtype=torch.float64)[:, None]  # noqa: E731
    inputs = NeuronStore("x", D_IN, mu=mu(D_IN), initial_live=D_IN, dtype=torch.float64)
    hidden = NeuronStore("h", H_MAX, mu=mu(H_MAX), initial_live=H_LIVE, dtype=torch.float64)
    linear = CSTLinear(inputs, hidden, store, GaussianKernel(0.2).double())
    boundary = CSTBoundary(linear, activation=F.gelu, normalize=normalize)
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(32, D_IN, dtype=torch.float64, generator=generator)
    return hidden, linear, boundary, x


def test_default_normalize_is_none_and_output_is_unchanged() -> None:
    hidden, linear, boundary, x = _world(normalize=None)

    assert boundary.normalize is None
    live = hidden.live_ids()
    manual = F.gelu(linear.pre_gate_rows(x)).index_select(1, live)
    torch.testing.assert_close(boundary(linear(x)).index_select(1, live), manual)


def test_normalize_is_rejected_unless_it_is_a_real_module() -> None:
    mu = lambda n: torch.linspace(0.0, 1.0, n, dtype=torch.float64)[:, None]  # noqa: E731
    with pytest.raises(TypeError, match="nn.Module"):
        CSTBoundary(
            CSTLinear(
                NeuronStore("x", D_IN, mu=mu(D_IN), initial_live=D_IN, dtype=torch.float64),
                NeuronStore("h", H_MAX, mu=mu(H_MAX), initial_live=H_LIVE, dtype=torch.float64),
                SynapseStore(
                    "layer", 1, 1, 4,
                    spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
                    dtype=torch.float64,
                ),
                GaussianKernel(0.2).double(),
            ),
            normalize=lambda t: t,
        )


def test_normalize_applies_to_the_pre_activation_and_owns_its_parameters() -> None:
    norm = nn.LayerNorm(H_MAX, dtype=torch.float64)
    hidden, linear, boundary, x = _world(normalize=norm)

    assert boundary.normalize is norm
    owned = dict(boundary.named_parameters())
    assert "normalize.weight" in owned and "normalize.bias" in owned
    assert owned["normalize.weight"] is norm.weight
    assert owned["normalize.bias"] is norm.bias

    # Applied before the activation, not after: activation(normalize(pre)),
    # not normalize(activation(pre)).
    pre = linear.pre_gate_rows(x)
    expected = F.gelu(norm(pre))
    live = hidden.live_ids()
    torch.testing.assert_close(
        boundary(linear(x)).index_select(1, live), expected.index_select(1, live)
    )

    wrong_order = norm(F.gelu(pre))
    assert not torch.allclose(
        boundary(linear(x)).index_select(1, live), wrong_order.index_select(1, live)
    )


def test_forward_and_activated_rows_feed_the_activation_the_same_tensor() -> None:
    """Live gates initialize at 1.0 (see NeuronStore), so at that value
    ``forward``'s ``activated * gate`` at a live column equals ``activated``
    itself exactly -- comparing it against ``activated_rows`` (which
    ``gate_tangent`` reconstructs from) directly exercises whether the two
    call sites were kept in sync when a normalizer sits in between."""
    norm = nn.LayerNorm(H_MAX, dtype=torch.float64)
    hidden, linear, boundary, x = _world(normalize=norm)
    live = hidden.live_ids()

    torch.testing.assert_close(
        boundary(linear(x)).index_select(1, live),
        boundary.activated_rows(x).index_select(1, live),
    )


def test_gate_tangent_reconstruction_matches_true_backprop_through_normalize() -> None:
    """``gate_tangent``'s gradient field must equal the real ``dL/dgamma``.
    Since ``gated = activated * gamma`` is exactly linear in ``gamma``,
    ``dL/dgamma`` at a live row is exactly ``dL/dgamma`` from ordinary
    autograd -- and it will disagree the moment the reconstruction and the
    forward pass normalize the pre-activation differently."""
    norm = nn.LayerNorm(H_MAX, dtype=torch.float64)
    hidden, linear, boundary, x = _world(normalize=norm)
    live = hidden.live_ids()

    out = boundary(linear(x))
    loss = out.square().sum()
    loss.backward()
    true_grad = hidden.gate.grad.index_select(0, live)

    g_gated = boundary.take_gate_grad()
    assert g_gated is not None
    reconstructed_grad, _ = boundary.gate_tangent(x, g_gated)
    torch.testing.assert_close(reconstructed_grad.index_select(0, live), true_grad)
