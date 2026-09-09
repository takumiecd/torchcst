from types import SimpleNamespace

import torch

from experiments.transported_parameter_rms import TransportedParameterRMS
from torchcst.optim.atom_grad import AtomGradientObservation


def context(index, jacobians):
    point = torch.full((1, 2), float(index), dtype=torch.float64)

    def cross(a, b, local=False):
        assert local
        return (jacobians[int(a[0, 0])].T @ jacobians[int(b[0, 0])]).unsqueeze(0)

    return SimpleNamespace(
        current_point=point, geometry=SimpleNamespace(cross=cross, visible_shape=(3, 1))
    )


def test_transport_discards_a_direction_no_longer_representable():
    eye = torch.eye(3, dtype=torch.float64)
    js = [eye[:, :2], eye[:, 1:]]
    old, new = context(0, js), context(1, js)
    rms = TransportedParameterRMS(0.5, eps=1e-8)
    first = rms.expand(
        rms.initialize(old),
        AtomGradientObservation(jg=torch.tensor([[2.0, 0.0]], dtype=torch.float64)),
        old,
        next_step=1,
    )
    second = rms.expand(
        first.pending_state,
        AtomGradientObservation(jg=torch.zeros(1, 2, dtype=torch.float64)),
        new,
        next_step=2,
    )
    torch.testing.assert_close(
        second.pending_state.value, torch.zeros_like(second.pending_state.value)
    )


def test_transport_preserves_visible_second_moment_under_basis_change():
    j = torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float64)
    js = [j, j @ rotation]
    old, new = context(0, js), context(1, js)
    rms = TransportedParameterRMS(0.9, eps=1e-8)
    g = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    first = rms.expand(
        rms.initialize(old),
        AtomGradientObservation(jg=(j.T @ g).unsqueeze(0)),
        old,
        next_step=1,
    )
    second = rms.expand(
        first.pending_state,
        AtomGradientObservation(jg=torch.zeros(1, 2, dtype=torch.float64)),
        new,
        next_step=2,
    )
    q0 = j @ first.pending_state.basis[0]
    q1 = js[1] @ second.pending_state.basis[0]
    expected = 0.9 * q0 @ first.pending_state.value[0] @ q0.T
    torch.testing.assert_close(q1 @ second.pending_state.value[0] @ q1.T, expected)
    assert torch.linalg.eigvalsh(second.metric.blocks[0]).min() >= -1e-12
