from types import SimpleNamespace

import torch

from experiments.parameter_rms import ParameterSquareEMA
from torchcst.optim.atom_grad import AtomGradientObservation


def test_raw_ema_squares_accumulated_parameter_gradient_and_corrects_bias():
    context = SimpleNamespace(
        current_point=torch.zeros(1, 2, dtype=torch.float64),
        geometry=SimpleNamespace(visible_shape=(2, 2)),
    )
    ema = ParameterSquareEMA(0.9, 1e-8, "raw", 0.01)
    state = ema.initialize(context)
    gradients = [
        torch.tensor([[2.0, -3.0]], dtype=torch.float64),
        torch.tensor([[-1.0, 4.0]], dtype=torch.float64),
    ]
    reference = torch.zeros_like(gradients[0])
    for step, g in enumerate(gradients, 1):
        result = ema.expand(
            state, AtomGradientObservation(jg=g), context, next_step=step
        )
        reference = 0.9 * reference + 0.1 * g.square()
        torch.testing.assert_close(
            result.metric.blocks.diagonal(dim1=-2, dim2=-1),
            (reference / (1 - 0.9**step)).sqrt() + 1e-8,
        )
        state = ema.compress(result, None, context)
        torch.testing.assert_close(state.value, reference)


def test_alpha_diagonal_scales_gradient_before_ema_and_maps_denominator_back():
    point = torch.zeros(1, 2, dtype=torch.float64)
    gram = torch.diag_embed(torch.tensor([[1.0, 4.0]], dtype=torch.float64))
    context = SimpleNamespace(
        current_point=point,
        geometry=SimpleNamespace(
            visible_shape=(2, 2),
            prepared=lambda _: SimpleNamespace(gram_blocks=lambda: gram),
        ),
    )
    ema = ParameterSquareEMA(0.5, 1e-8, "alpha_diagonal", 0.01)
    g = torch.tensor([[2.0, 6.0]], dtype=torch.float64)
    first = ema.expand(
        ema.initialize(context), AtomGradientObservation(jg=g), context, next_step=1
    )
    scale = torch.tensor([[1.01, 4.01]], dtype=torch.float64)
    torch.testing.assert_close(first.pending_state.value, 0.5 * (g / scale).square())
    # Move geometry before second observation: old EMA is intentionally not transported.
    gram.mul_(2)
    new_scale = torch.tensor([[2.01, 8.01]], dtype=torch.float64)
    second = ema.expand(
        first.pending_state, AtomGradientObservation(jg=-g), context, next_step=2
    )
    expected = 0.5 * first.pending_state.value + 0.5 * (g / new_scale).square()
    torch.testing.assert_close(
        second.metric.blocks.diagonal(dim1=-2, dim2=-1),
        new_scale * ((expected / 0.75).sqrt() + 1e-8),
    )
