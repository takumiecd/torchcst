import pytest
import torch
from test_quadratic_feature_gram import make_pair

from torchcst._derivatives import factored_taylor as ft
from torchcst._derivatives.factored_frame import FactoredFrameGeometry


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("block_size", [1, 3, 32])
@pytest.mark.parametrize("displaced", [False, True])
def test_blocked_gram_against_visible_oracle(gated, dtype, block_size, displaced):
    torch.manual_seed(71)
    problem, _ = make_pair(dtype, gated)
    original = problem.context.geometry
    point = problem.context.current_point
    geometry = FactoredFrameGeometry(original.derivatives)
    d = torch.randn_like(point) * 0.2 if displaced else torch.zeros_like(point)
    vector = torch.randn_like(point)
    frame = original.frame(point, d)
    # Independent oracle: visible Jacobian/Hessian, not factor Gram code.
    matrix = original.gram(frame).matrix
    expected = (matrix @ vector.flatten()).reshape_as(vector) + 1e-4 * vector
    actual = geometry.gram_matvec(frame, vector, block_size=block_size, damping=1e-4)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    if dtype == torch.float64:
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_blocked_gram_never_allocates_visible_or_full_gram():
    from torch.utils._python_dispatch import TorchDispatchMode

    k, p, inputs, outputs, block = 7, 4, 37, 41, 3
    shapes = [
        (k, inputs),
        (k, outputs),
        (k, inputs, p),
        (k, outputs, p),
        (k, inputs, p, p),
        (k, outputs, p, p),
    ]
    f = tuple(torch.randn(shape, dtype=torch.float64) for shape in shapes)
    d, vector = torch.randn(k, p), torch.randn(k, p)
    d, vector = d.double(), vector.double()

    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for tensor in torch.utils._pytree.tree_leaves(result):
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                    assert tuple(tensor.shape) not in {
                        (inputs, outputs),
                        (outputs, inputs),
                        (k * p, k * p),
                    }
            return result

    expected = (ft.frame_gram(f, d) @ vector.flatten()).reshape_as(vector)
    with Audit():
        actual = ft.frame_gram_matvec(f, d, vector, block_size=block)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_blocked_gram_cuda_without_host_sync():
    problem, _ = make_pair(torch.float64, True, device="cuda")
    geometry = FactoredFrameGeometry(problem.context.geometry.derivatives)
    point = problem.context.current_point
    d, vector = torch.randn_like(point) * 0.1, torch.randn_like(point)
    frame = geometry.frame(point, d)
    geometry.factor_local_derivatives(point)  # Cold capture outside audit.
    expected = (geometry.gram(frame).matrix @ vector.flatten()).reshape_as(vector)
    torch.cuda.set_sync_debug_mode("error")
    try:
        actual = geometry.gram_matvec(frame, vector, block_size=3)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.testing.assert_close(actual, expected)
