"""Streamed current/previous tangent products, including multiple visible tiles."""

import pytest
import torch

from tests.test_tangent_ops import jacobian
from torchcst import Chart, CSTLinear, FirstOrderAdamConfig
from torchcst.kernels import AmplitudeBandwidthSeparable
from torchcst.optim.moments import SeparableDiagonalMetric


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_streamed_cross_dense_oracle(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(21)
    site = CSTLinear(
        Chart.linspace(9, low=-1.0, high=1.0),
        Chart.linspace(19, low=-1.0, high=1.0),
        atoms=7,
        kernel=AmplitudeBandwidthSeparable(
            sigma_min=0.6, sigma_max=0.8
        ),
        dtype=dtype,
    ).to(device)
    ops = site.cst_derivatives().tangent_ops(
        gram_action="jvp_vjp", execution="triton" if device == "cuda" else "eager"
    )
    p = site.atoms.p.detach().clone()
    p[:, 0] = torch.linspace(-0.01, 0.01, 7, device=device, dtype=dtype)
    old = p + 0.01 * torch.randn_like(p)
    current, previous = ops.prepare(p), ops.prepare(old)
    j, jo = jacobian(site, p), jacobian(site, old)
    x = torch.randn_like(p)
    metric = SeparableDiagonalMetric(
        torch.rand(19, device=device, dtype=dtype),
        torch.rand(9, device=device, dtype=dtype),
        eps=0.12,
    )
    tol = 2e-5 if dtype == torch.float32 else 1e-10
    for weights in [None, metric]:
        d = 1 if weights is None else weights.diagonal().flatten()
        expected = j.T @ (d * (jo @ x.flatten()))
        torch.testing.assert_close(
            current.cross_gram_matvec(previous, x, metric=weights).flatten(),
            expected,
            rtol=tol,
            atol=tol,
        )
    expected = j.T @ j @ x.flatten() + 0.1 * x.flatten()
    torch.testing.assert_close(
        current._device_gram(x, damping=0.1).flatten(), expected, rtol=tol, atol=tol
    )
    inactive = torch.tensor(False, device=device)
    assert current._device_gram(x, damping=0.1, active=inactive).count_nonzero() == 0


@pytest.mark.parametrize(
    "options", [{"recompression_action": "bad"}, {"recompression_action": "jvp_vjp"}]
)
def test_invalid_stream_configuration(options):
    with pytest.raises(ValueError):
        FirstOrderAdamConfig(**options)
