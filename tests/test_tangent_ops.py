"""Dense autograd oracles independent of the prepared factor implementation."""

import copy

import pytest
import torch

from torchcst import Chart, CSTLinear
from torchcst.kernels import Amplitude, AmplitudeBandwidthSeparable, Gaussian, Separable
from torchcst.optim.moments import SeparableDiagonalMetric


def site_for(kind, dtype=torch.float64):
    base = Separable(input_profile=Gaussian(0.8), output_profile=Gaussian(0.6))
    kernel = base if kind == "plain" else Amplitude(base)
    if kind == "bandwidth":
        kernel = AmplitudeBandwidthSeparable(
            sigma_min=0.6, sigma_max=0.8
        )
    torch.manual_seed(17)
    site = CSTLinear(
        Chart.grid((2, 3)), Chart.linspace(4), atoms=5, kernel=kernel, dtype=dtype
    )
    with torch.no_grad():
        site.atoms.p.add_(0.15 * torch.randn_like(site.atoms.p))
        if kind != "plain":
            site.atoms.p[:, 0].copy_(
                torch.tensor([0.0, 1e-10, -0.2, 0.5, -1.0], dtype=dtype)
            )
        if kind == "bandwidth":
            site.atoms.p[:, 0].copy_(
                torch.tensor([0.0, 1e-10, -0.003, 0.005, -0.02], dtype=dtype)
            )
    return site


def jacobian(site, point):
    return torch.func.jacfwd(lambda p: site._materialize_atoms(p).sum(0))(
        point
    ).reshape(-1, point.numel())


@pytest.mark.parametrize("kind", ["plain", "amplitude", "bandwidth"])
@pytest.mark.parametrize("backend", ["specialized", "factor_autograd", "reference"])
@pytest.mark.parametrize("tile", [1, 3, 32])
def test_actions_match_autograd(kind, backend, tile):
    site = site_for(kind)
    point = site.atoms.p.detach().clone()
    old = point + 0.13 * torch.randn_like(point)
    ops = site.cst_derivatives().tangent_ops(backend=backend, atom_tile=tile)
    current, previous = ops.prepare(point), ops.prepare(old)
    j, js = jacobian(site, point), jacobian(site, old)
    x, y = torch.randn_like(point), torch.randn_like(point)
    force = torch.randn(4, 6, dtype=point.dtype)
    metric = SeparableDiagonalMetric(
        torch.rand(4, dtype=point.dtype), torch.rand(6, dtype=point.dtype), eps=0.12
    )
    torch.testing.assert_close(current.jvp(x).flatten(), j @ x.flatten())
    torch.testing.assert_close(current.vjp(force).flatten(), j.T @ force.flatten())
    torch.testing.assert_close(current.gram_matvec(x).flatten(), j.T @ j @ x.flatten())
    cross = current.cross_gram_matvec(previous, x)
    torch.testing.assert_close(cross.flatten(), j.T @ js @ x.flatten())
    torch.testing.assert_close(
        (y * cross).sum(), (x * previous.cross_gram_matvec(current, y)).sum()
    )
    expected = j.T @ (metric.diagonal().flatten()[:, None] * j)
    torch.testing.assert_close(
        current.weighted_gram_matvec(metric, x).flatten(), expected @ x.flatten()
    )
    expected_cross = j.T @ (metric.diagonal().flatten()[:, None] * js)
    torch.testing.assert_close(
        current.cross_gram_matvec(previous, x, metric=metric).flatten(),
        expected_cross @ x.flatten(),
    )
    gram = (j.T @ j).reshape(5, point.shape[1], 5, point.shape[1])
    torch.testing.assert_close(
        current.gram_blocks(), torch.stack([gram[k, :, k, :] for k in range(5)])
    )
    snapshot = current.gram_matvec(x).clone()
    point.add_(10)
    current.point.zero_()
    torch.testing.assert_close(current.gram_matvec(x), snapshot, rtol=0, atol=0)
    assert not current.jvp(x).requires_grad


def test_descriptor_compatibility_and_fixed_config_guard():
    site = site_for("bandwidth")
    ops = site.cst_derivatives().tangent_ops()
    p = site.atoms.p
    prepared = ops.prepare(p)
    with pytest.raises(ValueError, match="dtype/device"):
        ops.prepare(p.float())
    clone = copy.deepcopy(site)
    same = clone.cst_derivatives().tangent_ops().prepare(p)
    torch.testing.assert_close(
        prepared.cross_gram_matvec(same, p), prepared.gram_matvec(p)
    )
    clone.input_chart.coordinates.add_(0.1)
    changed = clone.cst_derivatives().tangent_ops().prepare(p)
    with pytest.raises(ValueError, match="incompatible"):
        prepared.cross_gram_matvec(changed, p)
    site.kernel.profile.sigma.add_(0.1)
    with pytest.raises(ValueError, match="configuration changed"):
        ops.prepare(p)


def test_fallback_and_fast_actions_do_not_call_dense_or_autograd(monkeypatch):
    site = site_for("amplitude")
    derivatives = site.cst_derivatives()
    derivatives._tangent_backend = None
    assert derivatives.tangent_ops().backend == "factor_autograd"
    with pytest.raises(ValueError, match="specialized"):
        derivatives.tangent_ops(backend="specialized")
    derivatives.factor_atoms = None
    assert derivatives.tangent_ops().backend == "reference"
    ops = site.cst_derivatives().tangent_ops(atom_tile=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("dense/autograd path called")

    for name in ("jacfwd", "jacrev", "jvp", "vjp", "hessian"):
        monkeypatch.setattr(torch.func, name, forbidden)
    monkeypatch.setattr(site, "_materialize_atoms", forbidden)
    a = ops.prepare(site.atoms.p)
    monkeypatch.setattr(a, "jvp", forbidden)
    monkeypatch.setattr(a, "vjp", forbidden)
    assert torch.isfinite(a.gram_matvec(torch.ones_like(site.atoms.p))).all()


def test_fast_gram_scratch_is_bounded_by_tile():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    site = site_for("amplitude")
    frame = site.cst_derivatives().tangent_ops(atom_tile=2).prepare(site.atoms.p)
    forbidden_shapes = {(20, 20), (5, 5), (4, 6)}

    class AllocationGuard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    assert tuple(tensor.shape) not in forbidden_shapes, (
                        func,
                        tensor.shape,
                    )
            return result

    metric = SeparableDiagonalMetric(
        torch.ones(4, dtype=torch.float64), torch.ones(6, dtype=torch.float64), eps=0.1
    )
    x = torch.ones_like(site.atoms.p)
    with AllocationGuard():
        frame.gram_matvec(x)
        frame.weighted_gram_matvec(metric, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", ["specialized", "factor_autograd", "reference"])
def test_cuda_weighted_transport_and_compression(dtype, backend):
    from torchcst._derivatives.tangent_solve import solve_compression

    site = site_for("bandwidth", dtype).to("cuda")
    p = site.atoms.p.detach().clone()
    old = p + 0.015 * torch.randn_like(p)
    ops = site.cst_derivatives().tangent_ops(backend=backend, atom_tile=3)
    current, previous = ops.prepare(p), ops.prepare(old)
    j, js = jacobian(site, p), jacobian(site, old)
    x = torch.randn_like(p)
    metric = SeparableDiagonalMetric(
        torch.rand(4, device=p.device, dtype=dtype),
        torch.rand(6, device=p.device, dtype=dtype),
        eps=0.1,
    )
    tolerance = 3e-5 if dtype == torch.float32 else 1e-10
    expected = j.T @ (metric.diagonal().flatten()[:, None] * js) @ x.flatten()
    torch.testing.assert_close(
        current.cross_gram_matvec(previous, x, metric=metric).flatten(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    alpha, result = solve_compression(
        current, x, damping=0.1, rtol=tolerance, max_iter=128
    )
    direct = torch.linalg.solve(
        j.T @ j + 0.1 * torch.eye(p.numel(), device=p.device, dtype=dtype), x.flatten()
    )
    torch.testing.assert_close(
        alpha.flatten(), direct, rtol=10 * tolerance, atol=10 * tolerance
    )
    assert result.converged
