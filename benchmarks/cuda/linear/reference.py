"""Independent normalized Triweight truth; contains no backend calls."""

import math

import torch


def oracle(p, sizes=(64, 4, 4), origin=(0.0, 0.0, 0.0), stored_dtype=None):
    # Lift stored FP32 clamp endpoints; high-coordinate CUDA tests use F64 truth.
    bounds = torch.tensor(
        [math.log(0.03), math.log(3.25)], dtype=stored_dtype or p.dtype
    ).to(p)
    sigma = p[:, 1].clamp(bounds[0], bounds[1]).exp()
    sites = torch.cartesian_prod(
        *[
            torch.arange(n, device=p.device, dtype=p.dtype) * s + o
            for n, s, o in zip(sizes, (1.0, 0.5, 0.5), origin)
        ]
    )
    delta = sites[None] - p[:, None, 2:]
    k = (1 - delta.square().sum(-1) / sigma[:, None].square()).clamp_min(0).pow(3)
    norm = torch.linalg.vector_norm(k, dim=1).clamp_min(
        float(torch.tensor(1e-6, dtype=stored_dtype or p.dtype))
    )
    return (p[:, 0, None] * k / norm[:, None]).sum(0).reshape(sizes[0], -1)


def mixed(dtype=torch.float64, device="cpu"):
    return torch.tensor(
        [
            [-0.3, math.log(3.0), 31.25, 0.63, 0.77],
            [0.2, math.log(3.25), 32.5, 0.53, 0.61],
            [0.1, math.log(0.03), 32.02978515625, 1.50355, 1.5],
            [0.0, math.log(0.03), 32.02978515625, 1.50355, 1.5],
            [-0.2, math.log(0.03), 16.0, 1.5, 1.5],
            [0.3, math.log(0.03), 7.5, 1.25, 1.25],
            [0.1, math.log(0.003), 8.0, 0.5, 0.5],
            [-0.1, math.log(10.0), 63.5, 0.5, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
