"""The native truncated backend: no W, no ``[rows, K]``, local kernels.

Each atom reads only the input neurons and writes only the output
neurons inside its ``radius``-sigma ball, so the map costs
``rows x K x (m_in + m_out)`` FLOPs -- below a dense GEMM whenever the
coordinate domain is wide relative to sigma (the lawful 20-30 sigma
regime).  This PyTorch implementation is the semantics oracle and
working fallback for the fused GPU kernel: a Triton implementation
replaces the gathered intermediates with register tiles but must match
this Function bit-for-bit in double precision.
"""

from __future__ import annotations

import torch
from torch import Tensor


def neighbor_tables(
    coords: Tensor, mu: Tensor, sigma: Tensor, radius: float
) -> tuple[Tensor, Tensor]:
    """Padded ``[K, m]`` neuron indices and their validity mask.

    Built from detached coordinates: the support boundary is not
    differentiated (retraction semantics) -- only kernel values on the
    surviving pairs are.  The padded width follows the widest atom,
    which is a host sync; a CUDA-graph-ready variant pins the width and
    rebuilds on a cadence (Verlet-style) instead.
    """
    with torch.no_grad():
        d2 = (coords[:, None, :] - mu[None]).square().sum(-1)
        mask = d2 <= (radius * sigma).square()
        width = max(int(mask.sum(dim=1).max()), 1)
        order = mask.to(torch.int8).argsort(
            dim=1, descending=True, stable=True
        )
        idx = order[:, :width]
        return idx, torch.gather(mask, 1, idx).to(coords.dtype)


class NativeTruncatedFunction(torch.autograd.Function):
    """Row map through truncated kernels with closed-form chunked backward.

    Backward saves only the atom parameters and neighbor tables and
    recomputes per-chunk with analytic Gaussian derivatives; peak memory
    is O(rows x chunk x m).  Gaussian kernels only; the caller enforces
    it.
    """

    CHUNK = 32

    @staticmethod
    def forward(ctx, x, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, idx_in, pad_in, idx_out, pad_out):
        rows = x.shape[0]
        m_in = idx_in.shape[1]
        y = x.new_zeros(rows, mu_out.shape[0])
        with torch.no_grad():
            for start in range(0, source.shape[0], NativeTruncatedFunction.CHUNK):
                sl = slice(start, start + NativeTruncatedFunction.CHUNK)
                kc = source[sl].shape[0]
                ki = torch.exp(
                    -(mu_in[idx_in[sl]] - source[sl][:, None, :])
                    .square().sum(-1) / (2.0 * sigma_in.square())
                ) * pad_in[sl]
                ko = torch.exp(
                    -(mu_out[idx_out[sl]] - target[sl][:, None, :])
                    .square().sum(-1) / (2.0 * sigma_out.square())
                ) * pad_out[sl]
                xg = x[:, idx_in[sl].reshape(-1)].reshape(rows, kc, m_in)
                u = torch.einsum("nkm,km->nk", xg, ki) * weights[sl]
                contrib = torch.einsum("nk,km->nkm", u, ko)
                y.index_add_(
                    1, idx_out[sl].reshape(-1), contrib.reshape(rows, -1)
                )
        ctx.save_for_backward(x, source, target, weights, mu_in, mu_out,
                              sigma_in, sigma_out, idx_in, pad_in, idx_out,
                              pad_out)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (x, source, target, weights, mu_in, mu_out, sigma_in, sigma_out,
         idx_in, pad_in, idx_out, pad_out) = ctx.saved_tensors
        rows = x.shape[0]
        m_in = idx_in.shape[1]
        m_out = idx_out.shape[1]
        g_x = torch.zeros_like(x)
        g_source = torch.zeros_like(source)
        g_target = torch.zeros_like(target)
        g_w = torch.zeros_like(weights)
        g_sig_in = torch.zeros_like(sigma_in)
        g_sig_out = torch.zeros_like(sigma_out)
        for start in range(0, source.shape[0], NativeTruncatedFunction.CHUNK):
            sl = slice(start, start + NativeTruncatedFunction.CHUNK)
            kc = source[sl].shape[0]
            w = weights[sl]
            diff_in = mu_in[idx_in[sl]] - source[sl][:, None, :]  # [k, m, ds]
            diff_out = mu_out[idx_out[sl]] - target[sl][:, None, :]
            d2_in = diff_in.square().sum(-1)
            d2_out = diff_out.square().sum(-1)
            ki = torch.exp(-d2_in / (2.0 * sigma_in.square())) * pad_in[sl]
            ko = torch.exp(-d2_out / (2.0 * sigma_out.square())) * pad_out[sl]
            xg = x[:, idx_in[sl].reshape(-1)].reshape(rows, kc, m_in)
            go = grad_out[:, idx_out[sl].reshape(-1)].reshape(rows, kc, m_out)
            u0 = torch.einsum("nkm,km->nk", xg, ki)   # pre-weight reads
            v = torch.einsum("nkm,km->nk", go, ko)    # downstream response
            g_w[sl] = torch.einsum("nk,nk->k", u0, v)
            vw = v * w
            g_ki = torch.einsum("nk,nkm->km", vw, xg)
            g_ko = torch.einsum("nk,nkm->km", u0 * w, go)
            # ki carries the pad mask, so padded lanes contribute zero here.
            g_source[sl] = (
                (g_ki * ki)[:, :, None] * diff_in
            ).sum(1) / sigma_in.square()
            g_target[sl] = (
                (g_ko * ko)[:, :, None] * diff_out
            ).sum(1) / sigma_out.square()
            g_sig_in = g_sig_in + (g_ki * ki * d2_in).sum() / sigma_in.pow(3)
            g_sig_out = g_sig_out + (
                (g_ko * ko * d2_out).sum() / sigma_out.pow(3)
            )
            gx_part = torch.einsum("nk,km->nkm", vw, ki)
            g_x.index_add_(
                1, idx_in[sl].reshape(-1), gx_part.reshape(rows, -1)
            )
        return (g_x, g_source, g_target, g_w, None, None, g_sig_in,
                g_sig_out, None, None, None, None)
