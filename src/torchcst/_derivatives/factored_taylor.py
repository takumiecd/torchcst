"""Exact Taylor contractions through small factor derivatives, never visible H."""

from functools import cache

import torch
from torch.func import jacfwd, vmap

from torchcst._runtime.graphs import CapturedCall


def factor_derivatives(factor_atoms, point):
    def one(p):
        u, v = factor_atoms(p.unsqueeze(0))
        return u[:, 0], v[:, 0]

    u, v = vmap(one)(point)
    ju, jv = vmap(jacfwd(one))(point)
    hu, hv = vmap(jacfwd(jacfwd(one)))(point)
    return u, v, ju, jv, hu, hv


def tangent(f, d):
    u, v, ju, jv, _, _ = f
    ud = torch.einsum("kip,kp->ki", ju, d)
    vd = torch.einsum("kop,kp->ko", jv, d)
    return vd.T @ u + v.T @ ud


def second(f, x, y):
    u, v, ju, jv, hu, hv = f
    ux = torch.einsum("kip,kp->ki", ju, x)
    uy = torch.einsum("kip,kp->ki", ju, y)
    vx = torch.einsum("kop,kp->ko", jv, x)
    vy = torch.einsum("kop,kp->ko", jv, y)
    uxy = torch.einsum("kipq,kp,kq->ki", hu, x, y)
    vxy = torch.einsum("kopq,kp,kq->ko", hv, x, y)
    return vxy.T @ u + v.T @ uxy + vx.T @ uy + vy.T @ ux


def displacement(f, d):
    return tangent(f, d) + 0.5 * second(f, d, d)


def pullback(f, force, d):
    u, v, ju, jv, hu, hv = f
    ud = torch.einsum("kip,kp->ki", ju, d)
    vd = torch.einsum("kop,kp->ko", jv, d)
    hud = torch.einsum("kipq,kq->kip", hu, d)
    hvd = torch.einsum("kopq,kq->kop", hv, d)
    return (
        torch.einsum("kip,ik->kp", ju, force.T @ (v + vd).T)
        + torch.einsum("kop,ok->kp", jv, force @ (u + ud).T)
        + torch.einsum("kip,ik->kp", hud, force.T @ v.T)
        + torch.einsum("kop,ok->kp", hvd, force @ u.T)
    )


def affine_pullback(f, force):
    u, v, ju, jv, hu, hv = f
    fu = (force @ u.T).T
    fv = (force.T @ v.T).T
    constant = torch.einsum("kip,ki->kp", ju, fv) + torch.einsum("kop,ko->kp", jv, fu)
    projected = (
        (force.T @ jv.permute(1, 0, 2).flatten(1))
        .reshape(u.shape[1], u.shape[0], -1)
        .permute(1, 0, 2)
    )
    mixed = torch.einsum("kip,kiq->kpq", ju, projected)
    blocks = (
        torch.einsum("kipq,ki->kpq", hu, fv)
        + torch.einsum("kopq,ko->kpq", hv, fu)
        + mixed
        + mixed.transpose(-1, -2)
    )
    return constant, blocks


def frame_gram(f, d):
    u, v, ju, jv, hu, hv = f
    ud = torch.einsum("kip,kp->ki", ju, d)
    vd = torch.einsum("kop,kp->ko", jv, d)
    hud = torch.einsum("kipq,kq->kip", hu, d)
    hvd = torch.einsum("kopq,kq->kop", hv, d)
    p = d.shape[1]
    # Each frame column is a sum of four rank-one terms.
    us = (
        u[:, None, :].expand(-1, p, -1),
        (ju + hud).transpose(1, 2),
        ud[:, None, :].expand(-1, p, -1),
        ju.transpose(1, 2),
    )
    vs = (
        (jv + hvd).transpose(1, 2),
        v[:, None, :].expand(-1, p, -1),
        jv.transpose(1, 2),
        vd[:, None, :].expand(-1, p, -1),
    )
    us = tuple(t.flatten(0, 1) for t in us)
    vs = tuple(t.flatten(0, 1) for t in vs)
    result = d.new_zeros(d.numel(), d.numel())
    for i in range(4):
        for j in range(4):
            result = result + (us[i] @ us[j].T) * (vs[i] @ vs[j].T)
    return 0.5 * (result + result.T)


def _frame_column_block(f, d, start, stop):
    """Four separated terms per column, generated only for this block."""
    u, v, ju, jv, hu, hv = f
    indices = torch.arange(start, stop, device=d.device)
    atoms, coordinates = indices // d.shape[1], indices % d.shape[1]
    ub, vb, db = u[atoms], v[atoms], d[atoms]
    jub, jvb = ju[atoms], jv[atoms]
    up = jub[torch.arange(stop - start, device=d.device), :, coordinates]
    vp = jvb[torch.arange(stop - start, device=d.device), :, coordinates]
    hud = (hu[atoms, :, coordinates, :] * db[:, None, :]).sum(-1)
    hvd = (hv[atoms, :, coordinates, :] * db[:, None, :]).sum(-1)
    ud = (jub * db[:, None, :]).sum(-1)
    vd = (jvb * db[:, None, :]).sum(-1)
    return (ub, up + hud, ud, up), (vp + hvd, vb, vp, vd)


def frame_gram_matvec(f, d, vector, *, block_size=32, damping=0.0):
    """Apply (B* B + damping I) without a visible matrix or full Gram.

    B is the displaced Taylor frame J + H[d, .]. All 16 cross terms
    are retained. With fixed P, extra eager inference workspace is
    O(block_size * (I + O) * P + block_size**2 + K*P), excluding the
    supplied factor derivatives. Arithmetic remains quadratic in K.
    This is an exact contraction, not a linear-system solver.
    """
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if vector.shape != d.shape:
        raise ValueError("vector must have the frame point shape")
    flat = vector.flatten()
    rows = []
    for start in range(0, d.numel(), block_size):
        stop = min(start + block_size, d.numel())
        left_u, left_v = _frame_column_block(f, d, start, stop)
        result = torch.zeros_like(flat[start:stop])
        for source in range(0, d.numel(), block_size):
            end = min(source + block_size, d.numel())
            right_u, right_v = _frame_column_block(f, d, source, end)
            for i in range(4):
                for j in range(4):
                    tile = (left_u[i] @ right_u[j].T) * (left_v[i] @ right_v[j].T)
                    result = result + tile @ flat[source:end]
        rows.append(result)
    return torch.cat(rows).reshape_as(vector) + damping * vector


def metric_diagonal(f, row, column, eps):
    u, v, ju, jv, _, _ = f

    def weighted(r, c):
        uu = (u.square() * c).sum(1)
        vv = (v.square() * r).sum(1)
        jj_u = (ju.square() * c[None, :, None]).sum(1)
        jj_v = (jv.square() * r[None, :, None]).sum(1)
        uj = (ju * (u * c)[:, :, None]).sum(1)
        vj = (jv * (v * r)[:, :, None]).sum(1)
        return uu[:, None] * jj_v + vv[:, None] * jj_u + 2 * uj * vj

    return weighted(row, column) + eps * weighted(
        torch.ones_like(row), torch.ones_like(column)
    )


@cache
def _runner(name):
    function = torch.compile(globals()[name], fullgraph=True, dynamic=False)
    return CapturedCall(function)


def _gram_flat(*args):
    return (frame_gram(args[:6], args[6]),)


def _transport_flat(*args):
    current, source = args[:6], args[6:12]
    d, alpha = args[12:]
    force = tangent(source, alpha) + second(source, d, alpha)
    return affine_pullback(current, force)


def call(name, *args):
    if args[0].device.type == "cuda":
        return _runner(name)(*args)
    return globals()[name](*args)
