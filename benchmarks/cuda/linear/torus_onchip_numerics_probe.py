"""Diagnostic output only; never called by a runtime execution path."""

import triton as tr
import triton.language as tl

from torchcst._backends.cuda.algorithms.linear.torus_profile_product.onchip.kernels import (
    _atom,
    _circle,
    _section,
)


@tr.jit
def probe(
    P,
    Prec,
    Maximum,
    Major,
    Minor,
    Circle,
    Sites,
    Centre,
    U,
    V,
    N: tl.constexpr,
    TRIG: tl.constexpr,
    T: tl.constexpr,
):
    (
        amp,
        inv,
        major,
        minor,
        arc,
        q0,
        q1,
        q2,
        _w1,
        _w2,
        rho2,
        sinc,
        _z0,
        _z1,
        _zr,
        _maximum,
        _r2,
    ) = _atom(P, Prec, Maximum, Major, Minor, TRIG)
    a, t = tl.program_id(0), tl.arange(0, T)
    tl.store(Centre + 8 * a, amp)
    tl.store(Centre + 8 * a + 1, inv)
    tl.store(Centre + 8 * a + 2, q0)
    tl.store(Centre + 8 * a + 3, q1)
    tl.store(Centre + 8 * a + 4, q2)
    tl.store(Centre + 8 * a + 5, rho2)
    tl.store(Centre + 8 * a + 6, sinc)
    tl.store(Centre + 8 * a + 7, major + minor * q0)
    for start in range(tl.cdiv(N, T)):
        i = start * T + t
        u, _da, _dq = _circle(Circle, i, N, arc, major, minor, q0, inv, TRIG)
        v, _d0, _d1, _d2 = _section(Sites, i, N, minor, q0, q1, q2, inv)
        tl.store(U + a * N + i, u, i < N)
        tl.store(V + a * N + i, v, i < N)
