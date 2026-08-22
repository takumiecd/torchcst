"""The materialized backend: build W per forward, lean on cuBLAS/cuDNN.

W is a compute intermediate, not state: rebuilt every forward (the atoms
move every optimizer step), freed after backward, never owning gradient
buffers or optimizer moments.  What persists is the atoms.

Two build modes per map family:

* the plain autograd product (:func:`linear_weight`, or the module's own
  einsum for conv) -- fine until the retained ``[features, K, d]`` kernel
  broadcasts hurt;
* a ``Lean*`` autograd Function -- closed-form chunked backward saving
  only the atom parameters, peak O(chunk) at any K.  Gaussian only.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..._geometry import squared_distance_matrix, squared_norm_last


def gaussian_columns(mu: Tensor, coords: Tensor, sigma: Tensor) -> Tensor:
    """``exp(-|mu_f - coord_k|^2 / 2 sigma^2)`` as ``[features, K]``."""
    return torch.exp(
        -squared_distance_matrix(mu, coords) / (2.0 * sigma.square())
    )


def _gaussian_pieces(mu: Tensor, coords: Tensor, sigma: Tensor):
    """(values, mu - coord, squared distance) for analytic derivatives."""
    diff = mu[:, None, :] - coords[None]
    d2 = squared_norm_last(diff)
    return torch.exp(-d2 / (2.0 * sigma.square())), diff, d2


def normalized_gaussian_columns(mu: Tensor, coords: Tensor, sigma: Tensor) -> Tensor:
    """Stable unit-L2 Gaussian columns as ``[features, K]``.

    Subtracting the nearest squared distance only rescales a column.  The
    following normalisation removes that scale exactly while keeping a value
    of one in every non-empty column, so remote atoms cannot underflow before
    they are normalised.
    """
    d2 = squared_distance_matrix(mu, coords)
    values = torch.exp(
        -(d2 - d2.amin(dim=0, keepdim=True)) / (2.0 * sigma.square())
    )
    return values / torch.linalg.vector_norm(values, dim=0, keepdim=True)


def _normalized_gaussian_pieces(mu: Tensor, coords: Tensor, sigma: Tensor):
    """(unit columns, mu - coord, centered squared distance).

    Centering ``d2`` is a per-column scale choice.  Its derivative lies in the
    radial column direction and is therefore deleted by the unit-sphere
    tangent projection in :func:`_normalized_column_backward`.
    """
    diff = mu[:, None, :] - coords[None]
    d2 = squared_norm_last(diff)
    centered = d2 - d2.amin(dim=0, keepdim=True)
    values = torch.exp(-centered / (2.0 * sigma.square()))
    unit = values / torch.linalg.vector_norm(values, dim=0, keepdim=True)
    return unit, diff, centered


def _normalized_column_backward(
    unit: Tensor,
    grad_unit: Tensor,
    diff: Tensor,
    centered_d2: Tensor,
    sigma: Tensor,
) -> tuple[Tensor, Tensor]:
    """Atom, chart and bandwidth gradients of unit Gaussian columns.

    For ``u = k / ||k||``, the differential is
    ``du = (I - uu.T) dk / ||k||``.  Multiplying the projected gradient by
    ``dk = k dlog(k)`` cancels ``||k||`` and leaves the stable score below;
    the unnormalised, potentially underflowing Gaussian is never needed.

    The same projection is what lets ``centered_d2``'s own dependence on the
    chart be dropped: subtracting a per-column minimum rescales the column,
    and a radial change is exactly what ``(I - uu.T)`` deletes -- for the
    chart's points as much as for the atoms'.
    """
    tangent = grad_unit - unit * (unit * grad_unit).sum(dim=0, keepdim=True)
    score = unit * tangent
    weighted = score[:, :, None] * diff
    # ``diff`` is ``mu - coord``, so the two endpoints of one distance differ
    # only in which axis is summed and in sign: the atom gathers over the
    # chart's points, a chart point gathers over the atoms.  The expensive
    # part is already built, so carrying the chart's gradient costs one more
    # reduction of a tensor that had to exist anyway.
    grad_coord = weighted.sum(0) / sigma.square()
    grad_mu = -weighted.sum(1) / sigma.square()
    grad_sigma = (score * centered_d2).sum() / sigma.pow(3)
    return grad_coord, grad_mu, grad_sigma


_compiled_normalized_gaussian_pieces = None
_compiled_normalized_column_backward = None
_compiled_normalized_gaussian_columns = None


def _l2_forward_columns(
    compiled: bool, mu: Tensor, coords: Tensor, sigma: Tensor
) -> Tensor:
    """Build normalized columns eagerly or with a lazily fused CUDA graph."""
    if not compiled or not coords.is_cuda:
        return normalized_gaussian_columns(mu, coords, sigma)
    global _compiled_normalized_gaussian_columns
    if _compiled_normalized_gaussian_columns is None:
        _compiled_normalized_gaussian_columns = torch.compile(
            normalized_gaussian_columns, fullgraph=True
        )
    return _compiled_normalized_gaussian_columns(mu, coords, sigma)


def _l2_backward_helpers(compiled: bool, source: Tensor):
    """Return eager or lazily Inductor-fused L2 backward primitives."""
    if not compiled or not source.is_cuda:
        return _normalized_gaussian_pieces, _normalized_column_backward
    global _compiled_normalized_gaussian_pieces
    global _compiled_normalized_column_backward
    if _compiled_normalized_gaussian_pieces is None:
        _compiled_normalized_gaussian_pieces = torch.compile(
            _normalized_gaussian_pieces, fullgraph=True
        )
        _compiled_normalized_column_backward = torch.compile(
            _normalized_column_backward, fullgraph=True
        )
    return (
        _compiled_normalized_gaussian_pieces,
        _compiled_normalized_column_backward,
    )


def linear_weight(
    k_in: Tensor,
    k_out: Tensor,
    weights: Tensor,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """``(k_out * w) @ k_in.T`` with optional reduced-precision contraction."""
    scaled = k_out * weights
    if compute_dtype is not None and weights.dtype != compute_dtype:
        return (
            scaled.to(compute_dtype)
            @ k_in.to(compute_dtype).transpose(0, 1)
        ).to(weights.dtype)
    return scaled @ k_in.transpose(0, 1)


def reject_learnable_chart(*charts: Tensor) -> None:
    """Refuse a learnable ``mu`` the lean backward cannot differentiate.

    The lean materializations hand back ``None`` for the chart coordinates,
    which autograd reads as "no gradient" rather than as an error: a chart
    made learnable here would train nothing and say nothing about it.  A
    learnable chart belongs on the plain autograd build (``lean=False``)
    until the closed forms carry ``mu`` too.
    """
    if any(chart is not None and chart.requires_grad for chart in charts):
        raise NotImplementedError(
            "a learnable neuron chart needs the autograd build: pass "
            "Materialized(lean=False), whose backward differentiates mu"
        )


class LeanLinearMaterialize(torch.autograd.Function):
    """Atoms -> dense ``[out, in]`` weight with closed-form chunked backward.

    Saves only the atom parameters and recomputes per-chunk in backward
    with analytic Gaussian derivatives; peak memory is O(chunk)
    regardless of K.  Gaussian kernels only; the caller enforces it.
    """

    CHUNK = 4096

    @staticmethod
    def forward(ctx, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, compute_dtype):
        reject_learnable_chart(mu_in, mu_out)
        n_out, n_in = mu_out.shape[0], mu_in.shape[0]
        weight = source.new_zeros(n_out, n_in)
        with torch.no_grad():
            for start in range(0, source.shape[0], LeanLinearMaterialize.CHUNK):
                sl = slice(start, start + LeanLinearMaterialize.CHUNK)
                ki = gaussian_columns(mu_in, source[sl], sigma_in)
                ko = gaussian_columns(mu_out, target[sl], sigma_out)
                weight += linear_weight(ki, ko, weights[sl], compute_dtype)
        ctx.save_for_backward(source, target, weights, mu_in, mu_out,
                              sigma_in, sigma_out)
        return weight

    @staticmethod
    def backward(ctx, grad_out):
        source, target, weights, mu_in, mu_out, sigma_in, sigma_out = (
            ctx.saved_tensors
        )
        g_source = torch.zeros_like(source)
        g_target = torch.zeros_like(target)
        g_w = torch.zeros_like(weights)
        g_sig_in = torch.zeros_like(sigma_in)
        g_sig_out = torch.zeros_like(sigma_out)
        for start in range(0, source.shape[0], LeanLinearMaterialize.CHUNK):
            sl = slice(start, start + LeanLinearMaterialize.CHUNK)
            w = weights[sl]
            ki, diff_in, d2_in = _gaussian_pieces(mu_in, source[sl], sigma_in)
            ko, diff_out, d2_out = _gaussian_pieces(
                mu_out, target[sl], sigma_out
            )
            m_in = grad_out.transpose(0, 1) @ ko                # [I, k]
            m_out = grad_out @ ki                               # [O, k]
            g_w[sl] = (m_in * ki).sum(0)
            dko = m_out * w[None]
            dki = m_in * w[None]
            g_target[sl] = (
                (dko * ko)[:, :, None] * diff_out
            ).sum(0) / sigma_out.square()
            g_source[sl] = (
                (dki * ki)[:, :, None] * diff_in
            ).sum(0) / sigma_in.square()
            g_sig_out = g_sig_out + (dko * ko * d2_out).sum() / sigma_out.pow(3)
            g_sig_in = g_sig_in + (dki * ki * d2_in).sum() / sigma_in.pow(3)
        return (g_source, g_target, g_w, None, None, g_sig_in, g_sig_out, None)


class LeanL2LinearMaterialize(torch.autograd.Function):
    """Exact chunked materialisation for unit-L2 Gaussian columns.

    Like :class:`LeanLinearMaterialize`, this builds the transient dense
    weight but retains only atom parameters.  Backward recomputes one chunk of
    normalised columns and applies the unit-sphere tangent projection, avoiding
    the full autograd path's ``[features, K, d]`` broadcasts.
    """

    CHUNK = 4096
    FULL_FORWARD_COLUMN_LIMIT = 32 * 1024 * 1024

    @staticmethod
    def forward(ctx, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, compute_dtype, compile_backward):
        n_out, n_in = mu_out.shape[0], mu_in.shape[0]
        weight = source.new_zeros(n_out, n_in)
        with torch.no_grad():
            column_elements = source.shape[0] * (n_in + n_out)
            if column_elements <= LeanL2LinearMaterialize.FULL_FORWARD_COLUMN_LIMIT:
                ki = _l2_forward_columns(
                    compile_backward, mu_in, source, sigma_in
                )
                ko = _l2_forward_columns(
                    compile_backward, mu_out, target, sigma_out
                )
                weight = linear_weight(ki, ko, weights, compute_dtype)
            else:
                for start in range(
                    0, source.shape[0], LeanL2LinearMaterialize.CHUNK
                ):
                    sl = slice(start, start + LeanL2LinearMaterialize.CHUNK)
                    ki = _l2_forward_columns(
                        compile_backward, mu_in, source[sl], sigma_in
                    )
                    ko = _l2_forward_columns(
                        compile_backward, mu_out, target[sl], sigma_out
                    )
                    weight += linear_weight(ki, ko, weights[sl], compute_dtype)
        ctx.save_for_backward(source, target, weights, mu_in, mu_out,
                              sigma_in, sigma_out)
        ctx.compile_backward = bool(compile_backward)
        return weight

    @staticmethod
    def backward(ctx, grad_out):
        source, target, weights, mu_in, mu_out, sigma_in, sigma_out = (
            ctx.saved_tensors
        )
        g_source = torch.zeros_like(source)
        g_target = torch.zeros_like(target)
        g_w = torch.zeros_like(weights)
        g_sig_in = torch.zeros_like(sigma_in)
        g_sig_out = torch.zeros_like(sigma_out)
        # Every chunk of atoms touches every chart point, so the chart
        # gradients accumulate across the loop rather than being written per
        # slice the way the atoms' are.
        learn_charts = mu_in.requires_grad or mu_out.requires_grad
        g_mu_in = torch.zeros_like(mu_in) if learn_charts else None
        g_mu_out = torch.zeros_like(mu_out) if learn_charts else None
        pieces, column_backward = _l2_backward_helpers(
            ctx.compile_backward, source
        )
        for start in range(0, source.shape[0], LeanL2LinearMaterialize.CHUNK):
            sl = slice(start, start + LeanL2LinearMaterialize.CHUNK)
            w = weights[sl]
            ki, diff_in, d2_in = pieces(
                mu_in, source[sl], sigma_in
            )
            ko, diff_out, d2_out = pieces(
                mu_out, target[sl], sigma_out
            )
            m_in = grad_out.transpose(0, 1) @ ko
            m_out = grad_out @ ki
            g_w[sl] = (m_in * ki).sum(0)
            g_source[sl], chunk_mu_in, chunk_sig_in = column_backward(
                ki, m_in * w[None], diff_in, d2_in, sigma_in
            )
            g_target[sl], chunk_mu_out, chunk_sig_out = column_backward(
                ko, m_out * w[None], diff_out, d2_out, sigma_out
            )
            g_sig_in = g_sig_in + chunk_sig_in
            g_sig_out = g_sig_out + chunk_sig_out
            if learn_charts:
                g_mu_in = g_mu_in + chunk_mu_in
                g_mu_out = g_mu_out + chunk_mu_out
        return (
            g_source, g_target, g_w,
            g_mu_in if mu_in.requires_grad else None,
            g_mu_out if mu_out.requires_grad else None,
            g_sig_in, g_sig_out, None, None,
        )


def bilinear_pieces(delta: Tensor, lo, hi, r: int):
    """Shared stencil decomposition: cells, fractional weights, box mask."""
    dy_raw, dx_raw = delta[:, 0], delta[:, 1]
    dy = dy_raw.clamp(lo[0], hi[0])
    dx = dx_raw.clamp(lo[1], hi[1])
    in_y = ((dy_raw >= lo[0]) & (dy_raw <= hi[0])).to(delta.dtype)
    in_x = ((dx_raw >= lo[1]) & (dx_raw <= hi[1])).to(delta.dtype)
    iy_f = dy.detach().floor().clamp(-r, r - 1)
    ix_f = dx.detach().floor().clamp(-r, r - 1)
    ay, ax = dy - iy_f, dx - ix_f
    span = 2 * r + 1
    iy, ix = iy_f.long(), ix_f.long()
    rows = torch.stack((iy, iy, iy + 1, iy + 1), dim=1) + r
    cols = torch.stack((ix, ix + 1, ix, ix + 1), dim=1) + r
    cells = rows * span + cols
    one = torch.ones_like(ay)
    vals = torch.stack(
        ((one - ay) * (one - ax), (one - ay) * ax, ay * (one - ax), ay * ax),
        dim=1,
    )
    return cells, vals, ay, ax, in_y, in_x, span


class LeanConvMaterialize(torch.autograd.Function):
    """Atoms -> dense conv kernel with closed-form, chunked backward.

    The conv analogue of :class:`LeanLinearMaterialize`: the source's
    trailing displacement axes expand to 4-cell bilinear stencils, and
    the accumulated kernel is ``[out, in, span, span]``.  Peak memory is
    O(chunk) regardless of K.  Gaussian kernels only.
    """

    CHUNK = 2048

    @staticmethod
    def forward(ctx, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, chart_d, r_int, off_lo, off_hi, compute_dtype):
        reject_learnable_chart(mu_in, mu_out)
        span = 2 * r_int + 1
        n_out, n_in = mu_out.shape[0], mu_in.shape[0]
        weight = source.new_zeros(n_out, n_in, span * span)
        with torch.no_grad():
            for start in range(0, source.shape[0], LeanConvMaterialize.CHUNK):
                sl = slice(start, start + LeanConvMaterialize.CHUNK)
                ki = gaussian_columns(mu_in, source[sl, :chart_d], sigma_in)
                ko = gaussian_columns(mu_out, target[sl], sigma_out)
                cells, vals, *_ = bilinear_pieces(
                    source[sl, chart_d:], off_lo, off_hi, r_int
                )
                stencil = source.new_zeros(
                    ki.shape[1], span * span
                ).scatter(1, cells, vals)
                scaled = (ko * weights[sl])[:, None, :] * (
                    stencil.transpose(0, 1)[None]
                )
                if compute_dtype is not None:
                    weight += torch.einsum(
                        "osk,ck->ocs",
                        scaled.to(compute_dtype),
                        ki.to(compute_dtype),
                    ).to(weight.dtype)
                else:
                    weight += torch.einsum("osk,ck->ocs", scaled, ki)
        ctx.save_for_backward(source, target, weights, mu_in, mu_out,
                              sigma_in, sigma_out)
        ctx.meta = (chart_d, r_int, off_lo, off_hi)
        return weight.reshape(n_out, n_in, span, span)

    @staticmethod
    def backward(ctx, grad_out):
        source, target, weights, mu_in, mu_out, sigma_in, sigma_out = (
            ctx.saved_tensors
        )
        chart_d, r_int, off_lo, off_hi = ctx.meta
        span = 2 * r_int + 1
        grads = grad_out.reshape(grad_out.shape[0], grad_out.shape[1], -1)
        g_source = torch.zeros_like(source)
        g_target = torch.zeros_like(target)
        g_w = torch.zeros_like(weights)
        g_sig_in = torch.zeros_like(sigma_in)
        g_sig_out = torch.zeros_like(sigma_out)
        for start in range(0, source.shape[0], LeanConvMaterialize.CHUNK):
            sl = slice(start, start + LeanConvMaterialize.CHUNK)
            w = weights[sl]
            ki, diff_in, d2_in = _gaussian_pieces(
                mu_in, source[sl, :chart_d], sigma_in
            )
            ko, diff_out, d2_out = _gaussian_pieces(
                mu_out, target[sl], sigma_out
            )
            cells, vals, ay, ax, in_y, in_x, _ = bilinear_pieces(
                source[sl, chart_d:], off_lo, off_hi, r_int
            )
            stencil = source.new_zeros(
                ki.shape[1], span * span
            ).scatter(1, cells, vals)
            m_stage = torch.einsum("ocs,ck->osk", grads, ki)   # [O, S, k]
            n_stage = torch.einsum("ocs,ok->csk", grads, ko)   # [C, S, k]
            a_stage = torch.einsum("osk,ok->sk", m_stage, ko)  # [S, k]
            g_w[sl] = torch.einsum("sk,ks->k", a_stage, stencil)
            dko = torch.einsum("osk,ks->ok", m_stage, stencil) * w[None]
            dki = torch.einsum("csk,ks->ck", n_stage, stencil) * w[None]
            g_target[sl] = (
                (dko * ko)[:, :, None] * diff_out
            ).sum(0) / sigma_out.square()
            g_source[sl, :chart_d] = (
                (dki * ki)[:, :, None] * diff_in
            ).sum(0) / sigma_in.square()
            g_sig_out = g_sig_out + (
                (dko * ko * d2_out).sum() / sigma_out.pow(3)
            )
            g_sig_in = g_sig_in + (
                (dki * ki * d2_in).sum() / sigma_in.pow(3)
            )
            d_cells = (a_stage * w[None]).transpose(0, 1).gather(1, cells)
            one = torch.ones_like(ay)
            dv_day = torch.stack((-(one - ax), -ax, one - ax, ax), dim=1)
            dv_dax = torch.stack((-(one - ay), one - ay, -ay, ay), dim=1)
            g_source[sl, chart_d] = (d_cells * dv_day).sum(1) * in_y
            g_source[sl, chart_d + 1] = (d_cells * dv_dax).sum(1) * in_x
        return (g_source, g_target, g_w, None, None, g_sig_in, g_sig_out,
                None, None, None, None, None)
