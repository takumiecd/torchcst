"""General CST linear map over continuous coordinates."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.representation import ContinuousKernel
from torchcst.storage import NeuronStore, SynapseStore

from .capture import register_capture_hook
from .cst_map import _ContinuousCSTMap


class _LeanLinearMaterialize(torch.autograd.Function):
    """Atoms -> dense weight with closed-form, chunked backward.

    The default autograd materialization retains every kernel-evaluation
    broadcast ``[features, K, d]`` for backward -- gigabytes across sites
    at large K.  This Function saves only the atom parameters and
    recomputes per-chunk in backward, with analytic Gaussian derivatives.
    Peak memory is O(chunk) regardless of K.

    Restriction: Gaussian kernels only (the closed-form derivative used
    here); the caller enforces it.
    """

    CHUNK = 4096

    @staticmethod
    def forward(ctx, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, compute_dtype):
        n_out, n_in = mu_out.shape[0], mu_in.shape[0]
        weight = source.new_zeros(n_out, n_in)
        with torch.no_grad():
            for start in range(0, source.shape[0], _LeanLinearMaterialize.CHUNK):
                sl = slice(start, start + _LeanLinearMaterialize.CHUNK)
                ki = torch.exp(
                    -(mu_in[:, None, :] - source[sl][None]).square().sum(-1)
                    / (2.0 * sigma_in.square())
                )
                ko = torch.exp(
                    -(mu_out[:, None, :] - target[sl][None]).square().sum(-1)
                    / (2.0 * sigma_out.square())
                )
                scaled = ko * weights[sl]
                if compute_dtype is not None:
                    weight += (
                        scaled.to(compute_dtype)
                        @ ki.to(compute_dtype).transpose(0, 1)
                    ).to(weight.dtype)
                else:
                    weight += scaled @ ki.transpose(0, 1)
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
        for start in range(0, source.shape[0], _LeanLinearMaterialize.CHUNK):
            sl = slice(start, start + _LeanLinearMaterialize.CHUNK)
            w = weights[sl]
            diff_in = mu_in[:, None, :] - source[sl][None]      # [I, k, ds]
            diff_out = mu_out[:, None, :] - target[sl][None]    # [O, k, dt]
            d2_in = diff_in.square().sum(-1)
            d2_out = diff_out.square().sum(-1)
            ki = torch.exp(-d2_in / (2.0 * sigma_in.square()))
            ko = torch.exp(-d2_out / (2.0 * sigma_out.square()))
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


class _NativeTruncated(torch.autograd.Function):
    """Row map through truncated kernels, no W and no ``[rows, K]``.

    Each atom reads only the ``m_in`` input neurons and writes only the
    ``m_out`` output neurons inside its ``support_radius``-sigma ball, so
    the map costs ``rows x K x (m_in + m_out)`` FLOPs -- below a dense
    GEMM whenever the coordinate domain is wide relative to sigma (the
    lawful 20-30 sigma regime).  Backward saves only the atom parameters
    and neighbor tables and recomputes per-chunk with analytic Gaussian
    derivatives; peak memory is O(rows x chunk x m).

    This is the semantics oracle and working fallback for the fused GPU
    kernel: a Triton implementation replaces the gathered intermediates
    with register tiles but must match this Function bit-for-bit in
    double precision.  Gaussian kernels only; the caller enforces it.
    """

    CHUNK = 32

    @staticmethod
    def forward(ctx, x, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, idx_in, pad_in, idx_out, pad_out):
        rows = x.shape[0]
        m_in = idx_in.shape[1]
        y = x.new_zeros(rows, mu_out.shape[0])
        with torch.no_grad():
            for start in range(0, source.shape[0], _NativeTruncated.CHUNK):
                sl = slice(start, start + _NativeTruncated.CHUNK)
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
        for start in range(0, source.shape[0], _NativeTruncated.CHUNK):
            sl = slice(start, start + _NativeTruncated.CHUNK)
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


class CSTLinear(_ContinuousCSTMap):
    """Apply a continuous CST measure to feature rows.

    Neuron coordinates are fixed floating buffers. Synapse source and target
    coordinates, atom weights, and global kernel bandwidths remain learnable.

    A CST layer is a composition, not a primitive: neurons come first
    (:meth:`torchcst.storage.NeuronStore.propose` for hidden populations, or
    data-supplied coordinates for pinned ones), synapses derive from the
    populations they connect (:meth:`torchcst.storage.SynapseStore.between`),
    and this module merely applies the composed site.

    Two execution paths deliver the same map:

    * **factored** -- ``((x @ k_in) * w) @ k_out.T``.  Per row it costs
      ``K (d_in + d_out)`` FLOPs, and autograd retains a ``[rows, K]``
      intermediate for backward.
    * **materialized** -- build ``W = (k_out * w) @ k_in.T`` once per
      forward and apply one GEMM.  Past the FLOP crossover
      ``K > d_in d_out / (d_in + d_out)`` this is both faster and lighter:
      no ``[rows, K]`` tensor ever exists, so activation memory matches a
      dense linear.

    * **native** -- ``support_radius=R`` (in sigma units) truncates each
      kernel at ``R sigma`` and routes rows through the gather-scatter
      :class:`_NativeTruncated` Function: no W is ever built and no
      ``[rows, K]`` exists.  Per row it costs ``K (m_in + m_out)`` where
      ``m`` is the neighbor count inside the ball -- in the lawful
      20-30 sigma coordinate domain that undercuts the dense GEMM itself,
      which neither other path can.  Truncation is an evaluation policy of
      the same represented map (per-entry error ``exp(-R^2/2)`` relative
      to the atom's peak), and coordinates stopped at the support edge
      keep retraction semantics: zero position gradient outside, exactly
      like the displacement-box clamp on :class:`OffsetCSTConv2d`.

    ``materialize="auto"`` (default) picks the factored/materialized
    crossover per forward from the live atom count; ``True``/``False``
    pin one; ``support_radius`` overrides both (and requires the default
    ``"auto"`` so the surface states one intent).  Large-K options travel
    from :class:`OffsetCSTConv2d`: ``lean_materialize=True`` swaps the
    materialization's autograd for a closed-form chunked backward
    (Gaussian kernels only, and ``track_mass=False`` because mass's kernel
    matrices would resurrect the memory this mode removes) -- the native
    path shares both restrictions; ``compute_dtype`` runs the
    materialization contraction in reduced precision while parameters,
    GEMM, and gradients stay in the parameter dtype.

    The neighbor tables are rebuilt per forward under ``no_grad`` (a
    ``K x features`` distance test, one materialization chunk's worth of
    work) and their padded width follows the widest atom, which is a host
    sync -- a CUDA-graph-ready variant pins the width and rebuilds on a
    cadence instead; that belongs to the fused-kernel arc.
    """

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel: ContinuousKernel,
        kernel_out: ContinuousKernel | None = None,
        *,
        track_mass: bool = True,
        materialize: bool | str = "auto",
        compute_dtype: torch.dtype | None = None,
        lean_materialize: bool = False,
        support_radius: float | None = None,
    ) -> None:
        super().__init__(
            in_neurons, out_neurons, synapses, kernel, kernel_out,
            track_mass=track_mass,
        )
        if materialize is not True and materialize is not False and (
            materialize != "auto"
        ):
            raise ValueError('materialize must be True, False, or "auto"')
        if compute_dtype is not None and not compute_dtype.is_floating_point:
            raise TypeError("compute_dtype must be a floating dtype or None")
        if lean_materialize:
            if self.kernel_in.family != "gaussian" or (
                self.kernel_out.family != "gaussian"
            ):
                raise ValueError("lean_materialize requires Gaussian kernels")
            if track_mass:
                raise ValueError("lean_materialize requires track_mass=False")
            if materialize is False:
                raise ValueError(
                    "lean_materialize requires the materialized path"
                )
        if support_radius is not None:
            if isinstance(support_radius, bool) or not isinstance(
                support_radius, (int, float)
            ):
                raise TypeError("support_radius must be a number or None")
            if not support_radius > 0.0:
                raise ValueError("support_radius must be positive")
            if materialize != "auto":
                raise ValueError(
                    "support_radius overrides path selection; leave "
                    'materialize="auto"'
                )
            if self.kernel_in.family != "gaussian" or (
                self.kernel_out.family != "gaussian"
            ):
                raise ValueError("support_radius requires Gaussian kernels")
            if track_mass:
                raise ValueError("support_radius requires track_mass=False")
        self.materialize = materialize
        self.compute_dtype = compute_dtype
        self.lean_materialize = lean_materialize
        self.support_radius = (
            float(support_radius) if support_radius is not None else None
        )

    def _neighbor_tables(
        self, source: Tensor, target: Tensor, mu_in: Tensor, mu_out: Tensor,
        sigma_in: Tensor, sigma_out: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Padded per-atom neuron indices inside the support ball.

        Built from detached coordinates: the support boundary is not
        differentiated (retraction semantics), only the kernel values on
        the surviving pairs are.
        """
        radius = self.support_radius
        assert radius is not None
        with torch.no_grad():
            tables = []
            for coords, mu, sigma in (
                (source, mu_in, sigma_in), (target, mu_out, sigma_out)
            ):
                d2 = (coords[:, None, :] - mu[None]).square().sum(-1)
                mask = d2 <= (radius * sigma).square()
                width = max(int(mask.sum(dim=1).max()), 1)
                order = mask.to(torch.int8).argsort(
                    dim=1, descending=True, stable=True
                )
                idx = order[:, :width]
                tables.append(idx)
                tables.append(torch.gather(mask, 1, idx).to(coords.dtype))
        return tables[0], tables[1], tables[2], tables[3]

    def _forward_native(self, x: Tensor) -> Tensor:
        source, target, weights = self._live_factors()
        source = source.to(device=x.device)
        target = target.to(device=x.device)
        weights = weights.to(device=x.device)
        mu_in = self.in_neurons.mu.to(source)
        mu_out = self.out_neurons.mu.to(target)
        sigma_in = self.kernel_in.sigma.to(source)
        sigma_out = self.kernel_out.sigma.to(target)
        idx_in, pad_in, idx_out, pad_out = self._neighbor_tables(
            source, target, mu_in, mu_out, sigma_in, sigma_out
        )
        rows = x.reshape(-1, self.in_features)
        output = _NativeTruncated.apply(
            rows, source, target, weights, mu_in, mu_out, sigma_in,
            sigma_out, idx_in, pad_in, idx_out, pad_out,
        )
        return output.reshape(*x.shape[:-1], self.out_features)

    def _materialize_now(self, count: int) -> bool:
        """The FLOP-crossover rule; ``True``/``False`` pins override it."""
        if self.materialize == "auto":
            return count * (self.in_features + self.out_features) > (
                self.in_features * self.out_features
            )
        return bool(self.materialize)

    def dense_weight(self) -> Tensor:
        """Materialize ``K_out diag(w) K_in.T`` -- the materialized path's W."""
        self._view()
        source, target, weights = self._live_factors()
        if self.lean_materialize:
            return _LeanLinearMaterialize.apply(
                source, target, weights,
                self.in_neurons.mu.to(source),
                self.out_neurons.mu.to(target),
                self.kernel_in.sigma.to(source),
                self.kernel_out.sigma.to(target),
                self.compute_dtype,
            )
        k_in, k_out = self._kernel_matrices(source, target)
        self._refresh_mass_scale(k_in, k_out)
        scaled = k_out * weights
        if self.compute_dtype is not None and weights.dtype != self.compute_dtype:
            return (
                scaled.to(self.compute_dtype)
                @ k_in.to(self.compute_dtype).transpose(0, 1)
            ).to(weights.dtype)
        return scaled @ k_in.transpose(0, 1)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError("x's final dimension must equal the input neuron width")
        view = self._view()
        count = int(self._cached_slots.shape[0])
        if self.support_radius is not None and count > 0:
            output = self._forward_native(x)
            if self._backward_context is not None:
                register_capture_hook(
                    output, self._backward_context, self.capture_site, x,
                    view.version,
                )
            return output
        if not self._materialize_now(count):
            return self._forward_rows(x)
        weight = self.dense_weight().to(dtype=x.dtype, device=x.device)
        output = F.linear(x, weight)
        if self._backward_context is not None:
            register_capture_hook(
                output, self._backward_context, self.capture_site, x, view.version
            )
        return output
