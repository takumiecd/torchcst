"""Unfactored CST convolution: channel charts times continuous displacement.

The FC-9 arc's measured form (reports in the sibling ``cst`` repository,
``scripts/fc9/``).  One atom carries a source coordinate on the *product*
domain ``input-channel chart × displacement box``, a target coordinate on the
output-channel chart, and an amplitude:

    atom = (s_chart ∈ R^d, Δ ∈ R², t ∈ R^d', w)

and the represented map is

    y[o](p) = Σ_a w_a · K(mu_out[o], t_a) · Σ_c K(mu_in[c], s_a) · x[c](p + Δ_a)

Two evaluation rules coexist on one coordinate vector.  The chart axes are
matched by the ordinary continuous kernel.  The displacement axes are *not*
given a chart or a kernel — the conv v1 lesson: a 3×3 tap lattice is ~5σ
quasi-discrete and a kernel-side coordinate moves through data-free vacuum.
Instead ``Δ`` acts on the data side as a bilinear read position, which is a
triangular kernel of exactly one pixel bandwidth evaluated on the *image's*
pixel lattice, so a dense position gradient exists by construction.  The
separable factorization of :class:`DepthwiseCSTConv2d` is not imposed; the
FC-9 measurements have the unfactored form matching it at parity everywhere,
scaling monotonically on the atom ladder, and beating it under churn.

Forward is the measured fast path: the atoms are assembled into the
equivalent dense kernel ``W[o, c, ky, kx]`` (each Δ expands to a 4-cell
bilinear stencil; one scatter + one einsum, O(K·C_in·C_out)) and the heavy
work is one ``F.conv2d`` at plain-conv cost, independent of the atom count.
The dense tensor is a per-forward compute intermediate — the parameterization
stays atomic, and gradients reach ``s`` (chart and displacement axes), ``t``,
``w``, and the bandwidth.

Because the displacement axes live inside ``domain_in``, every structural
mechanism works unchanged: birth candidates sampled uniform-in-box carry
their own Δ, scored birth ranks them through this module's
:meth:`candidate_weight_grads`, and the box's retraction law governs drift.
The forward additionally clamps Δ into its declared box (subgradient zero
outside), which keeps the materialized support static: ``R = 2·ceil(r) + 1``
where ``r`` is the box half-extent, so no forward pays a host sync and the
conv geometry cannot silently grow between structural events.

Like :class:`CSTLinear` this map is purely synaptic and gate-free: a neuron
gate belongs to whichever :class:`CSTBoundary` owns that boundary.  Row-space
capabilities (``pre_gate_rows``, ``input_row_energy``) have no meaning for a
spatial map and raise; boundary composition over this module is unsupported,
matching :class:`CSTConv2d`'s "a conv is its own endpoint" rule.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.representation import (
    ContinuousKernel,
    GaussianKernel,
    RepresentationSpec,
    TriangularKernel,
)
from torchcst.storage import NeuronStore, SynapseStore

from .capture import register_capture_hook
from .cst_map import _ContinuousCSTMap
from .cst_conv import _positive_pair
from .depthwise_conv import _seed_uniform

__all__ = ["OffsetCSTConv2d"]

_KERNELS = {"gaussian": GaussianKernel, "triangular": TriangularKernel}


def _axis_bounds(domain, dim: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-axis (lo, hi) tuples from a Box whose bounds may be scalar."""
    lo, hi = domain.bounds
    lo_t = (float(lo),) * dim if not isinstance(lo, (tuple, list)) else tuple(lo)
    hi_t = (float(hi),) * dim if not isinstance(hi, (tuple, list)) else tuple(hi)
    return lo_t, hi_t


def _bilinear_pieces(delta: Tensor, lo, hi, r: int):
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


class _LeanMaterialize(torch.autograd.Function):
    """Atoms -> dense kernel with closed-form, chunked backward.

    The default autograd path retains every kernel-evaluation broadcast
    ``[channels, K, d]`` and einsum intermediate for backward -- gigabytes
    across sites at large K.  This Function saves only the atom parameters
    and recomputes per-chunk in backward, with analytic Gaussian and
    bilinear derivatives.  Peak memory is O(chunk) regardless of K.

    Restriction: Gaussian kernels only (the closed-form derivative used
    here); the caller enforces it.
    """

    CHUNK = 2048

    @staticmethod
    def forward(ctx, source, target, weights, mu_in, mu_out, sigma_in,
                sigma_out, chart_d, r_int, off_lo, off_hi, compute_dtype):
        span = 2 * r_int + 1
        n_out, n_in = mu_out.shape[0], mu_in.shape[0]
        W = source.new_zeros(n_out, n_in, span * span)
        with torch.no_grad():
            for start in range(0, source.shape[0], _LeanMaterialize.CHUNK):
                sl = slice(start, start + _LeanMaterialize.CHUNK)
                s_chart = source[sl, :chart_d]
                ki = torch.exp(
                    -(mu_in[:, None, :] - s_chart[None]).square().sum(-1)
                    / (2.0 * sigma_in.square())
                )
                ko = torch.exp(
                    -(mu_out[:, None, :] - target[sl][None]).square().sum(-1)
                    / (2.0 * sigma_out.square())
                )
                cells, vals, *_ = _bilinear_pieces(
                    source[sl, chart_d:], off_lo, off_hi, r_int
                )
                P = source.new_zeros(s_chart.shape[0], span * span).scatter(
                    1, cells, vals
                )
                if compute_dtype is not None:
                    scaled = ((ko * weights[sl])[:, None, :]
                              * P.transpose(0, 1)[None]).to(compute_dtype)
                    W += torch.einsum(
                        "osk,ck->ocs", scaled, ki.to(compute_dtype)
                    ).to(W.dtype)
                else:
                    scaled = (ko * weights[sl])[:, None, :] * P.transpose(0, 1)[None]
                    W += torch.einsum("osk,ck->ocs", scaled, ki)
        ctx.save_for_backward(source, target, weights, mu_in, mu_out,
                              sigma_in, sigma_out)
        ctx.meta = (chart_d, r_int, off_lo, off_hi)
        return W.reshape(n_out, n_in, span, span)

    @staticmethod
    def backward(ctx, grad_out):
        source, target, weights, mu_in, mu_out, sigma_in, sigma_out = (
            ctx.saved_tensors
        )
        chart_d, r_int, off_lo, off_hi = ctx.meta
        span = 2 * r_int + 1
        G = grad_out.reshape(grad_out.shape[0], grad_out.shape[1], -1)
        g_source = torch.zeros_like(source)
        g_target = torch.zeros_like(target)
        g_w = torch.zeros_like(weights)
        g_sig_in = sigma_in.new_zeros(())
        g_sig_out = sigma_out.new_zeros(())
        for start in range(0, source.shape[0], _LeanMaterialize.CHUNK):
            sl = slice(start, start + _LeanMaterialize.CHUNK)
            s_chart = source[sl, :chart_d]
            w = weights[sl]
            diff_in = mu_in[:, None, :] - s_chart[None]        # [C, k, dc]
            diff_out = mu_out[:, None, :] - target[sl][None]   # [O, k, do]
            d2_in = diff_in.square().sum(-1)
            d2_out = diff_out.square().sum(-1)
            ki = torch.exp(-d2_in / (2.0 * sigma_in.square()))
            ko = torch.exp(-d2_out / (2.0 * sigma_out.square()))
            cells, vals, ay, ax, in_y, in_x, _ = _bilinear_pieces(
                source[sl, chart_d:], off_lo, off_hi, r_int
            )
            P = source.new_zeros(s_chart.shape[0], span * span).scatter(
                1, cells, vals
            )
            M = torch.einsum("ocs,ck->osk", G, ki)   # [O, S, k]
            N = torch.einsum("ocs,ok->csk", G, ko)   # [C, S, k]
            A = torch.einsum("osk,ok->sk", M, ko)    # [S, k]
            g_w[sl] = torch.einsum("sk,ks->k", A, P)
            dko = torch.einsum("osk,ks->ok", M, P) * w[None]
            dki = torch.einsum("csk,ks->ck", N, P) * w[None]
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
            dP_cells = (A * w[None]).transpose(0, 1).gather(1, cells)  # [k, 4]
            one = torch.ones_like(ay)
            dv_day = torch.stack((-(one - ax), -ax, one - ax, ax), dim=1)
            dv_dax = torch.stack((-(one - ay), one - ay, -ay, ay), dim=1)
            g_source[sl, chart_d] = (dP_cells * dv_day).sum(1) * in_y
            g_source[sl, chart_d + 1] = (dP_cells * dv_dax).sum(1) * in_x
        return (g_source, g_target, g_w, None, None, g_sig_in, g_sig_out,
                None, None, None, None, None)


class OffsetCSTConv2d(_ContinuousCSTMap):
    """Continuous CST conv whose atoms own a spatial displacement.

    Composition-first: the caller builds the channel populations and a
    synapse store on the product domain (:meth:`displacement_store`), and
    this module applies the composed site.  :meth:`propose` is the one-call
    lawful construction.  The displacement axes are the trailing two axes of
    the source coordinate, ordered ``(Δy, Δx)`` to match the dense kernel's
    ``(ky, kx)`` index order.
    """

    input_offset_axes = 2

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel: ContinuousKernel,
        *,
        kernel_out: ContinuousKernel | None = None,
        stride: int | tuple[int, int] = 1,
        track_mass: bool = True,
        compute_dtype: torch.dtype | None = None,
        lean_materialize: bool = False,
    ) -> None:
        super().__init__(
            in_neurons, out_neurons, synapses, kernel, kernel_out,
            track_mass=track_mass,
        )
        self.chart_d_in = synapses.d_in - self.input_offset_axes
        self.stride = _positive_pair(stride, "stride")
        lo, hi = _axis_bounds(synapses.spec.domain_in, synapses.d_in)
        off_lo, off_hi = lo[self.chart_d_in:], hi[self.chart_d_in:]
        if not all(low < 0.0 < high for low, high in zip(off_lo, off_hi)):
            raise ValueError(
                "displacement axes must straddle zero: an atom must be able "
                "to read at its own output position"
            )
        self._off_lo = off_lo
        self._off_hi = off_hi
        radius = max(max(-low for low in off_lo), max(high for high in off_hi))
        # Static support: Δ is clamped into its box in the forward, so the
        # materialized kernel is always (2·ceil(r)+1)² and no forward pays a
        # host sync to size it.
        self._r_int = max(1, int(math.ceil(radius)))
        self.in_channels = self.in_features
        self.out_channels = self.out_features
        # Optional reduced-precision materialization: the kernel-column /
        # stencil contraction runs in this dtype (tensor cores accumulate in
        # fp32, so W keeps ~fp32 fidelity); parameters, conv, and gradients
        # stay in the parameter dtype. None = full precision (default).
        if compute_dtype is not None and not compute_dtype.is_floating_point:
            raise TypeError("compute_dtype must be a floating dtype or None")
        self.compute_dtype = compute_dtype
        # Large-K mode: closed-form chunked backward that saves only the atom
        # parameters -- the default autograd path retains [channels, K, d]
        # kernel broadcasts and einsum intermediates (gigabytes at K ~ 10^5).
        # Gaussian-only (analytic derivative), and mass tracking is skipped
        # (its kernel matrices would resurrect the memory this mode removes).
        if lean_materialize:
            if self.kernel_in.family != "gaussian" or (
                self.kernel_out.family != "gaussian"
            ):
                raise ValueError("lean_materialize requires Gaussian kernels")
            if track_mass:
                raise ValueError(
                    "lean_materialize requires track_mass=False"
                )
        self.lean_materialize = lean_materialize

    # -- construction ----------------------------------------------------------

    @staticmethod
    def displacement_store(
        site: str,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        sigma: float,
        kernel_size: int | tuple[int, int],
        *,
        kernel: str = "gaussian",
        capacity: int | None = None,
        max_capacity: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SynapseStore:
        """A synapse store on ``input chart × displacement box``.

        The chart axes and the default capacity follow
        :meth:`SynapseStore.between` exactly (one atom per resolvable cell of
        the larger endpoint chart — displacement axes carry no kernel, so
        they add no cells).  The displacement box is ``[-k/2, k/2]`` per
        spatial axis in ``(Δy, Δx)`` order: the atom analogue of a k×k
        receptive field.
        """
        size = _positive_pair(kernel_size, "kernel_size")
        base = SynapseStore.between(
            site, in_neurons, out_neurons, sigma, kernel=kernel
        )
        lo, hi = _axis_bounds(base.spec.domain_in, base.d_in)
        lo_out, hi_out = _axis_bounds(base.spec.domain_out, base.d_out)
        half = (size[0] / 2.0, size[1] / 2.0)
        return SynapseStore(
            site,
            d_in=base.d_in + 2,
            d_out=base.d_out,
            capacity=capacity if capacity is not None else base.capacity,
            max_capacity=max_capacity,
            spec=RepresentationSpec.continuous(
                base.d_in + 2,
                base.d_out,
                bounds=(lo + tuple(-h for h in half), hi + half),
                bounds_out=(lo_out, hi_out),
                kernel=kernel,
            ),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def propose(
        cls,
        site: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        sigma: float,
        *,
        capacity_scale: float = 2.0,
        seed_atoms: bool = True,
        seed_weight_scale: float = 0.05,
        kernel: str = "gaussian",
        stride: int | tuple[int, int] = 1,
        generator: torch.Generator | None = None,
    ) -> "OffsetCSTConv2d":
        """Build the map on lawful charts, atoms seeded uniform-in-box.

        Conventions travel from :meth:`DepthwiseCSTConv2d.propose`: both
        channel populations from :meth:`NeuronStore.propose`, capacity
        ``capacity_scale ×`` the resolvable cells of the larger endpoint
        chart (default the measured 2×), seeds uniform over the *product*
        domain — so every seeded atom is born with its own displacement.
        The stem rule also travels: do not chart a population that carries
        data geometry (e.g. the RGB input).
        """
        if kernel not in _KERNELS:
            raise ValueError(
                f"kernel must be one of {sorted(_KERNELS)}, got {kernel!r}"
            )
        if not capacity_scale > 0.0:
            raise ValueError(
                f"capacity_scale must be positive, got {capacity_scale}"
            )
        rng = generator if generator is not None else torch.Generator()
        inputs = NeuronStore.propose(f"{site}.in", in_channels, sigma, generator=rng)
        outputs = NeuronStore.propose(
            f"{site}.out", out_channels, sigma, generator=rng
        )
        base = cls.displacement_store(
            site, inputs, outputs, sigma, kernel_size, kernel=kernel
        )
        synapses = cls.displacement_store(
            site,
            inputs,
            outputs,
            sigma,
            kernel_size,
            kernel=kernel,
            capacity=max(1, int(round(capacity_scale * base.capacity))),
        )
        if seed_atoms:
            _seed_uniform(synapses, synapses.capacity, seed_weight_scale, rng)
        return cls(
            inputs, outputs, synapses, _KERNELS[kernel](sigma), stride=stride
        )

    # -- representation --------------------------------------------------------

    def _kernel_matrices(self, source: Tensor, target: Tensor):
        """Kernel columns over the chart axes only; Δ axes act via stencils."""
        in_mu = self.in_neurons.mu.to(device=source.device, dtype=source.dtype)
        out_mu = self.out_neurons.mu.to(device=target.device, dtype=target.dtype)
        return (
            self.kernel_in(in_mu, source[:, : self.chart_d_in]),
            self.kernel_out(out_mu, target),
        )

    def _stencil(self, source: Tensor) -> Tensor:
        """Dense [N, R²] bilinear stencils of the source rows' Δ axes.

        Δ is clamped into its declared box (retraction semantics: zero
        subgradient outside), then split into an integer cell and a
        differentiable fractional weight over the 4 surrounding cells.  The
        4 cells of one atom are always distinct, so the scatter is
        collision-free and exact under autograd.
        """
        delta = source[:, self.chart_d_in:]
        dy = delta[:, 0].clamp(self._off_lo[0], self._off_hi[0])
        dx = delta[:, 1].clamp(self._off_lo[1], self._off_hi[1])
        r = self._r_int
        # The corner base cell must satisfy base+1 <= r. A box with integer
        # extent puts clamped displacements exactly on r (floor(r) + 1 would
        # leave the grid), so cap the base at r-1: the boundary atom then
        # carries fractional weight 1.0 on its far corner -- same value,
        # in bounds, gradient unchanged.
        iy_f = dy.detach().floor().clamp(-r, r - 1)
        ix_f = dx.detach().floor().clamp(-r, r - 1)
        ay, ax = dy - iy_f, dx - ix_f
        iy, ix = iy_f.long(), ix_f.long()
        span = 2 * r + 1
        rows = torch.stack((iy, iy, iy + 1, iy + 1), dim=1) + r
        cols = torch.stack((ix, ix + 1, ix, ix + 1), dim=1) + r
        one = torch.ones_like(ay)
        vals = torch.stack(
            ((one - ay) * (one - ax), (one - ay) * ax, ay * (one - ax), ay * ax),
            dim=1,
        )
        return torch.zeros(
            source.shape[0], span * span, dtype=source.dtype, device=source.device
        ).scatter(1, rows * span + cols, vals)

    def dense_weight(self) -> Tensor:
        """Materialize the equivalent dense kernel ``[C_out, C_in, R, R]``.

        Gate-free like every synaptic map: gates belong to boundaries.  This
        is the exact filter the forward convolves with, and the only place
        the dense tensor exists — a compute intermediate, not state.
        """
        self._view()
        source, target, weights = self._live_factors()
        if self.lean_materialize:
            return _LeanMaterialize.apply(
                source, target, weights,
                self.in_neurons.mu.to(source),
                self.out_neurons.mu.to(target),
                self.kernel_in.sigma.to(source),
                self.kernel_out.sigma.to(target),
                self.chart_d_in, self._r_int, self._off_lo, self._off_hi,
                self.compute_dtype,
            )
        k_in, k_out = self._kernel_matrices(source, target)
        stencil = self._stencil(source)
        self._refresh_mass_scale(
            k_in * torch.linalg.vector_norm(stencil, dim=1).detach(), k_out
        )
        span = 2 * self._r_int + 1
        # Contraction order matters at large K: the single three-factor einsum
        # is free to materialize an [out, in, K] intermediate (gigabytes at
        # K ~ 7000 on wide layers -- the measured fc10a OOM). Staging through
        # [out, span^2, K] keeps the peak at span^2/in_features of that.
        scaled = (k_out * weights)[:, None, :] * stencil.transpose(0, 1)[None]
        if self.compute_dtype is not None and weights.dtype != self.compute_dtype:
            weight = torch.einsum(
                "osk,ck->ocs",
                scaled.to(self.compute_dtype),
                k_in.to(self.compute_dtype),
            ).to(weights.dtype)
        else:
            weight = torch.einsum("osk,ck->ocs", scaled, k_in)
        return weight.reshape(self.out_features, self.in_features, span, span)

    # -- forward ---------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"x must have shape (batch, {self.in_channels}, height, width)"
            )
        view = self._view()
        weight = self.dense_weight().to(dtype=x.dtype, device=x.device)
        output = F.conv2d(x, weight, stride=self.stride, padding=self._r_int)
        if self._backward_context is not None:
            register_capture_hook(
                output, self._backward_context, self.capture_site, x, view.version
            )
        return output

    # -- capture consumers -----------------------------------------------------

    def _captured_weight_grad(self, x: Tensor, g_out: Tensor) -> Tensor:
        """``dL/dW`` of the materialized kernel from one captured pair."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("captured x must be a (batch, C_in, H, W) tensor")
        if g_out.ndim != 4 or g_out.shape[1] != self.out_channels:
            raise ValueError("captured g_out must be a (batch, C_out, H, W) tensor")
        span = 2 * self._r_int + 1
        return torch.nn.grad.conv2d_weight(
            x.detach(),
            (self.out_channels, self.in_channels, span, span),
            g_out.detach(),
            stride=self.stride,
            padding=(self._r_int, self._r_int),
        )

    def _weight_grad_inner(
        self, grad_w: Tensor, source: Tensor, target: Tensor
    ) -> Tensor:
        """Signed ``⟨dL/dW, atom direction⟩`` for each (source, target) row."""
        k_in, k_out = self._kernel_matrices(source, target)
        stencil = self._stencil(source)
        flat = grad_w.reshape(grad_w.shape[0], grad_w.shape[1], -1)
        partial = torch.einsum("ocs,cn->osn", flat, k_in)
        return torch.einsum("osn,on,ns->n", partial, k_out, stencil)

    def atom_grads(self, x: Tensor, g_out: Tensor) -> Tensor:
        """Return the signed update contribution for every live atom weight."""
        self._view()
        source, target, _ = self._live_factors()
        with torch.no_grad():
            grad_w = self._captured_weight_grad(x, g_out)
            return self._weight_grad_inner(
                grad_w,
                source.detach().to(grad_w),
                target.detach().to(grad_w),
            )

    def candidate_weight_grads(
        self,
        x: Tensor,
        g_out: Tensor,
        source: Tensor,
        target: Tensor,
        *,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Score zero-weight candidates; their Δ rides in the source rows."""
        if source.ndim != 2 or source.shape[1] != self.synapses.d_in:
            raise ValueError("source candidates have the wrong coordinate shape")
        if target.ndim != 2 or target.shape[1] != self.synapses.d_out:
            raise ValueError("target candidates have the wrong coordinate shape")
        if source.shape[0] != target.shape[0]:
            raise ValueError("source and target candidate counts must match")
        count = source.shape[0]
        if chunk_size is None:
            chunk_size = max(count, 1)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an int or None")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        with torch.no_grad():
            grad_w = self._captured_weight_grad(x, g_out)
            if count == 0:
                return self.synapses.w.detach().new_zeros(0).to(grad_w)
            source = source.detach().to(grad_w)
            target = target.detach().to(grad_w)
            values = [
                self._weight_grad_inner(
                    grad_w, source[start : start + chunk_size],
                    target[start : start + chunk_size],
                )
                for start in range(0, count, chunk_size)
            ]
        return torch.cat(values)

    # -- row-space capabilities do not exist on a spatial map ------------------

    def pre_gate_rows(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "OffsetCSTConv2d has no row boundary: a conv is its own endpoint "
            "and CSTBoundary composition over it is unsupported"
        )

    def input_row_energy(self) -> Tensor:
        raise NotImplementedError(
            "OffsetCSTConv2d has no row boundary; interface courts that need "
            "input_row_energy cannot sit at a spatial map's input"
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"support={2 * self._r_int + 1}, stride={self.stride}, "
            f"chart_d_in={self.chart_d_in}"
        )
