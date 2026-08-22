"""Adam moments measured in parameter or pullback-tangent coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor, nn

from .._validation import require_int
from ..representation import GaussianKernel, L2NormalizedColumns
from ..storage import SynapseStore
from . import metric
from .forces import PairRepulsion, SmoothRent

MomentSpace = Literal["parameter", "tangent"]
MetricForm = Literal["diag", "block"]


def _gaussian_unit_factor(
    mu: Tensor, centers: Tensor, sigma: Tensor
) -> tuple[Tensor, Tensor]:
    """Stable normalized Gaussian columns and their derivative factor."""
    diff = mu[:, None, :] - centers[None]
    squared_distance = diff.square().sum(-1)
    centered = squared_distance - squared_distance.amin(dim=0, keepdim=True)
    value = torch.exp(-centered / (2.0 * sigma.square()))
    unit = value / torch.linalg.vector_norm(value, dim=0, keepdim=True)
    return unit, -unit / sigma.square()


def _gaussian_l2_metric_block(
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    mass_sq: Tensor,
    sigma_in: Tensor,
    sigma_out: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pure-tensor Gaussian fast path for ``diag(J.T @ J)``."""
    unit_in, factor_in = _gaussian_unit_factor(mu_in, source, sigma_in)
    unit_out, factor_out = _gaussian_unit_factor(mu_out, target, sigma_out)
    return metric.gauged_jacobian_sq(
        unit_in=unit_in,
        factor_in=factor_in,
        unit_out=unit_out,
        factor_out=factor_out,
        mu_in=mu_in,
        mu_out=mu_out,
        source=source,
        target=target,
        mass_sq=mass_sq,
        per_axis=True,
    )


def _gaussian_l2_metric_gram(
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    mass_sq: Tensor,
    sigma_in: Tensor,
    sigma_out: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pure-tensor Gaussian fast path for the within-atom metric blocks."""
    unit_in, factor_in = _gaussian_unit_factor(mu_in, source, sigma_in)
    unit_out, factor_out = _gaussian_unit_factor(mu_out, target, sigma_out)
    return metric.gauged_jacobian_gram(
        unit_in=unit_in,
        factor_in=factor_in,
        unit_out=unit_out,
        factor_out=factor_out,
        mu_in=mu_in,
        mu_out=mu_out,
        source=source,
        target=target,
        mass_sq=mass_sq,
    )


def _inverse_sqrt(blocks: Tensor) -> Tensor:
    """Batched symmetric inverse square root of PSD ``[K, d, d]`` blocks."""
    eigenvalues, eigenvectors = torch.linalg.eigh(blocks)
    tiny = torch.finfo(blocks.dtype).tiny
    inverted = eigenvalues.clamp_min(tiny).rsqrt()
    return (eigenvectors * inverted[..., None, :]) @ eigenvectors.transpose(
        -1, -2
    )


_compiled_gaussian_l2_metric_block = None
_compiled_gaussian_l2_metric_gram = None


def _metric_block(
    module: nn.Module,
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    mass_sq: Tensor,
    *,
    compiled: bool,
) -> tuple[Tensor, Tensor]:
    """Evaluate one diagonal-metric block, optionally through ``compile``."""
    if (
        compiled
        and source.is_cuda
        and isinstance(module.kernel_in, GaussianKernel)
        and isinstance(module.kernel_out, GaussianKernel)
    ):
        global _compiled_gaussian_l2_metric_block
        if _compiled_gaussian_l2_metric_block is None:
            _compiled_gaussian_l2_metric_block = torch.compile(
                _gaussian_l2_metric_block, fullgraph=True
            )
        return _compiled_gaussian_l2_metric_block(
            mu_in,
            mu_out,
            source,
            target,
            mass_sq,
            module.kernel_in.sigma.detach().to(source),
            module.kernel_out.sigma.detach().to(target),
        )

    unit_in, factor_in = metric.normalized_columns(
        module.kernel_in, mu_in, source
    )
    unit_out, factor_out = metric.normalized_columns(
        module.kernel_out, mu_out, target
    )
    return metric.gauged_jacobian_sq(
        unit_in=unit_in,
        factor_in=factor_in,
        unit_out=unit_out,
        factor_out=factor_out,
        mu_in=mu_in,
        mu_out=mu_out,
        source=source,
        target=target,
        mass_sq=mass_sq,
        per_axis=True,
    )


def _metric_gram_block(
    module: nn.Module,
    mu_in: Tensor,
    mu_out: Tensor,
    source: Tensor,
    target: Tensor,
    mass_sq: Tensor,
    *,
    compiled: bool,
) -> tuple[Tensor, Tensor]:
    """Evaluate one within-atom metric-gram block, optionally compiled."""
    if (
        compiled
        and source.is_cuda
        and isinstance(module.kernel_in, GaussianKernel)
        and isinstance(module.kernel_out, GaussianKernel)
    ):
        global _compiled_gaussian_l2_metric_gram
        if _compiled_gaussian_l2_metric_gram is None:
            _compiled_gaussian_l2_metric_gram = torch.compile(
                _gaussian_l2_metric_gram, fullgraph=True
            )
        return _compiled_gaussian_l2_metric_gram(
            mu_in,
            mu_out,
            source,
            target,
            mass_sq,
            module.kernel_in.sigma.detach().to(source),
            module.kernel_out.sigma.detach().to(target),
        )

    unit_in, factor_in = metric.normalized_columns(
        module.kernel_in, mu_in, source
    )
    unit_out, factor_out = metric.normalized_columns(
        module.kernel_out, mu_out, target
    )
    return metric.gauged_jacobian_gram(
        unit_in=unit_in,
        factor_in=factor_in,
        unit_out=unit_out,
        factor_out=factor_out,
        mu_in=mu_in,
        mu_out=mu_out,
        source=source,
        target=target,
        mass_sq=mass_sq,
    )


class PullbackAdam:
    """Diagonal Adam for the learnable coordinates of one continuous CST site.

    ``moment_space`` names where Adam's exponential moving averages live:

    ``"parameter"``
        Accumulate the ordinary coordinate gradient ``g_theta``.  The Adam
        direction is mapped to a current unit-length tangent step with
        ``G_t**-1/2``.

    ``"tangent"``
        First form ``r_t = G_t**-1/2 g_theta`` and accumulate moments of that
        local orthonormal-tangent coefficient.  The resulting direction is
        mapped back to coordinates with the current ``G_t**-1/2``.  This is a
        moving-frame approximation: old moments are not parallel-transported
        when the tangent frame rotates.

    ``metric`` selects how much of the within-atom pullback metric ``G`` is:

    ``"diag"``
        The per-axis diagonal from
        :func:`torchcst.optim.metric.gauged_jacobian_sq`, exactly the
        historical behaviour.

    ``"block"``
        The full per-atom, per-side axis Gram ``R_ab = <d_a, d_b>`` from
        :func:`torchcst.optim.metric.gauged_jacobian_gram`.  Under L2 column
        normalisation the amplitude row and the source-target cross block are
        exactly zero, so these two small blocks are the complete within-atom
        metric; ``G**-1/2`` becomes a batched ``d x d`` symmetric inverse
        square root.

    Neither form is a claim that the full ``J.T @ J`` is diagonal: cross-atom
    coupling is always neglected.  Persistent state is coordinate-shaped;
    neither a dense represented weight nor a Jacobian-shaped moment is
    retained.

    The optional structural terms make this optimizer the site's single
    owner for the continuous lifecycle:

    ``rent=SmoothRent(...)`` (with ``lr_w``)
        The optimizer also owns ``synapses.w``.  The rent gradient joins the
        amplitude gradient *before* Adam's moments, so settlement is priced
        by carried map (``|dL/dw| > lam``); unprofitable amplitudes decay
        continuously and near-zero atoms keep a sign-coherent residue that
        turns their coordinate gradient into ascent of the squared
        birth-score field.  The outer optimizer must then exclude ``w`` too.

    ``repulsion=PairRepulsion(...)``
        The overlap penalty's gradient joins the coordinate gradients:
        atoms close on both sides repel, reserves spread over the chart,
        and column collisions are prevented instead of merged away.

    ``wall=True``
        After each step, coordinates are clamped into the store's declared
        coordinate domains, so wandering reserves stay on the chart.

    The owning optimizer must exclude ``module.synapses.s`` and ``.t`` and
    call :meth:`zero_grad` alongside its own ``zero_grad``.  Amplitudes and
    unrelated model parameters remain the owning optimizer's responsibility.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        moment_space: MomentSpace,
        metric: MetricForm = "diag",
        cap_sigma: float,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        damping: float = 1e-2,
        target_step: float = 0.01,
        rent: SmoothRent | None = None,
        lr_w: float | None = None,
        repulsion: PairRepulsion | None = None,
        wall: bool = False,
        seed: int = 0,
        subscribe: bool = True,
        chunk_elements: int = 1 << 24,
        compile_metric: bool = False,
    ) -> None:
        store = getattr(module, "synapses", None)
        kernel_in = getattr(module, "kernel_in", None)
        kernel_out = getattr(module, "kernel_out", None)
        if (
            not isinstance(store, SynapseStore)
            or kernel_in is None
            or kernel_out is None
        ):
            raise TypeError(
                "module must be a continuous CST map with .synapses and kernels"
            )
        if not isinstance(getattr(module, "gauge", None), L2NormalizedColumns):
            raise TypeError("PullbackAdam requires L2NormalizedColumns")
        if not isinstance(store.s, nn.Parameter) or not isinstance(
            store.t, nn.Parameter
        ):
            raise TypeError("PullbackAdam requires learnable coordinates")
        if moment_space not in ("parameter", "tangent"):
            raise ValueError(
                "moment_space must be 'parameter' or 'tangent'"
            )
        if metric not in ("diag", "block"):
            raise ValueError("metric must be 'diag' or 'block'")
        if (
            not isinstance(betas, tuple)
            or len(betas) != 2
            or not all(isinstance(beta, (int, float)) for beta in betas)
            or not all(0.0 <= beta < 1.0 for beta in betas)
        ):
            raise ValueError("betas must be a pair in [0, 1)")
        for value, name, allow_zero in (
            (cap_sigma, "cap_sigma", False),
            (target_step, "target_step", False),
            (eps, "eps", True),
            (damping, "damping", True),
        ):
            if not isinstance(value, (int, float)) or (
                value < 0 if allow_zero else value <= 0
            ):
                relation = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {relation} number")
        require_int(chunk_elements, "chunk_elements", minimum=1)
        if not isinstance(compile_metric, bool):
            raise TypeError("compile_metric must be a bool")
        if rent is not None and not isinstance(rent, SmoothRent):
            raise TypeError("rent must be a SmoothRent")
        if (rent is None) != (lr_w is None):
            raise ValueError("rent and lr_w come together: both or neither")
        if lr_w is not None and (
            not isinstance(lr_w, (int, float)) or lr_w <= 0
        ):
            raise ValueError("lr_w must be a positive number")
        if repulsion is not None and not isinstance(repulsion, PairRepulsion):
            raise TypeError("repulsion must be a PairRepulsion")
        if not isinstance(wall, bool):
            raise TypeError("wall must be a bool")
        require_int(seed, "seed", minimum=0)

        self.module = module
        self.store = store
        self.kernel_in = kernel_in
        self.kernel_out = kernel_out
        self.moment_space: MomentSpace = moment_space
        self.metric: MetricForm = metric
        self.cap_sigma = float(cap_sigma)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.damping = float(damping)
        self.target_step = float(target_step)
        self.chunk_elements = int(chunk_elements)
        self.compile_metric = compile_metric
        self.rent = rent
        self.lr_w = None if lr_w is None else float(lr_w)
        self.repulsion = repulsion
        self.wall = wall
        self.seed = int(seed)
        self._generator: torch.Generator | None = None

        self._capacity = store.capacity
        self.m_s: Tensor | None = None
        self.m_t: Tensor | None = None
        self.v_s: Tensor | None = None
        self.v_t: Tensor | None = None
        self.m_w: Tensor | None = None
        self.v_w: Tensor | None = None
        self.travel: Tensor | None = None
        self.step_count = 0
        self.eta: float | None = None
        if subscribe:
            store.followers().subscribe(self)

    def _materialize(self) -> None:
        self.m_s = torch.zeros_like(self.store.s)
        self.m_t = torch.zeros_like(self.store.t)
        self.v_s = torch.zeros_like(self.store.s)
        self.v_t = torch.zeros_like(self.store.t)
        if self.rent is not None:
            self.m_w = torch.zeros_like(self.store.w)
            self.v_w = torch.zeros_like(self.store.w)
        self.travel = torch.zeros(
            self.store.capacity,
            device=self.store.s.device,
            dtype=self.store.s.dtype,
        )

    def _pair_generator(self) -> torch.Generator:
        device = self.store.s.device
        if (
            self._generator is None
            or self._generator.device != device
        ):
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    @staticmethod
    def _live_values(tensor: Tensor, live: Tensor) -> Tensor:
        return tensor.index_select(0, live).reshape(-1)

    @staticmethod
    def _row_mask(parameter: Tensor, live: Tensor) -> Tensor:
        mask = torch.zeros(
            parameter.shape[0], 1,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        mask.index_fill_(0, live, 1.0)
        return mask

    @torch.no_grad()
    def _metric_diag(self, live: Tensor) -> tuple[Tensor, Tensor]:
        """Return the per-axis diagonal metric in full-capacity layout."""
        store = self.store
        diagonal_s = torch.zeros_like(store.s)
        diagonal_t = torch.zeros_like(store.t)
        if live.numel() == 0:
            return diagonal_s, diagonal_t

        mu_in = self.module.in_neurons.mu.to(store.s)
        mu_out = self.module.out_neurons.mu.to(store.t)
        width = max(mu_in.shape[0], mu_out.shape[0], 1)
        block_size = max(1, self.chunk_elements // width)
        for start in range(0, live.numel(), block_size):
            slots = live[start : start + block_size]
            source = store.s.index_select(0, slots)
            target = store.t.index_select(0, slots)
            mass_sq = store.w.detach().index_select(0, slots).square()
            block_s, block_t = _metric_block(
                self.module,
                mu_in,
                mu_out,
                source,
                target,
                mass_sq,
                compiled=self.compile_metric,
            )
            diagonal_s.index_copy_(0, slots, block_s)
            diagonal_t.index_copy_(0, slots, block_t)
        return diagonal_s, diagonal_t

    def _effective_diag(
        self, diagonal_s: Tensor, diagonal_t: Tensor, live: Tensor
    ) -> tuple[Tensor, Tensor]:
        values = torch.cat(
            [
                self._live_values(diagonal_s, live),
                self._live_values(diagonal_t, live),
            ]
        )
        tiny = torch.finfo(values.dtype).tiny
        reference = values.median().clamp_min(tiny)
        return (
            (diagonal_s + self.damping * reference).clamp_min(tiny),
            (diagonal_t + self.damping * reference).clamp_min(tiny),
        )

    @torch.no_grad()
    def _metric_gram(self, live: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-atom axis-Gram blocks in full-capacity layout."""
        store = self.store
        d_in = store.s.shape[1]
        d_out = store.t.shape[1]
        gram_s = store.s.new_zeros(store.s.shape[0], d_in, d_in)
        gram_t = store.t.new_zeros(store.t.shape[0], d_out, d_out)
        if live.numel() == 0:
            return gram_s, gram_t

        mu_in = self.module.in_neurons.mu.to(store.s)
        mu_out = self.module.out_neurons.mu.to(store.t)
        width = max(mu_in.shape[0], mu_out.shape[0], 1) * max(d_in, d_out, 1)
        block_size = max(1, self.chunk_elements // width)
        for start in range(0, live.numel(), block_size):
            slots = live[start : start + block_size]
            source = store.s.index_select(0, slots)
            target = store.t.index_select(0, slots)
            mass_sq = store.w.detach().index_select(0, slots).square()
            block_s, block_t = _metric_gram_block(
                self.module,
                mu_in,
                mu_out,
                source,
                target,
                mass_sq,
                compiled=self.compile_metric,
            )
            gram_s.index_copy_(0, slots, block_s)
            gram_t.index_copy_(0, slots, block_t)
        return gram_s, gram_t

    def _effective_gram(
        self, gram_s: Tensor, gram_t: Tensor, live: Tensor
    ) -> tuple[Tensor, Tensor]:
        diagonals = torch.cat(
            [
                self._live_values(gram_s.diagonal(dim1=-2, dim2=-1), live),
                self._live_values(gram_t.diagonal(dim1=-2, dim2=-1), live),
            ]
        )
        tiny = torch.finfo(diagonals.dtype).tiny
        reference = diagonals.median().clamp_min(tiny)

        def damped(gram: Tensor) -> Tensor:
            eye = torch.eye(
                gram.shape[-1], device=gram.device, dtype=gram.dtype
            )
            return gram + (self.damping * reference) * eye

        return damped(gram_s), damped(gram_t)

    def _whiteners(self, live: Tensor):
        """Per-side maps applying this step's effective ``G**-1/2``."""
        if self.metric == "diag":
            diagonal_s, diagonal_t = self._metric_diag(live)
            effective_s, effective_t = self._effective_diag(
                diagonal_s, diagonal_t, live
            )
            return (
                lambda values: values / effective_s.sqrt(),
                lambda values: values / effective_t.sqrt(),
            )
        gram_s, gram_t = self._metric_gram(live)
        effective_s, effective_t = self._effective_gram(
            gram_s, gram_t, live
        )
        inverse_s = _inverse_sqrt(effective_s)
        inverse_t = _inverse_sqrt(effective_t)
        return (
            lambda values: torch.einsum("kab,kb->ka", inverse_s, values),
            lambda values: torch.einsum("kab,kb->ka", inverse_t, values),
        )

    @torch.no_grad()
    def step(self, lr_scale: float = 1.0) -> None:
        """Consume current coordinate gradients and apply one Adam update."""
        if not isinstance(lr_scale, (int, float)) or lr_scale < 0:
            raise ValueError("lr_scale must be a non-negative number")
        store = self.store
        if store.s.grad is None or store.t.grad is None:
            return
        if self.rent is not None and store.w.grad is None:
            return
        live = store.live_slots().to(store.s.device)
        if live.numel() == 0:
            return
        if self.m_s is None:
            self._materialize()
        assert self.m_s is not None
        assert self.m_t is not None
        assert self.v_s is not None
        assert self.v_t is not None
        assert self.travel is not None

        whiten_s, whiten_t = self._whiteners(live)
        mask_s = self._row_mask(store.s, live)
        mask_t = self._row_mask(store.t, live)
        incoming_s = store.s.grad * mask_s
        incoming_t = store.t.grad * mask_t
        if self.repulsion is not None:
            repulsion_s, repulsion_t = self.repulsion.gradient(
                store.s.detach(),
                store.t.detach(),
                live,
                self.kernel_in.sigma.detach().to(store.s),
                self.kernel_out.sigma.detach().to(store.t),
                self._pair_generator(),
            )
            incoming_s = incoming_s + repulsion_s
            incoming_t = incoming_t + repulsion_t
        if self.moment_space == "tangent":
            incoming_s = whiten_s(incoming_s)
            incoming_t = whiten_t(incoming_t)

        self.step_count += 1
        self.m_s.mul_(self.beta1).add_(incoming_s, alpha=1.0 - self.beta1)
        self.m_t.mul_(self.beta1).add_(incoming_t, alpha=1.0 - self.beta1)
        self.v_s.mul_(self.beta2).addcmul_(
            incoming_s, incoming_s, value=1.0 - self.beta2
        )
        self.v_t.mul_(self.beta2).addcmul_(
            incoming_t, incoming_t, value=1.0 - self.beta2
        )
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        direction_s = (self.m_s / correction1) / (
            (self.v_s / correction2).sqrt() + self.eps
        )
        direction_t = (self.m_t / correction1) / (
            (self.v_t / correction2).sqrt() + self.eps
        )
        raw_s = whiten_s(direction_s) * mask_s
        raw_t = whiten_t(direction_t) * mask_t

        raw_norm = (raw_s.square().sum(1) + raw_t.square().sum(1)).sqrt()
        sigma = self.kernel_in.sigma.detach().to(raw_norm)
        if self.eta is None:
            median = raw_norm.index_select(0, live).median().clamp_min(1e-30)
            self.eta = float(self.target_step * sigma / median)

        delta_s = raw_s * (self.eta * float(lr_scale))
        delta_t = raw_t * (self.eta * float(lr_scale))
        delta_norm = (delta_s.square().sum(1) + delta_t.square().sum(1)).sqrt()
        cap = self.cap_sigma * sigma
        scale = (cap / delta_norm.clamp_min(1e-30)).clamp(max=1.0)
        delta_s.mul_(scale[:, None])
        delta_t.mul_(scale[:, None])
        store.s.sub_(delta_s)
        store.t.sub_(delta_t)
        self.travel.add_(
            (delta_s.square().sum(1) + delta_t.square().sum(1)).sqrt() / sigma
        )

        if self.rent is not None:
            assert self.m_w is not None and self.v_w is not None
            mask_w = mask_s[:, 0]
            incoming_w = (
                store.w.grad
                + self.rent.gradient(store.w.detach(), self.step_count)
            ) * mask_w
            self.m_w.mul_(self.beta1).add_(
                incoming_w, alpha=1.0 - self.beta1
            )
            self.v_w.mul_(self.beta2).addcmul_(
                incoming_w, incoming_w, value=1.0 - self.beta2
            )
            direction_w = (self.m_w / correction1) / (
                (self.v_w / correction2).sqrt() + self.eps
            )
            store.w.sub_(
                direction_w * (self.lr_w * float(lr_scale)) * mask_w
            )

        if self.wall:
            box_in = store.spec.domain_in
            box_out = store.spec.domain_out
            store.s.clamp_(
                min=store.s.new_tensor(box_in.lo_per_axis),
                max=store.s.new_tensor(box_in.hi_per_axis),
            )
            store.t.clamp_(
                min=store.t.new_tensor(box_out.lo_per_axis),
                max=store.t.new_tensor(box_out.hi_per_axis),
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear the gradients consumed by this optimizer."""
        parameters = [self.store.s, self.store.t]
        if self.rent is not None:
            parameters.append(self.store.w)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def mobility(self, threshold: float = 0.5) -> float:
        """Fraction of live atoms whose accumulated travel exceeds a sigma."""
        if self.travel is None:
            return 0.0
        live = self.store.live_slots().to(self.travel.device)
        if live.numel() == 0:
            return 0.0
        return float(
            (self.travel.index_select(0, live) > threshold).float().mean()
        )

    # ---- follower contract ------------------------------------------------

    _ROW_NAMES = ("m_s", "m_t", "v_s", "v_t", "m_w", "v_w", "travel")

    def _rows(self):
        return (
            tensor
            for tensor in (
                getattr(self, name) for name in self._ROW_NAMES
            )
            if tensor is not None
        )

    def _zero_rows(self, slots: Tensor, rows=None) -> None:
        if not isinstance(slots, Tensor):
            raise TypeError("slots must be a Tensor")
        slots = slots.detach().to("cpu")
        if slots.numel() and bool(
            ((slots < 0) | (slots >= self._capacity)).any()
        ):
            raise IndexError("PullbackAdam follower slots are outside capacity")
        for tensor in self._rows() if rows is None else rows:
            if tensor is not None and slots.numel():
                tensor.index_fill_(0, slots.to(tensor.device), 0)

    def grow(self, new_capacity: int) -> None:
        require_int(new_capacity, "new_capacity")
        if new_capacity < self._capacity:
            raise ValueError("PullbackAdam cannot shrink")
        if new_capacity == self._capacity:
            return
        for name in self._ROW_NAMES:
            tensor = getattr(self, name)
            if tensor is None:
                continue
            grown = tensor.new_zeros((new_capacity, *tensor.shape[1:]))
            grown[: self._capacity].copy_(tensor)
            setattr(self, name, grown)
        self._capacity = new_capacity

    def on_birth(self, slots: Tensor, lineage: Tensor) -> None:
        del lineage
        self._zero_rows(slots)

    def on_death(self, slots: Tensor) -> None:
        self._zero_rows(slots)

    def on_refit(self, slots: Tensor) -> None:
        """Amplitude refits invalidate only the amplitude moments."""
        if self.m_w is not None:
            self._zero_rows(slots, rows=(self.m_w, self.v_w))

    def on_remap(self, old_to_new: Tensor) -> None:
        mapping = old_to_new.detach().to("cpu")
        if mapping.numel() != self._capacity:
            raise ValueError("old_to_new must align with PullbackAdam capacity")
        old = torch.nonzero(mapping >= 0, as_tuple=False).flatten()
        if old.numel() and bool((mapping[old] >= self._capacity).any()):
            raise IndexError("PullbackAdam remap targets outside capacity")
        for name in self._ROW_NAMES:
            tensor = getattr(self, name)
            if tensor is None:
                continue
            remapped = torch.zeros_like(tensor)
            if old.numel():
                source = old.to(tensor.device)
                target = mapping[old].to(tensor.device)
                remapped.index_copy_(0, target, tensor.index_select(0, source))
            setattr(self, name, remapped)

    # ---- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, object]:
        def snapshot(tensor: Tensor | None) -> Tensor | None:
            return None if tensor is None else tensor.detach().cpu().clone()

        return {
            "schema": "torchcst-pullback-adam-v1",
            "moment_space": self.moment_space,
            "metric": self.metric,
            "owns_w": self.rent is not None,
            "capacity": self._capacity,
            "step_count": self.step_count,
            "eta": self.eta,
            "m_s": snapshot(self.m_s),
            "m_t": snapshot(self.m_t),
            "v_s": snapshot(self.v_s),
            "v_t": snapshot(self.v_t),
            "m_w": snapshot(self.m_w),
            "v_w": snapshot(self.v_w),
            "travel": snapshot(self.travel),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-pullback-adam-v1"
        ):
            raise ValueError("unsupported PullbackAdam state schema")
        if state.get("moment_space") != self.moment_space:
            raise ValueError("PullbackAdam moment_space does not match state")
        if state.get("metric", "diag") != self.metric:
            raise ValueError("PullbackAdam metric does not match state")
        if bool(state.get("owns_w", False)) != (self.rent is not None):
            raise ValueError(
                "PullbackAdam rent ownership does not match state"
            )
        capacity = require_int(
            state.get("capacity"), "PullbackAdam capacity", minimum=0
        )
        if capacity != self.store.capacity:
            raise ValueError("PullbackAdam state does not match store capacity")
        self._capacity = capacity
        self.step_count = require_int(
            state.get("step_count"), "PullbackAdam step_count", minimum=0
        )
        eta = state.get("eta")
        if eta is not None and not isinstance(eta, (int, float)):
            raise TypeError("PullbackAdam eta must be a number or None")
        self.eta = None if eta is None else float(eta)
        shapes = {
            "m_s": self.store.s.shape,
            "m_t": self.store.t.shape,
            "v_s": self.store.s.shape,
            "v_t": self.store.t.shape,
            "m_w": self.store.w.shape,
            "v_w": self.store.w.shape,
            "travel": (capacity,),
        }
        targets = {
            "m_s": self.store.s,
            "v_s": self.store.s,
            "m_t": self.store.t,
            "v_t": self.store.t,
            "m_w": self.store.w,
            "v_w": self.store.w,
            "travel": self.store.s,
        }
        for name, shape in shapes.items():
            value = state.get(name)
            if value is None:
                setattr(self, name, None)
                continue
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError(f"PullbackAdam {name} has an invalid shape")
            setattr(self, name, value.to(targets[name]).clone())
