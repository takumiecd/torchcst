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
MetricForm = Literal["diag", "block", "full"]


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
    columns: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate one diagonal-metric block, optionally through ``compile``."""
    if (
        compiled
        and source.is_cuda
        and type(module.kernel_in) is GaussianKernel
        and type(module.kernel_out) is GaussianKernel
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
        module.kernel_in, mu_in, source, columns
    )
    unit_out, factor_out = metric.normalized_columns(
        module.kernel_out, mu_out, target, columns
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
    columns: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate one within-atom metric-gram block, optionally compiled."""
    if (
        compiled
        and source.is_cuda
        and type(module.kernel_in) is GaussianKernel
        and type(module.kernel_out) is GaussianKernel
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
        module.kernel_in, mu_in, source, columns
    )
    unit_out, factor_out = metric.normalized_columns(
        module.kernel_out, mu_out, target, columns
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

    ``"full"``
        The exact coordinate Gram across every live atom in the site.  This
        includes source-source, source-target and target-target coupling but
        holds amplitudes fixed, so their separate optimiser remains unchanged.
        It is an intentionally expensive oracle implemented with the
        separable kernel factors rather than a materialised dense-map
        Jacobian.

    ``"diag"`` and ``"block"`` neglect cross-atom coupling.  Persistent
    state is coordinate-shaped in every form; ``"full"`` retains its dense
    Gram only for the duration of one step.

    ``betas`` belong to the coordinate moments.  When this optimizer also owns
    amplitudes, ``amplitude_betas`` can keep their scalar Adam clock separate;
    ``None`` preserves the historical behaviour of sharing ``betas``.

    The optional structural terms make this optimizer the site's single
    owner for the continuous lifecycle:

    ``decay=`` (with ``lr_w``)
        A decoupled price on the amplitudes, applied after the Adam step as
        ``w -= lr_w * decay * w`` -- AdamW's own answer to an adaptive
        denominator eating the price.  Independent of ``rent``: either one
        gives the optimizer ownership of ``synapses.w``, and they compose.
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

    ``decoupled_repulsion=PairRepulsion(...)``
        Apply the same structural force after the adaptive coordinate step,
        in the AdamW sense.  It never enters either coordinate moment, so a
        small repulsion cannot be amplified by ``1 / sqrt(v)`` on reserve
        atoms.  Its ``mu`` is therefore a direct coordinate-step scale.

    ``wall=True``
        After each step, coordinates are clamped into the store's declared
        coordinate domains, so wandering reserves stay on the chart.

    ``moment_distance=(tau_m, tau_v)``
        Forget coordinate moments by distance travelled, in joint
        source-target kernel-sigma units.  A step of length ``d``
        retains ``exp(-d / tau_m)`` of the first-moment history and
        ``exp(-d / tau_v)`` of the second-moment history.  Per-row normalising
        masses keep Adam's bias correction valid under this extra decay.  This
        is useful for mobile reserves whose old tangent frame ceases to be
        relevant after they cross the chart; ``None`` preserves ordinary
        step-clock Adam exactly.

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
        amplitude_betas: tuple[float, float] | None = None,
        eps: float = 1e-8,
        damping: float = 1e-2,
        target_step: float = 0.01,
        rent: SmoothRent | None = None,
        decay: float = 0.0,
        lr_w: float | None = None,
        repulsion: PairRepulsion | None = None,
        decoupled_repulsion: PairRepulsion | None = None,
        wall: bool = False,
        seed: int = 0,
        subscribe: bool = True,
        chunk_elements: int = 1 << 24,
        compile_metric: bool = False,
        moment_distance: tuple[float, float] | None = None,
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
        if metric not in ("diag", "block", "full"):
            raise ValueError("metric must be 'diag', 'block', or 'full'")
        for values, name in (
            (betas, "betas"),
            (amplitude_betas, "amplitude_betas"),
        ):
            if values is None and name == "amplitude_betas":
                continue
            if (
                not isinstance(values, tuple)
                or len(values) != 2
                or not all(isinstance(beta, (int, float)) for beta in values)
                or not all(0.0 <= beta < 1.0 for beta in values)
            ):
                raise ValueError(f"{name} must be a pair in [0, 1)")
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
        if moment_distance is not None and (
            not isinstance(moment_distance, tuple)
            or len(moment_distance) != 2
            or not all(
                isinstance(value, (int, float)) and value > 0
                for value in moment_distance
            )
        ):
            raise ValueError(
                "moment_distance must be a pair of positive numbers or None"
            )
        if rent is not None and not isinstance(rent, SmoothRent):
            raise TypeError("rent must be a SmoothRent")
        if not isinstance(decay, (int, float)) or decay < 0:
            raise ValueError("decay must be a non-negative number")
        owns_amplitudes = rent is not None or decay > 0
        if owns_amplitudes != (lr_w is not None):
            raise ValueError(
                "lr_w comes with amplitude ownership: pass it with rent "
                "and/or decay, and only then"
            )
        if lr_w is not None and (
            not isinstance(lr_w, (int, float)) or lr_w <= 0
        ):
            raise ValueError("lr_w must be a positive number")
        if repulsion is not None and not isinstance(repulsion, PairRepulsion):
            raise TypeError("repulsion must be a PairRepulsion")
        if decoupled_repulsion is not None and not isinstance(
            decoupled_repulsion, PairRepulsion
        ):
            raise TypeError("decoupled_repulsion must be a PairRepulsion")
        if repulsion is not None and decoupled_repulsion is not None:
            raise ValueError(
                "repulsion and decoupled_repulsion are mutually exclusive"
            )
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
        amplitude_betas = betas if amplitude_betas is None else amplitude_betas
        self.amplitude_beta1 = float(amplitude_betas[0])
        self.amplitude_beta2 = float(amplitude_betas[1])
        self.eps = float(eps)
        self.damping = float(damping)
        self.target_step = float(target_step)
        self.chunk_elements = int(chunk_elements)
        self.compile_metric = compile_metric
        self.moment_distance = (
            None
            if moment_distance is None
            else (float(moment_distance[0]), float(moment_distance[1]))
        )
        self.rent = rent
        self.decay = float(decay)
        #: Whether this optimizer updates ``synapses.w``.  A price is two
        #: independent choices -- who owns the amplitudes, and whether the
        #: price is charged inside the moments (``rent``) or outside them
        #: (``decay``) -- and they were entangled while ``rent`` alone
        #: decided ownership.  The owning optimizer must be excluded from
        #: the model optimizer's parameter groups either way.
        self.owns_amplitudes = owns_amplitudes
        self.lr_w = None if lr_w is None else float(lr_w)
        self.repulsion = repulsion
        self.decoupled_repulsion = decoupled_repulsion
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
        self.moment_mass1: Tensor | None = None
        self.moment_mass2: Tensor | None = None
        self.step_count = 0
        self.eta: float | None = None
        if subscribe:
            store.followers().subscribe(self)

    def _materialize(self) -> None:
        self.m_s = torch.zeros_like(self.store.s)
        self.m_t = torch.zeros_like(self.store.t)
        self.v_s = torch.zeros_like(self.store.s)
        self.v_t = torch.zeros_like(self.store.t)
        if self.owns_amplitudes:
            self.m_w = torch.zeros_like(self.store.w)
            self.v_w = torch.zeros_like(self.store.w)
        self.travel = torch.zeros(
            self.store.capacity,
            device=self.store.s.device,
            dtype=self.store.s.dtype,
        )
        if self.moment_distance is not None:
            self.moment_mass1 = torch.zeros_like(self.travel)
            self.moment_mass2 = torch.zeros_like(self.travel)

    def _moment_corrections(
        self, mask: Tensor
    ) -> tuple[Tensor | float, Tensor | float]:
        """Advance and return bias corrections for this step's live rows."""
        if self.moment_distance is None:
            return (
                1.0 - self.beta1**self.step_count,
                1.0 - self.beta2**self.step_count,
            )
        assert self.moment_mass1 is not None
        assert self.moment_mass2 is not None
        self.moment_mass1.mul_(self.beta1).add_(
            mask, alpha=1.0 - self.beta1
        )
        self.moment_mass2.mul_(self.beta2).add_(
            mask, alpha=1.0 - self.beta2
        )
        tiny = torch.finfo(mask.dtype).tiny
        return (
            self.moment_mass1.clamp_min(tiny)[:, None],
            self.moment_mass2.clamp_min(tiny)[:, None],
        )

    def _forget_moments_by_distance(self, distance: Tensor) -> None:
        """Decay history after transporting an atom by ``distance`` sigmas."""
        if self.moment_distance is None:
            return
        assert self.m_s is not None and self.m_t is not None
        assert self.v_s is not None and self.v_t is not None
        assert self.moment_mass1 is not None
        assert self.moment_mass2 is not None
        tau_m, tau_v = self.moment_distance
        retention1 = torch.exp(-distance / tau_m)
        retention2 = torch.exp(-distance / tau_v)
        self.m_s.mul_(retention1[:, None])
        self.m_t.mul_(retention1[:, None])
        self.v_s.mul_(retention2[:, None])
        self.v_t.mul_(retention2[:, None])
        self.moment_mass1.mul_(retention1)
        self.moment_mass2.mul_(retention2)

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

    def _selected_columns(self, slots: Tensor) -> dict[str, Tensor]:
        return {
            name: getattr(self.store, name).index_select(0, slots)
            for name in self.store.atom_column_names
        }

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
                columns=self._selected_columns(slots),
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
                columns=self._selected_columns(slots),
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

    @torch.no_grad()
    def _metric_full(self, live: Tensor) -> Tensor:
        """Return the full cross-atom coordinate Gram for live rows."""
        store = self.store
        count = live.numel()
        size = count * (store.s.shape[1] + store.t.shape[1])
        if count == 0:
            return store.s.new_zeros(size, size)
        source = store.s.index_select(0, live)
        target = store.t.index_select(0, live)
        columns = self._selected_columns(live)
        unit_in, factor_in = metric.normalized_columns(
            self.module.kernel_in,
            self.module.in_neurons.mu.to(source),
            source,
            columns,
        )
        unit_out, factor_out = metric.normalized_columns(
            self.module.kernel_out,
            self.module.out_neurons.mu.to(target),
            target,
            columns,
        )
        return metric.gauged_coordinate_jacobian_gram(
            unit_in=unit_in,
            factor_in=factor_in,
            unit_out=unit_out,
            factor_out=factor_out,
            mu_in=self.module.in_neurons.mu.to(source),
            mu_out=self.module.out_neurons.mu.to(target),
            source=source,
            target=target,
            mass=store.w.detach().index_select(0, live),
        )

    def _effective_full(self, gram: Tensor) -> Tensor:
        tiny = torch.finfo(gram.dtype).tiny
        reference = gram.diagonal().median().clamp_min(tiny)
        eye = torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        return gram + (self.damping * reference) * eye

    def _whitener(self, live: Tensor):
        """Map coordinate pairs through this step's effective ``G**-1/2``."""
        if self.metric == "diag":
            diagonal_s, diagonal_t = self._metric_diag(live)
            effective_s, effective_t = self._effective_diag(
                diagonal_s, diagonal_t, live
            )

            def diagonal(values_s: Tensor, values_t: Tensor):
                return (
                    values_s / effective_s.sqrt(),
                    values_t / effective_t.sqrt(),
                )

            return diagonal
        if self.metric == "block":
            gram_s, gram_t = self._metric_gram(live)
            effective_s, effective_t = self._effective_gram(
                gram_s, gram_t, live
            )
            inverse_s = _inverse_sqrt(effective_s)
            inverse_t = _inverse_sqrt(effective_t)

            def block(values_s: Tensor, values_t: Tensor):
                return (
                    torch.einsum("kab,kb->ka", inverse_s, values_s),
                    torch.einsum("kab,kb->ka", inverse_t, values_t),
                )

            return block

        effective = self._effective_full(self._metric_full(live))
        inverse = _inverse_sqrt(effective)
        count = live.numel()
        source_size = count * self.store.s.shape[1]

        def full(values_s: Tensor, values_t: Tensor):
            selected = torch.cat(
                [
                    values_s.index_select(0, live).reshape(-1),
                    values_t.index_select(0, live).reshape(-1),
                ]
            )
            transformed = inverse @ selected
            output_s = torch.zeros_like(values_s)
            output_t = torch.zeros_like(values_t)
            output_s.index_copy_(
                0, live, transformed[:source_size].reshape(
                    count, self.store.s.shape[1]
                )
            )
            output_t.index_copy_(
                0, live, transformed[source_size:].reshape(
                    count, self.store.t.shape[1]
                )
            )
            return output_s, output_t

        return full

    @torch.no_grad()
    def step(self, lr_scale: float = 1.0) -> None:
        """Consume current coordinate gradients and apply one Adam update."""
        if not isinstance(lr_scale, (int, float)) or lr_scale < 0:
            raise ValueError("lr_scale must be a non-negative number")
        store = self.store
        if store.s.grad is None or store.t.grad is None:
            return
        if self.owns_amplitudes and store.w.grad is None:
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

        whiten = self._whitener(live)
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
            incoming_s, incoming_t = whiten(incoming_s, incoming_t)

        self.step_count += 1
        self.m_s.mul_(self.beta1).add_(incoming_s, alpha=1.0 - self.beta1)
        self.m_t.mul_(self.beta1).add_(incoming_t, alpha=1.0 - self.beta1)
        self.v_s.mul_(self.beta2).addcmul_(
            incoming_s, incoming_s, value=1.0 - self.beta2
        )
        self.v_t.mul_(self.beta2).addcmul_(
            incoming_t, incoming_t, value=1.0 - self.beta2
        )
        correction1, correction2 = self._moment_corrections(mask_s[:, 0])
        direction_s = (self.m_s / correction1) / (
            (self.v_s / correction2).sqrt() + self.eps
        )
        direction_t = (self.m_t / correction1) / (
            (self.v_t / correction2).sqrt() + self.eps
        )
        raw_s, raw_t = whiten(direction_s, direction_t)
        raw_s.mul_(mask_s)
        raw_t.mul_(mask_t)

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

        if self.decoupled_repulsion is not None:
            repulsion_s, repulsion_t = self.decoupled_repulsion.gradient(
                store.s.detach(),
                store.t.detach(),
                live,
                self.kernel_in.sigma.detach().to(store.s),
                self.kernel_out.sigma.detach().to(store.t),
                self._pair_generator(),
            )
            repulsion_s.mul_(float(lr_scale))
            repulsion_t.mul_(float(lr_scale))
            repulsion_norm = (
                repulsion_s.square().sum(1)
                + repulsion_t.square().sum(1)
            ).sqrt()
            repulsion_scale = (
                cap / repulsion_norm.clamp_min(1e-30)
            ).clamp(max=1.0)
            repulsion_s.mul_(repulsion_scale[:, None])
            repulsion_t.mul_(repulsion_scale[:, None])
            store.s.sub_(repulsion_s)
            store.t.sub_(repulsion_t)
            delta_s.add_(repulsion_s)
            delta_t.add_(repulsion_t)
        distance = (
            delta_s.square().sum(1) + delta_t.square().sum(1)
        ).sqrt() / sigma
        self.travel.add_(distance)

        if self.owns_amplitudes:
            assert self.m_w is not None and self.v_w is not None
            mask_w = mask_s[:, 0]
            incoming_w = store.w.grad
            if self.rent is not None:
                incoming_w = incoming_w + self.rent.gradient(
                    store.w.detach(), self.step_count
                )
            incoming_w = incoming_w * mask_w
            self.m_w.mul_(self.amplitude_beta1).add_(
                incoming_w, alpha=1.0 - self.amplitude_beta1
            )
            self.v_w.mul_(self.amplitude_beta2).addcmul_(
                incoming_w, incoming_w, value=1.0 - self.amplitude_beta2
            )
            # Distance forgetting belongs to the moving coordinate frame.
            # Amplitudes live in a fixed scalar frame and retain ordinary
            # step-clock Adam, which also keeps this option orthogonal to the
            # site's amplitude optimizer.
            correction1_w = 1.0 - self.amplitude_beta1**self.step_count
            correction2_w = 1.0 - self.amplitude_beta2**self.step_count
            direction_w = (self.m_w / correction1_w) / (
                (self.v_w / correction2_w).sqrt() + self.eps
            )
            rate = self.lr_w * float(lr_scale)
            store.w.sub_(direction_w * rate * mask_w)
            if self.decay:
                # Decoupled, in AdamW's sense: the price never reaches the
                # moments, so `sqrt(v)` cannot normalise it away.  Charged
                # after the step and on live rows only.
                store.w.sub_(store.w.detach() * (rate * self.decay) * mask_w)

        self._forget_moments_by_distance(distance)

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
        if self.owns_amplitudes:
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

    _ROW_NAMES = (
        "m_s", "m_t", "v_s", "v_t", "m_w", "v_w", "travel",
        "moment_mass1", "moment_mass2",
    )

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
            "moment_distance": self.moment_distance,
            "owns_w": self.owns_amplitudes,
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
            "moment_mass1": snapshot(self.moment_mass1),
            "moment_mass2": snapshot(self.moment_mass2),
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
        saved_distance = state.get("moment_distance")
        if saved_distance is not None:
            saved_distance = tuple(saved_distance)
        if saved_distance != self.moment_distance:
            raise ValueError(
                "PullbackAdam moment_distance does not match state"
            )
        if bool(state.get("owns_w", False)) != self.owns_amplitudes:
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
            "moment_mass1": (capacity,),
            "moment_mass2": (capacity,),
        }
        targets = {
            "m_s": self.store.s,
            "v_s": self.store.s,
            "m_t": self.store.t,
            "v_t": self.store.t,
            "m_w": self.store.w,
            "v_w": self.store.w,
            "travel": self.store.s,
            "moment_mass1": self.store.s,
            "moment_mass2": self.store.s,
        }
        for name, shape in shapes.items():
            value = state.get(name)
            if value is None:
                setattr(self, name, None)
                continue
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError(f"PullbackAdam {name} has an invalid shape")
            setattr(self, name, value.to(targets[name]).clone())
