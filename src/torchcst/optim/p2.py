"""Dense-free primitives for atom-local P2 optimization.

The represented weight and its dense gradient are deliberately absent from
this module.  A caller supplies one scalar contribution per atom,

``score_k(theta_k) = <stopgrad(g_W), W_k(theta_k)>``,

computed through the factored CST forward.  Differentiating that scalar gives
the two sufficient statistics needed by the first-order-in-loss, second-order
in-map model:

``b_k = J_k.T g_W`` and ``C_k = g_W contract H_map,k``.

Both are atom sized.  The map is additive across atoms, so its contracted
Hessian has no cross-atom blocks and the storage cost is ``O(K p^2)`` rather
than ``O(n_out n_in)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor

from torchcst.compute import (
    BackwardContext,
    CSTLinear,
    CaptureBatch,
    Factored,
    flatten_capture_pair,
)


AtomScore = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ContractedP2Model:
    """Atom-local coefficients of ``b.T d + 1/2 d.T C d``."""

    linear: Tensor
    curvature: Tensor

    def __post_init__(self) -> None:
        if self.linear.ndim != 2:
            raise ValueError("linear must have shape [atoms, fields]")
        expected = (*self.linear.shape, self.linear.shape[1])
        if self.curvature.shape != expected:
            raise ValueError(
                f"curvature must have shape {expected}, got "
                f"{tuple(self.curvature.shape)}"
            )
        if self.curvature.device != self.linear.device:
            raise ValueError("linear and curvature must share a device")
        if self.curvature.dtype != self.linear.dtype:
            raise ValueError("linear and curvature must share a dtype")


def contracted_p2_model(atom_score: AtomScore, theta: Tensor) -> ContractedP2Model:
    """Differentiate independent scalar atom scores without materializing ``W``.

    ``theta`` has shape ``[K, p]`` and ``atom_score`` accepts one ``[p]`` row.
    The score must return a scalar and should contract a frozen upstream output
    gradient with that atom's factored output.  ``vmap`` then evaluates only
    the ``K`` independent ``p``-dimensional derivative problems; no
    ``[out, in]`` tensor or cross-atom Hessian is constructed.
    """

    if theta.ndim != 2:
        raise ValueError("theta must have shape [atoms, fields]")
    if not theta.is_floating_point():
        raise TypeError("theta must be floating point")

    gradient = torch.func.grad(atom_score)
    hessian = torch.func.jacfwd(gradient)
    linear = torch.vmap(gradient)(theta)
    curvature = torch.vmap(hessian)(theta)
    if linear.shape != theta.shape:
        raise ValueError("atom_score must return one scalar per atom")
    curvature = 0.5 * (curvature + curvature.transpose(-1, -2))
    return ContractedP2Model(linear=linear, curvature=curvature)


def contracted_cst_linear_p2_model(
    module: CSTLinear,
    x: Tensor,
    grad_output: Tensor,
) -> ContractedP2Model:
    """Build ``(b, C)`` from one factored :class:`CSTLinear` observation.

    The method intentionally rejects ``"auto"`` and materialized execution:
    the optimizer's memory contract must not depend on a crossover heuristic.
    The first implementation covers the core ``(w, s, t)`` atom family;
    factor-declared trainable atom columns will be added as explicit fields
    rather than silently omitted from the P2 model.
    """

    if not isinstance(module, CSTLinear):
        raise TypeError("module must be a CSTLinear")
    module._view()
    if not isinstance(module.backend, Factored):
        raise ValueError(
            "dense-free P2 optimization requires an explicit Factored backend"
        )
    if module.synapses.atom_column_names:
        raise NotImplementedError(
            "dense-free P2 does not yet support factor-declared atom columns"
        )
    x_flat, grad_flat = flatten_capture_pair(
        x, grad_output, module.in_features, module.out_features
    )
    source, target, weights = module._live_factors()
    source = source.detach().to(x_flat)
    target = target.detach().to(x_flat)
    weights = weights.detach().to(x_flat)
    theta = torch.cat((weights[:, None], source, target), dim=1)
    d_in = source.shape[1]
    mu_in = module.in_neurons.mu.detach().to(x_flat)
    mu_out = module.out_neurons.mu.detach().to(x_flat)

    def atom_score(atom: Tensor) -> Tensor:
        local_source = atom[1 : 1 + d_in][None, :]
        local_target = atom[1 + d_in :][None, :]
        k_in = module.gauge.columns(module.factor_in, mu_in, local_source, {})[:, 0]
        k_out = module.gauge.columns(module.factor_out, mu_out, local_target, {})[:, 0]
        activation = x_flat @ k_in
        output_projection = grad_flat @ k_out
        return atom[0] * (activation * output_projection).sum()

    return contracted_p2_model(atom_score, theta)


def cst_linear_p2_reducer(
    module: CSTLinear,
) -> Callable[[str, Tensor, Tensor, int], dict[str, Tensor]]:
    """Return a hook-time reducer that retains only atom-sized P2 statistics."""

    if not isinstance(module, CSTLinear):
        raise TypeError("module must be a CSTLinear")

    def reduce(
        site: str, x: Tensor, grad_output: Tensor, version: int
    ) -> dict[str, Tensor]:
        if site != module.capture_site:
            raise ValueError("P2 reducer received an observation for another site")
        if version != module.synapses.version:
            raise RuntimeError("P2 reducer received a stale synapse view")
        model = contracted_cst_linear_p2_model(module, x, grad_output)
        return {"linear": model.linear, "curvature": model.curvature}

    return reduce


class P2ModelEMA:
    """EMA of a stream of atom-local quadratic P2 models.

    A single decay averages the complete quadratic model, rather than silently
    assigning unrelated time horizons to its linear and curvature terms.
    Bias correction is exposed for diagnostics; multiplying both terms by the
    same positive scalar does not change an undamped trust-region minimizer.
    """

    def __init__(self, beta: float = 0.9) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("beta must be a number")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        self.beta = float(beta)
        self.linear: Tensor | None = None
        self.curvature: Tensor | None = None
        self.mass = 0.0

    def update(self, model: ContractedP2Model) -> ContractedP2Model:
        """Consume one detached batch model and return its corrected EMA."""

        incoming_linear = model.linear.detach()
        incoming_curvature = model.curvature.detach()
        if self.linear is None:
            self.linear = torch.zeros_like(incoming_linear)
            self.curvature = torch.zeros_like(incoming_curvature)
        assert self.curvature is not None
        if self.linear.shape != incoming_linear.shape:
            raise ValueError("linear shape changed across P2ModelEMA updates")
        if self.curvature.shape != incoming_curvature.shape:
            raise ValueError("curvature shape changed across P2ModelEMA updates")
        self.linear.mul_(self.beta).add_(incoming_linear, alpha=1.0 - self.beta)
        self.curvature.mul_(self.beta).add_(incoming_curvature, alpha=1.0 - self.beta)
        self.mass = self.beta * self.mass + (1.0 - self.beta)
        return self.average()

    def average(self) -> ContractedP2Model:
        """Return the bias-corrected current average."""

        if self.linear is None or self.curvature is None or self.mass == 0.0:
            raise RuntimeError("P2ModelEMA has not received a model")
        return ContractedP2Model(
            linear=self.linear / self.mass,
            curvature=self.curvature / self.mass,
        )

    def reset(self) -> None:
        """Forget the model history, e.g. after a structural mutation."""

        self.linear = None
        self.curvature = None
        self.mass = 0.0

    def state_dict(self) -> dict[str, object]:
        """Return checkpoint state without references to live tensors."""

        return {
            "beta": self.beta,
            "mass": self.mass,
            "linear": None if self.linear is None else self.linear.detach().clone(),
            "curvature": (
                None if self.curvature is None else self.curvature.detach().clone()
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a shape-consistent EMA checkpoint."""

        if not isinstance(state, dict):
            raise TypeError("P2ModelEMA state must be a dict")
        if set(state) != {"beta", "mass", "linear", "curvature"}:
            raise ValueError("P2ModelEMA state has unexpected fields")
        beta = state["beta"]
        mass = state["mass"]
        linear = state["linear"]
        curvature = state["curvature"]
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("P2ModelEMA beta state must be numeric")
        if not 0 <= float(beta) < 1:
            raise ValueError("P2ModelEMA beta state must lie in [0, 1)")
        if not isinstance(mass, (int, float)) or isinstance(mass, bool):
            raise TypeError("P2ModelEMA mass state must be numeric")
        if not 0 <= float(mass) <= 1:
            raise ValueError("P2ModelEMA mass state must lie in [0, 1]")
        if (linear is None) != (curvature is None):
            raise ValueError("P2ModelEMA tensors must both be present or absent")
        if linear is not None:
            if not isinstance(linear, Tensor) or not isinstance(curvature, Tensor):
                raise TypeError("P2ModelEMA tensor state must contain Tensors")
            ContractedP2Model(linear, curvature)
            linear = linear.detach().clone()
            curvature = curvature.detach().clone()
        elif float(mass) != 0.0:
            raise ValueError("empty P2ModelEMA state must have zero mass")
        self.beta = float(beta)
        self.mass = float(mass)
        self.linear = linear
        self.curvature = curvature


class P2StepEMA:
    """EMA of solved steps in dimensionless trust-region coordinates.

    This is intentionally a separate ablation from :class:`P2ModelEMA`.
    Averaging ``q = d / scales`` keeps the state independent of coordinate
    units and the returned step is projected back into the current trust
    region.  It is not a substitute for averaging the quadratic model.
    """

    def __init__(self, beta: float) -> None:
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("beta must be a number")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        self.beta = float(beta)
        self.dimensionless_step: Tensor | None = None
        self.mass = 0.0

    def update(self, step: Tensor, scales: Tensor, radius: float) -> Tensor:
        """Average one candidate and return a trust-region-feasible step."""

        scales = torch.broadcast_to(scales.to(step), step.shape)
        if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
            raise ValueError("scales must be finite and positive")
        if not isinstance(radius, (int, float)) or isinstance(radius, bool):
            raise TypeError("radius must be a number")
        if radius <= 0:
            raise ValueError("radius must be positive")
        incoming = (step.detach() / scales).detach()
        if self.dimensionless_step is None:
            self.dimensionless_step = torch.zeros_like(incoming)
        if self.dimensionless_step.shape != incoming.shape:
            raise ValueError("step shape changed across P2StepEMA updates")
        self.dimensionless_step.mul_(self.beta).add_(incoming, alpha=1.0 - self.beta)
        self.mass = self.beta * self.mass + (1.0 - self.beta)
        corrected = self.dimensionless_step / self.mass
        norm = torch.linalg.vector_norm(corrected)
        if bool(norm > radius):
            corrected = corrected * (radius / norm)
        return scales * corrected

    def reset(self) -> None:
        self.dimensionless_step = None
        self.mass = 0.0

    def state_dict(self) -> dict[str, object]:
        return {
            "beta": self.beta,
            "mass": self.mass,
            "dimensionless_step": (
                None
                if self.dimensionless_step is None
                else self.dimensionless_step.detach().clone()
            ),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise TypeError("P2StepEMA state must be a dict")
        if set(state) != {"beta", "mass", "dimensionless_step"}:
            raise ValueError("P2StepEMA state has unexpected fields")
        beta = state["beta"]
        mass = state["mass"]
        step = state["dimensionless_step"]
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise TypeError("P2StepEMA beta state must be numeric")
        if not 0 <= float(beta) < 1:
            raise ValueError("P2StepEMA beta state must lie in [0, 1)")
        if not isinstance(mass, (int, float)) or isinstance(mass, bool):
            raise TypeError("P2StepEMA mass state must be numeric")
        if not 0 <= float(mass) <= 1:
            raise ValueError("P2StepEMA mass state must lie in [0, 1]")
        if step is not None and not isinstance(step, Tensor):
            raise TypeError("P2StepEMA step state must be a Tensor")
        if step is None and float(mass) != 0.0:
            raise ValueError("empty P2StepEMA state must have zero mass")
        self.beta = float(beta)
        self.mass = float(mass)
        self.dimensionless_step = None if step is None else step.detach().clone()


@dataclass(frozen=True)
class TrustRegionResult:
    """Solution of one block-diagonal quadratic trust-region problem."""

    step: Tensor
    lagrange_multiplier: Tensor
    predicted_change: Tensor
    hit_boundary: bool
    accepted: bool = True
    actual_change: Tensor | None = None
    reduction_ratio: Tensor | None = None
    radius_before: float | None = None
    radius_after: float | None = None


def _quadratic_value(linear: Tensor, curvature: Tensor, step: Tensor) -> Tensor:
    return (linear * step).sum() + 0.5 * torch.einsum(
        "ki,kij,kj->", step, curvature, step
    )


def solve_block_trust_region(
    model: ContractedP2Model,
    radius: float,
    *,
    scales: Tensor | None = None,
    root_iterations: int = 80,
) -> TrustRegionResult:
    """Globally solve a block-diagonal quadratic over one ellipsoid.

    ``scales`` defines ``d = scales * q`` and the constraint ``||q|| <= radius``.
    Each atom block is diagonalised independently.  The shared KKT multiplier
    is found by scalar bisection, including the classical negative-curvature
    hard case where the gradient is orthogonal to the leftmost eigenspace.
    """

    if not isinstance(radius, (int, float)) or isinstance(radius, bool):
        raise TypeError("radius must be a number")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if root_iterations <= 0:
        raise ValueError("root_iterations must be positive")

    linear, curvature = model.linear, model.curvature
    if scales is None:
        scales = torch.ones_like(linear)
    else:
        scales = torch.broadcast_to(scales.to(linear), linear.shape)
        if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
            raise ValueError("scales must be finite and positive")

    q_linear = linear * scales
    q_curvature = curvature * scales.unsqueeze(-1) * scales.unsqueeze(-2)
    q_curvature = 0.5 * (q_curvature + q_curvature.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(q_curvature)
    eigen_gradient = torch.einsum("kji,kj->ki", eigenvectors, q_linear)
    flat_values = eigenvalues.flatten()
    flat_gradient = eigen_gradient.flatten()
    dtype = linear.dtype
    finfo = torch.finfo(dtype)
    spectral_scale = flat_values.abs().amax().clamp_min(1.0)
    tolerance = 128.0 * finfo.eps * spectral_scale
    minimum = flat_values.amin()
    zero = minimum.new_zeros(())

    def coefficients(multiplier: Tensor) -> Tensor:
        denominator = flat_values + multiplier
        safe = denominator.abs() > tolerance
        return torch.where(
            safe, -flat_gradient / denominator, torch.zeros_like(denominator)
        )

    positive_definite = bool(minimum > tolerance)
    if positive_definite:
        unconstrained = coefficients(zero)
        if bool(torch.linalg.vector_norm(unconstrained) <= radius):
            q_step = torch.einsum(
                "kij,kj->ki", eigenvectors, unconstrained.reshape_as(eigenvalues)
            )
            step = scales * q_step
            return TrustRegionResult(
                step=step,
                lagrange_multiplier=zero,
                predicted_change=_quadratic_value(linear, curvature, step),
                hit_boundary=False,
            )

    lower = torch.maximum(zero, -minimum)
    lower_coefficients = coefficients(lower)
    lower_norm = torch.linalg.vector_norm(lower_coefficients)
    hard_mask = (flat_values + lower).abs() <= tolerance
    hard_gradient = flat_gradient.masked_select(hard_mask)
    hard_case = (
        bool(minimum < -tolerance)
        and bool(lower_norm < radius)
        and bool((hard_gradient.abs() <= tolerance).all())
    )
    if hard_case:
        remaining = (lower.new_tensor(radius**2) - lower_norm.square()).clamp_min(0)
        index = int(torch.nonzero(hard_mask, as_tuple=False)[0])
        lower_coefficients[index] = remaining.sqrt()
        q_step = torch.einsum(
            "kij,kj->ki", eigenvectors, lower_coefficients.reshape_as(eigenvalues)
        )
        step = scales * q_step
        return TrustRegionResult(
            step=step,
            lagrange_multiplier=lower,
            predicted_change=_quadratic_value(linear, curvature, step),
            hit_boundary=True,
        )

    # Move just inside the positive-definite side of a singular lower bound.
    low = lower + tolerance
    high = torch.maximum(low + 1.0, spectral_scale)
    while bool(torch.linalg.vector_norm(coefficients(high)) > radius):
        high = 2.0 * high + 1.0
    for _ in range(root_iterations):
        middle = 0.5 * (low + high)
        if bool(torch.linalg.vector_norm(coefficients(middle)) > radius):
            low = middle
        else:
            high = middle
    multiplier = high
    solution = coefficients(multiplier)
    q_step = torch.einsum("kij,kj->ki", eigenvectors, solution.reshape_as(eigenvalues))
    step = scales * q_step
    return TrustRegionResult(
        step=step,
        lagrange_multiplier=multiplier,
        predicted_change=_quadratic_value(linear, curvature, step),
        hit_boundary=True,
    )


class CSTP2TrustRegion(torch.optim.Optimizer):
    """One-site dense-free P2 model-EMA trust-region optimizer.

    The optimizer consumes a :class:`~torchcst.compute.CaptureBatch` produced
    by :meth:`capture_context`; it never reads or constructs a represented
    dense weight.  This first implementation intentionally rejects structural
    mutations between construction and update.  Call :meth:`reset_model_ema`
    immediately after a mutation before resuming training.
    """

    def __init__(
        self,
        module: CSTLinear,
        *,
        radius: float,
        beta: float = 0.9,
        amplitude_scale: float = 1.0,
        curvature_scale: float = 1.0,
        step_beta: float | None = None,
        adaptive_radius: bool = False,
        acceptance_threshold: float = 0.1,
        min_radius: float = 1e-6,
        max_radius: float | None = None,
        root_iterations: int = 80,
    ) -> None:
        if not isinstance(module, CSTLinear):
            raise TypeError("module must be a CSTLinear")
        if not isinstance(module.backend, Factored):
            raise ValueError("CSTP2TrustRegion requires an explicit Factored backend")
        if not isinstance(radius, (int, float)) or isinstance(radius, bool):
            raise TypeError("radius must be a number")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if not isinstance(amplitude_scale, (int, float)) or isinstance(
            amplitude_scale, bool
        ):
            raise TypeError("amplitude_scale must be a number")
        if amplitude_scale <= 0:
            raise ValueError("amplitude_scale must be positive")
        if not isinstance(curvature_scale, (int, float)) or isinstance(
            curvature_scale, bool
        ):
            raise TypeError("curvature_scale must be a number")
        if not math.isfinite(curvature_scale) or curvature_scale < 0:
            raise ValueError("curvature_scale must be finite and non-negative")
        if not isinstance(root_iterations, int) or isinstance(root_iterations, bool):
            raise TypeError("root_iterations must be an int")
        if root_iterations <= 0:
            raise ValueError("root_iterations must be positive")
        if not isinstance(adaptive_radius, bool):
            raise TypeError("adaptive_radius must be bool")
        if not isinstance(acceptance_threshold, (int, float)) or isinstance(
            acceptance_threshold, bool
        ):
            raise TypeError("acceptance_threshold must be a number")
        if not 0 <= acceptance_threshold < 1:
            raise ValueError("acceptance_threshold must lie in [0, 1)")
        if not isinstance(min_radius, (int, float)) or isinstance(min_radius, bool):
            raise TypeError("min_radius must be a number")
        if min_radius <= 0 or min_radius > radius:
            raise ValueError("min_radius must be positive and at most radius")
        if max_radius is None:
            max_radius = 16.0 * float(radius)
        if not isinstance(max_radius, (int, float)) or isinstance(max_radius, bool):
            raise TypeError("max_radius must be a number")
        if not math.isfinite(max_radius) or max_radius < radius:
            raise ValueError("max_radius must be finite and at least radius")
        # contracted_cst_linear_p2_model owns the remaining family checks.
        super().__init__(
            [module.synapses.w, module.synapses.s, module.synapses.t],
            defaults={},
        )
        self.module = module
        self.radius = float(radius)
        self.amplitude_scale = float(amplitude_scale)
        self.curvature_scale = float(curvature_scale)
        self.adaptive_radius = adaptive_radius
        self.acceptance_threshold = float(acceptance_threshold)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.root_iterations = root_iterations
        self.model_ema = P2ModelEMA(beta)
        self.step_ema = None if step_beta is None else P2StepEMA(step_beta)
        self.step_ema_fallbacks = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self._store_version = module.synapses.version
        self.last_result: TrustRegionResult | None = None

    def capture_context(self, update_id: int) -> BackwardContext:
        """Create the reduced-only backward context for one optimizer update."""

        return BackwardContext(
            update_id,
            reducers={self.module.capture_site: cst_linear_p2_reducer(self.module)},
            raw_sites=set(),
        )

    def _batch_model(self, capture: CaptureBatch) -> ContractedP2Model:
        if not isinstance(capture, CaptureBatch):
            raise TypeError("capture must be a CaptureBatch")
        if any(item.site == self.module.capture_site for item in capture.observations):
            raise ValueError("P2 capture must not retain raw x/g_out observations")
        observations = [
            item for item in capture.reduced if item.site == self.module.capture_site
        ]
        if not observations:
            raise ValueError("capture contains no reduced P2 observation for this site")
        if any(item.version != self._store_version for item in observations):
            raise RuntimeError("capture contains a stale P2 observation")
        if any(item.micro_weight < 0 for item in observations):
            raise ValueError("P2 microbatch weights must be non-negative")
        total_weight = sum(item.micro_weight for item in observations)
        if total_weight <= 0:
            raise ValueError("P2 microbatch weights must have a positive sum")
        first = observations[0].values
        if set(first) != {"linear", "curvature"}:
            raise ValueError("P2 reduced payload has unexpected fields")
        linear = torch.zeros_like(first["linear"])
        curvature = torch.zeros_like(first["curvature"])
        for observation in observations:
            values = observation.values
            if set(values) != {"linear", "curvature"}:
                raise ValueError("P2 reduced payload has unexpected fields")
            linear.add_(values["linear"], alpha=observation.micro_weight)
            curvature.add_(values["curvature"], alpha=observation.micro_weight)
        return ContractedP2Model(
            linear=linear / total_weight,
            curvature=curvature / total_weight,
        )

    def _scales(self, reference: Tensor) -> Tensor:
        sigma_in = self.module.factor_in.sigma.detach().to(reference)
        sigma_out = self.module.factor_out.sigma.detach().to(reference)
        if sigma_in.numel() != 1 or sigma_out.numel() != 1:
            raise NotImplementedError(
                "CSTP2TrustRegion currently requires scalar factor bandwidths"
            )
        fields = reference.shape[1]
        expected = 1 + self.module.synapses.d_in + self.module.synapses.d_out
        if fields != expected:
            raise RuntimeError("P2 model fields do not match the synapse geometry")
        return torch.cat(
            (
                reference.new_tensor([self.amplitude_scale]),
                sigma_in.reshape(1).expand(self.module.synapses.d_in),
                sigma_out.reshape(1).expand(self.module.synapses.d_out),
            )
        )

    @torch.no_grad()
    def step(
        self,
        capture: CaptureBatch,
        closure=None,
        *,
        current_loss: Tensor | float | None = None,
    ) -> TrustRegionResult:
        """EMA one captured model, solve it, and update live ``(w, s, t)``."""

        if self.adaptive_radius and closure is None:
            raise ValueError("adaptive trust regions require a loss closure")
        if closure is None and current_loss is not None:
            raise ValueError("current_loss is only useful with a loss closure")
        if self.module.synapses.version != self._store_version:
            raise RuntimeError(
                "synapse structure changed; call reset_model_ema() before step"
            )
        batch_model = self._batch_model(capture)
        if self.curvature_scale != 1.0:
            batch_model = ContractedP2Model(
                linear=batch_model.linear,
                curvature=batch_model.curvature * self.curvature_scale,
            )
        averaged = self.model_ema.update(batch_model)
        scales = self._scales(averaged.linear)
        radius_before = self.radius
        result = solve_block_trust_region(
            averaged,
            self.radius,
            scales=scales,
            root_iterations=self.root_iterations,
        )
        if bool(result.predicted_change > 0):
            raise RuntimeError("trust-region solver returned a predicted ascent step")
        if self.step_ema is not None:
            smoothed_step = self.step_ema.update(result.step, scales, self.radius)
            smoothed_prediction = _quadratic_value(
                averaged.linear, averaged.curvature, smoothed_step
            )
            if bool(smoothed_prediction <= 0):
                boundary_threshold = self.radius * (
                    1.0 - 32.0 * torch.finfo(scales.dtype).eps
                )
                result = TrustRegionResult(
                    step=smoothed_step,
                    lagrange_multiplier=result.lagrange_multiplier,
                    predicted_change=smoothed_prediction,
                    hit_boundary=bool(
                        torch.linalg.vector_norm(smoothed_step / scales)
                        >= boundary_threshold
                    ),
                )
            else:
                self.step_ema_fallbacks += 1

        store = self.module.synapses
        slots = store.live_slots().to(store.w.device)
        step = result.step.to(store.w)
        if step.shape[0] != slots.numel():
            raise RuntimeError("P2 step no longer aligns with live synapses")
        d_in = store.d_in
        if self.adaptive_radius:
            before_w = store.w.index_select(0, slots).clone()
            before_s = store.s.index_select(0, slots).clone()
            before_t = store.t.index_select(0, slots).clone()
            if current_loss is None:
                old_loss = torch.as_tensor(closure(), device=step.device).detach()
            else:
                old_loss = torch.as_tensor(current_loss, device=step.device).detach()
            if old_loss.numel() != 1 or not bool(torch.isfinite(old_loss)):
                raise ValueError("loss closure must return one finite scalar")
        store.w.index_add_(0, slots, step[:, 0])
        store.s.index_add_(0, slots, step[:, 1 : 1 + d_in])
        store.t.index_add_(0, slots, step[:, 1 + d_in :])
        store.retract_coordinates()
        if self.adaptive_radius:
            new_loss = torch.as_tensor(closure(), device=step.device).detach()
            if new_loss.numel() != 1 or not bool(torch.isfinite(new_loss)):
                store.w.index_copy_(0, slots, before_w)
                store.s.index_copy_(0, slots, before_s)
                store.t.index_copy_(0, slots, before_t)
                raise ValueError("loss closure must return one finite scalar")
            actual_change = new_loss - old_loss
            predicted_reduction = (-result.predicted_change).clamp_min(
                torch.finfo(result.predicted_change.dtype).tiny
            )
            ratio = (-actual_change) / predicted_reduction
            accepted = bool(actual_change < 0) and bool(
                ratio >= self.acceptance_threshold
            )
            if not accepted:
                store.w.index_copy_(0, slots, before_w)
                store.s.index_copy_(0, slots, before_s)
                store.t.index_copy_(0, slots, before_t)
                self.radius = max(self.min_radius, 0.5 * self.radius)
                self.rejected_steps += 1
            else:
                self.accepted_steps += 1
                if bool(ratio < 0.25):
                    self.radius = max(self.min_radius, 0.5 * self.radius)
                elif bool(ratio > 0.75) and result.hit_boundary:
                    self.radius = min(self.max_radius, 2.0 * self.radius)
            result = TrustRegionResult(
                step=result.step if accepted else torch.zeros_like(result.step),
                lagrange_multiplier=result.lagrange_multiplier,
                predicted_change=result.predicted_change,
                hit_boundary=result.hit_boundary,
                accepted=accepted,
                actual_change=actual_change,
                reduction_ratio=ratio,
                radius_before=radius_before,
                radius_after=self.radius,
            )
        self.last_result = result
        return result

    def reset_model_ema(self) -> None:
        """Accept the current structure as a new frame and clear old history."""

        self.model_ema.reset()
        if self.step_ema is not None:
            self.step_ema.reset()
        self.step_ema_fallbacks = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self._store_version = self.module.synapses.version
        self.last_result = None

    def state_dict(self) -> dict[str, object]:
        """Include the P2 model history and solver configuration."""

        state = super().state_dict()
        state["cst_p2"] = {
            "schema": "torchcst-p2-trust-region-v2",
            "radius": self.radius,
            "amplitude_scale": self.amplitude_scale,
            "curvature_scale": self.curvature_scale,
            "adaptive_radius": self.adaptive_radius,
            "acceptance_threshold": self.acceptance_threshold,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "root_iterations": self.root_iterations,
            "store_version": self._store_version,
            "model_ema": self.model_ema.state_dict(),
            "step_ema": None if self.step_ema is None else self.step_ema.state_dict(),
            "step_ema_fallbacks": self.step_ema_fallbacks,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore optimizer state only into the same live CST structure."""

        if not isinstance(state_dict, dict):
            raise TypeError("optimizer state must be a dict")
        state = dict(state_dict)
        p2 = state.pop("cst_p2", None)
        if not isinstance(p2, dict):
            raise ValueError("optimizer state has no CST P2 payload")
        if p2.get("schema") != "torchcst-p2-trust-region-v2":
            raise ValueError("unsupported CST P2 optimizer state schema")
        if p2.get("store_version") != self.module.synapses.version:
            raise ValueError("CST P2 state targets a different store version")
        super().load_state_dict(state)
        radius = p2.get("radius")
        amplitude_scale = p2.get("amplitude_scale")
        curvature_scale = p2.get("curvature_scale")
        adaptive_radius = p2.get("adaptive_radius")
        acceptance_threshold = p2.get("acceptance_threshold")
        min_radius = p2.get("min_radius")
        max_radius = p2.get("max_radius")
        root_iterations = p2.get("root_iterations")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("invalid CST P2 radius state")
        if not isinstance(amplitude_scale, (int, float)) or amplitude_scale <= 0:
            raise ValueError("invalid CST P2 amplitude scale state")
        if (
            not isinstance(curvature_scale, (int, float))
            or isinstance(curvature_scale, bool)
            or not math.isfinite(curvature_scale)
            or curvature_scale < 0
        ):
            raise ValueError("invalid CST P2 curvature scale state")
        if not isinstance(root_iterations, int) or root_iterations <= 0:
            raise ValueError("invalid CST P2 root iteration state")
        if not isinstance(adaptive_radius, bool):
            raise ValueError("invalid CST P2 adaptive radius state")
        if (
            not isinstance(acceptance_threshold, (int, float))
            or isinstance(acceptance_threshold, bool)
            or not 0 <= acceptance_threshold < 1
        ):
            raise ValueError("invalid CST P2 acceptance threshold state")
        if (
            not isinstance(min_radius, (int, float))
            or isinstance(min_radius, bool)
            or min_radius <= 0
        ):
            raise ValueError("invalid CST P2 minimum radius state")
        if (
            not isinstance(max_radius, (int, float))
            or isinstance(max_radius, bool)
            or max_radius < radius
        ):
            raise ValueError("invalid CST P2 maximum radius state")
        model_state = p2.get("model_ema")
        if not isinstance(model_state, dict):
            raise ValueError("invalid CST P2 model EMA state")
        restored = P2ModelEMA()
        restored.load_state_dict(model_state)
        step_state = p2.get("step_ema")
        if step_state is None:
            restored_step = None
        elif isinstance(step_state, dict):
            restored_step = P2StepEMA(0.0)
            restored_step.load_state_dict(step_state)
        else:
            raise ValueError("invalid CST P2 step EMA state")
        step_ema_fallbacks = p2.get("step_ema_fallbacks")
        if not isinstance(step_ema_fallbacks, int) or step_ema_fallbacks < 0:
            raise ValueError("invalid CST P2 step EMA fallback state")
        accepted_steps = p2.get("accepted_steps")
        rejected_steps = p2.get("rejected_steps")
        if not isinstance(accepted_steps, int) or accepted_steps < 0:
            raise ValueError("invalid CST P2 accepted step state")
        if not isinstance(rejected_steps, int) or rejected_steps < 0:
            raise ValueError("invalid CST P2 rejected step state")
        if restored.linear is not None:
            expected_atoms = self.module.synapses.live_slots().numel()
            expected_fields = 1 + self.module.synapses.d_in + self.module.synapses.d_out
            if restored.linear.shape != (expected_atoms, expected_fields):
                raise ValueError("CST P2 model EMA does not match live atoms")
            parameter = self.module.synapses.w
            restored.linear = restored.linear.to(parameter)
            assert restored.curvature is not None
            restored.curvature = restored.curvature.to(parameter)
        if restored_step is not None and restored_step.dimensionless_step is not None:
            expected_atoms = self.module.synapses.live_slots().numel()
            expected_fields = 1 + self.module.synapses.d_in + self.module.synapses.d_out
            if restored_step.dimensionless_step.shape != (
                expected_atoms,
                expected_fields,
            ):
                raise ValueError("CST P2 step EMA does not match live atoms")
            restored_step.dimensionless_step = restored_step.dimensionless_step.to(
                self.module.synapses.w
            )
        self.radius = float(radius)
        self.amplitude_scale = float(amplitude_scale)
        self.curvature_scale = float(curvature_scale)
        self.adaptive_radius = adaptive_radius
        self.acceptance_threshold = float(acceptance_threshold)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.root_iterations = root_iterations
        self.model_ema = restored
        self.step_ema = restored_step
        self.step_ema_fallbacks = step_ema_fallbacks
        self.accepted_steps = accepted_steps
        self.rejected_steps = rejected_steps
        self._store_version = self.module.synapses.version
        self.last_result = None
