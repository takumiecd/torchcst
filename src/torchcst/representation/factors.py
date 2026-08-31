"""Continuous factors for the general CST composition path.

Vocabulary: an atom has one *kernel* on the joint coordinate domain -- the
product of the input and output charts -- and ``W_ji = sum_k w_k
kappa(z_ji - p_k)`` with ``z_ji = (mu_i, mu_j)`` and ``p_k = (s_k, t_k)``.
The implementation never evaluates that joint kernel directly: every family
here is separable, and a *factor* is one side's share of the split,
``kappa(z - p) = factor_in(mu_i - s) * factor_out(mu_j - t)``.  Separability
is what the factored backends' ``O(K (n_in + n_out))`` matvec relies on, so a
non-separable joint kernel is a future backend decision, not a new factor
family in this module.

Delta and dot products are intentionally absent here: they are the implicit
specialized factors implemented by :class:`torchcst.compute.EntryLinear` and
:class:`torchcst.compute.RankOneLinear`, respectively.  Keeping those paths
specialized avoids materializing general factor matrices for discrete entry
and factorized rank-one families.

The factors defined here are *global-bandwidth isotropic* profiles of the
squared endpoint distance.  They differ only in that profile, so
:class:`ContinuousFactor` owns the bandwidth, the validation, and the distance
computation and each concrete family supplies one function.  The delta factor
of the entry family is the ``sigma -> 0`` limit of a compact profile, so
:class:`TriangularFactor` is the only member of this module that reaches the
entry family continuously; :class:`GaussianFactor` cannot, because its support
is the whole domain at every positive bandwidth.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import ClassVar

import torch
from torch import Tensor, nn

from torchcst._validation import require_real

from .._geometry import squared_distance_matrix
from .domains import ParameterRole

#: Factor names that :class:`torchcst.representation.RepresentationSpec`
#: accepts as a continuous family.  Kept here so the spec and the factor
#: modules cannot disagree about which families exist.
CONTINUOUS_FACTORS: frozenset[str] = frozenset(
    {
        "gaussian",
        "maturity_gaussian",
        "triangular",
        "weight_gated_gaussian",
    }
)

#: Live family registry, keyed by :attr:`ContinuousFactor.family`.  Populated
#: by ``__init_subclass__`` so a factor defined outside this module is a
#: first-class family: its name validates in a spec and its per-atom column
#: declaration reaches the store without any edit here.  ``CONTINUOUS_FACTORS``
#: stays the frozen built-in set for callers that import it.
_FACTOR_FAMILIES: dict[str, type[ContinuousFactor]] = {}


def continuous_family_names() -> frozenset[str]:
    """Every continuous family name a spec will accept right now."""
    return frozenset(CONTINUOUS_FACTORS) | frozenset(_FACTOR_FAMILIES)


def family_atom_columns(name: str) -> tuple[AtomColumn, ...]:
    """Per-atom columns the named family requires beyond ``(s, t, w)``."""
    factor = _FACTOR_FAMILIES.get(name)
    return () if factor is None else tuple(factor.atom_columns)


@dataclass(frozen=True)
class AtomColumn:
    """One per-atom column a factor family requires beyond ``(s, t, w)``.

    A family that parameterises each atom with more than a position and an
    amplitude -- a per-atom bandwidth, a Gabor frequency and phase -- declares
    those columns here.  :class:`~torchcst.storage.SynapseStore` then installs
    each one as a capacity-shaped slot column that follows birth, death,
    remap, and capacity growth exactly like ``s`` and ``t`` do, and every view
    and birth op carries it under :attr:`name`.

    ``width`` is either a literal column width or a callable of the site's
    ``(d_in, d_out)``, so a frequency vector conjugate to both charts is
    declared once and sized per site.  ``init`` is the value a birth that does
    *not* mention the column is given, which is what lets an existing policy
    keep proposing plain ``(s, t, w)`` births at a site whose factor has extra
    columns: the atom is simply born at the family's neutral value.
    """

    name: str
    width: int | Callable[[int, int], int] = 1
    role: ParameterRole = ParameterRole.PARAMETER
    init: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("AtomColumn.name must be a Python identifier")
        if self.name in {"s", "t", "w", "mass", "ids", "lineage"}:
            raise ValueError(
                f"AtomColumn.name {self.name!r} collides with a core column"
            )
        if not isinstance(self.role, ParameterRole):
            raise TypeError("AtomColumn.role must be a ParameterRole")

    def resolve_width(self, d_in: int, d_out: int) -> int:
        """The concrete column width at a site of these chart dimensions."""
        width = self.width(d_in, d_out) if callable(self.width) else self.width
        width = int(width)
        if width <= 0:
            raise ValueError(
                f"AtomColumn {self.name!r} resolved to a non-positive width"
            )
        return width


@dataclass(frozen=True)
class FactorState:
    """Row-aligned atom values delivered explicitly to one factor side.

    ``centers`` is the coordinate owned by this factor side. ``amplitude`` is
    the shared core amplitude of the complete CST atom, not an extra store
    column.  A factor may ignore it (ordinary Gaussian) or use it as a true
    differentiable input (weight-gated Gaussian). ``extras`` contains only
    factor-declared per-atom columns.

    Keeping the amplitude as a named field avoids smuggling a core CST value
    through the extensible column mapping and gives second-order consumers an
    unambiguous local field order: amplitude first, then center coordinates.
    """

    centers: Tensor
    amplitude: Tensor | None
    extras: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.centers, Tensor):
            raise TypeError("FactorState.centers must be a Tensor")
        if self.centers.ndim != 2:
            raise ValueError("FactorState.centers must be rank 2")
        if not self.centers.is_floating_point():
            raise TypeError("FactorState.centers must be floating point")
        if self.amplitude is not None:
            if not isinstance(self.amplitude, Tensor):
                raise TypeError("FactorState.amplitude must be a Tensor or None")
            if self.amplitude.ndim != 1:
                raise ValueError("FactorState.amplitude must be rank 1")
            if self.amplitude.shape[0] != self.centers.shape[0]:
                raise ValueError(
                    "FactorState.amplitude must have one value per center"
                )
            if not self.amplitude.is_floating_point():
                raise TypeError("FactorState.amplitude must be floating point")
        if not isinstance(self.extras, Mapping):
            raise TypeError("FactorState.extras must be a mapping")
        checked: dict[str, Tensor] = {}
        for name, value in self.extras.items():
            if not isinstance(name, str) or not name:
                raise ValueError("FactorState extra names must be non-empty strings")
            if not isinstance(value, Tensor):
                raise TypeError(f"FactorState extra {name!r} must be a Tensor")
            if value.ndim != 2 or value.shape[0] != self.centers.shape[0]:
                raise ValueError(
                    f"FactorState extra {name!r} must have one row per center"
                )
            checked[name] = value
        object.__setattr__(self, "extras", checked)


@dataclass(frozen=True)
class FactorJet2:
    """A delivered column and its first two local derivatives.

    Shapes are ``[neurons, atoms]``, ``[neurons, atoms, q]`` and
    ``[neurons, atoms, q, q]``. The local field order is
    ``(amplitude, *center)``.  Input/output jets stay separate; the CST map
    composes them with the product rule to obtain same-atom mixed blocks.
    """

    value: Tensor
    jacobian: Tensor
    hessian: Tensor

    def __post_init__(self) -> None:
        if self.value.ndim != 2:
            raise ValueError("FactorJet2.value must have shape [neurons, atoms]")
        if self.jacobian.ndim != 3:
            raise ValueError(
                "FactorJet2.jacobian must have shape [neurons, atoms, fields]"
            )
        if self.hessian.ndim != 4:
            raise ValueError(
                "FactorJet2.hessian must have shape "
                "[neurons, atoms, fields, fields]"
            )
        expected_jacobian = (*self.value.shape, self.jacobian.shape[-1])
        if self.jacobian.shape != expected_jacobian:
            raise ValueError(
                f"FactorJet2.jacobian must have shape {expected_jacobian}"
            )
        expected_hessian = (
            *self.value.shape,
            self.jacobian.shape[-1],
            self.jacobian.shape[-1],
        )
        if self.hessian.shape != expected_hessian:
            raise ValueError(
                f"FactorJet2.hessian must have shape {expected_hessian}"
            )


def differentiate_columns(
    evaluate: Callable[[FactorState], Tensor],
    state: FactorState,
) -> FactorJet2:
    """Differentiate a pure delivered-column function atom by atom.

    This is the correctness fallback for factors and gauges. Analytic or
    fused implementations may override it without changing the public jet
    contract. Factor extras are row-aligned constants in this first P2 field
    layout; the optimized CST path currently supports core ``(w, s, t)``.
    """

    if state.amplitude is None:
        raise ValueError("second-order column jets require explicit amplitudes")
    theta = torch.cat((state.amplitude[:, None], state.centers), dim=1)
    extra_names = tuple(state.extras)
    extra_values = tuple(state.extras[name] for name in extra_names)

    def one_atom(local_theta: Tensor, *local_extras: Tensor) -> Tensor:
        local_state = FactorState(
            centers=local_theta[1:][None, :],
            amplitude=local_theta[:1],
            extras={
                name: value[None, :]
                for name, value in zip(extra_names, local_extras, strict=True)
            },
        )
        value = evaluate(local_state)
        if value.ndim != 2 or value.shape[1] != 1:
            raise ValueError("column evaluator must return one factor column")
        return value[:, 0]

    jacobian_fn = torch.func.jacfwd(one_atom, argnums=0)
    hessian_fn = torch.func.jacfwd(jacobian_fn, argnums=0)
    value = torch.vmap(one_atom)(theta, *extra_values).transpose(0, 1)
    jacobian = torch.vmap(jacobian_fn)(theta, *extra_values).permute(1, 0, 2)
    hessian = torch.vmap(hessian_fn)(theta, *extra_values).permute(1, 0, 2, 3)
    return FactorJet2(value=value, jacobian=jacobian, hessian=hessian)


class ContinuousFactor(nn.Module):
    """Global-bandwidth isotropic factor over a squared endpoint distance.

    The scalar ``sigma`` is an ``nn.Parameter`` when ``learnable=True`` and a
    buffer otherwise.  Per-atom bandwidths are outside step 9; a future v0.3
    implementation may revive the old ``SynapseStore.add_extra`` design for
    that purpose.

    ``forward`` validates that ``sigma`` is finite and positive after every
    mutation by default (``validate_sigma=True``, construction time is always
    validated regardless of this flag).  The successful reading is cached by
    tensor identity, mutation version, device, and dtype, so repeated use in a
    forward/evaluation window pays no device-to-host traffic.  The combined
    guard costs one host sync when it does run; ``torch._assert_async`` would
    avoid it but corrupts the CUDA context on failure, too weak a contract for
    a general library.  Pass
    ``validate_sigma=False`` only when a caller can prove sigma is positive
    and finite by construction on every write after ``__init__`` too (e.g.
    it is always written as ``exp(x)`` for some finite real ``x``, and
    nothing else ever touches the buffer/parameter).

    Subclasses set :attr:`family` to the name their representation spec
    declares and implement :meth:`profile`.
    """

    #: Spec-level family name; concrete factors must override it.
    family: ClassVar[str] = ""

    #: Per-atom columns this family needs beyond ``(s, t, w)``.  Empty for
    #: every family whose atom is fully described by a position and an
    #: amplitude, which is why declaring nothing leaves the store, the ops,
    #: and the views byte-for-byte as they were.
    atom_columns: ClassVar[tuple[AtomColumn, ...]] = ()

    #: Whether factor columns depend on the core synapse amplitude. Such a
    #: family still adds no atom column: callers deliver the live amplitude
    #: explicitly in :class:`FactorState`.
    amplitude_dependent: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        family = getattr(cls, "family", "")
        if family:
            _FACTOR_FAMILIES[family] = cls

    def __init__(
        self,
        sigma: float | Tensor,
        learnable: bool = True,
        *,
        validate_sigma: bool = True,
    ) -> None:
        super().__init__()
        if not self.family:
            raise TypeError("ContinuousFactor subclasses must define a family name")
        if not isinstance(learnable, bool):
            raise TypeError("learnable must be a bool")
        if not isinstance(validate_sigma, bool):
            raise TypeError("validate_sigma must be a bool")
        if isinstance(sigma, bool):
            raise TypeError("sigma must be a real scalar")
        value = torch.as_tensor(sigma)
        if value.numel() != 1 or value.is_complex():
            raise ValueError("sigma must be a real scalar")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        value = value.detach().reshape(()).clone()
        reading = float(value)
        if not isfinite(reading) or reading <= 0:
            raise ValueError("sigma must be finite and positive")
        if learnable:
            self.sigma = nn.Parameter(value)
        else:
            self.register_buffer("sigma", value)
        # Per-call sigma re-validation flag; the contract and the opt-out
        # conditions live in the class docstring.
        self._validate_sigma = validate_sigma
        self._validated_sigma_signature: tuple[object, ...] | None = None

    @property
    def learnable(self) -> bool:
        return isinstance(self.sigma, nn.Parameter)

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """Map squared endpoint distances to factor values."""
        raise NotImplementedError

    def profile_grad(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """``d profile / d squared_distance``.

        The family-dependent half of a coordinate Jacobian: for an atom at
        ``c`` read by a neuron at ``x``, ``d kappa / d c = profile_grad *
        2 (c - x)``, so a caller needing ``||d kappa / d c||^2`` forms
        ``4 * profile_grad**2 * squared_distance`` and never has to know
        which family it is holding.  :class:`~torchcst.optim.
        CoordPreconditioner` is the in-tree consumer.

        Families that cannot be differentiated in closed form leave this
        unimplemented; the consumer then raises instead of silently
        substituting a Gaussian derivative.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement profile_grad(); a "
            "consumer that needs coordinate derivatives cannot use this "
            "factor family"
        )

    def scaled_profile_pair(
        self, squared_distance: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Profile and its distance derivative under one shared column scale.

        Each returned column may be multiplied by an arbitrary positive
        factor, but the value and derivative must carry the *same* factor.
        Consumers project the derivative off the column before dividing by
        its norm, so both that factor and its position derivative disappear.
        This is the derivative-side counterpart of :meth:`scaled_columns`.

        The generic form is sufficient for bounded profiles.  Exponential
        families override it to move their range-restoring shift inside the
        exponent before either value is formed.
        """
        return (
            self.profile(squared_distance, sigma),
            self.profile_grad(squared_distance, sigma),
        )

    def scaled_column_pair(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Stable columns and ``d value / d squared_distance`` together.

        This is the column-aware form consumed by pullback metrics.  Radial
        global-bandwidth families inherit the generic implementation;
        per-atom-bandwidth families override it so the value and derivative
        see the same row-aligned atom columns.
        """
        sigma, centers = self._prepare(query, centers, columns)
        squared_distance = squared_distance_matrix(query, centers)
        return self.scaled_profile_pair(squared_distance, sigma)

    def overlap(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """Normalised atom-atom overlap ``<kappa_j, kappa_k> / ||kappa||^2``.

        The geometry factor ``rho_jk`` that twin-control courts price with:
        ``1`` for coincident atoms, decaying to ``0`` as they separate.  It is
        the *continuous* inner product of two factor bumps, not a sampled
        Gram -- courts that can afford the sampled version build a
        :class:`~torchcst.representation.gram.GramService` from delivered
        factor columns instead, and are factor-agnostic already.

        Only families with a closed-form self-correlation implement it.  The
        radial tent's is not elementary in general dimension, so
        :class:`TriangularFactor` deliberately inherits the raise: a court
        asked to price triangular twins must fail loudly rather than quote
        Gaussian numbers for a non-Gaussian site.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement overlap(); a court "
            "that prices geometric twin overlap cannot use this factor family"
        )

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """:meth:`forward`'s columns, each up to its own positive scale.

        Same shape and same direction as :meth:`forward`, but every column is
        free to carry an arbitrary positive factor, because each caller
        divides that factor back out -- ``L2NormalizedColumns`` normalises to
        unit L2 norm.  Giving up the scale buys range: an atom far from every
        neuron casts a column whose values underflow, and a column that has
        already flattened to zero cannot be rescaled back.  In bf16/fp16 a
        Gaussian starts losing columns around four sigma, well inside a
        lawful chart, so this is not a corner case.

        The generic form below divides by each column's largest magnitude,
        which is exact for any family whose values survive being computed at
        all -- the bounded profiles need nothing more.  A family built on an
        exponential must override and fold the shift *inside* the exponent,
        the same move softmax makes with its maximum.

        A compactly supported family may return an honestly zero column, for
        an atom outside every neuron's support.  That is a fact about the
        geometry rather than a numerical failure, and callers clamp instead
        of dividing by it.
        """
        return peak_scaled(self(query, centers, columns))

    def columns(self, query: Tensor, state: FactorState) -> Tensor:
        """Evaluate columns from an explicit row-aligned factor state."""

        if not isinstance(state, FactorState):
            raise TypeError("state must be a FactorState")
        return self(query, state.centers, state.extras)

    def stable_columns(self, query: Tensor, state: FactorState) -> Tensor:
        """Evaluate scale-free stable columns from explicit factor state."""

        if not isinstance(state, FactorState):
            raise TypeError("state must be a FactorState")
        return self.scaled_columns(query, state.centers, state.extras)

    def jet2(self, query: Tensor, state: FactorState) -> FactorJet2:
        """Return columns and derivatives in ``(amplitude, *center)`` order."""

        return differentiate_columns(
            lambda local: self.columns(query, local), state
        )

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Factor matrix between ``query`` rows and atom ``centers``.

        ``columns`` delivers the family's declared per-atom values
        (:attr:`atom_columns`), row-aligned with ``centers``: an atom's
        frequency or bandwidth cannot be recovered from its position, so a
        family that declares columns is handed them here rather than digging
        into the store.  The isotropic families ignore the argument, which is
        why every existing caller and subclass keeps working untouched -- a
        family that needs the values overrides this method.
        """
        sigma, centers = self._prepare(query, centers, columns)
        squared_distance = squared_distance_matrix(query, centers)
        return self.profile(squared_distance, sigma)

    def _prepare(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Validate a factor call; return ``sigma`` and device-matched centers.

        Split out of :meth:`forward` so a family that does not go through
        :meth:`profile` -- :class:`GaborFactor` -- still gets one shared
        validation path instead of a second, drifting copy.
        """
        if columns:
            for name, value in columns.items():
                if not isinstance(value, Tensor):
                    raise TypeError(f"column {name!r} must be a Tensor")
                if value.ndim != 2 or value.shape[0] != centers.shape[0]:
                    raise ValueError(
                        f"column {name!r} must have one row per center "
                        f"({centers.shape[0]}), got {tuple(value.shape)}"
                    )
        if not isinstance(query, Tensor) or not isinstance(centers, Tensor):
            raise TypeError("query and centers must be Tensors")
        if query.ndim != 2 or centers.ndim != 2:
            raise ValueError("query and centers must be rank-2 Tensors")
        if query.shape[1] != centers.shape[1]:
            raise ValueError("query and centers must share their coordinate dimension")
        if not query.is_floating_point() or not centers.is_floating_point():
            raise TypeError("continuous coordinates must have floating dtypes")
        sigma = self.sigma.to(device=query.device, dtype=query.dtype)
        if self._validate_sigma:
            signature = (
                id(self.sigma), self.sigma._version, sigma.device, sigma.dtype
            )
            if self._validated_sigma_signature != signature and not bool(
                torch.isfinite(sigma) & (sigma > 0)
            ):
                raise ValueError("sigma must remain finite and positive")
            self._validated_sigma_signature = signature
        return sigma, centers.to(device=query.device, dtype=query.dtype)


class GaussianFactor(ContinuousFactor):
    """Global-bandwidth isotropic Gaussian factor.

    ``kappa(query, centers) = exp(-||query-center||^2 / (2 sigma^2))``.  Its
    support is the whole domain for every positive ``sigma``: the tail never
    vanishes, which keeps a position gradient alive everywhere but also means
    the represented matrix is never structurally sparse.
    """

    family: ClassVar[str] = "gaussian"

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        return torch.exp(-squared_distance / (2.0 * sigma.square()))

    def profile_grad(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        return -self.profile(squared_distance, sigma) / (2.0 * sigma.square())

    def scaled_profile_pair(
        self, squared_distance: Tensor, sigma: Tensor
    ) -> tuple[Tensor, Tensor]:
        nearest = squared_distance.amin(dim=0, keepdim=True)
        value = torch.exp(
            -(squared_distance - nearest) / (2.0 * sigma.square())
        )
        return value, -value / (2.0 * sigma.square())

    def overlap(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        # <G_j, G_k> / ||G||^2 for two unit-height Gaussians of one
        # bandwidth: the correlation widens the variance to 2 sigma^2.
        return torch.exp(-squared_distance / (4.0 * sigma.square()))

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        # Subtracting each column's nearest squared distance inside the
        # exponent is exact -- it scales the column by exp(+d_min^2/2 sigma^2)
        # -- and it means the largest entry is always exactly one, so no
        # column ever underflows regardless of how far the atom has walked.
        sigma, centers = self._prepare(query, centers, columns)
        squared_distance = squared_distance_matrix(query, centers)
        nearest = squared_distance.amin(dim=0, keepdim=True)
        return torch.exp(-(squared_distance - nearest) / (2.0 * sigma.square()))


class WeightGatedGaussianFactor(ContinuousFactor):
    """Gaussian with a smooth amplitude-controlled precision.

    The commitment gate is

    ``g(w) = sigmoid((log(w^2 + eps^2) - 2 log(tau)) / temperature)``

    and precision interpolates linearly from the exploratory to the narrow
    Gaussian.  ``sigma_explore=inf`` is lawful and gives an exactly uniform
    column when the gate reaches zero.  The squared smooth magnitude makes
    the represented map differentiable at ``w=0``.

    Set ``gated=False`` to use the same representation family with an ordinary
    fixed Gaussian on one side of a CST map.  This permits, for example, a
    fixed input factor and weight-gated output factor while the store keeps
    one unambiguous family name.
    """

    family: ClassVar[str] = "weight_gated_gaussian"
    amplitude_dependent: ClassVar[bool] = True

    def __init__(
        self,
        sigma_narrow: float | Tensor,
        learnable: bool = False,
        *,
        tau: float = 0.005,
        temperature: float = 0.25,
        gate_eps: float = 1e-12,
        sigma_explore: float = math.inf,
        gated: bool = True,
        validate_sigma: bool = True,
    ) -> None:
        for value, name in (
            (tau, "tau"),
            (temperature, "temperature"),
            (gate_eps, "gate_eps"),
        ):
            require_real(value, name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(sigma_explore, bool) or not isinstance(
            sigma_explore, (int, float)
        ):
            raise TypeError("sigma_explore must be a real number")
        if math.isnan(float(sigma_explore)) or float(sigma_explore) <= 0.0:
            raise ValueError("sigma_explore must be positive or infinity")
        if not isinstance(gated, bool):
            raise TypeError("gated must be a bool")
        super().__init__(
            sigma_narrow,
            learnable,
            validate_sigma=validate_sigma,
        )
        if float(sigma_explore) <= float(self.sigma.detach()):
            raise ValueError("sigma_explore must exceed sigma_narrow")
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.gate_eps = float(gate_eps)
        self.sigma_explore = float(sigma_explore)
        self.gated = gated

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError(
            "WeightGatedGaussianFactor needs each atom's core amplitude"
        )

    def gate(self, amplitude: Tensor) -> Tensor:
        """Return the smooth commitment gate with amplitude's shape."""

        log_magnitude_sq = (amplitude.square() + self.gate_eps**2).log()
        logit = (
            log_magnitude_sq - 2.0 * math.log(self.tau)
        ) / self.temperature
        return torch.sigmoid(logit)

    def precision(self, amplitude: Tensor) -> Tensor:
        """Return one inverse-variance value per amplitude."""

        sigma = self.sigma.to(amplitude)
        if not self.gated:
            return torch.ones_like(amplitude) / sigma.square()
        narrow = sigma.reciprocal().square()
        explore = (
            amplitude.new_zeros(())
            if math.isinf(self.sigma_explore)
            else amplitude.new_tensor(self.sigma_explore).reciprocal().square()
        )
        return explore + (narrow - explore) * self.gate(amplitude)

    def effective_sigma(self, amplitude: Tensor) -> Tensor:
        """Physical per-atom bandwidth, including an infinite explorer."""

        return self.precision(amplitude).rsqrt()

    def _state_components(
        self, query: Tensor, state: FactorState
    ) -> tuple[Tensor, Tensor]:
        if state.amplitude is None:
            raise ValueError("WeightGatedGaussianFactor needs atom amplitudes")
        _, centers = self._prepare(query, state.centers, state.extras)
        amplitude = state.amplitude.to(centers).reshape(1, -1)
        squared_distance = squared_distance_matrix(query, centers)
        return squared_distance, self.precision(amplitude)

    def columns(self, query: Tensor, state: FactorState) -> Tensor:
        squared_distance, precision = self._state_components(query, state)
        return torch.exp(-0.5 * precision * squared_distance)

    def stable_columns(self, query: Tensor, state: FactorState) -> Tensor:
        squared_distance, precision = self._state_components(query, state)
        nearest = squared_distance.amin(dim=0, keepdim=True)
        return torch.exp(-0.5 * precision * (squared_distance - nearest))

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        raise TypeError(
            "WeightGatedGaussianFactor requires columns(query, FactorState)"
        )

    def scaled_column_pair(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise TypeError(
            "WeightGatedGaussianFactor requires stable_columns(query, FactorState)"
        )

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        raise TypeError(
            "WeightGatedGaussianFactor requires stable_columns(query, FactorState)"
        )


class MaturityGaussianFactor(ContinuousFactor):
    """Gaussian whose inverse width is an independent per-atom maturity.

    ``maturity`` is an unconstrained logit.  Its sigmoid interpolates squared
    inverse width between ``min_scale**2`` and ``max_scale**2``.  A negative
    birth logit therefore casts a broad exploratory column, while increasing
    maturity continuously recovers the ordinary Gaussian when
    ``max_scale=1``.
    """

    family: ClassVar[str] = "maturity_gaussian"
    atom_columns: ClassVar[tuple[AtomColumn, ...]] = (
        AtomColumn("maturity", width=1, init=-2.0),
    )

    def __init__(
        self,
        sigma: float | Tensor,
        learnable: bool = True,
        *,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
        validate_sigma: bool = True,
    ) -> None:
        for value, name in ((min_scale, "min_scale"), (max_scale, "max_scale")):
            require_real(value, name)
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not float(min_scale) < float(max_scale):
            raise ValueError("min_scale must be smaller than max_scale")
        super().__init__(sigma, learnable, validate_sigma=validate_sigma)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError(
            "MaturityGaussianFactor needs each atom's maturity column"
        )

    def inverse_width_scale_sq(self, maturity: Tensor) -> Tensor:
        """Squared inverse-width multiplier represented by ``maturity``.

        This small public reading keeps deterministic bandwidth controllers
        and trust-region diagnostics on exactly the same parameterisation as
        the factor's forward path.  The returned shape matches ``maturity``.
        """
        fraction = torch.sigmoid(maturity)
        return self.min_scale**2 + (
            self.max_scale**2 - self.min_scale**2
        ) * fraction

    def effective_sigma(self, maturity: Tensor) -> Tensor:
        """Per-atom physical bandwidth induced by ``maturity``."""
        sigma = self.sigma.to(device=maturity.device, dtype=maturity.dtype)
        return sigma / self.inverse_width_scale_sq(maturity).sqrt()

    def _components(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not columns or "maturity" not in columns:
            raise ValueError(
                "MaturityGaussianFactor needs the 'maturity' atom column"
            )
        sigma, centers = self._prepare(query, centers, columns)
        maturity = columns["maturity"].to(centers).reshape(1, -1)
        scale_sq = self.inverse_width_scale_sq(maturity)
        squared_distance = squared_distance_matrix(query, centers)
        return squared_distance, sigma, scale_sq

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        squared_distance, sigma, scale_sq = self._components(
            query, centers, columns
        )
        return torch.exp(
            -scale_sq * squared_distance / (2.0 * sigma.square())
        )

    def scaled_column_pair(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        squared_distance, sigma, scale_sq = self._components(
            query, centers, columns
        )
        nearest = squared_distance.amin(dim=0, keepdim=True)
        value = torch.exp(
            -scale_sq * (squared_distance - nearest)
            / (2.0 * sigma.square())
        )
        derivative = -scale_sq * value / (2.0 * sigma.square())
        return value, derivative

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        return self.scaled_column_pair(query, centers, columns)[0]


class TriangularFactor(ContinuousFactor):
    """Global-bandwidth isotropic triangular factor with compact support.

    ``kappa(query, centers) = relu(1 - ||query-center|| / sigma)``: exactly
    zero outside the radius-``sigma`` ball, so the represented matrix is
    structurally sparse and ``sigma -> 0`` approaches the entry family's delta
    factor.  The profile is radial rather than a per-axis product, mirroring
    the Gaussian's isotropy; unlike the Gaussian it does *not* factor over
    coordinate axes.

    Two non-smooth points are handled rather than smoothed away.  The knee at
    ``r == sigma`` is a subgradient in the same sense as ``relu``.  At
    ``r == 0`` the Euclidean norm has no gradient, so the squared distance is
    floored at the smallest positive normal of its dtype before the square
    root: below that floor ``clamp_min`` passes zero gradient, which is the
    correct subgradient at a coincident endpoint, and above it the ratio
    ``|q - c| / r`` stays bounded by one.  The floor perturbs the returned
    value by ``sqrt(tiny) / sigma``, far below every supported dtype's
    resolution.
    """

    family: ClassVar[str] = "triangular"

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        floor = torch.finfo(squared_distance.dtype).tiny
        distance = squared_distance.clamp_min(floor).sqrt()
        return torch.relu(1.0 - distance / sigma)

    def profile_grad(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        # d/d(r^2) relu(1 - r/sigma) = -1/(2 sigma r) inside the support.
        # The 1/r pole at a coincident endpoint is the one ``profile``
        # already floors, and it cancels in the ``4 g^2 r^2`` form every
        # in-tree consumer uses.
        floor = torch.finfo(squared_distance.dtype).tiny
        distance = squared_distance.clamp_min(floor).sqrt()
        inside = (distance < sigma).to(squared_distance.dtype)
        return -inside / (2.0 * sigma * distance)

    # ``overlap`` is deliberately not implemented: the radial tent's
    # self-correlation is not elementary in general dimension.


class GaborFactor(ContinuousFactor):
    """Gaussian envelope times a per-atom plane wave: an oscillating atom.

    ``kappa(x, c) = exp(-||x-c||^2 / 2 sigma^2) * cos(omega . (x - c) + phi)``
    with ``omega`` and ``phi`` carried per atom.  At ``omega == 0`` the column
    is ``cos(phi)`` times the Gaussian -- the same shape, and exactly
    :class:`GaussianFactor` when the phase is zero too -- so a Gabor site
    starts as an ordinary CST site and can only gain from there.  It does not
    *start* at zero phase, though; see :attr:`atom_columns` for the critical
    point that forces a quarter turn.

    The point of the extra parameters is what a purely positive bump cannot
    write.  Structure finer than ``sigma`` is representable in the Gaussian
    dictionary only as the near-cancellation of two large opposite atoms --
    the twin pairs whose amplitudes grow like ``exp((sigma/l)^2 / 2)`` -- and
    that is a way of forging an oscillation the dictionary does not stock.
    Stocking it makes the same structure an ``O(1)`` single atom.  The band
    this opens runs from ``1/sigma`` up to the chart's Nyquist ``pi/spacing``:
    a frequency the neuron grid cannot resolve buys nothing.

    ``side`` selects which chart this instance reads, because the composition
    ``W = K_out diag(w) K_in^T`` is a product of two per-side factors and a
    jointly modulated 2-D Gabor does not factor through it.  Each side
    therefore carries its own frequency and phase, and the represented atom is
    the separable product of two 1-D Gabors -- oscillation along the input
    chart, the output chart, or both.

    Two inherited raises are deliberate.  There is no radial ``profile``: the
    value depends on the *direction* of ``x - c`` (and on the phase), which a
    squared distance has already discarded, so consumers that reduce a factor
    to its radial profile -- ``CoordPreconditioner``, the closed-form
    backends -- refuse this family rather than silently using the envelope.
    And ``overlap`` stays unimplemented because two Gabor atoms at the same
    place with different frequencies are nearly orthogonal, not twins: an
    absorb court keyed on distance alone would merge distinct atoms, so it
    must fail loudly until it prices the full ``(mu, omega)`` address.
    """

    family: ClassVar[str] = "gabor"
    #: ``phi`` is born at a quarter turn, not at zero, and E-omega is why.
    #: At ``(omega, phi) == (0, 0)`` the derivative of the factor column with
    #: respect to *both* new parameters vanishes identically -- ``sin(0)`` at
    #: every displacement -- so that point is not a stationary point of some
    #: loss but a critical point of the parameterisation itself: no target and
    #: no position can produce a gradient there, and an atom born there stays
    #: at zero frequency forever.  A quarter turn costs nothing (at
    #: ``omega == 0`` the column is ``cos(phi)`` times the Gaussian, the same
    #: shape with a constant the amplitude absorbs) and restores the gradient
    #: as soon as the atom is not exactly centred on an even feature.
    #: Measured: ``|dL/domega|`` goes 0 -> 1.9e-2 at a 0.2-sigma offset, and
    #: Adam then carries omega from 0 to within 5% of a target frequency it
    #: could not previously see.
    atom_columns: ClassVar[tuple[AtomColumn, ...]] = (
        AtomColumn("omega_s", width=lambda d_in, _d_out: d_in),
        AtomColumn("phi_s", width=1, init=math.pi / 4),
        AtomColumn("omega_t", width=lambda _d_in, d_out: d_out),
        AtomColumn("phi_t", width=1, init=math.pi / 4),
    )

    def __init__(
        self,
        sigma: float | Tensor,
        learnable: bool = True,
        *,
        side: str = "in",
        validate_sigma: bool = True,
    ) -> None:
        if side not in ("in", "out"):
            raise ValueError('side must be "in" or "out"')
        super().__init__(sigma, learnable, validate_sigma=validate_sigma)
        self.side = side

    @property
    def column_names(self) -> tuple[str, str]:
        """The ``(frequency, phase)`` column names this instance reads."""
        return ("omega_s", "phi_s") if self.side == "in" else ("omega_t", "phi_t")

    def profile(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError(
            "GaborFactor has no radial profile: its value depends on the "
            "direction of the displacement and on the atom's phase, both of "
            "which a squared distance has discarded"
        )

    def envelope(self, squared_distance: Tensor, sigma: Tensor) -> Tensor:
        """The Gaussian envelope alone, without the modulation."""
        return torch.exp(-squared_distance / (2.0 * sigma.square()))

    def _components(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``(squared_distance, angle, sigma)`` -- what both column forms share."""
        frequency_name, phase_name = self.column_names
        if not columns or frequency_name not in columns or phase_name not in columns:
            raise ValueError(
                f"GaborFactor(side={self.side!r}) needs the {frequency_name!r} "
                f"and {phase_name!r} columns; the site's store must declare "
                "them (spec factor 'gabor') and the caller must deliver them"
            )
        sigma, centers = self._prepare(query, centers, columns)
        # The [N, K, d] displacement cube is unavoidable here -- the phase is
        # a signed inner product, not a function of the distance -- so this
        # family costs d times the isotropic families' factor memory.
        displacement = query[:, None, :] - centers[None, :, :]
        squared_distance = displacement.square().sum(-1)
        frequency = columns[frequency_name].to(displacement)
        phase = columns[phase_name].to(displacement)
        angle = (displacement * frequency[None, :, :]).sum(-1) + phase.reshape(1, -1)
        return squared_distance, angle, sigma

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        squared_distance, angle, sigma = self._components(query, centers, columns)
        return self.envelope(squared_distance, sigma) * torch.cos(angle)

    def scaled_columns(
        self,
        query: Tensor,
        centers: Tensor,
        columns: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        # The envelope carries the whole dynamic range, so shifting it is
        # enough; the fringe is already O(1).  The resulting column's peak is
        # not one -- cos may be small where the envelope is largest -- which
        # the contract allows, because the scale is divided out downstream.
        squared_distance, angle, sigma = self._components(query, centers, columns)
        nearest = squared_distance.amin(dim=0, keepdim=True)
        return self.envelope(squared_distance - nearest, sigma) * torch.cos(angle)



def peak_scaled(values: Tensor) -> Tensor:
    """Divide each column by its largest magnitude, leaving zero columns zero."""
    peak = values.abs().amax(dim=0, keepdim=True)
    return values / peak.clamp_min(torch.finfo(values.dtype).tiny)


#: What a twin-control court may price geometric overlap with: the site's own
#: factor (family-generic) or a bare Gaussian bandwidth (explicitly Gaussian).
OverlapScale = ContinuousFactor | float


def pairwise_overlap(scale: OverlapScale, squared_distance: Tensor) -> Tensor:
    """Geometry overlap ``rho`` for squared coordinate distances.

    ``scale`` is either the site's :class:`ContinuousFactor` -- the
    family-generic form, which delegates to :meth:`ContinuousFactor.overlap`
    and therefore *raises* rather than misprice a family with no closed-form
    self-correlation -- or a bare positive bandwidth, which selects the
    Gaussian ``exp(-d^2 / 4 sigma^2)`` explicitly.

    The float form is kept because these courts are configured with a scale
    rather than bound to a compute module (``propose`` sees only a view), and
    because it is the form every registered experiment used.  It is now an
    explicit request for Gaussian geometry, not an implicit assumption: a
    non-Gaussian site should hand the court its factor instead.
    """
    if isinstance(scale, ContinuousFactor):
        sigma = scale.sigma.detach().to(squared_distance)
        return scale.overlap(squared_distance, sigma)
    return torch.exp(-squared_distance / (4.0 * float(scale) ** 2))


def require_overlap_scale(value: object, name: str) -> OverlapScale:
    """Validate a court's overlap scale: a factor, or a positive real."""
    if isinstance(value, ContinuousFactor):
        return value
    return require_real(value, name, positive=True)
