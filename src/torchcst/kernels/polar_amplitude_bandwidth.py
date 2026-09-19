"""Polar amplitude/width coordinates with radial-angular decoupling."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel, Profile
from .gaussian import Gaussian


class PolarAmpWidth(Kernel):
    r"""A rank-one atom with angular amplitude and radial width control.

    The first two coordinates of each atom are ``(s, t)``.  They encode

    .. math::

        w = W s / \sqrt{s^2+t^2}, \qquad
        \alpha = \operatorname{clamp}((s^2+t^2-1)/3, 0, 1).

    Consequently ``w`` is invariant to radial rescaling while ``alpha`` is
    invariant to angular motion.  The shared input/output bandwidth is
    ``(1-alpha) L(w) + alpha U(w)``, where ``L`` and ``U`` meet at
    ``sigma_max`` for zero amplitude and approach ``sigma_min`` as the
    amplitude scale grows.

    The hard clamp keeps every forward bandwidth inside its configured
    bounds under arbitrary optimizers.  Coordinates are initialized on the
    unit circle, so ``alpha == 0`` initially.
    """

    def __init__(
        self,
        *,
        amplitude_max: float,
        sigma_min: float,
        sigma_max: float,
        w_c: float,
        kappa: float = 3.0,
        activity_gain: float = 1.0,
        radial_regularization: float = 0.1,
        profile: Profile | None = None,
    ) -> None:
        super().__init__()
        maximum_amplitude = self._positive_scalar(amplitude_max, name="amplitude_max")
        minimum = self._positive_scalar(sigma_min, name="sigma_min")
        maximum = self._positive_scalar(sigma_max, name="sigma_max")
        crossover = self._positive_scalar(w_c, name="w_c")
        separation = self._positive_scalar(kappa, name="kappa")
        gain = self._positive_scalar(activity_gain, name="activity_gain")
        regularization = self._nonnegative_scalar(
            radial_regularization, name="radial_regularization"
        )
        if maximum < minimum:
            raise ValueError("sigma_max must not be smaller than sigma_min")
        if separation <= 1:
            raise ValueError("kappa must be greater than 1")

        if profile is None:
            profile = Gaussian(minimum)
        elif not isinstance(profile, Profile):
            raise TypeError("profile must implement the Profile contract")
        if not hasattr(profile, "evaluate_with_precision") or not hasattr(
            profile, "tangent_with_precision"
        ):
            raise TypeError(
                "bandwidth profile must implement evaluate_with_precision "
                "and tangent_with_precision"
            )
        if not hasattr(profile, "sigma"):
            raise TypeError("bandwidth profile must expose a sigma buffer")
        if not torch.allclose(
            profile.sigma.detach().to(dtype=minimum.dtype).reshape(()),
            minimum,
        ):
            raise ValueError("profile.sigma must match sigma_min")

        self.profile = profile
        self.register_buffer("amplitude_max", maximum_amplitude)
        self.register_buffer("sigma_max", maximum)
        self.register_buffer("w_c", crossover)
        self.register_buffer("kappa", separation)
        self.register_buffer("activity_gain", gain)
        self.register_buffer("radial_regularization", regularization)

    @property
    def sigma_min(self) -> Tensor:
        return self.profile.sigma

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            2
            + self.profile.parameter_dim(input_chart)
            + self.profile.parameter_dim(output_chart)
        )

    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart,
        atoms: int,
        *,
        mode: AtomInit,
    ) -> Tensor:
        if mode not in ("balanced", "uniform"):
            raise ValueError("mode must be 'balanced' or 'uniform'")
        input_p = self.profile.initialize(input_chart, atoms, mode="uniform")
        output_p = self.profile.initialize(output_chart, atoms, mode=mode)

        amplitude = input_p.new_empty(atoms)
        amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
        maximum = self.amplitude_max.to(amplitude)
        # Keep initialization away from the angular critical points w = +/- W.
        ratio = (amplitude / maximum).clamp(-1 + 1e-6, 1 - 1e-6)
        polar = torch.stack((ratio, (1 - ratio.square()).sqrt()), dim=-1)
        return torch.cat((polar, input_p, output_p), dim=-1)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        phi_input, phi_output = self.factors(input_chart, output_chart, p)
        return torch.einsum("oa,ia->aoi", phi_output, phi_input)

    @property
    def supports_factorization(self) -> bool:
        return True

    def factors(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        polar, input_p, output_p = self._split(input_chart, output_chart, p)
        amplitude, alpha = self._amplitude_and_alpha(polar)
        sigma, _, _ = self._sigma_bounds(amplitude, alpha)
        # Width is a state derived from update history, not a task-loss degree
        # of freedom. Only the explicit radial regularizer may decrease alpha.
        precision = sigma.reciprocal().square().detach()
        phi_input = self.profile.evaluate_with_precision(
            input_chart, input_p, precision
        )
        phi_output = self.profile.evaluate_with_precision(
            output_chart, output_p, precision
        )
        return phi_input, phi_output * amplitude.unsqueeze(0)

    def amplitude(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
        """Return the bounded signed amplitude represented by each atom."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        amplitude, _ = self._amplitude_and_alpha(polar)
        return amplitude

    def bandwidth_alpha(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the radial interpolation coordinate in ``[0, 1]``."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        _, alpha = self._amplitude_and_alpha(polar)
        return alpha

    def bandwidth_bounds(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(L(w), U(w))`` for diagnostic use."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        amplitude, alpha = self._amplitude_and_alpha(polar)
        _, lower, upper = self._sigma_bounds(amplitude, alpha)
        return lower, upper

    def bandwidth_sigma(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the shared effective input/output sigma for each atom."""

        polar, _, _ = self._split(input_chart, output_chart, p)
        amplitude, alpha = self._amplitude_and_alpha(polar)
        sigma, _, _ = self._sigma_bounds(amplitude, alpha)
        return sigma

    def bandwidth_precision(
        self, input_chart: Chart, output_chart: Chart, p: Tensor
    ) -> Tensor:
        """Return the shared input/output precision for each atom."""

        return self.bandwidth_sigma(input_chart, output_chart, p).reciprocal().square()

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        """Project task motion tangentially, decay radius, then enforce 1<=q<=4."""

        self._split(input_chart, output_chart, p)
        if displacement.shape != p.shape:
            raise ValueError("displacement must match the atom parameter shape")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive")

        polar = self._project_polar(p[:, :2])
        raw = displacement[:, :2]
        radius_square = polar.square().sum(dim=-1, keepdim=True)
        radial_coefficient = (raw * polar).sum(dim=-1, keepdim=True) / radius_square
        tangent = raw - radial_coefficient * polar

        # Preserve the optimizer's angular proposal while allowing the radial
        # history clock to be calibrated independently. activity_gain=1 is
        # the original finite-chord rule q' = q + ||tangent||^2.
        chord = polar + tangent
        chord_q = chord.square().sum(dim=-1, keepdim=True)
        direction = chord / chord_q.sqrt()
        task_q = (
            radius_square
            + self.activity_gain.to(p) * tangent.square().sum(dim=-1, keepdim=True)
        ).clamp(1.0, 4.0)
        task_polar = direction * task_q.sqrt()

        # Exact gradient flow for R(q)=lambda/2*(q-1)^2 over time step_size:
        # y=(q-1)/q decays as exp(-4*lambda*t), keeping q in [1, 4].
        decay = torch.exp(
            -4.0 * self.radial_regularization.to(p) * p.new_tensor(step_size)
        )
        activity = (task_q - 1.0) / task_q
        regularized_q = 1.0 / (1.0 - activity * decay)
        regularized_polar = task_polar * (regularized_q / task_q).sqrt()

        result = p + displacement
        return torch.cat((regularized_polar, result[:, 2:]), dim=-1)

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected_dim = self.parameter_dim(input_chart, output_chart)
        if p.ndim != 2 or p.shape[1] != expected_dim:
            raise ValueError(f"p must have shape [atoms, {expected_dim}]")
        input_dim = self.profile.parameter_dim(input_chart)
        input_end = 2 + input_dim
        return p[:, :2], p[:, 2:input_end], p[:, input_end:]

    def _amplitude_and_alpha(self, polar: Tensor) -> tuple[Tensor, Tensor]:
        radius_square = polar.square().sum(dim=-1)
        # The exact polar map is used everywhere except the singular origin.
        safe_square = radius_square.clamp_min(torch.finfo(polar.dtype).tiny)
        amplitude = self.amplitude_max.to(polar) * polar[:, 0] / safe_square.sqrt()
        alpha = ((radius_square - 1.0) / 3.0).clamp(0.0, 1.0)
        return amplitude, alpha

    @staticmethod
    def _project_polar(polar: Tensor) -> Tensor:
        radius_square = polar.square().sum(dim=-1, keepdim=True)
        tiny = torch.finfo(polar.dtype).tiny
        safe_radius = radius_square.clamp_min(tiny).sqrt()
        fallback = torch.zeros_like(polar)
        fallback[:, 1] = 1.0
        unit = torch.where(radius_square > tiny, polar / safe_radius, fallback)
        radius = safe_radius.clamp(1.0, 2.0)
        return unit * radius

    def _sigma_bounds(
        self, amplitude: Tensor, alpha: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = (amplitude / self.w_c.to(amplitude)).square()
        delta = self.sigma_max.to(amplitude) - self.sigma_min.to(amplitude)
        kappa = self.kappa.to(amplitude)
        upper = self.sigma_min.to(amplitude) + delta * kappa / (kappa + x)
        lower = self.sigma_min.to(amplitude) + delta / (1.0 + kappa * x)
        sigma = lower + alpha * (upper - lower)
        return sigma, lower, upper

    @staticmethod
    def _positive_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _nonnegative_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return result

    def extra_repr(self) -> str:
        return (
            f"amplitude_max={self.amplitude_max.item():g}, "
            f"sigma_min={self.sigma_min.item():g}, "
            f"sigma_max={self.sigma_max.item():g}, "
            f"w_c={self.w_c.item():g}, kappa={self.kappa.item():g}, "
            f"activity_gain={self.activity_gain.item():g}, "
            f"radial_regularization={self.radial_regularization.item():g}, "
            f"profile={type(self.profile).__name__}, "
            f"supports_factorization={self.supports_factorization}"
        )
