"""Direct amplitude coordinates with an explicit bandwidth activity state."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit
from .polar_amplitude_bandwidth import PolarAmpWidth


class DirectAmpWidth(PolarAmpWidth):
    r"""A rank-one atom with direct amplitude and activity coordinates.

    The first two coordinates of each atom are ``(w, q)``.  They encode

    .. math::

        w \in [-W, W], \qquad
        \alpha = \operatorname{clamp}((q-1)/3, 0, 1).

    ``w`` is the task-loss degree of freedom and therefore receives Adam
    moments directly. ``q`` is a persistent activity state rather than a
    task-loss degree of freedom.  An accepted amplitude update is converted
    to the equivalent Polar chord length

    .. math::

        \lVert h\rVert^2 = q\tan^2(\Delta\theta), \qquad
        \theta = \arcsin(w/W),

    before applying the same activity growth and exact radial regularization
    used by :class:`PolarAmpWidth`.
    """

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
        amplitude = amplitude.clamp(-maximum, maximum)
        q = torch.ones_like(amplitude) + 3.0 * self.alpha_init.to(amplitude)
        direct = torch.stack((amplitude, q), dim=-1)
        return torch.cat((direct, input_p, output_p), dim=-1)

    def apply_parameter_update(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        displacement: Tensor,
        *,
        step_size: float,
    ) -> Tensor:
        """Apply a direct-amplitude proposal and advance its activity state."""

        direct, input_p, output_p = self._split(input_chart, output_chart, p)
        if displacement.shape != p.shape:
            raise ValueError("displacement must match the atom parameter shape")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive")

        maximum = self.amplitude_max.to(p)
        old_w = direct[:, 0].clamp(-maximum, maximum)
        old_q = direct[:, 1].clamp(1.0, 4.0)
        proposed_w = (old_w + displacement[:, 0]).clamp(-maximum, maximum)

        # A finite Polar chord can turn by strictly less than pi/2. Limit an
        # unusually large optimizer proposal to that same geometric domain,
        # then recompute the accepted amplitude from the accepted angle.
        old_theta = torch.asin((old_w / maximum).clamp(-1.0, 1.0))
        proposed_theta = torch.asin((proposed_w / maximum).clamp(-1.0, 1.0))
        angle_margin = math.sqrt(torch.finfo(p.dtype).eps)
        angle_limit = p.new_tensor(math.pi / 2.0 - angle_margin)
        delta_theta = (proposed_theta - old_theta).clamp(-angle_limit, angle_limit)
        accepted_theta = old_theta + delta_theta
        accepted_w = maximum * accepted_theta.sin()

        tangent_square = old_q * delta_theta.tan().square()
        energy = tangent_square
        if self.activity_mode == "time_energy":
            energy = energy / step_size
        dormant_weight = 1.0 / (
            1.0 + (accepted_w / self.w_c.to(p)).square()
        )
        dormant_q = (
            3.0
            * self.dormant_expansion_rate.to(p)
            * p.new_tensor(step_size)
            * dormant_weight
        )
        task_q = (
            old_q + self.activity_gain.to(p) * energy + dormant_q
        ).clamp(1.0, 4.0)

        # Exact gradient flow for R(q)=lambda/2*(q-1)^2 over step_size.
        decay = torch.exp(
            -4.0 * self.radial_regularization.to(p) * p.new_tensor(step_size)
        )
        activity = (task_q - 1.0) / task_q
        regularized_q = 1.0 / (1.0 - activity * decay)

        _, input_d, output_d = self._split(
            input_chart,
            output_chart,
            displacement,
        )
        updated_input = self.profile.apply_parameter_update(
            input_chart,
            input_p,
            input_d,
        )
        updated_output = self.profile.apply_parameter_update(
            output_chart,
            output_p,
            output_d,
        )
        updated_direct = torch.stack((accepted_w, regularized_q), dim=-1)
        return torch.cat((updated_direct, updated_input, updated_output), dim=-1)

    def project_parameter_gradient(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
        gradient: Tensor,
    ) -> Tensor:
        direct, input_p, output_p = self._split(input_chart, output_chart, p)
        del direct
        direct_g, input_g, output_g = self._split(
            input_chart,
            output_chart,
            gradient,
        )
        projected_direct = torch.stack(
            (direct_g[:, 0], torch.zeros_like(direct_g[:, 1])),
            dim=-1,
        )
        return torch.cat(
            (
                projected_direct,
                self.profile.project_gradient(input_chart, input_p, input_g),
                self.profile.project_gradient(output_chart, output_p, output_g),
            ),
            dim=-1,
        )

    def _amplitude_and_alpha(self, direct: Tensor) -> tuple[Tensor, Tensor]:
        maximum = self.amplitude_max.to(direct)
        amplitude = direct[:, 0].clamp(-maximum, maximum)
        q = direct[:, 1].clamp(1.0, 4.0)
        alpha = (q - 1.0) / 3.0
        return amplitude, alpha
