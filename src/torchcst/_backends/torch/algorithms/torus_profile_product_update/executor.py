"""The public intrinsic retraction expressed without host tensor reads."""

import math

import torch

from ..polar_update.executor import graph_update


def updated_parameters(kernel, geometry, previous, displacement, *, step_size):
    polar = graph_update(kernel, previous, displacement, step_size=step_size)[:, :2]
    centers = previous[:, 2:]
    arc_step = displacement[:, 2:3]
    if geometry.max_arc_step is not None:
        arc_step = arc_step.clamp(-geometry.max_arc_step, geometry.max_arc_step)
    # Eager retraction first converts R to a Python double. Preserve that
    # scalar operation order, then let each tensor operand round to its dtype.
    half_turn = geometry.major_radius.double() * math.pi
    arc = torch.remainder(
        centers[:, :1] + arc_step + half_turn.to(previous),
        (2 * half_turn).to(previous),
    ) - half_turn.to(previous)
    section = centers[:, 1:] + displacement[:, 3:]
    length = torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    limit = (geometry.minor_radius * (torch.pi - geometry.chart_margin)).to(section)
    section = section * (
        limit / length.clamp_min(torch.finfo(section.dtype).tiny)
    ).clamp_max(1)
    return torch.cat((polar, arc, section), dim=-1)
