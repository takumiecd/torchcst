"""Conservative section boxes, rebuilt from current prepared geometry."""

import torch


def section_bounds(section, width):
    return torch.stack(
        [torch.stack((part.amin(0), part.amax(0))) for part in section.split(width)]
    ).contiguous()
