"""Compare the compact intrinsic decoder and its VJP with Torch geometry."""

import json

import torch

from prototypes.block_strip_linear import BlockStripLinear
from prototypes.torus_decode_trainable import trainable_decode_intrinsic_torus


torch.manual_seed(47)
layer = BlockStripLinear((1024, 1024), (64, 64), 1024, device="cuda")
geometry = layer.strip.chart.geometry
center = layer.strip.atoms.p[:, 2:].detach().clone()
center[0, 1:] = 0
center.requires_grad_()
gradient = torch.randn(center.shape[0], 4, device="cuda")
reference = geometry.decode_centers(center)
(reference_grad,) = torch.autograd.grad((reference * gradient).sum(), center)
actual = trainable_decode_intrinsic_torus(geometry, center)
(actual_grad,) = torch.autograd.grad((actual * gradient).sum(), center)
print(json.dumps({
    "device": torch.cuda.get_device_name(),
    "forward_relative_l2": float((actual - reference).norm() / reference.norm()),
    "forward_max_abs": float((actual - reference).abs().max()),
    "gradient_relative_l2": float((actual_grad - reference_grad).norm() / reference_grad.norm()),
    "gradient_max_abs": float((actual_grad - reference_grad).abs().max()),
    "zero_section_gradient_max_abs": float((actual_grad[0] - reference_grad[0]).abs().max()),
}), flush=True)
