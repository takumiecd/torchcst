"""Check the anchor atom kernel against differentiable all-atom PyTorch sums."""

import torch

from prototypes.anchor_atom_training import _AnchorSamples, make_anchor_layout
from prototypes.block_streamed_backward import trainable_boxed_prepare
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.diagnose_cst_sampled_blocks import column_basis, evenly_spaced_indices
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan

torch.manual_seed(37)
torch.backends.cuda.matmul.allow_tf32 = False
n = 128
layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
layout = make_anchor_layout(layer, 16, 8)
plan = execution_plan(layer.strip)
boxes = station_site_boxes(plan.circle, plan.section, 64)
hints = balanced_home_columns(layer.strip)
packed, circle, section, offsets = trainable_boxed_prepare(
    layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
)
packed = packed.detach().clone().requires_grad_()
samples = _AnchorSamples.apply(packed, circle, section, offsets, layer, layout)
ds = torch.randn_like(samples)
(custom_grad,) = torch.autograd.grad((samples * ds).sum(), packed)

row_anchors = torch.tensor(evenly_spaced_indices(64, 16), device="cuda")
_, cols_list = column_basis(8)
col_anchors = torch.tensor(cols_list, device="cuda")
rho = section[col_anchors, 0]
z = section[col_anchors, 1]
w = section[col_anchors, 2]
ref = torch.zeros_like(samples)
for station in range(layer.strip.chart.tile_count):
    cosine = circle[station * 64 + row_anchors, 0]
    sine = circle[station * 64 + row_anchors, 1]
    coords = torch.stack(
        (
            cosine[:, None] * rho[None, :],
            sine[:, None] * rho[None, :],
            z[None, :].expand(16, -1),
            w[None, :].expand(16, -1),
        ),
        dim=-1,
    ).reshape(-1, 4)
    squared = (coords[:, None, :] - packed[None, :, 2:6]).square().sum(-1)
    gap = (1.0 - squared * packed[None, :, 1]).clamp_min(0.0)
    values = (gap * gap * gap * packed[None, :, 0]).sum(-1).reshape(16, 32)
    row = station // layer.column_groups
    col = station % layer.column_groups
    ref[row * 16 : (row + 1) * 16, col * 32 : (col + 1) * 32] = values
(reference_grad,) = torch.autograd.grad((ref * ds).sum(), packed)
indices = torch.tensor([0, 2, 3, 4, 5], device="cuda")
g_ref = reference_grad.index_select(1, indices)
g_custom = custom_grad.index_select(1, indices)
print(
    {
        "device": torch.cuda.get_device_name(),
        "sample_relative_l2": float(((samples - ref).norm() / ref.norm()).detach()),
        "sample_max_abs": float((samples - ref).abs().max().detach()),
        "packed_grad_relative_l2": float(
            ((g_custom - g_ref).norm() / g_ref.norm()).detach()
        ),
        "packed_grad_max_abs": float((g_custom - g_ref).abs().max().detach()),
        "packed_width_grad_max_abs": float(custom_grad[:, 1].abs().max().detach()),
    },
    flush=True,
)
