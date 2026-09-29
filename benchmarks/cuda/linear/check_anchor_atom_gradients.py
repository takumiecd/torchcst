"""Check the anchor atom kernel against differentiable all-atom PyTorch sums."""

import argparse

import torch

from experiments.cuda.linear.anchor_atom_training import (
    _AnchorSamples,
    make_anchor_layout,
    make_calibrated_anchor_layout,
)
from experiments.cuda.linear.block_streamed_backward import trainable_boxed_prepare
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan

parser = argparse.ArgumentParser()
parser.add_argument("--forward-lanes", type=int, choices=(1, 2, 4, 8), default=1)
parser.add_argument("--backward-lanes", type=int, choices=(1, 2, 4, 8), default=1)
parser.add_argument(
    "--list-mode", choices=("full_tile", "anchors"), default="full_tile"
)
parser.add_argument("--calibration")
parser.add_argument("--row-rank", type=int, default=16)
parser.add_argument("--column-rank", type=int, default=8)
args = parser.parse_args()
torch.manual_seed(37)
torch.backends.cuda.matmul.allow_tf32 = False
n = 128
layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
layout = (
    make_calibrated_anchor_layout(
        layer, args.calibration, args.row_rank, args.column_rank
    )
    if args.calibration
    else make_anchor_layout(layer, args.row_rank, args.column_rank)
)
plan = execution_plan(layer.strip)
boxes = station_site_boxes(plan.circle, plan.section, 64)
hints = balanced_home_columns(layer.strip)
packed, circle, section, offsets = trainable_boxed_prepare(
    layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
)
packed = packed.detach().clone().requires_grad_()
samples = _AnchorSamples.apply(
    packed,
    circle,
    section,
    offsets,
    layer,
    layout,
    args.forward_lanes,
    args.backward_lanes,
    args.list_mode,
)
ds = torch.randn_like(samples)
(custom_grad,) = torch.autograd.grad((samples * ds).sum(), packed)

row_anchors = torch.tensor(layout.row_indices, device="cuda")
col_anchors = torch.tensor(layout.col_indices, device="cuda")
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
            z[None, :].expand(layout.rows_per_block, -1),
            w[None, :].expand(layout.rows_per_block, -1),
        ),
        dim=-1,
    ).reshape(-1, 4)
    squared = (coords[:, None, :] - packed[None, :, 2:6]).square().sum(-1)
    gap = (1.0 - squared * packed[None, :, 1]).clamp_min(0.0)
    values = (
        (gap * gap * gap * packed[None, :, 0])
        .sum(-1)
        .reshape(layout.rows_per_block, layout.cols_per_block)
    )
    row = station // layer.column_groups
    col = station % layer.column_groups
    ref[
        row * layout.rows_per_block : (row + 1) * layout.rows_per_block,
        col * layout.cols_per_block : (col + 1) * layout.cols_per_block,
    ] = values
(reference_grad,) = torch.autograd.grad((ref * ds).sum(), packed)
indices = torch.tensor([0, 2, 3, 4, 5], device="cuda")
g_ref = reference_grad.index_select(1, indices)
g_custom = custom_grad.index_select(1, indices)
print(
    {
        "device": torch.cuda.get_device_name(),
        "forward_lanes": args.forward_lanes,
        "backward_lanes": args.backward_lanes,
        "list_mode": args.list_mode,
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
