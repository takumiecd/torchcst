"""Hierarchical cell-mass forward using the existing ChebyshevSquare profile.

Sigma keeps its existing width-control meaning and concentration floor. There
is no new density, width conversion or profile fitting. These are numeric
research stages, not the public CSTLinear dispatcher.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from .plan import AxisPlan, HierarchyPlan


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    sources = [
        directory / "bindings.cpp",
        directory / "preparation.cu",
        directory / "contraction.cu",
        directory / "backward.cu",
    ]
    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in [*sources, directory / "common.cuh"])
    ).hexdigest()[:12]
    return load(
        name=f"torchcst_cell_mass_hierarchy_{digest}",
        sources=[str(path) for path in sources],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


@dataclass(frozen=True)
class PreparedLayout:
    """Forward-scoped cache: indices/ranges only, never packed atom values.

    groups[g] = (output_level, input_level, output_owner, input_owner).
    rows[offsets[g]:offsets[g+1]] lists original atom rows in that group.
    Key/sort/count scratch is released before this object is returned.
    """

    rows: torch.Tensor
    groups: torch.Tensor
    offsets: torch.Tensor
    plan: HierarchyPlan
    _source: torch.Tensor = field(repr=False)
    _source_version: int = field(repr=False)

    def validate_for(self, atoms, plan):
        if (
            atoms is not self._source
            or atoms._version != self._source_version
            or plan != self.plan
        ):
            raise RuntimeError("atom values or plan changed; rebuild the layout cache")

    @property
    def tensor_bytes(self):
        """Retained cache payload, not an allocator peak or whole-step budget."""
        return sum(
            t.numel() * t.element_size() for t in (self.rows, self.groups, self.offsets)
        )


def prepare_layout(plan, atoms):
    """Classify, stable-sort indices, and retain sparse nonempty group ranges.

    Values remain in the original Parameter. A library sort is the initial
    implementation, not a permanent requirement to materialize all atom keys.
    Width/finite rejection reads one error word. unique_consecutive also has a
    dynamic output shape, so this version does not claim CUDA Graph support.
    """
    if type(plan) is not HierarchyPlan:
        raise TypeError("plan must be HierarchyPlan")
    if atoms.device.type != "cuda" or atoms.dtype not in (torch.float32, torch.float64):
        raise ValueError("atoms must be CUDA float32/float64")
    if atoms.ndim != 2 or atoms.shape[1] != 4 or not atoms.is_contiguous():
        raise ValueError("atoms must be contiguous [atom_count,4]")
    if atoms.shape[0] > 65536:
        raise ValueError("preparation supports at most 65536 atoms")

    native = _extension()
    input_config = plan.input.native_config()
    output_config = plan.output.native_config()
    keys, errors = native.classify_atoms(atoms, input_config, output_config)
    error_bits = errors.item()
    if error_bits & 1:
        raise ValueError("all atom fields must be finite")
    if error_bits & 2:
        raise ValueError("sigma must lie in (0, min(input period,output period)/3]")

    sorted_keys, rows = torch.sort(keys, stable=True)
    group_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    offsets = torch.empty(counts.numel() + 1, dtype=torch.int64, device=atoms.device)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    groups = native.decode_groups(group_keys, input_config, output_config)
    return PreparedLayout(rows, groups, offsets, plan, atoms, atoms._version)


def build_hierarchy(signal, axis, *, method="merge", team_size=4):
    """Return double Z[batch,node,1+2*M] for the banks declared by axis.

    One team cooperates on a single (node,batch,mode) sum. The merge route
    directly creates leaves, odd parent harmonics and the full-chart bank;
    matching even harmonics are merged level by level. The direct route is a
    comparison that generates every element from signal independently.

    This low-level stage has no autograd binding. It is used by the explicit
    forward/backward implementation, not a differentiable public operator.
    """
    if type(axis) is not AxisPlan:
        raise TypeError("axis must be AxisPlan")
    if method not in ("direct", "merge"):
        raise ValueError("method must be direct or merge")
    if type(team_size) is not int or team_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError("team_size must be 1,2,4,8,16 or 32")
    if signal.device.type != "cuda" or signal.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("signal must be CUDA float32/float64")
    if (
        signal.ndim != 2
        or signal.shape[1] != axis.site_count
        or not signal.is_contiguous()
    ):
        raise ValueError("signal must be contiguous [batch,site_count]")
    return _extension().build_hierarchy(
        signal, axis.native_config(), method == "merge", team_size
    )


def compute_h(plan, atoms, layout, zx, *, storage_dtype=torch.float64):
    """Complete H[batch,sorted_slot], before amplitude, from all retained tiles.

    One thread owns one atom; coefficients and mass normalization are generated
    in registers from the original sigma. This internal stage has no autograd.
    """
    if storage_dtype not in (torch.float32, torch.float64):
        raise ValueError("H/G storage must be float32 or float64")
    layout.validate_for(atoms, plan)
    return _extension().compute_h(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        zx,
        plan.input.native_config(),
        False,
        storage_dtype == torch.float32,
    )


def compute_c(plan, atoms, layout, h):
    """Complete output owner C[batch,node,mode] using 32-atom shared chunks.

    The small cross-block coefficient buffer is bank/node sized, not an
    atom-by-mode table. Each C element has one writer, with amplitude applied
    once. The next stage adds bank contributions after site expansion.
    """
    layout.validate_for(atoms, plan)
    return _extension().compute_c(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        h,
        plan.input.native_config(),
        plan.output.native_config(),
        False,
    )


def expand_c(c, axis, like, *, sites_per_block=32):
    """Expand each bank's C with cell-integrated bases, then add into Y.

    like supplies batch/device/dtype, not output coordinates. Chart seams use
    the retained window's unwrapped image; every Y element has one writer.
    """
    if type(axis) is not AxisPlan:
        raise TypeError("axis must be AxisPlan")
    if type(sites_per_block) is not int:
        raise ValueError("sites_per_block must be a power of two in [1,256]")
    return _extension().expand_c(c, like, axis.native_config(), sites_per_block)


def forward_only(
    plan, x, atoms, *, layout=None, z_method="merge", team_size=4, sites_per_block=32
):
    """Research forward: prepare -> Z -> H -> C -> Y, with no hidden fallback.

    Use hierarchical_linear for the explicit autograd binding; this diagnostic
    route requires detached inputs.
    """
    if x.requires_grad or atoms.requires_grad:
        raise RuntimeError(
            "forward_only needs detached inputs; use hierarchical_linear for backward"
        )
    if x.device != atoms.device or x.dtype != atoms.dtype:
        raise ValueError("X and atoms must share device and dtype")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("X must be finite")
    if layout is None:
        layout = prepare_layout(plan, atoms)
    else:
        layout.validate_for(atoms, plan)
    zx = build_hierarchy(x, plan.input, method=z_method, team_size=team_size)
    h = compute_h(plan, atoms, layout, zx)
    c = compute_c(plan, atoms, layout, h)
    return expand_c(c, plan.output, x, sites_per_block=sites_per_block)


def compute_g(plan, atoms, layout, zy, *, storage_dtype=torch.float64):
    """Project dY onto each output profile: G[batch,sorted_slot], without amp.

    This uses the same contraction as H, with output levels/owners/centers.
    G can be reused by dX and all four atom-parameter gradients.
    """
    if storage_dtype not in (torch.float32, torch.float64):
        raise ValueError("H/G storage must be float32 or float64")
    layout.validate_for(atoms, plan)
    return _extension().compute_h(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        zy,
        plan.output.native_config(),
        True,
        storage_dtype == torch.float32,
    )


def compute_input_c(plan, atoms, layout, g):
    """Accumulate amp*G into input owner C; no dX atomic or packed atom copy.

    Input owners are not contiguous in the output-sorted layout. The first
    implementation scans sparse group descriptors and loads matching ranges.
    This is a correctness baseline; an input-group index can replace that scan.
    """
    layout.validate_for(atoms, plan)
    return _extension().compute_c(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        g,
        plan.output.native_config(),
        plan.input.native_config(),
        True,
    )


def parameter_grad(plan, atoms, layout, zx, zy, h, g):
    """One thread per atom reduces all batches into its original [a,s,muo,mui].

    Analytic derivatives include the effective-radius floor and area-normalized
    coefficient quotient. Bank/owner selection is fixed to this forward's plan.
    H/G and Zx/Zy are reused, with derivative coefficients only in registers.
    """
    layout.validate_for(atoms, plan)
    return _extension().parameter_grad(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        zx,
        zy,
        h,
        g,
        plan.input.native_config(),
        plan.output.native_config(),
    )


class _HierarchicalLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, atoms, plan, layout, z_method, team_size, sites_per_block):
        zx = build_hierarchy(x, plan.input, method=z_method, team_size=team_size)
        h = compute_h(plan, atoms, layout, zx, storage_dtype=atoms.dtype)
        c = compute_c(plan, atoms, layout, h)
        y = expand_c(c, plan.output, x, sites_per_block=sites_per_block)
        ctx.plan = plan
        ctx.z_method = z_method
        ctx.team_size = team_size
        ctx.sites_per_block = sites_per_block
        ctx.source_version = atoms._version
        # save_for_backward checks atom AND layout-buffer mutations. Avoid
        # keeping the source x: Zx already contains the needed contractions.
        # Input-only backward does not need Zx/H, so release those immediately.
        saved = (atoms, layout.rows, layout.groups, layout.offsets)
        if ctx.needs_input_grad[1]:
            saved += (zx, h)
        ctx.save_for_backward(*saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        atoms, rows, groups, offsets, *forward_state = ctx.saved_tensors
        plan = ctx.plan
        layout = PreparedLayout(rows, groups, offsets, plan, atoms, ctx.source_version)
        layout.validate_for(atoms, plan)
        dy = dy.contiguous()
        zy = build_hierarchy(
            dy, plan.output, method=ctx.z_method, team_size=ctx.team_size
        )
        g = compute_g(plan, atoms, layout, zy, storage_dtype=atoms.dtype)
        dx = datoms = None
        if ctx.needs_input_grad[0]:
            cin = compute_input_c(plan, atoms, layout, g)
            dx = expand_c(cin, plan.input, dy, sites_per_block=ctx.sites_per_block)
            del cin
        if ctx.needs_input_grad[1]:
            zx, h = forward_state
            datoms = parameter_grad(plan, atoms, layout, zx, zy, h, g)
        return dx, datoms, None, None, None, None, None


def hierarchical_linear(
    plan, x, atoms, *, layout=None, z_method="merge", team_size=4, sites_per_block=32
):
    """Private research autograd route, using this forward's bank/owner choices.

    Forward saves Zx/H only if atom gradients are requested. Backward builds Zy
    and G once, then uses them for dX and parameter gradients. No dense W, dW,
    packed atom values, global coefficient table, or separate dH is allocated.
    H/G storage uses the input dtype; all contractions, Z and C remain double.
    Double backward and full-route CUDA Graph capture are not supported yet
    (preparation has dynamic output shapes and host checks).
    """
    if x.device != atoms.device or x.dtype != atoms.dtype:
        raise ValueError("X and atoms must share device and dtype")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("X must be finite")
    if layout is None:
        layout = prepare_layout(plan, atoms)
    else:
        layout.validate_for(atoms, plan)
    return _HierarchicalLinear.apply(
        x, atoms, plan, layout, z_method, team_size, sites_per_block
    )
