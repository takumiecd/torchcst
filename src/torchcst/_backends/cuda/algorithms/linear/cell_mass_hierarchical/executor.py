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
        b"".join(
            path.read_bytes()
            for path in [
                *sources,
                directory / "common.cuh",
                directory / "projection.cuh",
            ]
        )
    ).hexdigest()[:12]
    return load(
        name=f"torchcst_cell_mass_hierarchy_{digest}",
        sources=[str(path) for path in sources],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--ftz=false"],
        verbose=False,
    )


@dataclass(frozen=True)
class PreparedLayout:
    """Forward-scoped cache: indices/ranges only, never packed atom values.

    groups[g] = (output_level, input_level, output_owner, input_owner).
    Counting preparation includes empty Cartesian groups; compact counting
    includes only compatible bank pairs, and sort omits all empty groups.
    rows[offsets[g]:offsets[g+1]] lists original atom rows in that group.
    Optional reverse_slots maps input-owner order back to these forward slots;
    reverse_offsets gives the owner intervals. It never copies atom values.
    Key/sort/count scratch is released before this object is returned.
    """

    rows: torch.Tensor
    groups: torch.Tensor
    offsets: torch.Tensor
    plan: HierarchyPlan
    _source: torch.Tensor = field(repr=False)
    _source_version: int = field(repr=False)
    reverse_slots: torch.Tensor | None = field(default=None, repr=False)
    reverse_offsets: torch.Tensor | None = field(default=None, repr=False)
    method: str = "sort"

    @property
    def has_reverse(self):
        """Whether input-owner ranges have a cached view of forward slots."""
        return self.reverse_slots is not None and self.reverse_offsets is not None

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
            t.numel() * t.element_size()
            for t in (
                self.rows,
                self.groups,
                self.offsets,
                self.reverse_slots,
                self.reverse_offsets,
            )
            if t is not None
        )


def prepare_layout(plan, atoms, *, method="sort", signal=None, reverse=False):
    """Classify atom indices into lexicographic hierarchy/owner group ranges.

    The sort method retains only nonempty groups in stable order. Counting
    retains the complete Cartesian group table (at most 4096), including empty
    ranges. Compact counting retains only compatible bank pairs, also with
    fixed empty ranges; it currently has no reverse index. Counting methods
    do not promise stable order within a group. No method
    copies atom values from the original Parameter or retains all atom keys.
    Width/finite rejection reads one error word. Sort/unique has dynamic output
    shapes; host checks remain, so full-route CUDA Graph support is not claimed.
    """
    if type(reverse) is not bool:
        raise ValueError("reverse must be bool")
    if reverse and method != "counting":
        raise ValueError("reverse layout requires counting preparation")
    if method not in ("sort", "counting", "counting_compact"):
        raise ValueError("layout method must be sort, counting or counting_compact")
    if type(plan) is not HierarchyPlan:
        raise TypeError("plan must be HierarchyPlan")
    if atoms.device.type != "cuda" or atoms.dtype not in (torch.float32, torch.float64):
        raise ValueError("atoms must be CUDA float32/float64")
    if atoms.ndim != 2 or atoms.shape[1] != 4 or not atoms.is_contiguous():
        raise ValueError("atoms must be contiguous [atom_count,4]")
    if atoms.shape[0] > 65536:
        raise ValueError("preparation supports at most 65536 atoms")

    if signal is not None:
        if (
            not isinstance(signal, torch.Tensor)
            or signal.device != atoms.device
            or signal.dtype != atoms.dtype
            or signal.ndim != 2
            or signal.shape[1] != plan.input.site_count
            or not signal.is_contiguous()
        ):
            raise ValueError(
                "signal must be contiguous [batch,site_count] on atom device/dtype"
            )
        if method == "sort" and not bool(torch.isfinite(signal).all()):
            raise ValueError("X must be finite")
    native = _extension()
    input_config = plan.input.native_config()
    output_config = plan.output.native_config()
    if method == "counting_compact":
        if signal is None:
            rows, groups, offsets, errors = native.prepare_compact_layout(
                atoms, input_config, output_config
            )
        else:
            rows, groups, offsets, errors = native.prepare_compact_layout_signal(
                atoms, signal, input_config, output_config
            )
    elif method == "counting":
        if plan.input.node_count * plan.output.node_count > 4096:
            raise ValueError("counting layout supports at most 4096 Cartesian groups")
        if reverse:
            if signal is None:
                result = native.prepare_counting_layout_reverse(
                    atoms, input_config, output_config
                )
            else:
                result = native.prepare_counting_layout_reverse_signal(
                    atoms, signal, input_config, output_config
                )
            rows, groups, offsets, errors, reverse_slots, reverse_offsets = result
        elif signal is None:
            rows, groups, offsets, errors = native.prepare_counting_layout(
                atoms, input_config, output_config
            )
        else:
            rows, groups, offsets, errors = native.prepare_counting_layout_signal(
                atoms, signal, input_config, output_config
            )
    else:
        keys, errors = native.classify_atoms(atoms, input_config, output_config)
    error_bits = errors.item()
    if error_bits & 4:
        raise ValueError("X must be finite")
    if error_bits & 1:
        raise ValueError("all atom fields must be finite")
    if error_bits & 2:
        raise ValueError("sigma must lie in (0, min(input period,output period)/3]")

    if method in ("counting", "counting_compact"):
        return PreparedLayout(
            rows,
            groups,
            offsets,
            plan,
            atoms,
            atoms._version,
            reverse_slots=reverse_slots if reverse else None,
            reverse_offsets=reverse_offsets if reverse else None,
            method=method,
        )

    sorted_keys, rows = torch.sort(keys, stable=True)
    group_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    offsets = torch.empty(counts.numel() + 1, dtype=torch.int64, device=atoms.device)
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    groups = native.decode_groups(group_keys, input_config, output_config)
    return PreparedLayout(rows, groups, offsets, plan, atoms, atoms._version)


def build_hierarchy(signal, axis, *, method="merge", team_size=4, chunk_sites=0):
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
    if (
        type(chunk_sites) is not int
        or chunk_sites < 0
        or (chunk_sites and chunk_sites & (chunk_sites - 1))
    ):
        raise ValueError("chunk_sites must be zero or a positive power of two")
    if chunk_sites:
        return _extension().build_hierarchy_chunked(
            signal, axis.native_config(), method == "merge", team_size, chunk_sites
        )
    return _extension().build_hierarchy(
        signal, axis.native_config(), method == "merge", team_size
    )


def build_window_moments(z, axis):
    """Sum each owner's retained tiles in its periodic image, once per node.

    Run this only after the raw hierarchy is complete. The returned FP64 tensor
    has the same shape as Z, so pass windowed=True to every projection consumer.
    Neither hierarchy merge nor C expansion consumes this layout.
    """
    return _extension().build_window_moments(z, axis.native_config())


def _validate_compensated(compensated, atoms, projection_dtype, axis):
    if type(compensated) is not bool:
        raise ValueError("compensated must be bool")
    if compensated and (
        atoms.dtype != torch.float32
        or projection_dtype != torch.float32
        or axis.modes != 8
        or axis.tail_bound != 1e-3
    ):
        raise ValueError(
            "compensated values require M8, tail 1e-3 and FP32 atoms/projections"
        )


def _validate_compensated_cdot(compensated_cdot, atoms, projection_dtype, axis):
    if type(compensated_cdot) is not bool:
        raise ValueError("compensated_cdot must be bool")
    if compensated_cdot and (
        atoms.dtype != torch.float32
        or projection_dtype != torch.float32
        or axis.modes != 8
    ):
        raise ValueError("compensated C dot requires M8 and FP32 atoms/projections")


def compute_h(
    plan,
    atoms,
    layout,
    zx,
    *,
    storage_dtype=torch.float64,
    threads=128,
    windowed=False,
    compensated=False,
):
    """Complete H[batch,sorted_slot], before amplitude, from all retained tiles.

    One thread owns one atom; coefficients and mass normalization are generated
    in registers from the original sigma. Compensated values are opt-in;
    this internal stage has no autograd.
    """
    if storage_dtype not in (torch.float32, torch.float64):
        raise ValueError("H/G storage must be float32 or float64")
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("atom threads must be 32,64,128 or 256")
    if type(windowed) is not bool:
        raise ValueError("windowed must be bool")
    _validate_compensated(compensated, atoms, storage_dtype, plan.input)
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
        threads,
        windowed,
        compensated,
    )


def compute_c(
    plan,
    atoms,
    layout,
    h,
    *,
    splits=1,
    atoms_per_split=0,
    compensated=False,
    compensated_cdot=False,
):
    """Complete output owner C[batch,node,mode] using 32-atom shared chunks.

    The small cross-block coefficient buffer is bank/node sized, not an
    atom-by-mode table. Each C element has one writer, with amplitude applied
    once. The next stage adds bank contributions after site expansion.
    Compensated values evaluate the profile polynomial. The independent
    compensated_cdot option uses FP32 pairs only for guarded atom chunks;
    other chunks and all cross-chunk/cross-block sums remain FP64.
    """
    if type(splits) is not int or not 1 <= splits <= 64:
        raise ValueError("C splits must be an integer in [1,64]")
    if type(atoms_per_split) is not int or not 0 <= atoms_per_split <= 65536:
        raise ValueError("atoms_per_split must be an integer in [0,65536]")
    _validate_compensated(compensated, atoms, h.dtype, plan.output)
    _validate_compensated_cdot(compensated_cdot, atoms, h.dtype, plan.output)
    layout.validate_for(atoms, plan)
    native = _extension()
    arguments = (
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        h,
        plan.input.native_config(),
        plan.output.native_config(),
        False,
    )
    if splits == 1:
        return native.compute_c(*arguments, compensated, compensated_cdot)
    return native.compute_c_split(
        *arguments, splits, atoms_per_split, compensated, compensated_cdot
    )


def compute_h_c(
    plan,
    atoms,
    layout,
    zx,
    *,
    splits=1,
    atoms_per_split=0,
    windowed=False,
    compensated=False,
):
    """Compute H and output C together for M8 axes; H uses the atom dtype."""
    if type(splits) is not int or not 1 <= splits <= 64:
        raise ValueError("C splits must be an integer in [1,64]")
    if type(atoms_per_split) is not int or not 0 <= atoms_per_split <= 65536:
        raise ValueError("atoms_per_split must be an integer in [0,65536]")
    if type(windowed) is not bool:
        raise ValueError("windowed must be bool")
    if plan.input.modes != 8 or plan.output.modes != 8:
        raise ValueError("fused H/C requires M8 axes")
    _validate_compensated(compensated, atoms, atoms.dtype, plan.input)
    _validate_compensated(compensated, atoms, atoms.dtype, plan.output)
    layout.validate_for(atoms, plan)
    return _extension().compute_h_c(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        zx,
        plan.input.native_config(),
        plan.output.native_config(),
        splits,
        atoms_per_split,
        windowed,
        compensated,
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


def compute_g(
    plan,
    atoms,
    layout,
    zy,
    *,
    storage_dtype=torch.float64,
    threads=128,
    windowed=False,
    compensated=False,
):
    """Project dY onto each output profile: G[batch,sorted_slot], without amp.

    This uses the same contraction as H, with output levels/owners/centers.
    G can be reused by dX and all four atom-parameter gradients.
    Compensated values are opt-in for the supported FP32 output axis.
    """
    if storage_dtype not in (torch.float32, torch.float64):
        raise ValueError("H/G storage must be float32 or float64")
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("atom threads must be 32,64,128 or 256")
    if type(windowed) is not bool:
        raise ValueError("windowed must be bool")
    _validate_compensated(compensated, atoms, storage_dtype, plan.output)
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
        threads,
        windowed,
        compensated,
    )


def _validate_reverse_grain(layout, atoms_per_split):
    if type(atoms_per_split) is not int or not 0 <= atoms_per_split <= 65536:
        raise ValueError("atoms_per_split must be an integer in [0,65536]")
    if atoms_per_split and not layout.has_reverse:
        raise ValueError("atoms_per_split requires a reverse layout")


def compute_input_c(
    plan,
    atoms,
    layout,
    g,
    *,
    splits=1,
    atoms_per_split=0,
    compensated=False,
    compensated_cdot=False,
):
    """Accumulate amp*G into input owner C; no dX atomic or packed atom copy.

    A reverse index gathers forward slots in contiguous input-owner intervals.
    Without one, the reference route visits matching output-sorted group ranges.
    A positive atoms_per_split bounds active split count by each owner's work;
    zero retains the requested fixed split count.
    Compensated values evaluate the profile polynomial. Guarded C dot is a
    separate opt-in, with FP64 fallback and FP64 cross-chunk/cross-block sums.
    """
    if type(splits) is not int or not 1 <= splits <= 64:
        raise ValueError("C splits must be an integer in [1,64]")
    _validate_compensated(compensated, atoms, g.dtype, plan.input)
    _validate_compensated_cdot(compensated_cdot, atoms, g.dtype, plan.input)
    layout.validate_for(atoms, plan)
    _validate_reverse_grain(layout, atoms_per_split)
    native = _extension()
    if layout.has_reverse:
        return native.compute_input_c_reverse(
            atoms,
            layout.rows,
            layout.reverse_slots,
            layout.reverse_offsets,
            g,
            plan.input.native_config(),
            splits,
            atoms_per_split,
            compensated,
            compensated_cdot,
        )
    arguments = (
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        g,
        plan.output.native_config(),
        plan.input.native_config(),
        True,
    )
    if splits == 1:
        return native.compute_c(*arguments, compensated, compensated_cdot)
    return native.compute_c_split(
        *arguments, splits, atoms_per_split, compensated, compensated_cdot
    )


def parameter_grad(plan, atoms, layout, zx, zy, h, g, *, threads=128, windowed=False):
    """One thread per atom reduces all batches into its original [a,s,muo,mui].

    Analytic derivatives include the effective-radius floor and area-normalized
    coefficient quotient. Bank/owner selection is fixed to this forward's plan.
    H/G and Zx/Zy are reused, with derivative coefficients only in registers.
    """
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("atom threads must be 32,64,128 or 256")
    if type(windowed) is not bool:
        raise ValueError("windowed must be bool")
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
        threads,
        windowed,
    )


def compute_g_and_parameter_grad(
    plan,
    atoms,
    layout,
    zx,
    zy,
    h,
    *,
    threads=128,
    windowed=False,
    specialize_modes=True,
):
    """One thread builds rounded G and all atom gradients, sharing output work.

    G is stored in H's dtype before it participates in gradients, matching the
    separate kernels' intermediate precision. It remains available for dX.
    Specialization is used only when both axes have M8.
    """
    if type(threads) is not int or threads not in (32, 64, 128, 256):
        raise ValueError("atom threads must be 32,64,128 or 256")
    if type(windowed) is not bool:
        raise ValueError("windowed must be bool")
    if type(specialize_modes) is not bool:
        raise ValueError("specialize_modes must be bool")
    layout.validate_for(atoms, plan)
    return _extension().compute_g_and_parameter_grad(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        zx,
        zy,
        h,
        plan.input.native_config(),
        plan.output.native_config(),
        threads,
        windowed,
        specialize_modes,
    )


def compute_g_parameter_grad_cin(
    plan, atoms, layout, zx, zy, h, *, splits=1, windowed=False, atoms_per_split=0
):
    """Build Cin, saved rounded G and original-row atom gradients together.

    Each reverse owner/split visits an atom once. One lane computes its output
    projection and gradients, then reuses the input derivative coefficients for
    the shared C chunk. This native stage supports one to four batches only.
    """
    if type(splits) is not int or not 1 <= splits <= 64:
        raise ValueError("C splits must be an integer in [1,64]")
    if type(windowed) is not bool:
        raise ValueError("windowed must be bool")
    layout.validate_for(atoms, plan)
    _validate_reverse_grain(layout, atoms_per_split)
    if layout.has_reverse:
        return _extension().compute_g_parameter_grad_cin_reverse(
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            layout.reverse_slots,
            layout.reverse_offsets,
            zx,
            zy,
            h,
            plan.input.native_config(),
            plan.output.native_config(),
            splits,
            windowed,
            atoms_per_split,
        )
    return _extension().compute_g_parameter_grad_cin(
        atoms,
        layout.rows,
        layout.groups,
        layout.offsets,
        zx,
        zy,
        h,
        plan.input.native_config(),
        plan.output.native_config(),
        splits,
        windowed,
    )


class _HierarchicalLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        atoms,
        plan,
        layout,
        z_method,
        team_size,
        sites_per_block,
        c_splits,
        fuse_g_grad,
        z_chunk_sites,
        atom_threads,
        window_moments,
        fuse_backward_cin,
        atoms_per_split,
        compensated_values,
        fuse_h_c,
        specialize_g_grad,
        compensated_cdot,
    ):
        zx = build_hierarchy(
            x,
            plan.input,
            method=z_method,
            team_size=team_size,
            chunk_sites=z_chunk_sites,
        )
        if window_moments:
            # The assignment releases raw Z after the transformation. Save only
            # the immutable owner windows needed by H and parameter gradients.
            zx = build_window_moments(zx, plan.input)
        if fuse_h_c and plan.input.modes == 8 and plan.output.modes == 8:
            h, c = compute_h_c(
                plan,
                atoms,
                layout,
                zx,
                splits=c_splits,
                atoms_per_split=atoms_per_split,
                windowed=window_moments,
                compensated=compensated_values,
            )
        else:
            h = compute_h(
                plan,
                atoms,
                layout,
                zx,
                storage_dtype=atoms.dtype,
                threads=atom_threads,
                windowed=window_moments,
                compensated=compensated_values,
            )
            c = compute_c(
                plan,
                atoms,
                layout,
                h,
                splits=c_splits,
                atoms_per_split=atoms_per_split,
                compensated=compensated_values,
                compensated_cdot=compensated_cdot,
            )
        y = expand_c(c, plan.output, x, sites_per_block=sites_per_block)
        ctx.plan = plan
        ctx.z_method = z_method
        ctx.team_size = team_size
        ctx.sites_per_block = sites_per_block
        ctx.c_splits = c_splits
        ctx.fuse_g_grad = fuse_g_grad
        ctx.z_chunk_sites = z_chunk_sites
        ctx.atom_threads = atom_threads
        ctx.window_moments = window_moments
        ctx.fuse_backward_cin = fuse_backward_cin
        ctx.atoms_per_split = atoms_per_split
        ctx.compensated_values = compensated_values
        ctx.specialize_g_grad = specialize_g_grad
        ctx.compensated_cdot = compensated_cdot
        ctx.source_version = atoms._version
        ctx.layout_method = layout.method
        # save_for_backward checks atom AND layout-buffer mutations. Avoid
        # keeping the source x: Zx already contains the needed contractions.
        # Input-only backward does not need Zx/H, so release those immediately.
        saved = (
            atoms,
            layout.rows,
            layout.groups,
            layout.offsets,
            layout.reverse_slots,
            layout.reverse_offsets,
        )
        if ctx.needs_input_grad[1]:
            saved += (zx, h)
        ctx.save_for_backward(*saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        atoms, rows, groups, offsets, reverse_slots, reverse_offsets, *forward_state = (
            ctx.saved_tensors
        )
        plan = ctx.plan
        layout = PreparedLayout(
            rows,
            groups,
            offsets,
            plan,
            atoms,
            ctx.source_version,
            reverse_slots=reverse_slots,
            reverse_offsets=reverse_offsets,
            method=ctx.layout_method,
        )
        layout.validate_for(atoms, plan)
        dy = dy.contiguous()
        zy = build_hierarchy(
            dy,
            plan.output,
            method=ctx.z_method,
            team_size=ctx.team_size,
            chunk_sites=ctx.z_chunk_sites,
        )
        if ctx.window_moments:
            zy = build_window_moments(zy, plan.output)
        dx = datoms = cin = None
        if (
            ctx.fuse_backward_cin
            and ctx.needs_input_grad[0]
            and ctx.needs_input_grad[1]
            and 1 <= dy.shape[0] <= 4
        ):
            zx, h = forward_state
            cin, g, datoms = compute_g_parameter_grad_cin(
                plan,
                atoms,
                layout,
                zx,
                zy,
                h,
                splits=ctx.c_splits,
                windowed=ctx.window_moments,
                atoms_per_split=ctx.atoms_per_split,
            )
        elif ctx.needs_input_grad[1] and ctx.fuse_g_grad:
            zx, h = forward_state
            g, datoms = compute_g_and_parameter_grad(
                plan,
                atoms,
                layout,
                zx,
                zy,
                h,
                threads=ctx.atom_threads,
                windowed=ctx.window_moments,
                specialize_modes=ctx.specialize_g_grad,
            )
        else:
            g = compute_g(
                plan,
                atoms,
                layout,
                zy,
                storage_dtype=atoms.dtype,
                threads=ctx.atom_threads,
                windowed=ctx.window_moments,
                compensated=ctx.compensated_values,
            )
        if ctx.needs_input_grad[0]:
            if cin is None:
                cin = compute_input_c(
                    plan,
                    atoms,
                    layout,
                    g,
                    splits=ctx.c_splits,
                    atoms_per_split=ctx.atoms_per_split,
                    compensated=ctx.compensated_values,
                    compensated_cdot=ctx.compensated_cdot,
                )
            dx = expand_c(cin, plan.input, dy, sites_per_block=ctx.sites_per_block)
            del cin
        if ctx.needs_input_grad[1] and datoms is None:
            zx, h = forward_state
            datoms = parameter_grad(
                plan,
                atoms,
                layout,
                zx,
                zy,
                h,
                g,
                threads=ctx.atom_threads,
                windowed=ctx.window_moments,
            )
        return (
            dx,
            datoms,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def hierarchical_linear(
    plan,
    x,
    atoms,
    *,
    layout=None,
    z_method="merge",
    team_size=4,
    sites_per_block=32,
    c_splits=1,
    fuse_g_grad=False,
    layout_method="sort",
    z_chunk_sites=0,
    atom_threads=128,
    window_moments=False,
    fuse_backward_cin=False,
    reverse_layout=False,
    atoms_per_split=0,
    compensated_values=False,
    fuse_h_c=False,
    specialize_g_grad=True,
    compensated_cdot=False,
):
    """Private research autograd route, using this forward's bank/owner choices.

    Forward saves Zx/H only if atom gradients are requested. Backward builds Zy
    and G once, then uses them for dX and parameter gradients. No dense W, dW,
    packed atom values, global coefficient table, or separate dH is allocated.
    H/G storage uses the input dtype. Opt-in compensated values apply to H,
    output C, standalone G and separate input C; fused gradients stay double.
    The optional fused H/C forward is used for M8 axes, with separate fallback
    for other mode counts.
    specialize_g_grad controls standalone G/gradient fusion; fused Cin is unchanged.
    compensated_cdot applies only to separate output/input C generation; it
    keeps the FP64 fallback and is incompatible with the experimental fusions.
    Double backward and full-route CUDA Graph capture are not supported yet
    (preparation has dynamic output shapes and host checks).
    """
    if type(fuse_h_c) is not bool:
        raise ValueError("fuse_h_c must be bool")
    if type(specialize_g_grad) is not bool:
        raise ValueError("specialize_g_grad must be bool")
    _validate_compensated_cdot(compensated_cdot, atoms, atoms.dtype, plan.input)
    _validate_compensated_cdot(compensated_cdot, atoms, atoms.dtype, plan.output)
    if compensated_cdot and (fuse_h_c or fuse_backward_cin):
        raise ValueError("compensated C dot requires separate H/C and separate Cin")
    _validate_compensated(compensated_values, atoms, atoms.dtype, plan.input)
    _validate_compensated(compensated_values, atoms, atoms.dtype, plan.output)
    if type(reverse_layout) is not bool:
        raise ValueError("reverse_layout must be bool")
    if type(atoms_per_split) is not int or not 0 <= atoms_per_split <= 65536:
        raise ValueError("atoms_per_split must be an integer in [0,65536]")
    if atoms_per_split and not (
        reverse_layout or (layout is not None and layout.has_reverse)
    ):
        raise ValueError("atoms_per_split requires a reverse layout")
    if reverse_layout and layout is None and layout_method != "counting":
        raise ValueError("reverse layout requires counting preparation")
    if reverse_layout and layout is not None and not layout.has_reverse:
        raise ValueError("cached layout lacks reverse indices; rebuild it")
    if type(window_moments) is not bool:
        raise ValueError("window_moments must be bool")
    if type(atom_threads) is not int or atom_threads not in (32, 64, 128, 256):
        raise ValueError("atom threads must be 32,64,128 or 256")
    if layout_method not in ("sort", "counting", "counting_compact"):
        raise ValueError("layout method must be sort, counting or counting_compact")
    if type(c_splits) is not int or not 1 <= c_splits <= 64:
        raise ValueError("C splits must be an integer in [1,64]")
    if type(fuse_g_grad) is not bool:
        raise ValueError("fuse_g_grad must be bool")
    if type(fuse_backward_cin) is not bool:
        raise ValueError("fuse_backward_cin must be bool")
    # Larger/empty batches and one-sided gradients use the established route.
    # Enabling Cin fusion also enables G/gradient fusion for that fallback.
    fuse_g_grad = fuse_g_grad or fuse_backward_cin
    if x.device != atoms.device or x.dtype != atoms.dtype:
        raise ValueError("X and atoms must share device and dtype")
    if layout is None and layout_method in ("counting", "counting_compact"):
        # One error word validates both X and atoms; no separate all/item sync.
        layout = prepare_layout(
            plan, atoms, method=layout_method, signal=x, reverse=reverse_layout
        )
    else:
        if not bool(torch.isfinite(x).all()):
            raise ValueError("X must be finite")
        if layout is None:
            layout = prepare_layout(plan, atoms, method=layout_method)
        else:
            layout.validate_for(atoms, plan)
    return _HierarchicalLinear.apply(
        x,
        atoms,
        plan,
        layout,
        z_method,
        team_size,
        sites_per_block,
        c_splits,
        fuse_g_grad,
        z_chunk_sites,
        atom_threads,
        window_moments,
        fuse_backward_cin,
        atoms_per_split,
        compensated_values,
        fuse_h_c,
        specialize_g_grad,
        compensated_cdot,
    )
