"""Readable five-kernel CUDA route for the physical cell-mass prototype.

This numeric research entry is deliberately not registered with CSTLinear or
the public dispatcher. It accepts physical rows, not an old Polar/L2 checkpoint.
The JIT compiler is imported only when this CUDA entry is actually executed.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.kernels.cell_mass import validate_input_structure
from torchcst.kernels.cell_mass import CellMassLinearSpec


def _axis_config(axis, coefficient_method="polynomial"):
    if coefficient_method not in ("polynomial", "dft"):
        raise ValueError("coefficient_method must be polynomial or dft")
    return [
        axis.site_count,
        axis.spacing,
        axis.tile_sites,
        axis.origin,
        axis.profile.modes,
        axis.profile.tail_bound,
        int(coefficient_method == "polynomial"),
        axis.period_tiles,
    ]


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    sources = [directory / "bindings.cpp", directory / "kernels.cu"]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:12]
    return load(
        name=f"torchcst_cell_mass_{digest}",
        sources=[str(p) for p in sources],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def _validate_cuda_support(spec, x, atoms):
    if x.device.type != "cuda":
        raise ValueError("cell-mass CUDA entry requires CUDA inputs")
    if atoms.shape[0] > 65536:
        raise ValueError("CUDA prototype supports at most 65536 atoms")
    for axis in (spec.input, spec.output):
        if axis.site_count > 1024 or axis.site_count & (axis.site_count - 1):
            raise ValueError("first CUDA prototype requires power-of-two sites <=1024")
        if axis.profile.modes > 16:
            raise ValueError("first CUDA prototype supports at most 16 Fourier modes")
        if axis.tile_sites & (axis.tile_sites - 1):
            raise ValueError("CUDA prototype requires power-of-two tile_sites")


def prepare_layout(spec, atoms):
    """Sort only indices; keep the sole parameter array in its original order.

    The first supported case has one fixed Fourier bank per axis. Whole-cell
    tile boundaries start at origin-spacing/2. The key is
    (output center tile, input center tile); multiple width banks will extend it
    with a hierarchy/bank prefix. No atom or coefficient values are packed.
    This standalone helper is also the Torch sorting reference. Normal forward
    uses prepare_inputs_and_layout to combine key generation with value checks.
    """
    input_leaf = spec.input.tile_sites
    output_leaf = spec.output.tile_sites
    qi = (
        (
            (
                (atoms[:, 3] - spec.input.origin).remainder(spec.input.period)
                + spec.input.spacing / 2
            ).remainder(spec.input.period)
            / (input_leaf * spec.input.spacing)
        )
        .floor()
        .long()
    )
    qo = (
        (
            (
                (atoms[:, 2] - spec.output.origin).remainder(spec.output.period)
                + spec.output.spacing / 2
            ).remainder(spec.output.period)
            / (output_leaf * spec.output.spacing)
        )
        .floor()
        .long()
    )
    key = qo * (spec.input.site_count // input_leaf) + qi
    return torch.argsort(key, stable=True)


def prepare_inputs_and_layout(
    native, x, atoms, input_config, output_config, *, with_groups=False
):
    """Generate only sort keys and validate values in one CUDA traversal.

    Structural checks have already run on the host. The single scalar read
    below preserves finite/width rejection before Z/H/Y can consume bad values.
    Keys and the error word are temporary; only sorted row indices survive.
    """
    keys, error_word = native.prepare_inputs(x, atoms, input_config, output_config)
    errors = error_word.item()
    if errors & 3:
        raise ValueError("X and atoms must be finite")
    if errors & 4:
        raise ValueError("sigma must lie in (0, min(profile_period)/3]")
    rows = torch.argsort(keys, stable=True)
    if not with_groups:
        return rows
    # A group is one exact (output owner,input owner) pair. The sorted row
    # ranges serve both directions without a second permutation or atom copy.
    input_tiles = int(input_config[0] / input_config[2])
    output_tiles = int(output_config[0] / output_config[2])
    boundaries = torch.arange(
        input_tiles * output_tiles + 1, device=keys.device, dtype=torch.int64
    )
    offsets = torch.searchsorted(keys[rows], boundaries)
    return rows, offsets


class _CellMassFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        atoms,
        spec,
        z_method,
        z_threads_per_element,
        coefficient_method,
        sites_per_block,
        atom_splits,
        atom_threads_per_block,
        cache_projection_derivatives,
        double_intermediates,
        owner_coefficient_reduction,
        fuse_g_parameter_grad,
    ):
        native = _extension()
        input_config = _axis_config(spec.input, coefficient_method)
        output_config = _axis_config(spec.output, coefficient_method)
        prepared = prepare_inputs_and_layout(
            native, x, atoms, input_config, output_config,
            with_groups=owner_coefficient_reduction,
        )
        if owner_coefficient_reduction:
            rows, group_offsets = prepared
        else:
            rows, group_offsets = prepared, None

        # Same stream: H is complete before its consumers start. Z is global;
        # atom parameters are never copied into a second global parameter array.
        zx = native.build_z(
            x, input_config, z_method == "merge", z_threads_per_element, double_intermediates
        )
        if cache_projection_derivatives:
            h, h_sigma, h_center = native.compute_h_with_derivatives(
                x, atoms, rows, zx, input_config, atom_threads_per_block
            )
            del zx  # Cached pullbacks no longer need the input hierarchy.
        else:
            h = native.compute_h(x, atoms, rows, zx, input_config, atom_threads_per_block)
        if owner_coefficient_reduction:
            y = native.compute_y_from_owners(
                h, atoms, rows, group_offsets, input_config, output_config,
                sites_per_block, atom_splits,
            )
        else:
            y = native.compute_y(h, atoms, rows, output_config, sites_per_block, atom_splits)

        ctx.spec = spec
        ctx.z_method = z_method
        ctx.z_threads_per_element = z_threads_per_element
        ctx.coefficient_method = coefficient_method
        ctx.sites_per_block = sites_per_block
        ctx.atom_splits = atom_splits
        ctx.atom_threads_per_block = atom_threads_per_block
        ctx.cache_projection_derivatives = cache_projection_derivatives
        ctx.double_intermediates = double_intermediates
        ctx.owner_coefficient_reduction = owner_coefficient_reduction
        ctx.fuse_g_parameter_grad = fuse_g_parameter_grad
        if cache_projection_derivatives:
            # Keep original X/atoms for their autograd mutation/version checks.
            ctx.save_for_backward(x, atoms, rows, h, h_sigma, h_center, group_offsets)
        elif fuse_g_parameter_grad:
            # H is consumed by Y above. The fused pullback reprojects its value
            # and derivatives, so backward only retains the input hierarchy.
            ctx.save_for_backward(x, atoms, rows, zx, group_offsets)
        else:
            ctx.save_for_backward(x, atoms, rows, zx, h, group_offsets)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        native = _extension()
        if ctx.cache_projection_derivatives:
            x, atoms, rows, h, h_sigma, h_center, group_offsets = ctx.saved_tensors
        elif ctx.fuse_g_parameter_grad:
            x, atoms, rows, zx, group_offsets = ctx.saved_tensors
        else:
            x, atoms, rows, zx, h, group_offsets = ctx.saved_tensors
        # Only dY may need canonicalization. Saved X/atoms are original tensors;
        # autograd's version checks reject mutation before their backward use.
        dy = grad_output.contiguous()
        input_config = _axis_config(ctx.spec.input, ctx.coefficient_method)
        output_config = _axis_config(ctx.spec.output, ctx.coefficient_method)

        zy = native.build_z(
            dy, output_config, ctx.z_method == "merge", ctx.z_threads_per_element,
            ctx.double_intermediates
        )
        if ctx.cache_projection_derivatives:
            g, g_sigma, g_center = native.compute_g_with_derivatives(
                dy, atoms, rows, zy, output_config, ctx.atom_threads_per_block
            )
            del zy
        elif ctx.fuse_g_parameter_grad:
            g, grad_atoms = native.compute_g_parameter_grad(
                x, dy, atoms, rows, zx, zy, input_config, output_config,
                ctx.atom_threads_per_block,
            )
        else:
            g = native.compute_g(
                dy, atoms, rows, zy, output_config, ctx.atom_threads_per_block
            )
        if ctx.owner_coefficient_reduction:
            dx = native.compute_dx_from_owners(
                g, atoms, rows, group_offsets, input_config, output_config,
                ctx.sites_per_block, ctx.atom_splits,
            )
        else:
            dx = native.compute_dx(
                g, atoms, rows, input_config, ctx.sites_per_block, ctx.atom_splits
            )
        if ctx.cache_projection_derivatives:
            grad_atoms = native.compute_parameter_grad_from_cache(
                atoms, rows, h, g, h_sigma, h_center, g_sigma, g_center,
                ctx.atom_threads_per_block,
            )
        elif not ctx.fuse_g_parameter_grad:
            grad_atoms = native.compute_parameter_grad(
                x, dy, atoms, rows, h, g, zx, zy, input_config, output_config,
                ctx.atom_threads_per_block,
            )
        return dx, grad_atoms, None, None, None, None, None, None, None, None, None, None, None


def physical_linear(
    spec: CellMassLinearSpec,
    x,
    atoms,
    *,
    z_method="direct",
    z_threads_per_element=4,
    coefficient_method="polynomial",
    sites_per_block=32,
    atom_splits=1,
    atom_threads_per_block=128,
    cache_projection_derivatives=False,
    double_intermediates=False,
    owner_coefficient_reduction=False,
    fuse_g_parameter_grad=False,
):
    """Run the first-order tile-window approximation with continuous mass.

    Validation reads one error word on the host; this research
    entry does not yet support CUDA Graph capture. sites_per_block is an execution
    limit for Y/dX (powers of two from 8 to 256), separate from the profile's tile_sites and period.
    atom_splits (powers of two from 1 to 64) partitions their atom reduction across blocks.
    Split outputs use temporary global storage and a unique-writer reduction;
    they are not saved for backward. Neither knob changes the mathematical axis.
    atom_threads_per_block (16/32/64/128) controls H/G/parameter-gradient blocks;
    each thread still owns a different atom. Z and Y/dX keep 128-thread blocks.
    cache_projection_derivatives saves four [batch,atom] float64 pullbacks so
    parameter gradients reuse H/G work. This trades bounded memory for recompute;
    no per-site/per-mode Jacobian is stored. The default retains recomputation.
    double_intermediates preserves Z as float64 and, when caching derivatives,
    also preserves H/G as float64. Recomputed pullbacks use their double values
    directly. Inputs, outputs and parameter gradients retain the input dtype.
    This can matter when a near-zero gradient is amplified by AdamW.
    owner_coefficient_reduction builds begin/end metadata for exact owner groups.
    Each atom generates C once for its owner; bounded double C partials are
    combined across neighbouring owners before a single site expansion.
    It supports up to 128 atom splits without a second atom permutation.
    fuse_g_parameter_grad writes G while recomputing parameter pullbacks and
    releases forward H after Y; it is available with derivative recomputation.
    Each block covers at most four profile tiles, independent of site width.
    Build/JIT time is separate from execution timing. Support is explicit.
    """
    validate_input_structure(spec, x, atoms)
    _validate_cuda_support(spec, x, atoms)
    if type(owner_coefficient_reduction) is not bool:
        raise ValueError("owner_coefficient_reduction must be bool")
    if type(fuse_g_parameter_grad) is not bool:
        raise ValueError("fuse_g_parameter_grad must be bool")
    if type(double_intermediates) is not bool:
        raise ValueError("double_intermediates must be bool")
    if type(cache_projection_derivatives) is not bool:
        raise ValueError("cache_projection_derivatives must be bool")
    if fuse_g_parameter_grad and cache_projection_derivatives:
        raise ValueError("fuse_g_parameter_grad requires derivative recomputation")
    if z_method not in ("direct", "merge"):
        raise ValueError("z_method must be direct or merge")
    if (
        type(z_threads_per_element) is not int
        or not 1 <= z_threads_per_element <= 128
        or z_threads_per_element & (z_threads_per_element - 1)
    ):
        raise ValueError("Z thread count must be a power of two in [1,128]")
    if coefficient_method not in ("polynomial", "dft"):
        raise ValueError("coefficient_method must be polynomial or dft")
    if type(sites_per_block) is not int or sites_per_block not in (8, 16, 32, 64, 128, 256):
        raise ValueError("sites_per_block must be a power of two in [8,256]")
    allowed_splits = (1, 2, 4, 8, 16, 32, 64, 128) if owner_coefficient_reduction else (1, 2, 4, 8, 16, 32, 64)
    if type(atom_splits) is not int or atom_splits not in allowed_splits:
        raise ValueError("atom_splits must be a supported power of two (at most 64, or 128 for owner C)")
    if type(atom_threads_per_block) is not int or atom_threads_per_block not in (
        16,
        32,
        64,
        128,
    ):
        raise ValueError("atom_threads_per_block must be 16, 32, 64 or 128")
    return _CellMassFunction.apply(
        x,
        atoms,
        spec,
        z_method,
        z_threads_per_element,
        coefficient_method,
        sites_per_block,
        atom_splits,
        atom_threads_per_block,
        cache_projection_derivatives,
        double_intermediates,
        owner_coefficient_reduction,
        fuse_g_parameter_grad,
    )
