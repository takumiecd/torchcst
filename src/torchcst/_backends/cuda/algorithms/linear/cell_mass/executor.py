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

from torchcst._backends.torch.kernels.cell_mass import validate_inputs
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
    if atoms.shape[0] > 1024:
        raise ValueError("CUDA prototype supports at most 1024 atoms")
    for axis in (spec.input, spec.output):
        if axis.site_count > 128 or axis.site_count & (axis.site_count - 1):
            raise ValueError("first CUDA prototype requires power-of-two sites <=128")
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
    ):
        native = _extension()
        rows = prepare_layout(spec, atoms)
        input_config = _axis_config(spec.input, coefficient_method)
        output_config = _axis_config(spec.output, coefficient_method)

        # Same stream: H is complete before its consumers start. Z is global;
        # atom parameters are never copied into a second global parameter array.
        zx = native.build_z(x, input_config, z_method == "merge", z_threads_per_element)
        h = native.compute_h(x, atoms, rows, zx, input_config, atom_threads_per_block)
        y = native.compute_y(h, atoms, rows, output_config, sites_per_block, atom_splits)

        ctx.spec = spec
        ctx.z_method = z_method
        ctx.z_threads_per_element = z_threads_per_element
        ctx.coefficient_method = coefficient_method
        ctx.sites_per_block = sites_per_block
        ctx.atom_splits = atom_splits
        ctx.atom_threads_per_block = atom_threads_per_block
        ctx.save_for_backward(x, atoms, rows, zx, h)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        native = _extension()
        x, atoms, rows, zx, h = ctx.saved_tensors
        # Only dY may need canonicalization. Saved X/atoms are original tensors;
        # autograd's version checks reject mutation before their backward use.
        dy = grad_output.contiguous()
        input_config = _axis_config(ctx.spec.input, ctx.coefficient_method)
        output_config = _axis_config(ctx.spec.output, ctx.coefficient_method)

        zy = native.build_z(
            dy, output_config, ctx.z_method == "merge", ctx.z_threads_per_element
        )
        g = native.compute_g(
            dy, atoms, rows, zy, output_config, ctx.atom_threads_per_block
        )
        dx = native.compute_dx(
            g, atoms, rows, input_config, ctx.sites_per_block, ctx.atom_splits
        )
        grad_atoms = native.compute_parameter_grad(
            x,
            dy,
            atoms,
            rows,
            h,
            g,
            zx,
            zy,
            input_config,
            output_config,
            ctx.atom_threads_per_block,
        )
        return dx, grad_atoms, None, None, None, None, None, None, None


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
):
    """Run the first-order tile-window approximation with continuous mass.

    Validation currently reads reduction results on the host; this research
    entry does not yet support CUDA Graph capture. sites_per_block is an execution
    limit for Y/dX (8, 16 or 32), separate from the profile's tile_sites and period.
    atom_splits (1, 2, 4 or 8) partitions their atom reduction across blocks.
    Split outputs use temporary global storage and a unique-writer reduction;
    they are not saved for backward. Neither knob changes the mathematical axis.
    atom_threads_per_block (16/32/64/128) controls H/G/parameter-gradient blocks;
    each thread still owns a different atom. Z and Y/dX keep 128-thread blocks.
    Tiny inputs retain the existing four-leaf bound. Build/JIT time is separate
    from execution timing. All inputs and support restrictions are explicit.
    """
    validate_inputs(spec, x, atoms)
    _validate_cuda_support(spec, x, atoms)
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
    if type(sites_per_block) is not int or sites_per_block not in (8, 16, 32):
        raise ValueError("sites_per_block must be 8, 16 or 32")
    if type(atom_splits) is not int or atom_splits not in (1, 2, 4, 8):
        raise ValueError("atom_splits must be 1, 2, 4 or 8")
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
    )
