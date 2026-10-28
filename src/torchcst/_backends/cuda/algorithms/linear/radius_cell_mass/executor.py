"""Opt-in native radius pipeline, with explicit and inspectable stage boundaries.

This first implementation saves complete responses and input moments for
backward. It establishes a numerical baseline, not a low-memory speed result.
It never materializes weights, packed atoms, or global atom Fourier tables.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from .algorithm import RadiusCellMassAlgorithm
from .plan import HierarchyPlan
from .recipe import RadiusCellMassRecipe


@lru_cache(maxsize=1)
def _extension():
    """Compile only this radius implementation; never load the sigma binary."""
    from torch.utils.cpp_extension import load

    directory = Path(__file__).parent
    sources = [directory / "pipeline.cu", directory / "common.cuh"]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:12]
    return load(
        name=f"torchcst_radius_cell_mass_{digest}",
        sources=[str(sources[0])],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--ftz=false"],
        verbose=False,
    )


def prepare_layout(plan, atoms, *, reverse=False):
    """Return original-row indices and first-tile begin/end offsets.

    Sort order is (output bank/tile, input bank/tile), reversed for dX. The
    numerical atom Parameter stays in place. Temporary keys/counts die here;
    the returned rows and offsets are the only retained routing arrays.

    One error-word read rejects invalid radii/parameters before native gathers.
    This correctness baseline is eager-only and does not claim Graph capture.
    """
    if type(reverse) is not bool or type(plan) is not HierarchyPlan:
        raise ValueError("expected a radius hierarchy and bool reverse")
    keys, error = _extension().classify_atoms(
        atoms, plan.input.native_config(), plan.output.native_config()
    )
    if error.item():
        raise ValueError(
            "atoms must be finite, with radii in (0, chart half-period] and representable radius/tile ratios"
        )
    input_node = keys.remainder(plan.input.node_count)
    output_node = torch.div(keys, plan.input.node_count, rounding_mode="floor")
    if reverse:
        keys = input_node * plan.output.node_count + output_node
        owner, count = input_node, plan.input.node_count
    else:
        owner, count = output_node, plan.output.node_count
    rows = torch.argsort(keys, stable=True)
    counts = torch.bincount(owner, minlength=count)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return rows, offsets


def build_moments(signal, axis, *, method="merge"):
    """Build [batch, bank-node, real basis] cell-integrated moments.

    Direct evaluates every frequency from signal. Merge inherits DC and
    harmonics divisible by merge_factor from children; unmatched frequencies
    still read signal. Levels depend on children, while elements at a level
    are parallel. Both methods compute the same finite Fourier sums.
    """
    if method not in ("direct", "merge"):
        raise ValueError("moment method must be direct or merge")
    return _extension().build_moments(
        signal.contiguous(), axis.native_config(), method == "merge"
    )


class _RadiusLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, atoms, plan, recipe):
        native = _extension()
        rows, offsets = prepare_layout(plan, atoms)

        # Input tile/basis owners publish moments first. Response threads then
        # finish every input-tile contribution of their own atom and batch.
        input_moments = build_moments(x, plan.input, method=recipe.moment_method)
        atom_response = native.compute_response(
            atoms, input_moments, plan.input.native_config(), False
        )

        # Each output-tile owner reads final responses in sorted atom ranges.
        # output_coeffs is forward scratch and is not checkpointed for backward.
        output_coeffs = native.compute_coefficients(
            atoms, rows, offsets, atom_response, plan.output.native_config(), True
        )
        y = native.expand_coefficients(output_coeffs, x, plan.output.native_config())
        ctx.plan, ctx.recipe = plan, recipe
        ctx.save_for_backward(atoms, input_moments, atom_response)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        atoms, input_moments, atom_response = ctx.saved_tensors
        plan, recipe = ctx.plan, ctx.recipe
        native = _extension()

        # The transposed path uses exactly the same moment/response builders.
        grad_output = grad_output.contiguous()
        grad_output_moments = build_moments(
            grad_output, plan.output, method=recipe.moment_method
        )
        grad_response = native.compute_response(
            atoms, grad_output_moments, plan.output.native_config(), True
        )
        grad_atoms = None
        if ctx.needs_input_grad[1]:
            grad_atoms = native.parameter_grad(
                atoms,
                input_moments,
                grad_output_moments,
                atom_response,
                grad_response,
                plan.input.native_config(),
                plan.output.native_config(),
            )

        grad_input = None
        if ctx.needs_input_grad[0]:
            rows, offsets = prepare_layout(plan, atoms, reverse=True)
            grad_input_coeffs = native.compute_coefficients(
                atoms, rows, offsets, grad_response, plan.input.native_config(), False
            )
            grad_input = native.expand_coefficients(
                grad_input_coeffs, grad_output, plan.input.native_config()
            )
        return grad_input, grad_atoms, None, None


def radius_linear(plan, x, atoms, *, recipe=None):
    """Execute a bounded first-order research route through an explicit Plan.

    Initialize CSTLinear with the CPU reference and then move it to CUDA.
    This stage supports N<=128, M<=16 and A<=4096, without autocast/Graph or
    higher-order derivatives. These limits are execution choices, not API math.
    """
    if type(plan) is not HierarchyPlan:
        raise TypeError("expected a radius HierarchyPlan")
    if recipe is None:
        recipe = RadiusCellMassRecipe()
    RadiusCellMassAlgorithm().validate_recipe(recipe)
    if (
        x.device.type != "cuda"
        or atoms.device != x.device
        or x.dtype not in (torch.float32, torch.float64)
        or atoms.dtype != x.dtype
        or x.ndim != 2
        or x.shape[1] != plan.input.site_count
        or atoms.ndim != 2
        or atoms.shape[1] != 4
        or atoms.shape[0] > 4096
    ):
        raise ValueError("expected matching CUDA float32/64 signal and [atom,4] rows")
    work_per_batch = max(
        plan.input.node_count * plan.input.basis_count,
        plan.output.node_count * plan.output.basis_count,
        atoms.shape[0],
    )
    if x.shape[0] * work_per_batch > 2**31 - 128:
        raise ValueError("radius baseline exceeds its native 32-bit work-index limit")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("radius baseline requires execution without autocast")
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("radius baseline has not qualified deterministic execution")
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("radius baseline uses eager preparation, not CUDA Graphs")
        return _RadiusLinear.apply(x.contiguous(), atoms.contiguous(), plan, recipe)
