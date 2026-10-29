"""Opt-in native radius pipeline, with explicit and inspectable stage boundaries.

The reference saves input moments; hybrid routes omit direct levels. Projection
builds moments/response once, then output tiles reuse the completed global values
and fuse C/site expansion in shared. These are numerical candidates, not complete
training-step speed results. No weights, packed atoms or global atom Fourier
tables are materialized.
"""

import hashlib
from functools import cache, lru_cache
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
    source = directory / "pipeline.cu"
    sources = [source, *sorted(directory.glob("*.cuh"))]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()[:12]
    return load(
        name=f"torchcst_radius_cell_mass_{digest}",
        sources=[str(source)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--ftz=false"],
        verbose=False,
    )


def _classify_layout_keys(plan, atoms):
    """Classify current atoms once, validating on device without a CPU wait.

    Each integer key contains the two bank/tile owners, never atom values.
    A fused forward may save these keys for its backward, whose saved atom
    Tensor already enforces PyTorch's version check before they are reused.
    Invalid CUDA values raise a device-side assertion, reported at a later CUDA
    API or synchronization point. Shape/configuration errors stay CPU errors.
    """
    return _extension().classify_atoms_async(
        atoms, plan.input.native_config(), plan.output.native_config()
    )


def prepare_layout_checked(plan, atoms, *, reverse=False):
    """Synchronous diagnostic path; never called by normal Linear execution.

    This preserves immediate ValueError reporting for deliberate invalid-input
    inspection. Training uses device validation through prepare_layout instead.
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
    return _ranges_from_layout_keys(plan, keys, reverse=reverse)


def _ranges_from_layout_keys(plan, keys, *, reverse=False):
    """Sort validated owner keys into forward or reverse original-row ranges.

    This does not decode atom parameters or read an error word to the CPU.
    It is safe only for keys from the same unchanged forward atom Tensor.
    """
    if reverse:
        input_node = keys.remainder(plan.input.node_count)
        output_node = torch.div(keys, plan.input.node_count, rounding_mode="floor")
        keys = input_node * plan.output.node_count + output_node
        stride, count = plan.output.node_count, plan.input.node_count
    else:
        stride, count = plan.input.node_count, plan.output.node_count
    sorted_keys, rows = torch.sort(keys, stable=True)
    # Composite keys for owner q lie in [q*stride, (q+1)*stride).
    # Searching these fixed boundaries also represents empty owner ranges.
    # Unlike bincount, the output size needs no maximum read from the GPU.
    boundaries = torch.arange(
        0, (count + 1) * stride, stride, dtype=keys.dtype, device=keys.device
    )
    offsets = torch.searchsorted(sorted_keys, boundaries, right=False)
    return rows, offsets


def prepare_layout(plan, atoms, *, reverse=False):
    """Return original-row indices and first-tile begin/end offsets.

    Sort order is (output bank/tile, input bank/tile), reversed for dX.
    Standalone preparation releases its keys. Fused autograd separately saves
    one owner key per atom when dX is needed, avoiding a second classification
    in backward. No atom values are packed or read back to the CPU.
    Preparation remains eager-only; fixed boundaries replace owner counting.
    """
    if type(reverse) is not bool or type(plan) is not HierarchyPlan:
        raise ValueError("expected a radius hierarchy and bool reverse")
    keys = _classify_layout_keys(plan, atoms)
    return _ranges_from_layout_keys(plan, keys, reverse=reverse)


def build_moments(signal, axis, *, method="merge", start_level=0):
    """Build [batch, bank-node, real basis] cell-integrated moments.

    Direct evaluates every frequency from signal. Merge inherits DC and
    harmonics divisible by merge_factor from children; unmatched frequencies
    still read signal. Levels depend on children, while elements at a level
    are parallel. Both methods compute the same finite Fourier sums.
    """
    if method not in ("direct", "merge"):
        raise ValueError("moment method must be direct or merge")
    axis.prefix_nodes(start_level)
    if start_level:
        return _extension().build_compact_moments(
            signal.contiguous(), axis.native_config(), method == "merge", start_level
        )
    return _extension().build_moments(
        signal.contiguous(), axis.native_config(), method == "merge"
    )


@cache
def _moment_stream(device_index):
    """A reusable auxiliary stream; it reserves no SMs or tensor workspace."""
    return torch.cuda.Stream(device=device_index)


def _complete_hybrid_response(signal, atoms, axis, start_level, recipe, output_side):
    """Fork independent direct and moment work, then join final responses.

    Low/high kernels write disjoint atom entries of the same response tensor.
    The join precedes every coefficient/gradient consumer. Stream separation
    permits concurrency; it does not guarantee concurrent SM residency.
    """
    native = _extension()
    config = axis.native_config()
    response = torch.empty(
        (signal.shape[0], atoms.shape[0]), device=signal.device, dtype=torch.float64
    )
    if recipe.schedule == "overlap":
        parent = torch.cuda.current_stream(signal.device)
        auxiliary = _moment_stream(signal.device.index)
        # Fork AFTER response allocation but BEFORE the direct kernel. Waiting
        # after launching direct would accidentally serialize the two branches.
        auxiliary.wait_stream(parent)
        native.direct_response_into(
            signal, atoms, response, config, output_side, start_level
        )
        with torch.cuda.stream(auxiliary):
            signal.record_stream(auxiliary)
            atoms.record_stream(auxiliary)
            response.record_stream(auxiliary)
            moments = build_moments(
                signal, axis, method=recipe.moment_method, start_level=start_level
            )
            native.complete_response(
                atoms, moments, response, config, output_side, start_level
            )
        parent.wait_stream(auxiliary)
        moments.record_stream(parent)
    else:
        native.direct_response_into(
            signal, atoms, response, config, output_side, start_level
        )
        moments = build_moments(
            signal, axis, method=recipe.moment_method, start_level=start_level
        )
        native.complete_response(
            atoms, moments, response, config, output_side, start_level
        )
    return moments, response


class _BlockLocalRadiusLinear(torch.autograd.Function):
    """Shared forward projection/expansion, reverse, then atom gradients.

    prepare_layout supplies routing metadata outside these compute kernels.
    Each transform builds its projection caches only once, then every output
    interval consumes those caches. Coefficients stay in shared scratch.
    """

    @staticmethod
    def forward(ctx, x, atoms, plan, recipe):
        native = _extension()
        layout_keys = _classify_layout_keys(plan, atoms)
        rows, offsets = _ranges_from_layout_keys(plan, layout_keys)
        forward = (
            native.shared_forward_parallel
            if recipe.schedule == "block_local_parallel"
            else native.shared_forward
        )
        y, input_moments, atom_response = forward(
            x,
            atoms,
            rows,
            offsets,
            plan.input.native_config(),
            plan.output.native_config(),
            recipe.input_direct_levels,
            recipe.output_direct_levels,
            recipe.moment_method == "merge",
            recipe.output_tile_sites,
        )
        ctx.plan, ctx.recipe = plan, recipe
        # X/dY are still used for derivatives of direct atoms. The hierarchy
        # moments and H are actual completed caches, not empty placeholders.
        # Only dX needs reverse routing. This is 8*A bytes of integer metadata,
        # not a second atom table; atom-only backward retains no routing keys.
        ctx.save_for_backward(
            x,
            atoms,
            input_moments,
            atom_response,
            layout_keys if ctx.needs_input_grad[0] else None,
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, atoms, input_moments, atom_response, layout_keys = ctx.saved_tensors
        plan, recipe = ctx.plan, ctx.recipe
        native = _extension()
        dy = dy.contiguous()
        grad_input = None
        if ctx.needs_input_grad[0]:
            rows, offsets = _ranges_from_layout_keys(plan, layout_keys, reverse=True)
            # The SAME transform runs in reverse: one dY projection produces
            # moments/G, then all dX tiles reuse G. Cin stays in shared memory.
            grad_input, grad_output_moments, grad_response = (
                native.shared_backward_parallel
                if recipe.schedule == "block_local_parallel"
                else native.shared_backward
            )(
                dy,
                atoms,
                rows,
                offsets,
                plan.input.native_config(),
                plan.output.native_config(),
                recipe.input_direct_levels,
                recipe.output_direct_levels,
                recipe.moment_method == "merge",
                recipe.output_tile_sites,
            )
        elif ctx.needs_input_grad[1]:
            # Atom-only backward publishes the two required caches in ONE
            # projection kernel; it has no dX expansion or routing sort.
            project = (
                native.shared_project_parallel
                if recipe.schedule == "block_local_parallel"
                else native.shared_project
            )
            grad_output_moments, grad_response = project(
                dy,
                atoms,
                plan.output.native_config(),
                True,
                recipe.output_direct_levels,
                recipe.moment_method == "merge",
            )
        else:
            return None, None, None, None

        grad_atoms = None
        if ctx.needs_input_grad[1]:
            parameter_grad = (
                native.hybrid_parameter_grad_parallel
                if recipe.schedule == "block_local_parallel"
                else native.hybrid_parameter_grad
            )
            grad_atoms = parameter_grad(
                atoms,
                x,
                dy,
                input_moments,
                grad_output_moments,
                atom_response,
                grad_response,
                plan.input.native_config(),
                plan.output.native_config(),
                recipe.input_direct_levels,
                recipe.output_direct_levels,
            )
        return grad_input, grad_atoms, None, None


class _HybridRadiusLinear(torch.autograd.Function):
    """Staged sequential/stream-overlap comparisons to the fused transform."""

    @staticmethod
    def forward(ctx, x, atoms, plan, recipe):
        native = _extension()
        rows, offsets = prepare_layout(plan, atoms)
        moments, response = _complete_hybrid_response(
            x, atoms, plan.input, recipe.input_direct_levels, recipe, False
        )
        coefficients = native.compact_coefficients(
            atoms,
            rows,
            offsets,
            response,
            plan.output.native_config(),
            True,
            recipe.output_direct_levels,
        )
        y = native.expand_hybrid(
            atoms,
            rows,
            offsets,
            response,
            coefficients,
            x,
            plan.output.native_config(),
            True,
            recipe.output_direct_levels,
        )
        ctx.plan, ctx.recipe = plan, recipe
        ctx.save_for_backward(x, atoms, moments, response)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, atoms, input_moments, response = ctx.saved_tensors
        plan, recipe = ctx.plan, ctx.recipe
        native = _extension()
        dy = dy.contiguous()
        # Prepare routing before the expensive projections, so its error-word
        # validation does not force a CPU wait for an already-launched G.
        rows, offsets = (
            prepare_layout(plan, atoms, reverse=True)
            if ctx.needs_input_grad[0]
            else (None, None)
        )
        output_moments, grad_response = _complete_hybrid_response(
            dy, atoms, plan.output, recipe.output_direct_levels, recipe, True
        )
        # Finish dX first. Its output moments and complete G remain live for
        # atom gradients; neither is rebuilt by the parameter-gradient stage.
        grad_input = None
        if ctx.needs_input_grad[0]:
            coefficients = native.compact_coefficients(
                atoms,
                rows,
                offsets,
                grad_response,
                plan.input.native_config(),
                False,
                recipe.input_direct_levels,
            )
            grad_input = native.expand_hybrid(
                atoms,
                rows,
                offsets,
                grad_response,
                coefficients,
                dy,
                plan.input.native_config(),
                False,
                recipe.input_direct_levels,
            )
            # Cin is not an input to atom gradients. Release it before the
            # parameter-gradient stage while retaining the shared moments/G.
            del coefficients

        grad_atoms = None
        if ctx.needs_input_grad[1]:
            grad_atoms = native.hybrid_parameter_grad(
                atoms,
                x,
                dy,
                input_moments,
                output_moments,
                response,
                grad_response,
                plan.input.native_config(),
                plan.output.native_config(),
                recipe.input_direct_levels,
                recipe.output_direct_levels,
            )
        return grad_input, grad_atoms, None, None


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
        # The dX path produces the shared output moments/G once. Consume those
        # same tensors for atom gradients after dX, without another projection.
        grad_input = None
        if ctx.needs_input_grad[0]:
            rows, offsets = prepare_layout(plan, atoms, reverse=True)
            grad_input_coeffs = native.compute_coefficients(
                atoms, rows, offsets, grad_response, plan.input.native_config(), False
            )
            grad_input = native.expand_coefficients(
                grad_input_coeffs, grad_output, plan.input.native_config()
            )
            del grad_input_coeffs

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
    zi = plan.input.active_nodes(recipe.input_direct_levels) * plan.input.basis_count
    co = plan.output.active_nodes(recipe.output_direct_levels) * plan.output.basis_count
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
    if recipe.schedule in ("block_local", "block_local_parallel"):
        if 8 * (atoms.shape[0] + max(zi, co)) > 48 * 1024:
            raise ValueError("block-local shared storage exceeds 48 KiB")
        if x.shape[0] > 65535:
            raise ValueError("block-local batch grid exceeds 65535")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("radius baseline requires execution without autocast")
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("radius baseline has not qualified deterministic execution")
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("radius baseline uses eager preparation, not CUDA Graphs")
        if recipe.schedule in ("block_local", "block_local_parallel"):
            function = _BlockLocalRadiusLinear
        elif (
            recipe.input_direct_levels
            or recipe.output_direct_levels
            or recipe.schedule == "overlap"
        ):
            function = _HybridRadiusLinear
        else:
            function = _RadiusLinear
        return function.apply(x.contiguous(), atoms.contiguous(), plan, recipe)
