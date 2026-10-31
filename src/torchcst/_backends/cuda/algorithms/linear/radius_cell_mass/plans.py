"""Explicit research Plans; the automatic selector is unchanged."""

from torchcst._backends.schema import ExecutionPlan

from .recipe import RadiusCellMassRecipe

REFERENCE = ExecutionPlan("cuda_radius_cell_mass", "v1", RadiusCellMassRecipe())
DIRECT = ExecutionPlan("cuda_radius_cell_mass", "v1", RadiusCellMassRecipe("direct"))

# DIRECT above means direct *moment construction*. These plans instead bypass
# moments/coeffs for low levels, while retaining exactly the same cell masses.
HYBRID_SERIAL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(input_direct_levels=2, output_direct_levels=2),
)
HYBRID_OVERLAP = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2, output_direct_levels=2, schedule="overlap"
    ),
)
BLOCK_LOCAL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2, output_direct_levels=2, schedule="block_local"
    ),
)

BLOCK_LOCAL_PARALLEL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2, output_direct_levels=2, schedule="block_local_parallel"
    ),
)


BLOCK_LOCAL_CACHED_PARALLEL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="block_local_cached_parallel",
    ),
)


BLOCK_LOCAL_BATCHED_CACHED_PARALLEL = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="block_local_batched_cached_parallel",
    ),
)


# Experimental general projection + tile-local expansion; no native host entry.
GLOBAL_BATCHED_CACHED = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_batched_cached",
    ),
)


# Response-only attribution against GLOBAL_BATCHED_CACHED's serial H/G stages.
GLOBAL_BATCHED_RESPONSE4 = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_batched_cached_response4",
    ),
)


# Keep response4 as the scalar-gradient baseline; opt in to four-row dP here.
GLOBAL_BATCHED_RESPONSE4_GRAD4 = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_batched_cached_response4_grad4",
    ),
)


# Response4 plus current-P bank work skipping; full global Z layout is retained.
GLOBAL_BATCHED_ACTIVE_MOMENTS = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_batched_cached_active_moments",
    ),
)


# Independently qualified active Z and grad4 compose behind a new explicit Plan.
GLOBAL_ACTIVE_GRAD4 = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_batched_cached_active_moments_grad4",
    ),
)


# General global stages behind C++ host autograd; mode4 is reserved for tile-local Z.
GLOBAL_ACTIVE_GRAD4_NATIVE = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_batched_cached_active_moments_grad4_native",
        native_autograd=True,
        native_entry=True,
    ),
)


# Bounded first cohort: owner-local complete responses and coefficient partials.
GLOBAL_OWNER_RESPONSE_COEFFICIENTS = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_owner_response_coefficients",
        native_autograd=True,
        native_entry=True,
    ),
)


# New numerical route: complete center-derivative responses replace saved Zx.
# Old owner6 and every default continue to select their original kernels.
GLOBAL_OWNER_RESPONSE_DERIVATIVES = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_owner_response_derivatives",
        native_autograd=True,
        native_entry=True,
    ),
)


# Explicit mode9 repacks only temporary owner C; all floating math is mode7.
GLOBAL_OWNER_COMPACT_PARTIALS = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="global_owner_compact_partials",
        native_autograd=True,
        native_entry=True,
    ),
)


# Experimental: C/direct cutoff2, retained global Z starts at level5 instead.
# Complete FP64 H/G are still built once and reused by all same-side consumers.
LOCAL_TILE_MOMENTS = ExecutionPlan(
    "cuda_radius_cell_mass",
    "v1",
    RadiusCellMassRecipe(
        input_direct_levels=2,
        output_direct_levels=2,
        schedule="block_local_tile_moments",
        output_tile_sites=32,
        native_autograd=True,
        native_entry=True,
    ),
)
