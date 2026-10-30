#pragma once
#include "hybrid.cuh"
#include <algorithm>

namespace radius_cell_mass {

constexpr int BLOCK_LOCAL_C_CHUNK = 32;
// As in the legacy cell-mass contraction, a padded coefficient-column stride
// lets lanes read distinct modes of the same atom without a 32-way bank clash.
constexpr int BLOCK_LOCAL_C_STRIDE = BLOCK_LOCAL_C_CHUNK + 1;

// One warp owns one output node. Each lane evaluates a different atom once,
// then lanes independently sum their coefficient mode in sorted-slot order.
// This is warp-private scratch: no CTA barrier is needed between chunks, and
// all 32 lanes participate even for partial chunks and the M=16 extra mode.
template<class T>
__device__ inline void block_local_coefficient_tile(
    const T *p, const int64_t *rows, const int64_t *offsets,
    const double *response, double *out, double *columns,
    int node, Axis axis, bool output_side) {
    constexpr unsigned MASK = 0xffffffffu;
    int lane = threadIdx.x % BLOCK_LOCAL_C_CHUNK;
    int features = axis.basis_count();
    Bank bank = axis.banks[bank_for_node(node, axis)];
    int tile = node - bank.node_offset;
    double accumulator = 0, last_accumulator = 0;
    for (int part = 0; part < bank.count; ++part) {
        int first_node = bank.node_offset + positive_mod(tile - part, bank.tile_count);
        int64_t begin = offsets[first_node], end = offsets[first_node + 1];
        for (int64_t chunk = begin; chunk < end; chunk += BLOCK_LOCAL_C_CHUNK) {
            int64_t slot = chunk + lane;
            if (slot < end) {
                int atom = rows[slot];
                double R = radius(p, atom, axis);
                double original = center(p, atom, axis, output_side);
                Membership member = classify(R, original, axis);
                // Keep each repeated wrapped range as its own unwrapped image.
                // P may exceed the chart period, so tile alone cannot set phase.
                int unwrapped_tile = member.first_tile + part;
                double displacement = wrapped_center(original, axis) -
                    unwrapped_tile * bank.tile_sites * axis.spacing;
                double c[17], dr[17];
                profile_coefficients(R, bank.period, axis.modes, c, dr);
                double value = amplitude(p, atom, axis) * response[atom];
                columns[lane] = value * c[0];
                HarmonicPhase phase(2 * PI * displacement / bank.period);
                for (int k = 1; k <= axis.modes; ++k) {
                    double cp, sp;
                    phase.next(cp, sp);
                    columns[(2 * k - 1) * BLOCK_LOCAL_C_STRIDE + lane] =
                        value * 2 * c[k] * cp;
                    columns[2 * k * BLOCK_LOCAL_C_STRIDE + lane] =
                        value * 2 * c[k] * sp;
                }
            }
            __syncwarp(MASK);
            int retained = end - chunk < BLOCK_LOCAL_C_CHUNK
                ? int(end - chunk) : BLOCK_LOCAL_C_CHUNK;
            if (lane < features)
                for (int atom = 0; atom < retained; ++atom)
                    accumulator += columns[lane * BLOCK_LOCAL_C_STRIDE + atom];
            // A 33-feature M=16 bank has one more mode than warp lanes.
            if (lane == 0 && features > BLOCK_LOCAL_C_CHUNK)
                for (int atom = 0; atom < retained; ++atom)
                    last_accumulator += columns[BLOCK_LOCAL_C_CHUNK * BLOCK_LOCAL_C_STRIDE + atom];
            __syncwarp(MASK); // Finish reads before this warp reuses columns.
        }
    }
    if (lane < features) out[lane] = accumulator;
    if (lane == 0 && features > BLOCK_LOCAL_C_CHUNK)
        out[BLOCK_LOCAL_C_CHUNK] = last_accumulator;
}

// One CTA completes the dependency chain for one batch row and output interval.
// H and input moments are deliberately recomputed by every output CTA. There is
// no global C table, no inter-block wait, and no atomic accumulation.
// The moment and coefficient regions share scratch only after all H is final.
template<class T>
__global__ void block_local_kernel(
    const T *x, const T *p, const int64_t *rows, const int64_t *offsets,
    T *y, double *saved_z, double *saved_h, int atoms, Axis input, Axis output,
    int input_start, int output_start, bool merge, int output_tile_sites,
    int scratch_elements, int coefficient_warps,
    bool projection_output_side, bool emit_output) {
    extern __shared__ double shared[];
    double *scratch = shared;
    double *response = shared + scratch_elements;
    int b = blockIdx.y;
    int first_site = blockIdx.x * output_tile_sites;
    int end_site = min(first_site + output_tile_sites, output.sites);
    const T *signal = x + int64_t(b) * input.sites;
    int input_prefix = prefix_nodes(input, input_start);
    int input_nodes = input.node_count - input_prefix;
    int input_features = input.basis_count();
    int moment_elements = input_nodes * input_features;

    if (merge) {
        // compact_moment_element reads children only from earlier levels. Every
        // thread reaches each barrier, including threads with no assigned work.
        for (int level = input_start; level < input.bank_count / 2; ++level) {
            for (int index = threadIdx.x; index < moment_elements; index += blockDim.x) {
                int node = input_prefix + index / input_features;
                Bank bank = input.banks[bank_for_node(node, input)];
                if (bank.level == level)
                    scratch[index] = compact_moment_element(
                        signal, scratch, 0, node, index % input_features,
                        input, input_start, true);
            }
            __syncthreads();
        }
    } else {
        for (int index = threadIdx.x; index < moment_elements; index += blockDim.x) {
            int node = input_prefix + index / input_features;
            scratch[index] = compact_moment_element(
                signal, scratch, 0, node, index % input_features,
                input, input_start, false);
        }
        __syncthreads();
    }

    // Save completed moments before scratch is reused for coefficients.
    // Only interval zero writes global state; every CTA consumes its own copy.
    if (blockIdx.x == 0)
        for (int index = threadIdx.x; index < moment_elements; index += blockDim.x)
            saved_z[int64_t(b) * moment_elements + index] = scratch[index];

    for (int atom = threadIdx.x; atom < atoms; atom += blockDim.x) {
        double R = radius(p, atom, input);
        double original_center = center(p, atom, input, projection_output_side);
        double mu = wrapped_center(original_center, input);
        Membership member = classify(R, original_center, input);
        Bank bank = input.banks[member.bank];
        double coefficients[17], derivative[17], dR, dmu;
        profile_coefficients(R, bank.period, input.modes, coefficients, derivative);
        double value = bank.level < input_start
            ? direct_project(signal, 0, input, member, mu,
                             coefficients, derivative, dR, dmu)
            : compact_project(scratch, 0, input, member, mu,
                              coefficients, derivative, input_start, dR, dmu);
        response[atom] = value;
        // H has exactly one global writer, even when output CTAs overlap atoms.
        if (blockIdx.x == 0) saved_h[int64_t(b) * atoms + atom] = value;
    }
    __syncthreads();

    if (!emit_output) return;

    int output_prefix = prefix_nodes(output, output_start);
    int output_nodes = output.node_count - output_prefix;
    int output_features = output.basis_count();
    if (coefficient_warps) {
        int warp = threadIdx.x / BLOCK_LOCAL_C_CHUNK;
        if (warp < coefficient_warps) {
            double *columns = response + atoms +
                warp * output_features * BLOCK_LOCAL_C_STRIDE;
            int job = 0;
            for (int bi = 2 * output_start; bi < output.bank_count; ++bi) {
                Bank bank = output.banks[bi];
                int first_tile = first_site / bank.tile_sites;
                int tile_count = (end_site - 1) / bank.tile_sites - first_tile + 1;
                // Distribute consecutive overlapping jobs, rather than all
                // chart nodes: otherwise coarse nodes with aligned offsets
                // concentrate on one warp when output intervals are small.
                int first_job = warp - job % coefficient_warps;
                if (first_job < 0) first_job += coefficient_warps;
                for (int local = first_job; local < tile_count; local += coefficient_warps) {
                    int node = bank.node_offset + first_tile + local;
                    block_local_coefficient_tile(
                        p, rows, offsets, response,
                        scratch + int64_t(node - output_prefix) * output_features, columns,
                        node, output, !projection_output_side);
                }
                job += tile_count;
            }
        }
    } else {
        // Preserve every previously valid configuration when no warp workspace
        // fits alongside H and the compact coefficient/moment scratch.
        for (int local_node = threadIdx.x; local_node < output_nodes; local_node += blockDim.x) {
            int node = output_prefix + local_node;
            Bank bank = output.banks[bank_for_node(node, output)];
            int tile_first = (node - bank.node_offset) * bank.tile_sites;
            if (tile_first >= end_site || tile_first + bank.tile_sites <= first_site) continue;
            compact_coefficient_tile(
                p, rows, offsets, response,
                scratch + int64_t(local_node) * output_features,
                atoms, 0, node, output, !projection_output_side);
        }
    }
    __syncthreads();

    for (int site = first_site + threadIdx.x; site < end_site; site += blockDim.x) {
        double value = hybrid_site(
            p, rows, offsets, response, scratch, atoms, 0, site,
            output, !projection_output_side, output_start);
        y[int64_t(b) * output.sites + site] = T(value);
    }
}

// Staging route: fixed 128-thread CTAs and a hard 48-KiB shared-memory budget.
// Axis arguments here name projection and expansion, independent of direction.
inline std::vector<Tensor> block_local_apply(
    Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input_config, Config output_config,
    int input_start, int output_start, bool merge, int output_tile_sites,
    bool projection_output_side, bool emit_output) {
    check_matrix(x);
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_shared_shape(p, input, output);
    TORCH_CHECK(x.device() == p.device() && x.scalar_type() == p.scalar_type() &&
                x.size(1) == input.sites, "signal and atoms must match radius projection axis/device/dtype");
    TORCH_CHECK(input_start >= 0 && input_start <= input.bank_count / 2 &&
                output_start >= 0 && output_start <= output.bank_count / 2,
                "block-local starts must lie in [0, hierarchy levels]");
    TORCH_CHECK(output_tile_sites >= 1 && output_tile_sites <= 128,
                "output_tile_sites must lie in [1,128]");
    if (emit_output) check_ranges(rows, offsets, p, output);
    // Saved tensors feed helpers with native int indexing. Check before allocation,
    // including zero active nodes, which still have response storage.
    TORCH_CHECK(x.size(0) <= 65535, "block-local batch exceeds grid.y limit");
    int input_nodes = input.node_count - prefix_nodes(input, input_start);
    int input_elements = input_nodes * input.basis_count();
    int output_elements = emit_output
        ? (output.node_count - prefix_nodes(output, output_start)) * output.basis_count() : 0;
    TORCH_CHECK(input_elements == 0 || x.size(0) <= (INT_MAX - THREADS) / input_elements,
                "block-local compact work exceeds native index limit");
    TORCH_CHECK(p.size(0) == 0 || x.size(0) <= (INT_MAX - THREADS) / p.size(0),
                "block-local response work exceeds native index limit");
    int scratch_elements = std::max(input_elements, output_elements);
    size_t shared_bytes = (size_t(scratch_elements) + p.size(0)) * sizeof(double);
    TORCH_CHECK(shared_bytes <= 48 * 1024,
                "block-local radius route exceeds the 48-KiB shared-memory budget");
    // Extra C workspace is optional and never narrows the original route's
    // support. Projection-only calls allocate none. Use fewer active C warps
    // when needed, or the original serial node contraction when none fit.
    int coefficient_warps = 0;
    if (emit_output && output_elements) {
        size_t warp_bytes = size_t(output.basis_count()) * BLOCK_LOCAL_C_STRIDE * sizeof(double);
        coefficient_warps = int(std::min(size_t(THREADS / BLOCK_LOCAL_C_CHUNK),
            (48 * 1024 - shared_bytes) / warp_bytes));
        shared_bytes += coefficient_warps * warp_bytes;
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto z = torch::empty({x.size(0), input_nodes, input.basis_count()}, x.options().dtype(torch::kDouble));
    auto h = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    Tensor y;
    if (emit_output) y = torch::empty({x.size(0), output.sites}, x.options());
    if (!x.size(0)) return emit_output ? std::vector<Tensor>{y, z, h} : std::vector<Tensor>{z, h};
    dim3 grid(emit_output ? (output.sites + output_tile_sites - 1) / output_tile_sites : 1, x.size(0));
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "radius_block_local_apply", [&] {
        block_local_kernel<<<grid, THREADS, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(),
            emit_output ? rows.data_ptr<int64_t>() : nullptr,
            emit_output ? offsets.data_ptr<int64_t>() : nullptr,
            emit_output ? y.data_ptr<scalar_t>() : nullptr, z.data_ptr<double>(), h.data_ptr<double>(),
            p.size(0), input, output, input_start, output_start, merge,
            output_tile_sites, scratch_elements, coefficient_warps, projection_output_side, emit_output);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return emit_output ? std::vector<Tensor>{y, z, h} : std::vector<Tensor>{z, h};
}

inline std::vector<Tensor> block_local_forward(
    Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input_config, Config output_config,
    int input_start, int output_start, bool merge, int output_tile_sites) {
    return block_local_apply(x, p, rows, offsets, input_config, output_config,
        input_start, output_start, merge, output_tile_sites, false, true);
}

inline std::vector<Tensor> block_local_backward(
    Tensor dy, Tensor p, Tensor rows, Tensor offsets,
    Config input_config, Config output_config,
    int input_start, int output_start, bool merge, int output_tile_sites) {
    return block_local_apply(dy, p, rows, offsets, output_config, input_config,
        output_start, input_start, merge, output_tile_sites, true, true);
}

inline std::vector<Tensor> block_local_project(
    Tensor signal, Tensor p, Config config, bool output_side, int start, bool merge) {
    return block_local_apply(signal, p, Tensor(), Tensor(), config, config,
        start, start, merge, 1, output_side, false);
}

} // namespace radius_cell_mass
