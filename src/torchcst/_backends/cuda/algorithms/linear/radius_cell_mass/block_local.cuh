#pragma once
#include "hybrid.cuh"
#include <algorithm>

namespace radius_cell_mass {

// One CTA completes the dependency chain for one batch row and output interval.
// H and input moments are deliberately recomputed by every output CTA. There is
// no global C table, no inter-block wait, and no atomic accumulation.
// The moment and coefficient regions share scratch only after all H is final.
template<class T>
__global__ void block_local_kernel(
    const T *x, const T *p, const int64_t *rows, const int64_t *offsets,
    T *y, double *saved_z, double *saved_h, int atoms, Axis input, Axis output,
    int input_start, int output_start, bool merge, int output_tile_sites,
    int scratch_elements, bool projection_output_side, bool emit_output) {
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
    for (int local_node = threadIdx.x; local_node < output_nodes; local_node += blockDim.x) {
        int node = output_prefix + local_node;
        Bank bank = output.banks[bank_for_node(node, output)];
        int tile = node - bank.node_offset;
        int tile_first = tile * bank.tile_sites;
        // A site reads only its containing node in each bank. Other nodes are
        // intentionally left unwritten and must never be read by this CTA.
        if (tile_first >= end_site || tile_first + bank.tile_sites <= first_site) continue;
        compact_coefficient_tile(
            p, rows, offsets, response,
            scratch + int64_t(local_node) * output_features,
            atoms, 0, node, output, !projection_output_side);
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
            output_tile_sites, scratch_elements, projection_output_side, emit_output);
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
