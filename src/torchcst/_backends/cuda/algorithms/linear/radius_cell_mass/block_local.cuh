#pragma once
#include "hybrid.cuh"
#include <algorithm>

namespace radius_cell_mass {

// One CTA completes the dependency chain for one batch row and output interval.
// H and input moments are deliberately recomputed by every output CTA. There is
// no global moment/C table, no inter-block wait, and no atomic accumulation.
// The moment and coefficient regions share scratch only after all H is final.
template<class T>
__global__ void block_local_kernel(
    const T *x, const T *p, const int64_t *rows, const int64_t *offsets,
    T *y, double *saved_h, int atoms, Axis input, Axis output,
    int input_start, int output_start, bool merge, int output_tile_sites,
    int scratch_elements) {
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

    for (int atom = threadIdx.x; atom < atoms; atom += blockDim.x) {
        double R = radius(p, atom, input);
        double original_center = center(p, atom, input, false);
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
            atoms, 0, node, output, true);
    }
    __syncthreads();

    for (int site = first_site + threadIdx.x; site < end_site; site += blockDim.x) {
        double value = hybrid_site(
            p, rows, offsets, response, scratch, atoms, 0, site,
            output, true, output_start);
        y[int64_t(b) * output.sites + site] = T(value);
    }
}

// Staging route: fixed 128-thread CTAs and a hard 48-KiB shared-memory budget.
// The bound is per CTA and excludes allocator/native-extension overhead; it is
// not a complete-step memory or performance claim.
inline std::vector<Tensor> block_local_forward(
    Tensor x, Tensor p, Tensor rows, Tensor offsets,
    Config input_config, Config output_config,
    int input_start, int output_start, bool merge, int output_tile_sites) {
    check_matrix(x);
    check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(x.device() == p.device() && x.scalar_type() == p.scalar_type() &&
                x.size(1) == input.sites, "signal and atoms must match radius input axis/device/dtype");
    TORCH_CHECK(input_start >= 0 && input_start <= input.bank_count / 2 &&
                output_start >= 0 && output_start <= output.bank_count / 2,
                "block-local starts must lie in [0, hierarchy levels]");
    TORCH_CHECK(output_tile_sites >= 1 && output_tile_sites <= 128,
                "output_tile_sites must lie in [1,128]");
    TORCH_CHECK(rows.device() == p.device() && offsets.device() == p.device() &&
                rows.scalar_type() == torch::kLong && offsets.scalar_type() == torch::kLong &&
                rows.is_contiguous() && offsets.is_contiguous() &&
                rows.dim() == 1 && offsets.dim() == 1 &&
                rows.numel() == p.size(0) && offsets.numel() == output.node_count + 1,
                "invalid block-local output routing");
    // CUDA grid.y has a 65535 limit. All per-CTA indexing is tiny under the
    // shared-memory guard, but batch addressing and global H use int64_t.
    TORCH_CHECK(x.size(0) <= 65535, "block-local batch exceeds grid.y limit");
    int input_elements = (input.node_count - prefix_nodes(input, input_start)) * input.basis_count();
    int output_elements = (output.node_count - prefix_nodes(output, output_start)) * output.basis_count();
    int scratch_elements = std::max(input_elements, output_elements);
    size_t shared_bytes = (size_t(scratch_elements) + p.size(0)) * sizeof(double);
    TORCH_CHECK(shared_bytes <= 48 * 1024,
                "block-local radius route exceeds the 48-KiB shared-memory budget");
    c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({x.size(0), output.sites}, x.options());
    auto h = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    if (!x.size(0)) return {y, h};
    dim3 grid((output.sites + output_tile_sites - 1) / output_tile_sites, x.size(0));
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "radius_block_local_forward", [&] {
        block_local_kernel<<<grid, THREADS, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(),
            offsets.data_ptr<int64_t>(), y.data_ptr<scalar_t>(), h.data_ptr<double>(),
            p.size(0), input, output, input_start, output_start, merge,
            output_tile_sites, scratch_elements);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, h};
}

} // namespace radius_cell_mass
