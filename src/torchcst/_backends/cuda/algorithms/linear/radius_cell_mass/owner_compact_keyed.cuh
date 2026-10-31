#pragma once
#include "owner_compact_partials.cuh"
#include "owner_key_masks.cuh"

namespace radius_cell_mass {
// Mode10 composes qualified mask metadata with literal mode9 floating kernels.
// Complete H/DH and G/DG, compact C, direct cells and final gradients are old.
std::vector<Tensor> owner_compact_keyed_transform(Tensor x, Tensor p, Tensor keys, Tensor rows,
    Tensor offsets, Config input_config, Config output_config,
    bool reverse, bool need_derivative) {
    // Keys retain the original input/output orientation in both directions.
    // All saved-X/P/H/DH/key metadata/version checks precede reverse sorting.
    const Config &source_config = reverse ? output_config : input_config;
    const Config &destination_config = reverse ? input_config : output_config;
    check_atoms(p);
    Axis source = parse_axis(source_config), destination = parse_axis(destination_config);
    check_hybrid_signal(x, p, source); check_ranges(rows, offsets, p, destination);
    check_owner_axes(source, destination, x.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    auto z = build_compact_moments_from_keys(x, p, keys, input_config,
        output_config, reverse, true, 2);
    auto response = torch::empty({x.size(0), p.size(0)}, x.options().dtype(torch::kDouble));
    auto derivative = need_derivative ? torch::empty_like(response) : Tensor();
    auto layout = compact_owner_partial_layout(destination, 2);
    TORCH_CHECK(x.size(0) <= (INT_MAX - THREADS) / layout.total_fields, "compact owner work limit");
    auto partials = torch::empty({x.size(0), layout.total_fields},
        x.options().dtype(torch::kDouble));
    auto y = torch::empty({x.size(0), destination.sites}, x.options());
    if (x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_response_derivatives", [&] {
            owner_compact_partials_kernel<scalar_t><<<destination.node_count, THREADS,
                OWNER_SHARED_BYTES, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), z.data_ptr<double>(), response.data_ptr<double>(),
                need_derivative ? derivative.data_ptr<double>() : nullptr,
                partials.data_ptr<double>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
                p.size(0), x.size(0), source, destination, 2, reverse, need_derivative, layout);
            owner_compact_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
                at::cuda::getCurrentCUDAStream()>>>(partials.data_ptr<double>(), offsets.data_ptr<int64_t>(),
                y.data_ptr<scalar_t>(), x.size(0), destination, 2, layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, response, derivative};
}

} // namespace radius_cell_mass
