#pragma once
#include "owner_compact_keyed.cuh"
namespace radius_cell_mass {
std::vector<Tensor> owner_compact_fused_reverse(Tensor x, Tensor p, Tensor keys, Tensor rows,
                                                Tensor offsets, Config input_config,
                                                Config output_config, Tensor h, Tensor dh) {
    const bool reverse = true, need_derivative = true;
    // Keys retain the original input/output orientation in both directions.
    // All saved-X/P/H/DH/key metadata/version checks precede reverse sorting.
    const Config &source_config = reverse ? output_config : input_config;
    const Config &destination_config = reverse ? input_config : output_config;
    check_atoms(p);
    Axis source = parse_axis(source_config), destination = parse_axis(destination_config);
    check_hybrid_signal(x, p, source);
    check_ranges(rows, offsets, p, destination);
    check_owner_axes(source, destination, x.size(0), 2, 2, true, 32);
    c10::cuda::CUDAGuard guard(p.device());
    auto z =
        build_compact_moments_from_keys(x, p, keys, input_config, output_config, reverse, true, 2);
    // Preserve Polar signed zero in the empty-batch byte contract as well.
    auto gradient =
        x.size(0) ? torch::empty_like(p)
                  : owner_derivative_parameter_grad(p, h, h, dh, dh, input_config, output_config);
    auto layout = compact_owner_partial_layout(destination, 2);
    TORCH_CHECK(x.size(0) <= (INT_MAX - THREADS) / layout.total_fields, "compact owner work limit");
    auto partials =
        torch::empty({x.size(0), layout.total_fields}, x.options().dtype(torch::kDouble));
    auto y = torch::empty({x.size(0), destination.sites}, x.options());
    if (x.size(0)) {
        AT_DISPATCH_FLOATING_TYPES(p.scalar_type(), "radius_owner_response_derivatives", [&] {
            owner_compact_partials_kernel<scalar_t, true>
                <<<destination.node_count, THREADS, OWNER_SHARED_BYTES,
                   at::cuda::getCurrentCUDAStream()>>>(
                    p.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), z.data_ptr<double>(), nullptr,
                    nullptr, partials.data_ptr<double>(), rows.data_ptr<int64_t>(),
                    offsets.data_ptr<int64_t>(), p.size(0), x.size(0), source, destination, 2,
                    reverse, need_derivative, layout, h.data_ptr<double>(), dh.data_ptr<double>(),
                    gradient.data_ptr<scalar_t>());
            owner_compact_expand_kernel<scalar_t><<<(y.numel() + THREADS - 1) / THREADS, THREADS, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                partials.data_ptr<double>(), offsets.data_ptr<int64_t>(), y.data_ptr<scalar_t>(),
                x.size(0), destination, 2, layout);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {y, gradient};
}

} // namespace radius_cell_mass
