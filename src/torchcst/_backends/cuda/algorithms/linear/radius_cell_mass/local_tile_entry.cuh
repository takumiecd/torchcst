#pragma once
#include "local_tile_moments.cuh"
namespace radius_cell_mass {
void check_local_tile_entry(Tensor x, Tensor p, Config input_config,
    Config output_config, int input_start, int output_start,
    int tile_sites, int schedule) {
    check_matrix(x); check_atoms(p);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    const bool global = schedule == 5;
    if (!global) check_shared_shape(p, input, output);
    TORCH_CHECK(x.device() == p.device() && x.scalar_type() == p.scalar_type() &&
        x.size(1) == input.sites, "expected matching CUDA signal and atoms");
    int zi = (input.node_count - checked_start(input, input_start)) * input.basis_count();
    int co = (output.node_count - checked_start(output, output_start)) * output.basis_count();
    TORCH_CHECK(tile_sites >= 1 && tile_sites <= 128 &&
        ((schedule >= 0 && schedule <= 4) || global),
        "invalid native radius launch configuration");
    int work = std::max({input.node_count * input.basis_count(),
        output.node_count * output.basis_count(), int(p.size(0))});
    TORCH_CHECK(x.size(0) <= 65535 && x.size(0) * work <= INT_MAX - THREADS,
        "radius batch/grid work exceeds native index limit");
    if (schedule == 4) check_local_tile_axes(input, output, x.size(0), input_start, output_start, tile_sites);
    else TORCH_CHECK(global || 8 * (p.size(0) + std::max(zi, co)) <= 48 * 1024,
        "block-local shared storage exceeds 48 KiB");
    if (schedule == 2) {
        for (int side = 0; side < 2; ++side) {
            const Axis &axis = side ? output : input;
            int elements = side ? co : zi;
            int minimum = 8 * elements + 32 * ((axis.modes + 3) * 8 + 4);
            if (elements) minimum += axis.basis_count() * 33 * 8;
            TORCH_CHECK(minimum <= 48 * 1024,
                "coefficient cache shared storage exceeds 48 KiB");
        }
    }
    if (schedule == 3 || schedule == 4 || global) {
        batched_coefficient_layout(input, input_start, tile_sites);
        batched_coefficient_layout(output, output_start, tile_sites);
    }
    TORCH_CHECK(!at::autocast::is_autocast_enabled(at::kCUDA),
        "radius baseline requires execution without autocast");
    TORCH_CHECK(!at::globalContext().deterministicAlgorithms(),
        "radius baseline has not qualified deterministic execution");
    c10::cuda::CUDAGuard guard(x.device());
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone,
        "radius baseline uses eager preparation, not CUDA Graphs");
}

} // namespace radius_cell_mass
