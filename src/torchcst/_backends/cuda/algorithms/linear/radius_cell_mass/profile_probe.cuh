#pragma once

// Diagnostic only: normal projection/expansion never exports coefficients.
namespace radius_cell_mass {
__global__ void profile_value_comparison_kernel(const double *radius_period,
    double *full, double *values, double *derivatives, int count, int modes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    double c[17], dr[17], v[17];
    double R = radius_period[2 * row], P = radius_period[2 * row + 1];
    profile_coefficients(R, P, modes, c, dr);
    profile_values(R, P, modes, v);
    for (int k = 0; k <= modes; ++k) {
        int64_t index = int64_t(row) * (modes + 1) + k;
        full[index] = c[k]; values[index] = v[k]; derivatives[index] = dr[k];
    }
}
std::vector<Tensor> profile_value_comparison(Tensor radius_period, int modes) {
    check_matrix(radius_period);
    TORCH_CHECK(radius_period.scalar_type() == torch::kDouble && radius_period.size(1) == 2,
        "profile probe expects FP64 (radius, period) rows");
    TORCH_CHECK(modes >= 1 && modes <= 16 && radius_period.numel() <= INT_MAX - THREADS,
        "invalid profile probe size or modes");
    c10::cuda::CUDAGuard guard(radius_period.device());
    auto full = torch::empty({radius_period.size(0), int64_t(modes + 1)}, radius_period.options());
    auto values = torch::empty_like(full), derivatives = torch::empty_like(full);
    if (radius_period.size(0)) {
        profile_value_comparison_kernel<<<(radius_period.size(0) + THREADS - 1) / THREADS,
            THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(radius_period.data_ptr<double>(),
            full.data_ptr<double>(), values.data_ptr<double>(), derivatives.data_ptr<double>(),
            radius_period.size(0), modes);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {full, values, derivatives};
}
} // namespace radius_cell_mass
