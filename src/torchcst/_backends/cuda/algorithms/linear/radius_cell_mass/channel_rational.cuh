#pragma once

#ifdef RADIUS_RATIONAL_WIDE
namespace radius_cell_mass {

__global__ void channel_decode_rational(const float *rows, float *physical, float *jacobian,
    int atoms, double low, double birth, double high, double amplitude_max, float encoded_low) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const float *p = rows + 4 * atom;
    double a = p[0], q = p[1];
    bool valid = fabs(a) <= amplitude_max && 0 <= q && q <= 1 &&
        isfinite(p[2]) && isfinite(p[3]);
    CUDA_KERNEL_ASSERT_MSG(valid, "invalid rational radius Atom");
    if (!valid) return;

    double scaled = a / 0.3, envelope = 1 + 3 * (scaled * scaled);
    double lower = low + (birth - low) / envelope;
    double log_lower = log(lower), log_high = log(high);
    double raw = exp((1 - q) * log_lower + q * log_high);
    double lower_da = -(birth - low) * (6 * scaled / 0.3) / (envelope * envelope);
    jacobian[2 * atom] = float(raw * (1 - q) * lower_da / lower);
    jacobian[2 * atom + 1] = float(raw * (log_high - log_lower));

    // Match the exact endpoint encoding with the interior analytic Jacobian.
    double radius = q == 0 ? lower : (q == 1 ? high : raw);
    float *out = physical + 4 * atom;
    out[0] = p[0];
    out[1] = fminf(float(high), fmaxf(encoded_low, float(radius)));
    out[2] = p[2];
    out[3] = p[3];
}

__global__ void channel_rational_cotangent(const float *cotangent, const float *jacobian,
    float *grad, int atoms) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const float *d = cotangent + 4 * atom;
    float *out = grad + 4 * atom;
    out[0] = d[0] + d[1] * jacobian[2 * atom];
    out[1] = d[1] * jacobian[2 * atom + 1];
    out[2] = d[2];
    out[3] = d[3];
}

std::vector<Tensor> packed_channel_rational_decode(
    Tensor rows, double low, double birth, int cap, double amplitude_max) {
    check_channel_atoms(rows);
    TORCH_CHECK(cap >= 1 && cap <= 64 && 0.5 <= low && low <= birth && birth <= cap &&
        0 < amplitude_max && amplitude_max <= 32, "invalid channel rational radius bounds");
    c10::cuda::CUDAGuard guard(rows.device());
    float encoded_low = float(low);
    if (double(encoded_low) < low) encoded_low = std::nextafter(encoded_low, INFINITY);
    auto physical = torch::empty_like(rows);
    auto jacobian = torch::empty({rows.size(0), rows.size(1), 2}, rows.options());
    int atoms = rows.size(0) * rows.size(1);
    const auto stream = at::cuda::getCurrentCUDAStream();
    channel_decode_rational<<<(atoms + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        rows.data_ptr<float>(), physical.data_ptr<float>(), jacobian.data_ptr<float>(),
        atoms, low, birth, cap, amplitude_max, encoded_low);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {physical, jacobian};
}

Tensor packed_channel_rational_grad(Tensor cotangent, Tensor jacobian) {
    check_channel_atoms(cotangent);
    packed_check_tensor(jacobian, cotangent, {cotangent.size(0), cotangent.size(1), 2});
    c10::cuda::CUDAGuard guard(cotangent.device());
    auto grad = torch::empty_like(cotangent);
    int atoms = cotangent.size(0) * cotangent.size(1);
    const auto stream = at::cuda::getCurrentCUDAStream();
    channel_rational_cotangent<<<(atoms + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        cotangent.data_ptr<float>(), jacobian.data_ptr<float>(), grad.data_ptr<float>(), atoms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

} // namespace radius_cell_mass
#endif
