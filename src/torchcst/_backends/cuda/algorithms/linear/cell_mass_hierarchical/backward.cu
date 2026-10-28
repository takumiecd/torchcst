// Analytic atom gradients. Each thread owns a different atom and writes its
// original Parameter row once; batch reduction stays in double registers.
#include "projection.cuh"
using namespace cell_mass_hierarchy;

namespace {
// Polar has no task derivative through width. Calculate each projection and
// center derivative together, avoiding the unused beta/sigma derivative work.
// H/G used for parameter gradients stay in double registers rather than being
// rounded to their globally saved activation dtype.
template <int FixedModes, bool Compensated>
__device__ inline void polar_projection(const double *z, int batches, int begin,
    const Axis &axis, const Bank &bank, int owner, const Profile &profile,
    bool windowed, double *value, double *dcenter) {
    HarmonicStream<false, Compensated> stream(profile, FixedModes ? FixedModes : axis.modes);
    int retained = windowed || bank.full_chart ? 1 : 3;
    int first = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    int modes = FixedModes ? FixedModes : axis.modes;
#pragma unroll
    for (int harmonic = 0; harmonic <= modes; ++harmonic) {
        double cc, cs, unused_c, unused_s;
        stream.next(cc, cs, unused_c, unused_s);
        double omega = profile.omega * harmonic;
        for (int offset = 0; offset < retained; ++offset) {
            double rs = 0, rc = 1;
            int node = bank.node_offset + owner;
            if (!windowed) {
                int raw = first + offset;
                int tile = positive_mod(raw, bank.tile_count);
                image_rotation(bank, harmonic, (raw - tile) / bank.tile_count, rs, rc);
                node = bank.node_offset + tile;
            }
            for (int b = 0; b < BATCH_TILE && begin + b < batches; ++b) {
                int base = ((begin + b) * axis.node_count + node) * axis.basis_count();
                if (harmonic == 0) {
                    value[b] += cc * z[base];
                } else {
                    double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                    double rotated_c = zc * rc - zs * rs;
                    double rotated_s = zs * rc + zc * rs;
                    value[b] += cc * rotated_c + cs * rotated_s;
                    dcenter[b] += omega * (-cs * rotated_c + cc * rotated_s);
                }
            }
        }
    }
}

// Each lane owns one column and reads only values that it wrote itself.
// The cache needs no cross-thread synchronization, including a partial block.
// Values remain FP64; center derivatives are formed from the same cc/cs below.
template <int FixedModes, bool Compensated>
__device__ inline void cache_polar_coefficients(const Profile &profile, double *coefficients,
                                               int coefficient_stride) {
    HarmonicStream<false, Compensated> stream(profile, FixedModes);
#pragma unroll
    for (int harmonic = 0; harmonic <= FixedModes; ++harmonic) {
        double cc, cs, unused_c, unused_s;
        stream.next(cc, cs, unused_c, unused_s);
        coefficients[(harmonic == 0 ? 0 : 2 * harmonic - 1) * coefficient_stride] = cc;
        if (harmonic != 0)
            coefficients[2 * harmonic * coefficient_stride] = cs;
    }
}

template <int FixedModes>
__device__ inline void polar_projection_cached(const double *z, int batches, int begin,
    const Axis &axis, const Bank &bank, int owner, const Profile &profile,
    bool windowed, const double *coefficients, int coefficient_stride,
    double *value, double *dcenter) {
    int retained = windowed || bank.full_chart ? 1 : 3;
    int first = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    int modes = FixedModes ? FixedModes : axis.modes;
#pragma unroll
    for (int harmonic = 0; harmonic <= modes; ++harmonic) {
        double cc = coefficients[(harmonic == 0 ? 0 : 2 * harmonic - 1) * coefficient_stride];
        double cs = harmonic == 0 ? 0 : coefficients[2 * harmonic * coefficient_stride];
        double omega = profile.omega * harmonic;
        for (int offset = 0; offset < retained; ++offset) {
            double rs = 0, rc = 1;
            int node = bank.node_offset + owner;
            if (!windowed) {
                int raw = first + offset;
                int tile = positive_mod(raw, bank.tile_count);
                image_rotation(bank, harmonic, (raw - tile) / bank.tile_count, rs, rc);
                node = bank.node_offset + tile;
            }
            for (int b = 0; b < BATCH_TILE && begin + b < batches; ++b) {
                int base = ((begin + b) * axis.node_count + node) * axis.basis_count();
                if (harmonic == 0) {
                    value[b] += cc * z[base];
                } else {
                    double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                    double rotated_c = zc * rc - zs * rs;
                    double rotated_s = zs * rc + zc * rs;
                    value[b] += cc * rotated_c + cs * rotated_s;
                    dcenter[b] += omega * (-cs * rotated_c + cc * rotated_s);
                }
            }
        }
    }
}

// One warp owns one input-owner split and visits its reverse slots in exactly
// the same chunk/atom order as ragged_input_kernel. Each lane owns a distinct
// atom through every batch slab, so its FP64 parameter reduction is unchanged.
template <bool DebugG>
__global__ void fused_polar_grad_cin_kernel(
    const float *atoms, const int64_t *rows, const int *groups, const int64_t *offsets,
    const int *reverse_slots, const int64_t *reverse_offsets, const double *zx,
    const double *zy, const int *counts, const int *task_offsets, double *partial,
    float *gradient, float *debug_g, int atom_count, int group_count, int batches,
    Axis input, Axis output, bool windowed) {
    constexpr int CHUNK = 32, STRIDE = 33, BASIS = 17;
    int task = blockIdx.x;
    if (task >= task_offsets[input.node_count]) return; // Uniform CTA return.
    int first_node = 0, last_node = input.node_count + 1;
    while (first_node < last_node) {
        int middle = (first_node + last_node) / 2;
        if (task_offsets[middle] <= task) first_node = middle + 1;
        else last_node = middle;
    }
    int node = first_node - 1;
    int active = task_offsets[node + 1] - task_offsets[node];
    int local_split = task - task_offsets[node], count = counts[node];
    int first = static_cast<int>(reverse_offsets[node]) +
        static_cast<int>(static_cast<int64_t>(count) * local_split / active);
    int last = static_cast<int>(reverse_offsets[node]) +
        static_cast<int>(static_cast<int64_t>(count) * (local_split + 1) / active);
    int bank_index = 0;
    while (bank_index + 1 < input.bank_count && node >= input.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank ib = input.banks[bank_index];
    int input_owner = node - ib.node_offset, lane = threadIdx.x;
    extern __shared__ double shared[];
    double *input_coefficients = shared;
    double *output_coefficients = input_coefficients + BASIS * STRIDE;
    double *scaled_g = output_coefficients + BASIS * STRIDE;
    double *amplitudes = scaled_g + BATCH_TILE * STRIDE;
    double *totals = amplitudes + STRIDE;
    for (int work = lane; work < batches * BASIS; work += CHUNK) totals[work] = 0;
    __syncthreads();
    for (int chunk = first; chunk < last; chunk += CHUNK) {
        int position = chunk + lane;
        bool valid = position < last;
        int slot = 0, original = 0, output_owner = 0;
        Bank ob{};
        Profile ip{}, op{};
        if (valid) {
            slot = reverse_slots[position];
            original = rows[slot];
            int group = group_for_slot(offsets, group_count, slot);
            output_owner = groups[4 * group + 2];
            ob = output.banks[bank_for_level(output, groups[4 * group])];
            amplitudes[lane] = atom_amplitude(atoms, original, input);
            ip = make_profile<false, 8, true>(input, ib, atom_sigma(atoms, original, input),
                atom_center(atoms, original, input, false), input_owner);
            op = make_profile<false, 8, true>(output, ob, atom_sigma(atoms, original, output),
                atom_center(atoms, original, output, true), output_owner);
            cache_polar_coefficients<8, true>(ip, input_coefficients + lane, STRIDE);
            cache_polar_coefficients<8, true>(op, output_coefficients + lane, STRIDE);
        } else {
            // C reads every atom column; zero coefficients as well as G so no
            // padding lane can introduce a 0*uninitialized/NaN contribution.
            for (int mode = 0; mode < BASIS; ++mode) {
                input_coefficients[mode * STRIDE + lane] = 0;
                output_coefficients[mode * STRIDE + lane] = 0;
            }
            amplitudes[lane] = 0;
        }
        double da = 0, dmi = 0, dmo = 0;
        for (int begin = 0; begin < batches; begin += BATCH_TILE) {
            double hv[BATCH_TILE] = {}, gv[BATCH_TILE] = {};
            double dhi[BATCH_TILE] = {}, dgo[BATCH_TILE] = {};
            if (valid) {
                polar_projection_cached<8>(zx, batches, begin, input, ib, input_owner,
                    ip, windowed, input_coefficients + lane, STRIDE, hv, dhi);
                polar_projection_cached<8>(zy, batches, begin, output, ob, output_owner,
                    op, windowed, output_coefficients + lane, STRIDE, gv, dgo);
            }
            double amplitude = amplitudes[lane];
            for (int b = 0; b < BATCH_TILE; ++b) {
                double scaled = 0;
                if (valid && begin + b < batches) {
                    float rounded_g = static_cast<float>(gv[b]);
                    scaled = amplitude * static_cast<double>(rounded_g);
                    if constexpr (DebugG)
                        debug_g[(begin + b) * atom_count + slot] = rounded_g;
                    // The task pullback uses unrounded H/G, never scaled_g.
                    da += hv[b] * gv[b];
                    dmi += amplitude * gv[b] * dhi[b];
                    dmo += amplitude * hv[b] * dgo[b];
                }
                scaled_g[b * STRIDE + lane] = scaled;
            }
            __syncthreads();
            // Transpose atom-lane producers to (batch,mode) consumers. The
            // 32-term order and carried FP64 total match separate ragged Cin.
            for (int work = lane; work < BATCH_TILE * BASIS; work += CHUNK) {
                int b = work / BASIS, mode = work % BASIS;
                if (begin + b < batches) {
                    int index = (begin + b) * BASIS + mode;
                    double value = totals[index];
                    for (int atom = 0; atom < CHUNK; ++atom)
                        value += scaled_g[b * STRIDE + atom] * input_coefficients[mode * STRIDE + atom];
                    totals[index] = value;
                }
            }
            // Covers both the next G slab and the next chunk's coefficients.
            __syncthreads();
        }
        if (valid) store_atom_gradient(atoms, gradient, original, input, da, 0, dmi, dmo);
    }
    for (int work = lane; work < batches * BASIS; work += CHUNK)
        partial[static_cast<int64_t>(task) * batches * BASIS + work] = totals[work];
}


template <typename atom_t, int FixedModes, bool Compensated, bool CacheCoefficients = false>
__global__ void polar_g_parameter_grad_kernel(const atom_t *atoms, const int64_t *rows,
    const int *groups, const int64_t *offsets, const double *zx, const double *zy,
    atom_t *g, atom_t *gradient, int atom_count, int group_count, int batches,
    Axis input, Axis output, bool windowed) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count) return;
    int original = rows[slot];
    int group = group_for_slot(offsets, group_count, slot);
    int input_owner = groups[4 * group + 3], output_owner = groups[4 * group + 2];
    Bank ib = input.banks[bank_for_level(input, groups[4 * group + 1])];
    Bank ob = output.banks[bank_for_level(output, groups[4 * group])];
    double amplitude = atom_amplitude(atoms, original, input);
    Profile ip = make_profile<false, FixedModes, Compensated>(input, ib,
        atom_sigma(atoms, original, input), atom_center(atoms, original, input, false), input_owner);
    Profile op = make_profile<false, FixedModes, Compensated>(output, ob,
        atom_sigma(atoms, original, output), atom_center(atoms, original, output, true), output_owner);
    double *input_coefficients = nullptr, *output_coefficients = nullptr;
    if constexpr (CacheCoefficients) {
        static_assert(FixedModes == 8, "Polar coefficient cache specializes M8");
        extern __shared__ double coefficient_cache[];
        input_coefficients = coefficient_cache + threadIdx.x;
        output_coefficients = coefficient_cache + (1 + 2 * FixedModes) * blockDim.x + threadIdx.x;
        cache_polar_coefficients<FixedModes, Compensated>(ip, input_coefficients, blockDim.x);
        cache_polar_coefficients<FixedModes, Compensated>(op, output_coefficients, blockDim.x);
    }
    double da = 0, dmi = 0, dmo = 0;
    for (int begin = 0; begin < batches; begin += BATCH_TILE) {
        double hv[BATCH_TILE] = {}, gv[BATCH_TILE] = {};
        double dhi[BATCH_TILE] = {}, dgo[BATCH_TILE] = {};
        if constexpr (CacheCoefficients) {
            polar_projection_cached<FixedModes>(zx, batches, begin, input, ib, input_owner,
                ip, windowed, input_coefficients, blockDim.x, hv, dhi);
            polar_projection_cached<FixedModes>(zy, batches, begin, output, ob, output_owner,
                op, windowed, output_coefficients, blockDim.x, gv, dgo);
        } else {
            polar_projection<FixedModes, Compensated>(zx, batches, begin, input, ib, input_owner,
                                                    ip, windowed, hv, dhi);
            polar_projection<FixedModes, Compensated>(zy, batches, begin, output, ob, output_owner,
                                                    op, windowed, gv, dgo);
        }
        for (int b = 0; b < BATCH_TILE && begin + b < batches; ++b) {
            g[(begin + b) * atom_count + slot] = static_cast<atom_t>(gv[b]);
            da += hv[b] * gv[b];
            dmi += amplitude * gv[b] * dhi[b];
            dmo += amplitude * hv[b] * dgo[b];
        }
    }
    store_atom_gradient(atoms, gradient, original, input, da, 0, dmi, dmo);
}

template <typename scalar_t, typename projection_t>
__global__ void parameter_grad_kernel(const scalar_t *atoms, const int64_t *rows, const int *groups,
                                      const int64_t *offsets, const double *zx, const double *zy,
                                      const projection_t *h, const projection_t *g,
                                      scalar_t *gradient, int atom_count, int group_count,
                                      int batches, Axis input, Axis output, bool windowed) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int original = rows[slot];
    int group = group_for_slot(offsets, group_count, slot);
    int input_owner = groups[4 * group + 3], output_owner = groups[4 * group + 2];
    Bank input_bank = input.banks[bank_for_level(input, groups[4 * group + 1])];
    Bank output_bank = output.banks[bank_for_level(output, groups[4 * group])];
    double amplitude = atom_amplitude(atoms, original, input);
    double sigma = atom_sigma(atoms, original, input);
    double output_sigma = atom_sigma(atoms, original, output);
    double input_center = atom_center(atoms, original, input, false);
    double output_center = atom_center(atoms, original, output, true);
    double da = 0, ds = 0, dmu_input = 0, dmu_output = 0;
    for (int begin = 0; begin < batches; begin += BATCH_TILE) {
        double dh_mu[BATCH_TILE] = {}, dh_sigma[BATCH_TILE] = {};
        double dg_mu[BATCH_TILE] = {}, dg_sigma[BATCH_TILE] = {};
        project_derivatives(zx, batches, begin, input, input_bank, input_owner, sigma, input_center,
                            dh_mu, dh_sigma, windowed);
        project_derivatives(zy, batches, begin, output, output_bank, output_owner, output_sigma,
                            output_center, dg_mu, dg_sigma, windowed);
        for (int b = 0; b < BATCH_TILE && begin + b < batches; ++b) {
            double hv = h[(begin + b) * atom_count + slot];
            double gv = g[(begin + b) * atom_count + slot];
            da += hv * gv;
            ds += amplitude * (gv * dh_sigma[b] + hv * dg_sigma[b]);
            dmu_input += amplitude * gv * dh_mu[b];
            dmu_output += amplitude * hv * dg_mu[b];
        }
    }
    store_atom_gradient(atoms, gradient, original, input, da, ds, dmu_input, dmu_output);
}

template <typename scalar_t, typename projection_t, int FixedModes, bool Compensated = false,
          bool ReuseShape = false>
__global__ void compute_g_parameter_grad_kernel(
    const scalar_t *atoms, const int64_t *rows, const int *groups, const int64_t *offsets,
    const double *zx, const double *zy, const projection_t *h, projection_t *g, scalar_t *gradient,
    int atom_count, int group_count, int batches, Axis input, Axis output, bool windowed) {
    static_assert(!ReuseShape || FixedModes == 8, "shape cache requires M8 specialization");
    // Every thread owns one column; the cache contains no cross-thread data.
    extern __shared__ double magnitudes[];
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int original = rows[slot];
    int group = group_for_slot(offsets, group_count, slot);
    int input_owner = groups[4 * group + 3], output_owner = groups[4 * group + 2];
    Bank input_bank = input.banks[bank_for_level(input, groups[4 * group + 1])];
    Bank output_bank = output.banks[bank_for_level(output, groups[4 * group])];
    double amplitude = atom_amplitude(atoms, original, input);
    double sigma = atom_sigma(atoms, original, input);
    double output_sigma = atom_sigma(atoms, original, output);
    double input_center = atom_center(atoms, original, input, false);
    double output_center = atom_center(atoms, original, output, true);
    Profile output_profile = make_profile<true, FixedModes, Compensated>(
        output, output_bank, output_sigma, output_center, output_owner);
    // Reuse the common width shape only when all of its dependencies match.
    // Center/image handling still uses the input axis and owner independently.
    bool same_shape = sigma == output_sigma && input.modes == output.modes && input.radius_ratio4 == output.radius_ratio4 &&
                      input_bank.period == output_bank.period &&
                      input_bank.inverse_period == output_bank.inverse_period &&
                      input_bank.omega == output_bank.omega;
    Profile input_profile;
    if (same_shape) {
        input_profile = output_profile;
        input_profile.center = profile_center(input, input_bank, input_center, input_owner);
    } else {
        input_profile = make_profile<true, FixedModes, Compensated>(input, input_bank, sigma,
                                                                    input_center, input_owner);
    }
    double beta_sigma_out = output_profile.beta_sigma;
    double beta_sigma_in = input_profile.beta_sigma;
    double da = 0, ds = 0, dmu_input = 0, dmu_output = 0;
    for (int begin = 0; begin < batches; begin += BATCH_TILE) {
        double gv[BATCH_TILE] = {};
        if constexpr (ReuseShape) {
            project_weighted_derivatives<true, FixedModes, false, Compensated, true>(
                zy, batches, begin, output, output_bank, output_owner, output_profile,
                beta_sigma_out, h, atom_count, slot, amplitude, gv, dmu_output, ds, windowed,
                nullptr, 0, 0, magnitudes, blockDim.x, threadIdx.x);
        } else {
            project_weighted_derivatives<true, FixedModes, false, Compensated>(
                zy, batches, begin, output, output_bank, output_owner, output_profile,
                beta_sigma_out, h, atom_count, slot, amplitude, gv, dmu_output, ds, windowed);
        }
        for (int b = 0; b < BATCH_TILE && begin + b < batches; ++b) {
            projection_t rounded_g = static_cast<projection_t>(gv[b]);
            g[(begin + b) * atom_count + slot] = rounded_g;
            double h_value = h[(begin + b) * atom_count + slot];
            double g_value = static_cast<double>(rounded_g);
            da += h_value * g_value;
            gv[b] = amplitude * g_value;
        }
        if constexpr (ReuseShape) {
            if (same_shape) {
                project_cached_input_derivatives<FixedModes>(
                    zx, batches, begin, input, input_bank, input_owner, input_profile,
                    beta_sigma_in, h, atom_count, slot, amplitude, gv, dmu_input, ds, windowed,
                    magnitudes, blockDim.x, threadIdx.x);
            } else {
                project_weighted_derivatives<false, FixedModes, false, Compensated>(
                    zx, batches, begin, input, input_bank, input_owner, input_profile,
                    beta_sigma_in, h, atom_count, slot, amplitude, gv, dmu_input, ds, windowed);
            }
        } else {
            project_weighted_derivatives<false, FixedModes, false, Compensated>(
                zx, batches, begin, input, input_bank, input_owner, input_profile, beta_sigma_in, h,
                atom_count, slot, amplitude, gv, dmu_input, ds, windowed);
        }
    }
    store_atom_gradient(atoms, gradient, original, input, da, ds, dmu_input, dmu_output);
}
} // namespace

torch::Tensor parameter_grad(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                             torch::Tensor offsets, torch::Tensor zx, torch::Tensor zy,
                             torch::Tensor h, torch::Tensor g, std::vector<double> input_config,
                             std::vector<double> output_config, int threads, bool windowed) {
    TORCH_CHECK(power_of_two(threads) && threads >= 32 && threads <= 256, "invalid atom threads");
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_moments(zx, atoms, input);
    check_moments(zy, atoms, output);
    TORCH_CHECK(zx.size(0) == zy.size(0), "Zx/Zy batch mismatch");
    for (const auto &projection : {h, g}) {
        TORCH_CHECK(projection.is_contiguous() && projection.device() == atoms.device() &&
                        (projection.scalar_type() == torch::kFloat64 ||
                         projection.scalar_type() == torch::kFloat32) &&
                        projection.dim() == 2 && projection.size(0) == zx.size(0) &&
                        projection.size(1) == atoms.size(0) && projection.numel() <= INT_MAX,
                    "invalid H/G");
    }
    TORCH_CHECK(h.scalar_type() == g.scalar_type(), "H/G storage dtype mismatch");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto gradient = torch::empty_like(atoms);
    if (atoms.size(0) == 0)
        return gradient;
    TORCH_CHECK(groups.size(0) > 0, "nonempty atoms need groups");
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_parameter_grad_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_parameter_grad_storage", [&] {
            parameter_grad_kernel<atom_t, scalar_t>
                <<<(atoms.size(0) + threads - 1) / threads, threads, 0, stream>>>(
                    atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                    offsets.data_ptr<int64_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(),
                    h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), gradient.data_ptr<atom_t>(),
                    atoms.size(0), groups.size(0), zx.size(0), input, output, windowed);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gradient;
}

std::vector<torch::Tensor> compute_g_and_parameter_grad(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor zx, torch::Tensor zy, torch::Tensor h, std::vector<double> input_config,
    std::vector<double> output_config, int threads, bool windowed, bool specialize_modes,
    bool compensated_grad, bool reuse_shape_coefficients) {
    TORCH_CHECK(power_of_two(threads) && threads >= 32 && threads <= 256, "invalid atom threads");
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(!reuse_shape_coefficients ||
                    (specialize_modes && input.modes == 8 && output.modes == 8 &&
                     atoms.scalar_type() == torch::kFloat32 && h.scalar_type() == torch::kFloat32),
                "shape coefficient reuse requires specialized M8 axes and FP32 atoms/H");
    TORCH_CHECK(!compensated_grad ||
                    (input.modes == 8 && output.modes == 8 && input.tail_bound == 1e-3 &&
                     output.tail_bound == 1e-3 && atoms.scalar_type() == torch::kFloat32 &&
                     h.scalar_type() == torch::kFloat32),
                "compensated gradients require both axes M8, tail 1e-3 and FP32 atoms/H");
    check_moments(zx, atoms, input);
    check_moments(zy, atoms, output);
    TORCH_CHECK(zx.size(0) == zy.size(0), "Zx/Zy batch mismatch");
    TORCH_CHECK(h.is_contiguous() && h.device() == atoms.device() &&
                    (h.scalar_type() == torch::kFloat64 || h.scalar_type() == torch::kFloat32) &&
                    h.dim() == 2 && h.size(0) == zx.size(0) && h.size(1) == atoms.size(0) &&
                    h.numel() <= INT_MAX,
                "invalid H");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto g = torch::empty_like(h);
    auto gradient = torch::empty_like(atoms);
    if (atoms.size(0) == 0)
        return {g, gradient};
    TORCH_CHECK(groups.size(0) > 0, "nonempty atoms need groups");
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_fused_grad_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_fused_grad_storage", [&] {
            if constexpr (std::is_same_v<atom_t, float> && std::is_same_v<scalar_t, float>) {
                if (reuse_shape_coefficients) {
                    size_t shared_bytes = 16 * threads * sizeof(double);
                    if (compensated_grad) {
                        compute_g_parameter_grad_kernel<atom_t, scalar_t, 8, true, true>
                            <<<(atoms.size(0) + threads - 1) / threads, threads, shared_bytes,
                               stream>>>(atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(),
                                         groups.data_ptr<int>(), offsets.data_ptr<int64_t>(),
                                         zx.data_ptr<double>(), zy.data_ptr<double>(),
                                         h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
                                         gradient.data_ptr<atom_t>(), atoms.size(0), groups.size(0),
                                         zx.size(0), input, output, windowed);
                    } else {
                        compute_g_parameter_grad_kernel<atom_t, scalar_t, 8, false, true>
                            <<<(atoms.size(0) + threads - 1) / threads, threads, shared_bytes,
                               stream>>>(atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(),
                                         groups.data_ptr<int>(), offsets.data_ptr<int64_t>(),
                                         zx.data_ptr<double>(), zy.data_ptr<double>(),
                                         h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
                                         gradient.data_ptr<atom_t>(), atoms.size(0), groups.size(0),
                                         zx.size(0), input, output, windowed);
                    }
                    return;
                }
                if (compensated_grad) {
                    if (specialize_modes) {
                        compute_g_parameter_grad_kernel<atom_t, scalar_t, 8, true>
                            <<<(atoms.size(0) + threads - 1) / threads, threads, 0, stream>>>(
                                atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(),
                                groups.data_ptr<int>(), offsets.data_ptr<int64_t>(),
                                zx.data_ptr<double>(), zy.data_ptr<double>(),
                                h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
                                gradient.data_ptr<atom_t>(), atoms.size(0), groups.size(0),
                                zx.size(0), input, output, windowed);
                    } else {
                        compute_g_parameter_grad_kernel<atom_t, scalar_t, 0, true>
                            <<<(atoms.size(0) + threads - 1) / threads, threads, 0, stream>>>(
                                atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(),
                                groups.data_ptr<int>(), offsets.data_ptr<int64_t>(),
                                zx.data_ptr<double>(), zy.data_ptr<double>(),
                                h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
                                gradient.data_ptr<atom_t>(), atoms.size(0), groups.size(0),
                                zx.size(0), input, output, windowed);
                    }
                    return;
                }
            }
            if (specialize_modes && input.modes == 8 && output.modes == 8) {
                compute_g_parameter_grad_kernel<atom_t, scalar_t, 8>
                    <<<(atoms.size(0) + threads - 1) / threads, threads, 0, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                        offsets.data_ptr<int64_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(),
                        h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), gradient.data_ptr<atom_t>(),
                        atoms.size(0), groups.size(0), zx.size(0), input, output, windowed);
            } else {
                compute_g_parameter_grad_kernel<atom_t, scalar_t, 0>
                    <<<(atoms.size(0) + threads - 1) / threads, threads, 0, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                        offsets.data_ptr<int64_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(),
                        h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), gradient.data_ptr<atom_t>(),
                        atoms.size(0), groups.size(0), zx.size(0), input, output, windowed);
            }
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g, gradient};
}

// Dedicated Polar pullback: no saved H, no global FP64 projection table, and
// no width task gradient. Rounded G is returned solely for the dX contraction.
std::vector<torch::Tensor> compute_polar_g_and_parameter_grad(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor zx, torch::Tensor zy, std::vector<double> input_config,
    std::vector<double> output_config, int threads, bool windowed, bool compensated,
    bool cache_coefficients) {
    TORCH_CHECK(power_of_two(threads) && threads >= 32 && threads <= 256, "invalid atom threads");
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(input.polar && output.polar, "precise Polar pullback requires Polar axes");
    TORCH_CHECK(!cache_coefficients || (input.modes == 8 && output.modes == 8 &&
        atoms.scalar_type() == torch::kFloat32 && (threads == 32 || threads == 64)),
        "Polar coefficient cache requires FP32 M8 and 32/64 threads");
    TORCH_CHECK(!compensated || (input.modes == 8 && output.modes == 8 &&
        input.tail_bound == 1e-3 && output.tail_bound == 1e-3 &&
        atoms.scalar_type() == torch::kFloat32), "invalid compensated Polar projection");
    check_moments(zx, atoms, input); check_moments(zy, atoms, output);
    TORCH_CHECK(zx.size(0) == zy.size(0) && zx.size(0) * atoms.size(0) <= INT_MAX,
                "invalid Polar projection batch/index size");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto g = torch::empty({zx.size(0), atoms.size(0)}, atoms.options());
    auto gradient = torch::empty_like(atoms);
    if (atoms.size(0) == 0) return {g, gradient};
    TORCH_CHECK(groups.size(0) > 0, "nonempty atoms need groups");
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "precise_polar_pullback", [&] {
        auto launch = [&](auto mode_tag, auto compensated_tag, auto cache_tag) {
            constexpr int modes = decltype(mode_tag)::value;
            constexpr bool use_compensated = decltype(compensated_tag)::value;
            constexpr bool use_cache = decltype(cache_tag)::value;
            size_t shared = use_cache ? 2 * (1 + 2 * modes) * threads * sizeof(double) : 0;
            polar_g_parameter_grad_kernel<scalar_t, modes, use_compensated, use_cache>
                <<<(atoms.size(0) + threads - 1) / threads, threads, shared, stream>>>(
                    atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                    offsets.data_ptr<int64_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(),
                    g.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), atoms.size(0),
                    groups.size(0), zx.size(0), input, output, windowed);
        };
        if (cache_coefficients) {
            // The checked cache contract is FP32 only. Do not instantiate
            // unsupported FP64 cache kernels during AT_DISPATCH expansion.
            if constexpr (std::is_same<scalar_t, float>::value) {
                if (compensated)
                    launch(std::integral_constant<int, 8>{}, std::true_type{}, std::true_type{});
                else
                    launch(std::integral_constant<int, 8>{}, std::false_type{}, std::true_type{});
            }
        } else if (compensated)
            launch(std::integral_constant<int, 8>{}, std::true_type{}, std::false_type{});
        else if (input.modes == 8 && output.modes == 8)
            launch(std::integral_constant<int, 8>{}, std::false_type{}, std::false_type{});
        else launch(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g, gradient};
}

// Untimed compiler/runtime diagnostics for the FP32 M8 Polar pullback.
// Occupancy API output is a prediction, not measured achieved occupancy.
template <bool CacheCoefficients, bool Compensated>
static std::vector<int64_t> polar_g_grad_resources_impl(int threads) {
    size_t shared = CacheCoefficients ? 2 * 17 * threads * sizeof(double) : 0;
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,
        polar_g_parameter_grad_kernel<float, 8, Compensated, CacheCoefficients>));
    int active_blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
        polar_g_parameter_grad_kernel<float, 8, Compensated, CacheCoefficients>, threads, shared));
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return {attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.sharedSizeBytes), attributes.maxThreadsPerBlock,
        active_blocks, static_cast<int64_t>(shared), properties.multiProcessorCount};
}

std::vector<int64_t> polar_g_grad_kernel_resources(int threads, bool cache_coefficients,
                                                 bool compensated) {
    TORCH_CHECK(threads == 32 || threads == 64, "resource query requires 32/64 threads");
    if (cache_coefficients) {
        if (compensated) return polar_g_grad_resources_impl<true, true>(threads);
        return polar_g_grad_resources_impl<true, false>(threads);
    }
    if (compensated) return polar_g_grad_resources_impl<false, true>(threads);
    return polar_g_grad_resources_impl<false, false>(threads);
}

// The contraction translation unit owns the established ragged metadata and
// reduction kernels. These internal launch wrappers share their exact schedule.
void launch_cell_mass_reverse_ragged_metadata(torch::Tensor reverse_offsets,
    torch::Tensor counts, torch::Tensor task_offsets, const Axis &axis, int splits, int grain);
void launch_cell_mass_ragged_reduce(torch::Tensor task_offsets, torch::Tensor partial,
    torch::Tensor c);

static size_t fused_polar_cin_shared_bytes(int batches) {
    return sizeof(double) * (2 * 17 * 33 + BATCH_TILE * 33 + 33 + batches * 17);
}

std::vector<torch::Tensor> compute_polar_grad_cin(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor reverse_slots, torch::Tensor reverse_offsets, torch::Tensor zx,
    torch::Tensor zy, std::vector<double> input_config, std::vector<double> output_config,
    int splits, int grain, bool windowed, bool debug_g) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(atoms.scalar_type() == torch::kFloat32 && input.polar && output.polar &&
        input.modes == 8 && output.modes == 8 && input.tail_bound == 1e-3 &&
        output.tail_bound == 1e-3, "fused Polar Cin requires FP32 Polar M8/tail1e-3 axes");
    TORCH_CHECK(splits >= 1 && splits <= 256, "splits must lie in [1,256]");
    TORCH_CHECK(grain >= 1 && grain <= 65536, "grain must lie in [1,65536]");
    check_moments(zx, atoms, input); check_moments(zy, atoms, output);
    TORCH_CHECK(zx.size(0) == zy.size(0) && zx.size(0) <= 32,
                "fused Polar Cin supports matching batches in [0,32]");
    TORCH_CHECK(reverse_slots.is_contiguous() && reverse_slots.device() == atoms.device() &&
        reverse_slots.dim() == 1 && reverse_slots.scalar_type() == torch::kInt32 &&
        reverse_slots.numel() == atoms.size(0), "invalid reverse slots");
    TORCH_CHECK(reverse_offsets.is_contiguous() && reverse_offsets.device() == atoms.device() &&
        reverse_offsets.dim() == 1 && reverse_offsets.scalar_type() == torch::kInt64 &&
        reverse_offsets.numel() == input.node_count + 1, "invalid reverse offsets");
    int64_t batches = zx.size(0), nodes = input.node_count, basis = input.basis_count();
    TORCH_CHECK(nodes <= 2048 && batches <= INT_MAX / (nodes * basis), "Cin exceeds index capacity");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto c = torch::empty({batches, nodes, basis}, atoms.options().dtype(torch::kFloat64));
    auto gradient = torch::empty_like(atoms);
    auto g = debug_g ? torch::empty({batches, atoms.size(0)}, atoms.options())
                     : torch::empty({0}, atoms.options());
    if (batches == 0 || atoms.size(0) == 0) {
        // In particular, B=0 with A>0 has a zero task gradient.
        c.zero_(); gradient.zero_();
        return {gradient, c, g};
    }
    TORCH_CHECK(groups.size(0) > 0, "nonempty atoms need groups");
    int64_t task_cap = std::min(nodes * splits, (atoms.size(0) + grain - 1) / grain + nodes);
    TORCH_CHECK(task_cap > 0 && task_cap <= INT_MAX / (batches * basis),
                "fused Polar Cin partial exceeds index capacity");
    auto ints = atoms.options().dtype(torch::kInt32);
    auto counts = torch::empty({nodes}, ints);
    auto task_offsets = torch::empty({nodes + 1}, ints);
    auto partial = torch::empty({task_cap, batches, basis}, c.options());
    launch_cell_mass_reverse_ragged_metadata(reverse_offsets, counts, task_offsets, input, splits, grain);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto launch = [&](auto debug_tag) {
        constexpr bool save_g = decltype(debug_tag)::value;
        fused_polar_grad_cin_kernel<save_g><<<task_cap, 32, fused_polar_cin_shared_bytes(batches), stream>>>(
            atoms.data_ptr<float>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
            offsets.data_ptr<int64_t>(), reverse_slots.data_ptr<int>(), reverse_offsets.data_ptr<int64_t>(),
            zx.data_ptr<double>(), zy.data_ptr<double>(), counts.data_ptr<int>(), task_offsets.data_ptr<int>(),
            partial.data_ptr<double>(), gradient.data_ptr<float>(), save_g ? g.data_ptr<float>() : nullptr,
            atoms.size(0), groups.size(0), batches, input, output, windowed);
    };
    if (debug_g) launch(std::true_type{}); else launch(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_cell_mass_ragged_reduce(task_offsets, partial, c);
    return {gradient, c, g};
}

template <bool DebugG>
static std::vector<int64_t> fused_polar_cin_resources_impl(int batches) {
    size_t shared = fused_polar_cin_shared_bytes(batches);
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, fused_polar_grad_cin_kernel<DebugG>));
    int active_blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, fused_polar_grad_cin_kernel<DebugG>, 32, shared));
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return {attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes),
        static_cast<int64_t>(attributes.sharedSizeBytes), attributes.maxThreadsPerBlock,
        active_blocks, static_cast<int64_t>(shared), properties.multiProcessorCount};
}

std::vector<int64_t> fused_polar_cin_kernel_resources(int batches, bool debug_g) {
    TORCH_CHECK(batches >= 0 && batches <= 32, "resource query requires batches in [0,32]");
    if (debug_g) return fused_polar_cin_resources_impl<true>(batches);
    return fused_polar_cin_resources_impl<false>(batches);
}
