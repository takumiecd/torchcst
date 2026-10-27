// Analytic atom gradients. Each thread owns a different atom and writes its
// original Parameter row once; batch reduction stays in double registers.
#include "common.cuh"
using namespace cell_mass_hierarchy;

namespace {
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
    double amplitude = atoms[4 * original], sigma = atoms[4 * original + 1];
    double input_center = atoms[4 * original + 3], output_center = atoms[4 * original + 2];
    double da = 0, ds = 0, dmu_input = 0, dmu_output = 0;
    for (int begin = 0; begin < batches; begin += BATCH_TILE) {
        double dh_mu[BATCH_TILE] = {}, dh_sigma[BATCH_TILE] = {};
        double dg_mu[BATCH_TILE] = {}, dg_sigma[BATCH_TILE] = {};
        project_derivatives(zx, batches, begin, input, input_bank, input_owner, sigma, input_center,
                            dh_mu, dh_sigma, windowed);
        project_derivatives(zy, batches, begin, output, output_bank, output_owner, sigma,
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
    gradient[4 * original] = static_cast<scalar_t>(da);
    gradient[4 * original + 1] = static_cast<scalar_t>(ds);
    gradient[4 * original + 2] = static_cast<scalar_t>(dmu_output);
    gradient[4 * original + 3] = static_cast<scalar_t>(dmu_input);
}

// Accumulate a weighted projection derivative in scalar registers. On the
// output side, calculate G at the same time; input-side weights are the stored
// rounded G multiplied by amplitude. Gradients reassociate FP64 batch sums.
template <bool Value, int FixedModes, typename projection_t>
__device__ inline void
project_weighted_derivatives(const double *z, int batches, int batch_begin, const Axis &axis,
                             const Bank &bank, int owner, const Profile &profile, double beta_sigma,
                             const projection_t *h, int atom_count, int slot, double amplitude,
                             double *g_value, double &dcenter, double &dsigma, bool windowed) {
    // A windowed node already contains the rotated sum of its retained tiles.
    int retained = windowed || bank.full_chart ? 1 : 3;
    int first_tile = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    int mode_count = FixedModes ? FixedModes : axis.modes;
    HarmonicStream<true> stream(profile, mode_count);
    // The M=8 instantiation has a compile-time trip count. The generic
    // instantiation retains the runtime-selected profile mode count.
#pragma unroll
    for (int harmonic = 0; harmonic <= mode_count; ++harmonic) {
        double cc, cs, cc_beta, cs_beta;
        stream.next(cc, cs, cc_beta, cs_beta);
        double omega = profile.omega * harmonic;
        for (int offset = 0; offset < retained; ++offset) {
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            double rs, rc;
            image_rotation(bank, harmonic, image, rs, rc);
            int node = bank.node_offset + tile;
            double weighted_cos = 0, weighted_sin = 0;
            for (int b = 0; b < BATCH_TILE && batch_begin + b < batches; ++b) {
                int base = ((batch_begin + b) * axis.node_count + node) * axis.basis_count();
                if (harmonic == 0) {
                    if (Value)
                        g_value[b] += cc * z[base];
                    continue;
                }
                double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                if (Value) {
                    double rotated_cos = zc * rc - zs * rs;
                    double rotated_sin = zs * rc + zc * rs;
                    g_value[b] += cc * rotated_cos + cs * rotated_sin;
                    double weight =
                        amplitude * static_cast<double>(h[(batch_begin + b) * atom_count + slot]);
                    weighted_cos += weight * rotated_cos;
                    weighted_sin += weight * rotated_sin;
                } else {
                    weighted_cos += g_value[b] * zc;
                    weighted_sin += g_value[b] * zs;
                }
            }
            if (harmonic != 0) {
                if (!Value) {
                    // Rotation is shared across batches, so apply it once to
                    // their weighted Z sum. This reassociates FP64 addition.
                    double rotated_cos = weighted_cos * rc - weighted_sin * rs;
                    weighted_sin = weighted_sin * rc + weighted_cos * rs;
                    weighted_cos = rotated_cos;
                }
                dcenter += omega * (-cs * weighted_cos + cc * weighted_sin);
                dsigma += beta_sigma * (cc_beta * weighted_cos + cs_beta * weighted_sin);
            }
        }
    }
}

template <typename scalar_t, typename projection_t, int FixedModes>
__global__ void compute_g_parameter_grad_kernel(
    const scalar_t *atoms, const int64_t *rows, const int *groups, const int64_t *offsets,
    const double *zx, const double *zy, const projection_t *h, projection_t *g, scalar_t *gradient,
    int atom_count, int group_count, int batches, Axis input, Axis output, bool windowed) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    // Axis is passed by value; specialize only this kernel's local metadata.
    if (FixedModes) {
        input.modes = FixedModes;
        output.modes = FixedModes;
    }
    int original = rows[slot];
    int group = group_for_slot(offsets, group_count, slot);
    int input_owner = groups[4 * group + 3], output_owner = groups[4 * group + 2];
    Bank input_bank = input.banks[bank_for_level(input, groups[4 * group + 1])];
    Bank output_bank = output.banks[bank_for_level(output, groups[4 * group])];
    double amplitude = atoms[4 * original], sigma = atoms[4 * original + 1];
    double input_center = atoms[4 * original + 3], output_center = atoms[4 * original + 2];
    Profile output_profile =
        make_profile<true>(output, output_bank, sigma, output_center, output_owner);
    Profile input_profile = make_profile<true>(input, input_bank, sigma, input_center, input_owner);
    double beta_sigma_out = output_profile.beta_sigma;
    double beta_sigma_in = input_profile.beta_sigma;
    double da = 0, ds = 0, dmu_input = 0, dmu_output = 0;
    for (int begin = 0; begin < batches; begin += BATCH_TILE) {
        double gv[BATCH_TILE] = {};
        project_weighted_derivatives<true, FixedModes>(
            zy, batches, begin, output, output_bank, output_owner, output_profile, beta_sigma_out,
            h, atom_count, slot, amplitude, gv, dmu_output, ds, windowed);
        for (int b = 0; b < BATCH_TILE && begin + b < batches; ++b) {
            projection_t rounded_g = static_cast<projection_t>(gv[b]);
            g[(begin + b) * atom_count + slot] = rounded_g;
            double h_value = h[(begin + b) * atom_count + slot];
            double g_value = static_cast<double>(rounded_g);
            da += h_value * g_value;
            gv[b] = amplitude * g_value;
        }
        project_weighted_derivatives<false, FixedModes>(
            zx, batches, begin, input, input_bank, input_owner, input_profile, beta_sigma_in, h,
            atom_count, slot, amplitude, gv, dmu_input, ds, windowed);
    }
    gradient[4 * original] = static_cast<scalar_t>(da);
    gradient[4 * original + 1] = static_cast<scalar_t>(ds);
    gradient[4 * original + 2] = static_cast<scalar_t>(dmu_output);
    gradient[4 * original + 3] = static_cast<scalar_t>(dmu_input);
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

std::vector<torch::Tensor>
compute_g_and_parameter_grad(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                             torch::Tensor offsets, torch::Tensor zx, torch::Tensor zy,
                             torch::Tensor h, std::vector<double> input_config,
                             std::vector<double> output_config, int threads, bool windowed) {
    TORCH_CHECK(power_of_two(threads) && threads >= 32 && threads <= 256, "invalid atom threads");
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
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
            if (input.modes == 8 && output.modes == 8) {
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
