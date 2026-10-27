// Existing profile mathematics, connected to varying-period Z and owner C.
#include "common.cuh"
using namespace cell_mass_hierarchy;

namespace {
constexpr int CHUNK = 32;
constexpr int MAX_ACCUMULATORS = (BATCH_TILE * 33 + CHUNK - 1) / CHUNK;

// One thread completes H for a different atom and up to four batches. The
// amplitude is not included. Neighbour Z tiles are rotated to the same
// unwrapped chart image as the atom's retained one-period window.
template <typename scalar_t, typename projection_t>
__global__ void compute_h_kernel(const scalar_t *atoms, const int64_t *rows, const int *groups,
                                 const int64_t *offsets, const double *z, projection_t *h,
                                 int atom_count, int group_count, int batches, Axis axis,
                                 bool output_side) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int group = group_for_slot(offsets, group_count, slot);
    int owner = groups[4 * group + (output_side ? 2 : 3)];
    Bank bank = axis.banks[bank_for_level(axis, groups[4 * group + (output_side ? 0 : 1)])];
    int original = rows[slot];
    Profile profile =
        make_profile(axis, bank, static_cast<double>(atoms[4 * original + 1]),
                     static_cast<double>(atoms[4 * original + (output_side ? 2 : 3)]), owner);
    double accum[BATCH_TILE] = {};
    int batch_begin = blockIdx.y * BATCH_TILE;
    int retained = bank.full_chart ? 1 : 3;
    int first_tile = bank.full_chart ? 0 : owner - 1;
    for (int harmonic = 0; harmonic <= axis.modes; ++harmonic) {
        double cc, cs;
        coefficient_pair(profile, axis.modes, harmonic, cc, cs);
        for (int offset = 0; offset < retained; ++offset) {
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            double rs = 0, rc = 1;
            if (harmonic != 0 && image != 0)
                sincos(2 * PI * harmonic / bank.period * image * axis.period, &rs, &rc);
            int node = bank.node_offset + tile;
            for (int local = 0; local < BATCH_TILE && batch_begin + local < batches; ++local) {
                int base = ((batch_begin + local) * axis.node_count + node) * axis.basis_count();
                if (harmonic == 0) {
                    accum[local] += cc * z[base];
                } else {
                    double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
                    accum[local] += cc * (zc * rc - zs * rs) + cs * (zs * rc + zc * rs);
                }
            }
        }
    }
    for (int local = 0; local < BATCH_TILE && batch_begin + local < batches; ++local)
        h[(batch_begin + local) * atom_count + slot] = accum[local];
}

// Lower bound on a (h_out,h_in,q_out) prefix. Input owners within this prefix
// are contiguous, so each output owner reads one atom interval per input bank.
__device__ int lower_group(const int *groups, int count, int ho, int hi, int qo) {
    int first = 0, last = count;
    while (first < last) {
        int middle = (first + last) / 2;
        const int *g = groups + 4 * middle;
        bool smaller = g[0] < ho || (g[0] == ho && (g[1] < hi || (g[1] == hi && g[2] < qo)));
        if (smaller)
            first = middle + 1;
        else
            last = middle;
    }
    return first;
}

// One block owns one output bank/owner. A lane builds the coefficients of a
// different atom in a 32-atom chunk. Shared holds only this chunk's coefficients
// and amp*H. C has one writer per element, without atom-sized partial C arrays.
template <typename scalar_t, typename projection_t>
__global__ void compute_c_kernel(const scalar_t *atoms, const int64_t *rows, const int *groups,
                                 const int64_t *offsets, const projection_t *h, double *c,
                                 int atom_count, int group_count, int batches, Axis input,
                                 Axis output, bool reverse) {
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_h = shared + CHUNK * output.basis_count();
    __shared__ int range_begin, range_end;
    int node = blockIdx.x, index = 0;
    while (index + 1 < output.bank_count && node >= output.banks[index + 1].node_offset)
        ++index;
    Bank bank = output.banks[index];
    int owner = node - bank.node_offset, lane = threadIdx.x;
    int batch_begin = blockIdx.y * BATCH_TILE;
    int elements = BATCH_TILE * output.basis_count();
    double accum[MAX_ACCUMULATORS] = {};
    // Forward uses contiguous output prefixes. Reverse initially scans sparse
    // group descriptors, selecting (input_level,input_owner); atom values stay
    // in the original Parameter and no transposed atom copy is allocated.
    int ranges = reverse ? group_count : input.bank_count;
    for (int range = 0; range < ranges; ++range) {
        if (lane == 0) {
            if (reverse) {
                bool matches =
                    groups[4 * range + 1] == bank.level && groups[4 * range + 3] == owner;
                range_begin = matches ? offsets[range] : 0;
                range_end = matches ? offsets[range + 1] : 0;
            } else {
                int hi = input.banks[range].level;
                range_begin = offsets[lower_group(groups, group_count, bank.level, hi, owner)];
                range_end = offsets[lower_group(groups, group_count, bank.level, hi, owner + 1)];
            }
        }
        __syncthreads();
        for (int chunk = range_begin; chunk < range_end; chunk += CHUNK) {
            int slot = chunk + lane;
            bool valid = slot < range_end;
            int original = valid ? rows[slot] : 0;
            if (valid) {
                Profile profile = make_profile(
                    output, bank, static_cast<double>(atoms[4 * original + 1]),
                    static_cast<double>(atoms[4 * original + (reverse ? 3 : 2)]), owner);
                for (int harmonic = 0; harmonic <= output.modes; ++harmonic) {
                    double cc, cs;
                    coefficient_pair(profile, output.modes, harmonic, cc, cs);
                    if (harmonic == 0)
                        coefficients[lane] = cc;
                    else {
                        coefficients[(2 * harmonic - 1) * CHUNK + lane] = cc;
                        coefficients[2 * harmonic * CHUNK + lane] = cs;
                    }
                }
            } else {
                for (int mode = 0; mode < output.basis_count(); ++mode)
                    coefficients[mode * CHUNK + lane] = 0;
            }
            for (int local = 0; local < BATCH_TILE; ++local)
                scaled_h[local * CHUNK + lane] =
                    valid && batch_begin + local < batches
                        ? static_cast<double>(atoms[4 * original]) *
                              h[(batch_begin + local) * atom_count + slot]
                        : 0;
            __syncthreads();
            for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
                int batch = work / output.basis_count(), mode = work % output.basis_count();
                for (int atom = 0; atom < CHUNK; ++atom)
                    accum[a] += scaled_h[batch * CHUNK + atom] * coefficients[mode * CHUNK + atom];
            }
            __syncthreads(); // Readers finish before the shared chunk is replaced.
        }
        __syncthreads(); // Finish all range_end reads before lane 0 changes it.
    }
    for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
        int local = work / output.basis_count(), mode = work % output.basis_count();
        if (batch_begin + local < batches)
            c[((batch_begin + local) * output.node_count + node) * output.basis_count() + mode] =
                accum[a];
    }
}

// Unique output site/batch ownership: add contributions from each bank after
// evaluating C in that bank's basis. No Y atomic or mixed-basis C summation.
template <typename scalar_t>
__global__ void expand_c_kernel(const double *c, scalar_t *y, int batches, Axis axis, int width) {
    for (int work = threadIdx.x; work < width * BATCH_TILE; work += blockDim.x) {
        int site = blockIdx.x * width + work % width;
        int batch = blockIdx.y * BATCH_TILE + work / width;
        if (site >= axis.sites || batch >= batches)
            continue;
        double total = 0;
        for (int index = 0; index < axis.bank_count; ++index) {
            Bank bank = axis.banks[index];
            int tile = site / bank.tile_sites;
            int retained = bank.full_chart ? 1 : 3;
            for (int delta = 0; delta < retained; ++delta) {
                int owner = bank.full_chart ? 0 : positive_mod(tile - 1 + delta, bank.tile_count);
                int raw_tile = bank.full_chart
                                   ? 0
                                   : owner - 1 + positive_mod(tile - (owner - 1), bank.tile_count);
                int image = bank.full_chart ? 0 : (raw_tile - tile) / bank.tile_count;
                int base =
                    (batch * axis.node_count + bank.node_offset + owner) * axis.basis_count();
                total += c[base] * axis.spacing;
                double position = site * axis.spacing + image * axis.period;
                for (int harmonic = 1; harmonic <= axis.modes; ++harmonic) {
                    double omega = 2 * PI * harmonic / bank.period;
                    double factor = 2 * sin(omega * axis.spacing / 2) / omega;
                    double sine, cosine;
                    sincos(omega * position, &sine, &cosine);
                    total += factor *
                             (c[base + 2 * harmonic - 1] * cosine + c[base + 2 * harmonic] * sine);
                }
            }
        }
        y[batch * axis.sites + site] = static_cast<scalar_t>(total);
    }
}

} // namespace

torch::Tensor compute_h(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                        torch::Tensor offsets, torch::Tensor z, std::vector<double> config,
                        bool output_side, bool float_storage) {
    check_layout(atoms, rows, groups, offsets);
    Axis axis = parse_axis(config);
    check_moments(z, atoms, axis);
    TORCH_CHECK(z.size(0) * atoms.size(0) <= INT_MAX, "H exceeds index capacity");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto h = torch::empty({z.size(0), atoms.size(0)},
                          z.options().dtype(float_storage ? torch::kFloat32 : torch::kFloat64));
    if (h.numel() == 0)
        return h;
    TORCH_CHECK(groups.size(0) > 0, "nonempty atoms need groups");
    dim3 grid((atoms.size(0) + THREADS - 1) / THREADS, (z.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_compute_h_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_compute_h_storage", [&] {
            compute_h_kernel<atom_t, scalar_t><<<grid, THREADS, 0, stream>>>(
                atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                offsets.data_ptr<int64_t>(), z.data_ptr<double>(), h.data_ptr<scalar_t>(),
                atoms.size(0), groups.size(0), z.size(0), axis, output_side);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
}

torch::Tensor compute_c(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                        torch::Tensor offsets, torch::Tensor h, std::vector<double> input_config,
                        std::vector<double> output_config, bool reverse) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(h.is_contiguous() && h.device() == atoms.device() && h.dim() == 2 &&
                    (h.scalar_type() == torch::kFloat64 || h.scalar_type() == torch::kFloat32) &&
                    h.size(1) == atoms.size(0) && h.numel() <= INT_MAX,
                "invalid H");
    TORCH_CHECK(h.size(0) <= INT_MAX / (output.node_count * output.basis_count()),
                "C exceeds index capacity");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto c = torch::empty({h.size(0), output.node_count, output.basis_count()},
                          h.options().dtype(torch::kFloat64));
    if (c.numel() == 0)
        return c;
    dim3 grid(output.node_count, (h.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    size_t shared = sizeof(double) * CHUNK * (output.basis_count() + BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_compute_c_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_compute_c_storage", [&] {
            compute_c_kernel<atom_t, scalar_t><<<grid, CHUNK, shared, stream>>>(
                atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                offsets.data_ptr<int64_t>(), h.data_ptr<scalar_t>(), c.data_ptr<double>(),
                atoms.size(0), groups.size(0), h.size(0), input, output, reverse);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor expand_c(torch::Tensor c, torch::Tensor like, std::vector<double> config, int width) {
    check_cuda_float_matrix(like);
    Axis axis = parse_axis(config);
    check_moments(c, like, axis);
    TORCH_CHECK(c.size(0) == like.size(0), "batch mismatch");
    TORCH_CHECK(width >= 1 && width <= 256 && power_of_two(width), "invalid output block width");
    c10::cuda::CUDAGuard guard(like.device());
    auto y = torch::empty({c.size(0), axis.sites}, like.options());
    if (y.numel() == 0)
        return y;
    dim3 grid((axis.sites + width - 1) / width, (c.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(like.scalar_type(), "hierarchical_expand_c", [&] {
        expand_c_kernel<<<grid, THREADS, 0, stream>>>(c.data_ptr<double>(), y.data_ptr<scalar_t>(),
                                                      c.size(0), axis, width);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
