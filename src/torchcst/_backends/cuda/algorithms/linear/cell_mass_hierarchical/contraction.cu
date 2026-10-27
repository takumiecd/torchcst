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
    HarmonicStream<false> stream(profile, axis.modes);
    double accum[BATCH_TILE] = {};
    int batch_begin = blockIdx.y * BATCH_TILE;
    int retained = bank.full_chart ? 1 : 3;
    int first_tile = bank.full_chart ? 0 : owner - 1;
    for (int harmonic = 0; harmonic <= axis.modes; ++harmonic) {
        double cc, cs, unused_cc_beta, unused_cs_beta;
        stream.next(cc, cs, unused_cc_beta, unused_cs_beta);
        for (int offset = 0; offset < retained; ++offset) {
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            double rs, rc;
            image_rotation(bank, harmonic, image, rs, rc);
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
                HarmonicStream<false> stream(profile, output.modes);
                for (int harmonic = 0; harmonic <= output.modes; ++harmonic) {
                    double cc, cs, unused_cc_beta, unused_cs_beta;
                    stream.next(cc, cs, unused_cc_beta, unused_cs_beta);
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

// Consume the intersection of one group's sorted-slot range and this split's
// interval in the concatenation of all groups belonging to the owner. All
// lanes take the same branches and participate in every shared-memory barrier.
template <typename scalar_t, typename projection_t>
__device__ void split_consume_range(const scalar_t *atoms, const int64_t *rows,
                                    const projection_t *h, int atom_count, int batches, Axis output,
                                    Bank bank, int owner, bool reverse, int batch_begin, int begin,
                                    int end, int split_begin, int split_end, int &prefix,
                                    double *coefficients, double *scaled_h,
                                    double (&accum)[MAX_ACCUMULATORS]) {
    int length = end - begin;
    int first = max(0, split_begin - prefix);
    int last = min(length, split_end - prefix);
    prefix += length;
    int lane = threadIdx.x;
    int elements = BATCH_TILE * output.basis_count();
    for (int chunk = begin + first; chunk < begin + last; chunk += CHUNK) {
        int slot = chunk + lane;
        bool valid = slot < begin + last;
        int original = valid ? rows[slot] : 0;
        if (valid) {
            Profile profile =
                make_profile(output, bank, static_cast<double>(atoms[4 * original + 1]),
                             static_cast<double>(atoms[4 * original + (reverse ? 3 : 2)]), owner);
            HarmonicStream<false> stream(profile, output.modes);
            for (int harmonic = 0; harmonic <= output.modes; ++harmonic) {
                double cc, cs, unused_cc_beta, unused_cs_beta;
                stream.next(cc, cs, unused_cc_beta, unused_cs_beta);
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
            scaled_h[local * CHUNK + lane] = valid && batch_begin + local < batches
                                                 ? static_cast<double>(atoms[4 * original]) *
                                                       h[(batch_begin + local) * atom_count + slot]
                                                 : 0;
        __syncthreads();
        for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
            int batch = work / output.basis_count(), mode = work % output.basis_count();
            for (int atom = 0; atom < CHUNK; ++atom)
                accum[a] += scaled_h[batch * CHUNK + atom] * coefficients[mode * CHUNK + atom];
        }
        __syncthreads();
    }
}

// One warp per (owner, batch tile, split). The temporary is bounded by the
// number of owners and splits, never by atoms or groups. Two descriptor passes
// count the owner's atoms then partition their concatenation evenly: even a
// collection of short groups is distributed across splits. Reverse lookup uses
// one lane per descriptor and a warp ballot, avoiding barriers for nonmatches.
// Passed by value as CUDA kernel arguments, so no device pointer-table allocation.
// The tensors that own these addresses stay alive until both kernels are queued.
struct SplitPages {
    double *pointers[64];
    int page_splits;
};

template <typename scalar_t, typename projection_t>
__global__ void compute_c_split_kernel(const scalar_t *atoms, const int64_t *rows,
                                       const int *groups, const int64_t *offsets,
                                       const projection_t *h, SplitPages pages, int atom_count,
                                       int group_count, int batches, Axis input, Axis output,
                                       bool reverse, bool cartesian, int splits) {
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_h = shared + CHUNK * output.basis_count();
    int node = blockIdx.x, bank_index = 0;
    while (bank_index + 1 < output.bank_count && node >= output.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = output.banks[bank_index];
    int owner = node - bank.node_offset;
    int lane = threadIdx.x;
    int batch_begin = blockIdx.y * BATCH_TILE;
    int total = 0, split_begin = 0, split_end = 0;
    double accum[MAX_ACCUMULATORS] = {};
    for (int pass = 0; pass < 2; ++pass) {
        int prefix = 0;
        if (reverse && cartesian) {
            // Groups are ordered (output bank,input bank,output owner,input
            // owner). Here output is the target input axis, while input is the
            // other (original output) axis. Visit its banks/owners in the same
            // order as the sparse scan, and address the target owner directly.
            for (int range = 0; range < input.bank_count; ++range) {
                Bank other = input.banks[range];
                int first = other.node_offset * output.node_count +
                            other.tile_count * bank.node_offset + owner;
                for (int other_owner = 0; other_owner < other.tile_count; ++other_owner) {
                    int group = first + other_owner * bank.tile_count;
                    int begin = offsets[group], end = offsets[group + 1];
                    if (pass == 0)
                        prefix += end - begin;
                    else
                        split_consume_range(atoms, rows, h, atom_count, batches, output, bank,
                                            owner, reverse, batch_begin, begin, end, split_begin,
                                            split_end, prefix, coefficients, scaled_h, accum);
                }
            }
        } else if (reverse) {
            for (int base = 0; base < group_count; base += CHUNK) {
                int group = base + lane;
                bool matches = group < group_count && groups[4 * group + 1] == bank.level &&
                               groups[4 * group + 3] == owner;
                unsigned mask = __ballot_sync(0xffffffffu, matches);
                while (mask) {
                    int g = base + __ffs(mask) - 1;
                    mask &= mask - 1;
                    int begin = offsets[g], end = offsets[g + 1];
                    if (pass == 0)
                        prefix += end - begin;
                    else
                        split_consume_range(atoms, rows, h, atom_count, batches, output, bank,
                                            owner, reverse, batch_begin, begin, end, split_begin,
                                            split_end, prefix, coefficients, scaled_h, accum);
                }
            }
        } else {
            for (int range = 0; range < input.bank_count; ++range) {
                Bank other = input.banks[range];
                int first, last;
                if (cartesian) {
                    first = bank.node_offset * input.node_count +
                            bank.tile_count * other.node_offset + owner * other.tile_count;
                    last = first + other.tile_count;
                } else {
                    first = lower_group(groups, group_count, bank.level, other.level, owner);
                    last = lower_group(groups, group_count, bank.level, other.level, owner + 1);
                }
                int begin = offsets[first], end = offsets[last];
                if (pass == 0)
                    prefix += end - begin;
                else
                    split_consume_range(atoms, rows, h, atom_count, batches, output, bank, owner,
                                        reverse, batch_begin, begin, end, split_begin, split_end,
                                        prefix, coefficients, scaled_h, accum);
            }
        }
        if (pass == 0) {
            total = prefix;
            split_begin = static_cast<int>(static_cast<int64_t>(total) * blockIdx.z / splits);
            split_end = static_cast<int>(static_cast<int64_t>(total) * (blockIdx.z + 1) / splits);
        }
    }
    int elements = BATCH_TILE * output.basis_count();
    for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
        int local = work / output.basis_count(), mode = work % output.basis_count();
        if (batch_begin + local < batches) {
            int local_split = blockIdx.z % pages.page_splits;
            int page = blockIdx.z / pages.page_splits;
            int64_t index = ((static_cast<int64_t>(local_split) * batches + batch_begin + local) *
                                 output.node_count +
                             node) *
                                output.basis_count() +
                            mode;
            pages.pointers[page][index] = accum[a];
        }
    }
}

__global__ void reduce_c_splits_kernel(SplitPages pages, double *c, int64_t elements, int splits) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements)
        return;
    double value = 0;
    for (int split = 0; split < splits; ++split)
        value += pages.pointers[split / pages.page_splits]
                               [static_cast<int64_t>(split % pages.page_splits) * elements + i];
    c[i] = value;
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
            int bases[3], image_residues[3];
            for (int delta = 0; delta < retained; ++delta) {
                int owner = bank.full_chart ? 0 : positive_mod(tile - 1 + delta, bank.tile_count);
                int raw_tile = bank.full_chart
                                   ? 0
                                   : owner - 1 + positive_mod(tile - (owner - 1), bank.tile_count);
                int image = bank.full_chart ? 0 : (raw_tile - tile) / bank.tile_count;
                bases[delta] =
                    (batch * axis.node_count + bank.node_offset + owner) * axis.basis_count();
                image_residues[delta] =
                    bank.full_chart ? 0 : positive_mod(image * bank.tile_count, 3);
                total += c[bases[delta]] * axis.spacing;
            }
            // For a three-tile bank, an image shift has phase
            // 2*pi*k*image*tile_count/3. Its residue gives an exact choice of
            // the three roots of unity; the full-chart image is always zero.
            // Rotate owner coefficients into the canonical site frame first,
            // then evaluate that site's basis once for all retained owners.
            double omega = 2 * PI / bank.period;
            double sin_step, cos_step, sin_half_step, cos_half_step;
            sincos(omega * (site * axis.spacing), &sin_step, &cos_step);
            sincos(omega * axis.spacing / 2, &sin_half_step, &cos_half_step);
            double sine = sin_step, cosine = cos_step;
            double half_sine = sin_half_step, half_cosine = cos_half_step;
            constexpr double SQRT3_OVER_2 = 0.86602540378443864676;
            for (int harmonic = 1; harmonic <= axis.modes; ++harmonic) {
                double canonical_cos = 0, canonical_sin = 0;
                for (int delta = 0; delta < retained; ++delta) {
                    int residue = (harmonic * image_residues[delta]) % 3;
                    double rc = residue == 0 ? 1 : -0.5;
                    double rs = residue == 0 ? 0 : (residue == 1 ? SQRT3_OVER_2 : -SQRT3_OVER_2);
                    int base = bases[delta];
                    double cc = c[base + 2 * harmonic - 1];
                    double cs = c[base + 2 * harmonic];
                    canonical_cos += cc * rc + cs * rs;
                    canonical_sin += cs * rc - cc * rs;
                }
                double factor = 2 * half_sine / (omega * harmonic);
                total += factor * (canonical_cos * cosine + canonical_sin * sine);
                // Advance both the site phase and the cell half-step phase in
                // FP64; J_k uses the latter, including the 1/k factor above.
                double next_sine = sine * cos_step + cosine * sin_step;
                cosine = cosine * cos_step - sine * sin_step;
                sine = next_sine;
                double next_half_sine = half_sine * cos_half_step + half_cosine * sin_half_step;
                half_cosine = half_cosine * cos_half_step - half_sine * sin_half_step;
                half_sine = next_half_sine;
            }
        }
        y[batch * axis.sites + site] = static_cast<scalar_t>(total);
    }
}

} // namespace

torch::Tensor compute_h(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                        torch::Tensor offsets, torch::Tensor z, std::vector<double> config,
                        bool output_side, bool float_storage, int threads) {
    check_layout(atoms, rows, groups, offsets);
    Axis axis = parse_axis(config);
    check_moments(z, atoms, axis);
    TORCH_CHECK(z.size(0) * atoms.size(0) <= INT_MAX, "H exceeds index capacity");
    TORCH_CHECK(power_of_two(threads) && threads >= 32 && threads <= 256, "invalid atom threads");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto h = torch::empty({z.size(0), atoms.size(0)},
                          z.options().dtype(float_storage ? torch::kFloat32 : torch::kFloat64));
    if (h.numel() == 0)
        return h;
    TORCH_CHECK(groups.size(0) > 0, "nonempty atoms need groups");
    dim3 grid((atoms.size(0) + threads - 1) / threads, (z.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_compute_h_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_compute_h_storage", [&] {
            compute_h_kernel<atom_t, scalar_t><<<grid, threads, 0, stream>>>(
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

// Research candidate: fixed owner/split partials, then one writer per C element.
// S=1 remains available through compute_c for direct baseline comparison.
torch::Tensor compute_c_split(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                              torch::Tensor offsets, torch::Tensor h,
                              std::vector<double> input_config, std::vector<double> output_config,
                              bool reverse, int splits) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(splits >= 1 && splits <= 64, "splits must lie in [1,64]");
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
    // Small-allocation pages avoid crossing the allocator's 1 MiB boundary when
    // one C fits. If a single C exceeds it, one split occupies an oversized page.
    constexpr int64_t PAGE_BYTES = 1024 * 1024;
    int64_t c_bytes = c.numel() * static_cast<int64_t>(sizeof(double));
    int page_splits =
        static_cast<int>(std::min<int64_t>(splits, std::max<int64_t>(1, PAGE_BYTES / c_bytes)));
    SplitPages pages{};
    pages.page_splits = page_splits;
    std::vector<torch::Tensor> partial_pages;
    partial_pages.reserve((splits + page_splits - 1) / page_splits);
    for (int begin = 0; begin < splits; begin += page_splits) {
        int count = std::min(page_splits, splits - begin);
        partial_pages.push_back(
            torch::empty({count, h.size(0), output.node_count, output.basis_count()}, c.options()));
        pages.pointers[partial_pages.size() - 1] = partial_pages.back().data_ptr<double>();
    }
    dim3 grid(output.node_count, (h.size(0) + BATCH_TILE - 1) / BATCH_TILE, splits);
    size_t shared = sizeof(double) * CHUNK * (output.basis_count() + BATCH_TILE);
    // Counting layouts contain every Cartesian group, including empty ones;
    // a fully populated unique sorted layout has the same complete order.
    bool cartesian = groups.size(0) == input.node_count * output.node_count;
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_split_c_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_split_c_storage", [&] {
            compute_c_split_kernel<atom_t, scalar_t><<<grid, CHUNK, shared, stream>>>(
                atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                offsets.data_ptr<int64_t>(), h.data_ptr<scalar_t>(), pages, atoms.size(0),
                groups.size(0), h.size(0), input, output, reverse, cartesian, splits);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t elements = c.numel();
    reduce_c_splits_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        pages, c.data_ptr<double>(), elements, splits);
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
