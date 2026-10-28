// Existing profile mathematics, connected to varying-period Z and owner C.
#include "projection.cuh"
#include <type_traits>
using namespace cell_mass_hierarchy;

namespace {
constexpr int CHUNK = 32;
// Each logical row still contains 32 atoms. The extra shared-memory column
// changes only the physical row stride, reducing conflicts when lanes read
// different coefficient modes for the same atom.
constexpr int SHARED_STRIDE = CHUNK + 1;
constexpr int MAX_ACCUMULATORS = (BATCH_TILE * 33 + CHUNK - 1) / CHUNK;

// Engineering error budget for the experimental dot route, not a universal
// floating-point proof. Invalid or unrepresentable budgets disable conversion.
struct CdotBudget {
    double absolute;
    double factor;
};

// A whole-call guard prevents a large FP64 prefix from preceding an otherwise
// eligible DS chunk. The result stays on device and is consumed on this stream.
__global__ void cdot_preflight_kernel(const float *atoms, const int64_t *rows,
                                      const float *projection, int atom_count, int64_t elements,
                                      int *unsafe, Axis axis) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements)
        return;
    int slot = i % atom_count;
    int original = rows[slot];
    double value = atom_amplitude(atoms, original, axis) * projection[i];
    if (!isfinite(value) || fabs(value) > 16)
        atomicOr(unsafe, 1);
}

inline torch::Tensor make_cdot_preflight(const torch::Tensor &atoms, const torch::Tensor &rows,
                                         const torch::Tensor &projection, cudaStream_t stream, Axis axis) {
    auto flag = torch::zeros({1}, atoms.options().dtype(torch::kInt32));
    if (projection.numel() > 0) {
        cdot_preflight_kernel<<<(projection.numel() + 255) / 256, 256, 0, stream>>>(
            atoms.data_ptr<float>(), rows.data_ptr<int64_t>(), projection.data_ptr<float>(),
            atoms.size(0), projection.numel(), flag.data_ptr<int>(), axis);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return flag;
}

inline CdotBudget cdot_budget(int64_t atoms, int splits, int grain, const Axis &target, int ranges,
                              bool legacy_reverse) {
    if (atoms == 0)
        return {0, 0};
    int64_t split_nodes = static_cast<int64_t>(splits) * target.node_count;
    int64_t active_bound =
        grain ? std::min(split_nodes, (atoms + grain - 1) / grain + target.node_count)
              : split_nodes;
    int64_t chunks = legacy_reverse
                         ? atoms
                         : std::min(atoms, (atoms + CHUNK - 1) / CHUNK + active_bound * ranges);
    int64_t evenly_split = (atoms + splits - 1) / splits;
    int64_t max_split = grain ? std::max<int64_t>(grain, evenly_split) : evenly_split;
    double budget = ((5e-8 / static_cast<double>(chunks)) / target.basis_count()) / target.spacing;
    double factor = 1024.0 * std::ldexp(1.0, -48) +
                    8.0 * std::ldexp(1.0, -53) *
                        (max_split + splits + 3 * target.bank_count * target.basis_count());
    if (!(std::isfinite(budget) && budget >= DBL_MIN && std::isfinite(factor)))
        return {0, 0};
    return {budget, factor};
}

// Pack the two FP32 components into one shared 8-byte cell. These bit patterns
// are never used as FP64 values: some can encode a NaN when viewed as double.
__device__ inline double pack_cdot_pair(DoubleSingle value) {
    unsigned long long bits = static_cast<unsigned long long>(__float_as_uint(value.lo)) << 32;
    bits |= __float_as_uint(value.hi);
    return __longlong_as_double(static_cast<long long>(bits));
}

__device__ inline DoubleSingle unpack_cdot_pair(double payload) {
    unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(payload));
    return {__uint_as_float(static_cast<unsigned int>(bits)),
            __uint_as_float(static_cast<unsigned int>(bits >> 32))};
}

// All 32 lanes call this after writing their own FP64 column, before the
// existing chunk barrier. Unsafe values remain untouched for the FP64 route.
__device__ inline bool prepare_cdot_chunk(double *coefficients, double *scaled, int basis,
                                          CdotBudget budget, double maximum_accumulator,
                                          bool invalid_accumulator) {
    int lane = threadIdx.x;
    double maximum_coefficient = 0, maximum_scaled = 0;
    bool unsafe = invalid_accumulator || !isfinite(maximum_accumulator) ||
                  !(budget.absolute > 0 && isfinite(budget.absolute));
    constexpr double LOWER = 0x1p-40, UPPER = 0x1p40;
    for (int mode = 0; mode < basis; ++mode) {
        double value = coefficients[mode * SHARED_STRIDE + lane];
        double magnitude = fabs(value);
        unsafe |= !isfinite(value) || (magnitude != 0 && (magnitude < LOWER || magnitude > UPPER));
        maximum_coefficient = fmax(maximum_coefficient, magnitude);
    }
    for (int batch = 0; batch < BATCH_TILE; ++batch) {
        double value = scaled[batch * SHARED_STRIDE + lane];
        double magnitude = fabs(value);
        unsafe |= !isfinite(value) || (magnitude != 0 && (magnitude < LOWER || magnitude > UPPER));
        maximum_scaled = fmax(maximum_scaled, magnitude);
    }
    unsigned bad = __ballot_sync(0xffffffffu, unsafe);
    for (int offset = 16; offset > 0; offset /= 2) {
        maximum_coefficient =
            fmax(maximum_coefficient, __shfl_xor_sync(0xffffffffu, maximum_coefficient, offset));
        maximum_scaled = fmax(maximum_scaled, __shfl_xor_sync(0xffffffffu, maximum_scaled, offset));
        maximum_accumulator =
            fmax(maximum_accumulator, __shfl_xor_sync(0xffffffffu, maximum_accumulator, offset));
    }
    double sum_bound = CHUNK * maximum_coefficient * maximum_scaled;
    double error_bound =
        budget.factor * sum_bound + 0x1p-120 + 64.0 * 0x1p-53 * maximum_accumulator;
    bool safe = bad == 0 && isfinite(error_bound) && error_bound <= budget.absolute;
    if (safe) {
        for (int mode = 0; mode < basis; ++mode) {
            int i = mode * SHARED_STRIDE + lane;
            coefficients[i] = pack_cdot_pair(ds_from_double(coefficients[i]));
        }
        for (int batch = 0; batch < BATCH_TILE; ++batch) {
            int i = batch * SHARED_STRIDE + lane;
            scaled[i] = pack_cdot_pair(ds_from_double(scaled[i]));
        }
    }
    return safe;
}

// FP64 survives across chunks. A failed certification uses exactly the
// existing FP64 dot over the original shared values; only safe chunks are DS.
template <bool DotCompensated>
__device__ inline void accumulate_cdot(double &accum, const double *coefficients,
                                       const double *scaled, int batch, int mode, bool safe) {
    if constexpr (DotCompensated) {
        if (safe) {
            DoubleSingle local = {0, 0};
            for (int atom = 0; atom < CHUNK; ++atom) {
                DoubleSingle a = unpack_cdot_pair(scaled[batch * SHARED_STRIDE + atom]);
                DoubleSingle b = unpack_cdot_pair(coefficients[mode * SHARED_STRIDE + atom]);
                local = ds_add(local, ds_multiply(a, b));
            }
            accum += ds_to_double(local);
            return;
        }
    }
    for (int atom = 0; atom < CHUNK; ++atom)
        accum += scaled[batch * SHARED_STRIDE + atom] * coefficients[mode * SHARED_STRIDE + atom];
}

// One harmonic of H. The caller keeps the stream and accumulators in
// registers; every harmonic advances the stream exactly once.
template <bool Compensated>
__device__ __forceinline__ void
accumulate_h_harmonic(HarmonicStream<false, Compensated> &stream, int harmonic, const double *z,
                      int batches, int batch_begin, const Axis &axis, const Bank &bank,
                      int first_tile, int retained, bool windowed, int window_node,
                      double (&accum)[BATCH_TILE]) {
    double cc, cs, unused_cc_beta, unused_cs_beta;
    stream.next(cc, cs, unused_cc_beta, unused_cs_beta);
    for (int offset = 0; offset < retained; ++offset) {
        double rs, rc;
        int node;
        if (windowed) {
            // Window moments already use the owner's unwrapped chart image.
            node = window_node;
            rs = 0;
            rc = 1;
        } else {
            int raw_tile = first_tile + offset;
            int tile = positive_mod(raw_tile, bank.tile_count);
            int image = (raw_tile - tile) / bank.tile_count;
            image_rotation(bank, harmonic, image, rs, rc);
            node = bank.node_offset + tile;
        }
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

template <bool Compensated>
__device__ __forceinline__ void store_c_harmonic(HarmonicStream<false, Compensated> &stream,
                                                 int harmonic, int lane, double *coefficients) {
    double cc, cs, unused_cc_beta, unused_cs_beta;
    stream.next(cc, cs, unused_cc_beta, unused_cs_beta);
    if (harmonic == 0)
        coefficients[lane] = cc;
    else {
        coefficients[(2 * harmonic - 1) * SHARED_STRIDE + lane] = cc;
        coefficients[2 * harmonic * SHARED_STRIDE + lane] = cs;
    }
}

template <int FixedModes, bool Compensated = false>
__device__ __forceinline__ void store_c_coefficients(const Profile &profile, int modes, int lane,
                                                     double *coefficients) {
    HarmonicStream<false, Compensated> stream(profile, FixedModes ? FixedModes : modes);
    if (FixedModes == 8) {
#pragma unroll
        for (int harmonic = 0; harmonic <= 8; ++harmonic)
            store_c_harmonic<Compensated>(stream, harmonic, lane, coefficients);
    } else {
#pragma unroll 1
        for (int harmonic = 0; harmonic <= modes; ++harmonic)
            store_c_harmonic<Compensated>(stream, harmonic, lane, coefficients);
    }
}

// One thread completes H for a different atom and up to four batches. The
// amplitude is not included. Neighbour Z tiles are rotated to the same
// unwrapped chart image as the atom's retained one-period window.
template <typename scalar_t, typename projection_t, int FixedModes, bool Compensated = false>
__global__ void compute_h_kernel(const scalar_t *atoms, const int64_t *rows, const int *groups,
                                 const int64_t *offsets, const double *z, projection_t *h,
                                 int atom_count, int group_count, int batches, Axis axis,
                                 bool output_side, bool windowed) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= atom_count)
        return;
    int group = group_for_slot(offsets, group_count, slot);
    int owner = groups[4 * group + (output_side ? 2 : 3)];
    Bank bank = axis.banks[bank_for_level(axis, groups[4 * group + (output_side ? 0 : 1)])];
    int original = rows[slot];
    Profile profile = make_profile<false, FixedModes, Compensated>(
        axis, bank, atom_sigma(atoms, original, axis),
        atom_center(atoms, original, axis, output_side), owner);
    HarmonicStream<false, Compensated> stream(profile, FixedModes ? FixedModes : axis.modes);
    double accum[BATCH_TILE] = {};
    int batch_begin = blockIdx.y * BATCH_TILE;
    // Window Z has already combined and rotated the retained tiles into the
    // owner's chart. Read that owner once; its computed image is zero, so the
    // harmonic helper does not rotate the window a second time.
    int retained = (windowed || bank.full_chart) ? 1 : 3;
    int first_tile = windowed ? owner : (bank.full_chart ? 0 : owner - 1);
    int window_node = bank.node_offset + owner;
    if (FixedModes == 8) {
#pragma unroll
        for (int harmonic = 0; harmonic <= 8; ++harmonic)
            accumulate_h_harmonic<Compensated>(stream, harmonic, z, batches, batch_begin, axis,
                                               bank, first_tile, retained, windowed, window_node,
                                               accum);
    } else {
#pragma unroll 1
        for (int harmonic = 0; harmonic <= axis.modes; ++harmonic)
            accumulate_h_harmonic<Compensated>(stream, harmonic, z, batches, batch_begin, axis,
                                               bank, first_tile, retained, windowed, window_node,
                                               accum);
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
// different atom in a 32-atom chunk. Shared holds only this chunk's
// coefficients and amp*H. C has one writer per element, without atom-sized
// partial C arrays.
template <typename scalar_t, typename projection_t, int FixedModes, bool Compensated = false,
          bool DotCompensated = false>
__global__ void compute_c_kernel(const scalar_t *atoms, const int64_t *rows, const int *groups,
                                 const int64_t *offsets, const projection_t *h, double *c,
                                 int atom_count, int group_count, int batches, Axis input,
                                 Axis output, bool reverse, CdotBudget budget,
                                 const int *preflight) {
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_h = shared + SHARED_STRIDE * output.basis_count();
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
                Profile profile = make_profile<false, FixedModes, Compensated>(
                    output, bank, atom_sigma(atoms, original, output),
                    atom_center(atoms, original, output, !reverse), owner);
                store_c_coefficients<FixedModes, Compensated>(profile, output.modes, lane,
                                                              coefficients);
            } else {
                for (int mode = 0; mode < output.basis_count(); ++mode)
                    coefficients[mode * SHARED_STRIDE + lane] = 0;
            }
            for (int local = 0; local < BATCH_TILE; ++local)
                scaled_h[local * SHARED_STRIDE + lane] =
                    valid && batch_begin + local < batches
                        ? atom_amplitude(atoms, original, output) *
                              h[(batch_begin + local) * atom_count + slot]
                        : 0;
            bool safe = false;
            if constexpr (DotCompensated) {
                if (*preflight == 0) {
                    double maximum = 0;
                    bool invalid = false;
                    for (int a = 0; a < MAX_ACCUMULATORS; ++a) {
                        invalid |= !isfinite(accum[a]);
                        maximum = fmax(maximum, fabs(accum[a]));
                    }
                    safe = prepare_cdot_chunk(coefficients, scaled_h, output.basis_count(), budget,
                                              maximum, invalid);
                }
            }
            __syncthreads();
            for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
                int batch = work / output.basis_count(), mode = work % output.basis_count();
                if constexpr (DotCompensated) {
                    accumulate_cdot<true>(accum[a], coefficients, scaled_h, batch, mode, safe);
                } else {
                    for (int atom = 0; atom < CHUNK; ++atom)
                        accum[a] += scaled_h[batch * SHARED_STRIDE + atom] *
                                    coefficients[mode * SHARED_STRIDE + atom];
                }
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
template <typename scalar_t, typename projection_t, int FixedModes, bool Compensated = false,
          bool DotCompensated = false>
__device__ void
split_consume_range(const scalar_t *atoms, const int64_t *rows, const projection_t *h,
                    int atom_count, int batches, Axis output, Bank bank, int owner, bool reverse,
                    int batch_begin, int begin, int end, int split_begin, int split_end,
                    int &prefix, double *coefficients, double *scaled_h,
                    double (&accum)[MAX_ACCUMULATORS], CdotBudget budget, const int *preflight) {
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
            Profile profile = make_profile<false, FixedModes, Compensated>(
                output, bank, atom_sigma(atoms, original, output),
                atom_center(atoms, original, output, !reverse), owner);
            store_c_coefficients<FixedModes, Compensated>(profile, output.modes, lane,
                                                          coefficients);
        } else {
            for (int mode = 0; mode < output.basis_count(); ++mode)
                coefficients[mode * SHARED_STRIDE + lane] = 0;
        }
        for (int local = 0; local < BATCH_TILE; ++local)
            scaled_h[local * SHARED_STRIDE + lane] =
                valid && batch_begin + local < batches
                    ? atom_amplitude(atoms, original, output) *
                          h[(batch_begin + local) * atom_count + slot]
                    : 0;
        bool safe = false;
        if constexpr (DotCompensated) {
            if (*preflight == 0) {
                double maximum = 0;
                bool invalid = false;
                for (int a = 0; a < MAX_ACCUMULATORS; ++a) {
                    invalid |= !isfinite(accum[a]);
                    maximum = fmax(maximum, fabs(accum[a]));
                }
                safe = prepare_cdot_chunk(coefficients, scaled_h, output.basis_count(), budget,
                                          maximum, invalid);
            }
        }
        __syncthreads();
        for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
            int batch = work / output.basis_count(), mode = work % output.basis_count();
            if constexpr (DotCompensated) {
                accumulate_cdot<true>(accum[a], coefficients, scaled_h, batch, mode, safe);
            } else {
                for (int atom = 0; atom < CHUNK; ++atom)
                    accum[a] += scaled_h[batch * SHARED_STRIDE + atom] *
                                coefficients[mode * SHARED_STRIDE + atom];
            }
        }
        __syncthreads();
    }
}

// One warp per (owner, batch tile, split). The temporary is bounded by the
// number of owners and splits, never by atoms or groups. Two descriptor passes
// count the owner's atoms then partition their concatenation evenly: even a
// collection of short groups is distributed across splits. Reverse lookup uses
// one lane per descriptor and a warp ballot, avoiding barriers for nonmatches.
// Passed by value as CUDA kernel arguments, so no device pointer-table
// allocation. The tensors that own these addresses stay alive until both
// kernels are queued.
struct SplitPages {
    double *pointers[64];
    int page_splits;
};

// Classification assigns (previous tile width, current tile width] to each
// local bank; the full-chart bank has the same lower bound and ends at the
// dtype-rounded common sigma limit used by atom validation.
__device__ inline bool compatible_sigma_banks(const Axis &a, int ai, const Axis &b, int bi,
                                              double maximum_sigma) {
    // Disjoint sigma intervals imply an impossible pair only for a shared
    // width law. Independent Polar sides may legitimately occupy that pair.
    if (a.polar && (a.sigma_min != b.sigma_min || a.sigma_birth != b.sigma_birth ||
                    a.sigma_max != b.sigma_max || a.upper_floor != b.upper_floor))
        return true;
    double al = ai == 0 ? 0 : a.banks[ai - 1].tile_sites * a.spacing;
    double bl = bi == 0 ? 0 : b.banks[bi - 1].tile_sites * b.spacing;
    double au = a.banks[ai].full_chart ? maximum_sigma : a.banks[ai].tile_sites * a.spacing;
    double bu = b.banks[bi].full_chart ? maximum_sigma : b.banks[bi].tile_sites * b.spacing;
    return fmin(au, bu) > fmax(al, bl);
}

template <typename scalar_t, typename projection_t, int FixedModes, bool Compensated = false,
          bool DotCompensated = false>
__global__ void
compute_c_split_kernel(const scalar_t *atoms, const int64_t *rows, const int *groups,
                       const int64_t *offsets, const projection_t *h, SplitPages pages,
                       int atom_count, int group_count, int batches, Axis input, Axis output,
                       bool reverse, bool cartesian, int splits, int atoms_per_split,
                       CdotBudget budget, const int *preflight) {
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_h = shared + SHARED_STRIDE * output.basis_count();
    int node = blockIdx.x, bank_index = 0;
    while (bank_index + 1 < output.bank_count && node >= output.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = output.banks[bank_index];
    int owner = node - bank.node_offset;
    int lane = threadIdx.x;
    int batch_begin = blockIdx.y * BATCH_TILE;
    // Match classify_atoms: float32's representable upper endpoint may be
    // slightly larger than the exact double period/3.
    double maximum_sigma =
        static_cast<double>(static_cast<scalar_t>(fmin(input.period, output.period) / 3));
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
                if (!compatible_sigma_banks(output, bank_index, input, range, maximum_sigma))
                    continue;
                int first = other.node_offset * output.node_count +
                            other.tile_count * bank.node_offset + owner;
                for (int other_owner = 0; other_owner < other.tile_count; ++other_owner) {
                    int group = first + other_owner * bank.tile_count;
                    int begin = offsets[group], end = offsets[group + 1];
                    if (pass == 0)
                        prefix += end - begin;
                    else
                        split_consume_range<scalar_t, projection_t, FixedModes, Compensated,
                                            DotCompensated>(
                            atoms, rows, h, atom_count, batches, output, bank, owner, reverse,
                            batch_begin, begin, end, split_begin, split_end, prefix, coefficients,
                            scaled_h, accum, budget, preflight);
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
                        split_consume_range<scalar_t, projection_t, FixedModes, Compensated,
                                            DotCompensated>(
                            atoms, rows, h, atom_count, batches, output, bank, owner, reverse,
                            batch_begin, begin, end, split_begin, split_end, prefix, coefficients,
                            scaled_h, accum, budget, preflight);
                }
            }
        } else {
            for (int range = 0; range < input.bank_count; ++range) {
                Bank other = input.banks[range];
                if (cartesian &&
                    !compatible_sigma_banks(output, bank_index, input, range, maximum_sigma))
                    continue;
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
                    split_consume_range<scalar_t, projection_t, FixedModes, Compensated,
                                        DotCompensated>(
                        atoms, rows, h, atom_count, batches, output, bank, owner, reverse,
                        batch_begin, begin, end, split_begin, split_end, prefix, coefficients,
                        scaled_h, accum, budget, preflight);
            }
        }
        if (pass == 0) {
            total = prefix;
            int active = atoms_per_split
                             ? min(splits, max(1, (total + atoms_per_split - 1) / atoms_per_split))
                             : splits;
            // Every lane computed the same owner count. Inactive blocks can
            // return together after writing all their partials as zero.
            if (blockIdx.z >= active) {
                int elements = BATCH_TILE * output.basis_count();
                int local_split = blockIdx.z % pages.page_splits;
                int page = blockIdx.z / pages.page_splits;
                for (int work = lane; work < elements; work += CHUNK) {
                    int batch = batch_begin + work / output.basis_count();
                    int mode = work % output.basis_count();
                    if (batch < batches) {
                        int64_t index = ((static_cast<int64_t>(local_split) * batches + batch) *
                                             output.node_count +
                                         node) *
                                            output.basis_count() +
                                        mode;
                        pages.pointers[page][index] = 0;
                    }
                }
                return;
            }
            split_begin = static_cast<int>(static_cast<int64_t>(total) * blockIdx.z / active);
            split_end = static_cast<int>(static_cast<int64_t>(total) * (blockIdx.z + 1) / active);
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

// Forward-only experiment: a thread owns one atom, computes H in registers,
// saves its rounded value, and supplies that same value to the C chunk.
// Shared-shape atoms use one magnitude stream and independent center phases.
template <typename atom_t, bool Compensated, int ForwardBatchTile = BATCH_TILE,
          bool ReuseSigma = false, bool KnownInputBank = false>
__device__ void fused_forward_atom(const atom_t *atoms, const int64_t *rows, const int *groups,
                                   const int64_t *offsets, const double *zx, atom_t *h,
                                   int atom_count, int group_count, int batches, int batch_begin,
                                   const Axis &input, const Axis &output, const Bank &output_bank,
                                   int output_owner, int slot, int lane, bool windowed,
                                   double *coefficients, double *scaled_h, const Bank *known_input_bank = nullptr) {
    int original = rows[slot];
    Bank input_bank;
    int input_owner;
    if constexpr (KnownInputBank) {
        // Every atom in this contiguous range has the caller's input bank.
        input_bank = *known_input_bank;
        input_owner = input_bank.full_chart ? 0 : owner_from_center(
            atom_center(atoms, original, input, false), input, input_bank.tile_sites);
    } else {
        int group = group_for_slot(offsets, group_count, slot);
        input_owner = groups[4 * group + 3];
        input_bank = input.banks[bank_for_level(input, groups[4 * group + 1])];
    }
    double sigma = atom_sigma(atoms, original, input);
    double output_sigma;
    if constexpr (ReuseSigma)
        output_sigma = same_sigma_decode(input, output) ? sigma : atom_sigma(atoms, original, output);
    else
        output_sigma = atom_sigma(atoms, original, output);
    Profile input_profile = make_profile<false, 8, Compensated>(
        input, input_bank, sigma, atom_center(atoms, original, input, false), input_owner);
    bool same_shape = sigma == output_sigma && input.modes == output.modes && input.radius_ratio4 == output.radius_ratio4 &&
                      input_bank.period == output_bank.period &&
                      input_bank.inverse_period == output_bank.inverse_period &&
                      input_bank.omega == output_bank.omega;
    double step_sine = 0, step_cosine = 1, sine = 0, cosine = 1;
    if (same_shape) {
        double output_center = profile_center(
            output, output_bank, atom_center(atoms, original, output, true), output_owner);
        // The magnitude is shared, but the output center has its own phase.
        sincos(output_bank.omega * output_center, &step_sine, &step_cosine);
    }
    HarmonicStream<false, Compensated> stream(input_profile, 8);
    double sums[ForwardBatchTile] = {};
    int retained = windowed || input_bank.full_chart ? 1 : 3;
    int first_tile = windowed ? input_owner : (input_bank.full_chart ? 0 : input_owner - 1);
    int window_node = input_bank.node_offset + input_owner;
#pragma unroll
    for (int harmonic = 0; harmonic <= 8; ++harmonic) {
        double cc, cs, unused_cc_beta, unused_cs_beta, magnitude;
        stream.next(cc, cs, unused_cc_beta, unused_cs_beta, &magnitude);
        if (same_shape) {
            if (harmonic == 0) {
                coefficients[lane] = cc;
            } else {
                double next_cosine = cosine * step_cosine - sine * step_sine;
                sine = sine * step_cosine + cosine * step_sine;
                cosine = next_cosine;
                coefficients[(2 * harmonic - 1) * SHARED_STRIDE + lane] = magnitude * cosine;
                coefficients[2 * harmonic * SHARED_STRIDE + lane] = magnitude * sine;
            }
        }
        for (int offset = 0; offset < retained; ++offset) {
            double rs, rc;
            int node;
            if (windowed) {
                node = window_node;
                rs = 0;
                rc = 1;
            } else {
                int raw_tile = first_tile + offset;
                int tile = positive_mod(raw_tile, input_bank.tile_count);
                int image = (raw_tile - tile) / input_bank.tile_count;
                image_rotation(input_bank, harmonic, image, rs, rc);
                node = input_bank.node_offset + tile;
            }
            for (int b = 0; b < ForwardBatchTile && batch_begin + b < batches; ++b) {
                int base = ((batch_begin + b) * input.node_count + node) * input.basis_count();
                if (harmonic == 0) {
                    sums[b] += cc * zx[base];
                } else {
                    double zc = zx[base + 2 * harmonic - 1], zs = zx[base + 2 * harmonic];
                    sums[b] += cc * (zc * rc - zs * rs) + cs * (zs * rc + zc * rs);
                }
            }
        }
    }
    if (!same_shape) {
        Profile profile = make_profile<false, 8, Compensated>(
            output, output_bank, output_sigma, atom_center(atoms, original, output, true), output_owner);
        store_c_coefficients<8, Compensated>(profile, 8, lane, coefficients);
    }
    double amplitude = atom_amplitude(atoms, original, input);
    for (int b = 0; b < ForwardBatchTile; ++b) {
        if (batch_begin + b < batches) {
            atom_t rounded = static_cast<atom_t>(sums[b]);
            h[(batch_begin + b) * atom_count + slot] = rounded;
            scaled_h[b * SHARED_STRIDE + lane] = amplitude * static_cast<double>(rounded);
        } else {
            scaled_h[b * SHARED_STRIDE + lane] = 0;
        }
    }
}

// Every atom in the owner's concatenated bank ranges belongs to exactly one
// split. Padding lanes also reach all barriers, with zero shared contributions.
template <typename atom_t, bool Compensated, int ForwardBatchTile = BATCH_TILE,
          bool ReuseSigma = false, bool KnownInputBank = false>
__device__ void fused_forward_range(const atom_t *atoms, const int64_t *rows, const int *groups,
                                    const int64_t *offsets, const double *zx, atom_t *h,
                                    int atom_count, int group_count, int batches, int batch_begin,
                                    const Axis &input, const Axis &output, const Bank &bank,
                                    int owner, int begin, int end, int split_begin, int split_end,
                                    int &prefix, bool windowed, double *coefficients,
                                    double *scaled_h, double *totals, const Bank *known_input_bank = nullptr) {
    int length = end - begin;
    int first = max(0, split_begin - prefix), last = min(length, split_end - prefix);
    prefix += length;
    int lane = threadIdx.x, elements = ForwardBatchTile * output.basis_count();
    for (int chunk = begin + first; chunk < begin + last; chunk += CHUNK) {
        int slot = chunk + lane;
        if (slot < begin + last) {
            fused_forward_atom<atom_t, Compensated, ForwardBatchTile, ReuseSigma, KnownInputBank>(
                atoms, rows, groups, offsets, zx, h, atom_count, group_count, batches, batch_begin,
                input, output, bank, owner, slot, lane, windowed, coefficients, scaled_h,
                known_input_bank);
        } else {
            for (int mode = 0; mode < output.basis_count(); ++mode)
                coefficients[mode * SHARED_STRIDE + lane] = 0;
            for (int b = 0; b < ForwardBatchTile; ++b)
                scaled_h[b * SHARED_STRIDE + lane] = 0;
        }
        __syncthreads();
        for (int work = lane; work < elements; work += CHUNK) {
            int batch = work / output.basis_count(), mode = work % output.basis_count();
            double value = totals[work];
            for (int atom = 0; atom < CHUNK; ++atom)
                value += scaled_h[batch * SHARED_STRIDE + atom] *
                         coefficients[mode * SHARED_STRIDE + atom];
            totals[work] = value;
        }
        __syncthreads();
    }
}

template <typename atom_t, bool Compensated>
__global__ void fused_forward_kernel(const atom_t *atoms, const int64_t *rows, const int *groups,
                                     const int64_t *offsets, const double *zx, atom_t *h,
                                     SplitPages pages, int atom_count, int group_count, int batches,
                                     Axis input, Axis output, bool cartesian, int splits,
                                     int atoms_per_split, bool windowed) {
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_h = shared + SHARED_STRIDE * output.basis_count();
    double *totals = scaled_h + SHARED_STRIDE * BATCH_TILE;
    int node = blockIdx.x, bank_index = 0;
    while (bank_index + 1 < output.bank_count && node >= output.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = output.banks[bank_index];
    int owner = node - bank.node_offset, lane = threadIdx.x;
    int batch_begin = blockIdx.y * BATCH_TILE, elements = BATCH_TILE * output.basis_count();
    double maximum_sigma =
        static_cast<double>(static_cast<atom_t>(fmin(input.period, output.period) / 3));
    int split_begin = 0, split_end = 0;
    for (int pass = 0; pass < 2; ++pass) {
        int prefix = 0;
        for (int range = 0; range < input.bank_count; ++range) {
            Bank other = input.banks[range];
            if (cartesian &&
                !compatible_sigma_banks(output, bank_index, input, range, maximum_sigma))
                continue;
            int first, last;
            if (cartesian) {
                first = bank.node_offset * input.node_count + bank.tile_count * other.node_offset +
                        owner * other.tile_count;
                last = first + other.tile_count;
            } else {
                first = lower_group(groups, group_count, bank.level, other.level, owner);
                last = lower_group(groups, group_count, bank.level, other.level, owner + 1);
            }
            int begin = offsets[first], end = offsets[last];
            if (pass == 0)
                prefix += end - begin;
            else
                fused_forward_range<atom_t, Compensated>(
                    atoms, rows, groups, offsets, zx, h, atom_count, group_count, batches,
                    batch_begin, input, output, bank, owner, begin, end, split_begin, split_end,
                    prefix, windowed, coefficients, scaled_h, totals);
        }
        if (pass == 0) {
            int active = atoms_per_split
                             ? min(splits, max(1, (prefix + atoms_per_split - 1) / atoms_per_split))
                             : splits;
            int local_split = blockIdx.z % pages.page_splits, page = blockIdx.z / pages.page_splits;
            if (blockIdx.z >= active) {
                // This branch is uniform, before any block barrier.
                for (int work = lane; work < elements; work += CHUNK) {
                    int batch = batch_begin + work / output.basis_count();
                    int mode = work % output.basis_count();
                    if (batch < batches) {
                        int64_t i = ((static_cast<int64_t>(local_split) * batches + batch) *
                                         output.node_count +
                                     node) *
                                        output.basis_count() +
                                    mode;
                        pages.pointers[page][i] = 0;
                    }
                }
                return;
            }
            split_begin = static_cast<int>(static_cast<int64_t>(prefix) * blockIdx.z / active);
            split_end = static_cast<int>(static_cast<int64_t>(prefix) * (blockIdx.z + 1) / active);
            for (int work = lane; work < elements; work += CHUNK)
                totals[work] = 0;
            __syncthreads();
        }
    }
    int local_split = blockIdx.z % pages.page_splits, page = blockIdx.z / pages.page_splits;
    for (int work = lane; work < elements; work += CHUNK) {
        int batch = batch_begin + work / output.basis_count(), mode = work % output.basis_count();
        if (batch < batches) {
            int64_t i =
                ((static_cast<int64_t>(local_split) * batches + batch) * output.node_count + node) *
                    output.basis_count() +
                mode;
            pages.pointers[page][i] = totals[work];
        }
    }
}

// Shared per-atom calculation for legacy grouped and reverse-slot producers.
template <typename atom_t, typename projection_t, int FixedModes>
__device__ inline void
fused_grad_cin_atom(const atom_t *atoms, const int64_t *rows, const int *groups,
                    const projection_t *h, const double *zx, const double *zy, projection_t *g,
                    atom_t *gradient, int atom_count, int batches, const Axis &input,
                    const Axis &output, const Bank &input_bank, int input_owner, int group,
                    int slot, int lane, bool windowed, double *coefficients, double *scaled_g) {
    int original = rows[slot];
    int output_owner = groups[4 * group + 2];
    Bank output_bank = output.banks[bank_for_level(output, groups[4 * group])];
    double amplitude = atom_amplitude(atoms, original, input);
    double sigma = atom_sigma(atoms, original, input);
    double output_sigma = atom_sigma(atoms, original, output);
    double output_center = atom_center(atoms, original, output, true);
    double input_center = atom_center(atoms, original, input, false);
    Profile out_profile =
        make_profile<true, FixedModes>(output, output_bank, output_sigma, output_center, output_owner);
    bool same_shape = sigma == output_sigma && input.modes == output.modes && input.radius_ratio4 == output.radius_ratio4 &&
                      input_bank.period == output_bank.period &&
                      input_bank.inverse_period == output_bank.inverse_period &&
                      input_bank.omega == output_bank.omega;
    Profile in_profile;
    if (same_shape) {
        in_profile = out_profile;
        in_profile.center = profile_center(input, input_bank, input_center, input_owner);
    } else {
        in_profile =
            make_profile<true, FixedModes>(input, input_bank, sigma, input_center, input_owner);
    }
    double da = 0, ds = 0, dmu_input = 0, dmu_output = 0;
    double gv[BATCH_TILE] = {};
    project_weighted_derivatives<true, FixedModes>(
        zy, batches, 0, output, output_bank, output_owner, out_profile, out_profile.beta_sigma, h,
        atom_count, slot, amplitude, gv, dmu_output, ds, windowed);
    for (int b = 0; b < BATCH_TILE; ++b) {
        if (b < batches) {
            projection_t rounded = static_cast<projection_t>(gv[b]);
            g[b * atom_count + slot] = rounded;
            double g_value = static_cast<double>(rounded);
            da += static_cast<double>(h[b * atom_count + slot]) * g_value;
            gv[b] = amplitude * g_value;
            scaled_g[b * SHARED_STRIDE + lane] = gv[b];
        } else {
            gv[b] = 0;
            scaled_g[b * SHARED_STRIDE + lane] = 0;
        }
    }
    project_weighted_derivatives<false, FixedModes, true>(
        zx, batches, 0, input, input_bank, input_owner, in_profile, in_profile.beta_sigma, h,
        atom_count, slot, amplitude, gv, dmu_input, ds, windowed, coefficients, SHARED_STRIDE,
        lane);
    store_atom_gradient(atoms, gradient, original, input, da, ds, dmu_input, dmu_output);
}

// One reverse-layout group belongs to this input owner. Each valid lane owns
// a different atom and writes its G and all four gradients exactly once.
// Every lane, including padding lanes, participates in the shared barriers.
template <typename atom_t, typename projection_t, int FixedModes>
__device__ void fused_grad_cin_range(const atom_t *atoms, const int64_t *rows, const int *groups,
                                     const projection_t *h, const double *zx, const double *zy,
                                     projection_t *g, atom_t *gradient, int atom_count, int batches,
                                     const Axis &input, const Axis &output, const Bank &input_bank,
                                     int input_owner, int group, int begin, int end,
                                     int split_begin, int split_end, int &prefix, bool windowed,
                                     double *coefficients, double *scaled_g, double *totals) {
    int length = end - begin;
    int first = max(0, split_begin - prefix);
    int last = min(length, split_end - prefix);
    prefix += length;
    int lane = threadIdx.x;
    int elements = BATCH_TILE * input.basis_count();
    for (int chunk = begin + first; chunk < begin + last; chunk += CHUNK) {
        int slot = chunk + lane;
        bool valid = slot < begin + last;
        if (valid) {
            fused_grad_cin_atom<atom_t, projection_t, FixedModes>(
                atoms, rows, groups, h, zx, zy, g, gradient, atom_count, batches, input, output,
                input_bank, input_owner, group, slot, lane, windowed, coefficients, scaled_g);
        } else {
            for (int mode = 0; mode < input.basis_count(); ++mode)
                coefficients[mode * SHARED_STRIDE + lane] = 0;
            for (int b = 0; b < BATCH_TILE; ++b)
                scaled_g[b * SHARED_STRIDE + lane] = 0;
        }
        __syncthreads();
        for (int work = lane; work < elements; work += CHUNK) {
            int batch = work / input.basis_count(), mode = work % input.basis_count();
            double total = totals[work];
            for (int atom = 0; atom < CHUNK; ++atom)
                total += scaled_g[batch * SHARED_STRIDE + atom] *
                         coefficients[mode * SHARED_STRIDE + atom];
            totals[work] = total;
        }
        __syncthreads();
    }
}

template <typename atom_t, typename projection_t, int FixedModes>
__global__ void fused_grad_cin_kernel(const atom_t *atoms, const int64_t *rows, const int *groups,
                                      const int64_t *offsets, const double *zx, const double *zy,
                                      const projection_t *h, projection_t *g, atom_t *gradient,
                                      SplitPages pages, int atom_count, int group_count,
                                      int batches, Axis input, Axis output, bool cartesian,
                                      int splits, bool windowed) {
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_g = shared + SHARED_STRIDE * input.basis_count();
    double *totals = scaled_g + SHARED_STRIDE * BATCH_TILE;
    int lane = threadIdx.x;
    int elements = BATCH_TILE * input.basis_count();
    for (int work = lane; work < elements; work += CHUNK)
        totals[work] = 0;
    __syncthreads();
    int node = blockIdx.x, bank_index = 0;
    while (bank_index + 1 < input.bank_count && node >= input.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = input.banks[bank_index];
    int owner = node - bank.node_offset;
    double maximum_sigma =
        static_cast<double>(static_cast<atom_t>(fmin(input.period, output.period) / 3));
    int split_begin = 0, split_end = 0;
    for (int pass = 0; pass < 2; ++pass) {
        int prefix = 0;
        if (cartesian) {
            for (int range = 0; range < output.bank_count; ++range) {
                Bank other = output.banks[range];
                if (!compatible_sigma_banks(input, bank_index, output, range, maximum_sigma))
                    continue;
                int first = other.node_offset * input.node_count +
                            other.tile_count * bank.node_offset + owner;
                for (int other_owner = 0; other_owner < other.tile_count; ++other_owner) {
                    int group = first + other_owner * bank.tile_count;
                    int begin = offsets[group], end = offsets[group + 1];
                    if (pass == 0)
                        prefix += end - begin;
                    else
                        fused_grad_cin_range<atom_t, projection_t, FixedModes>(
                            atoms, rows, groups, h, zx, zy, g, gradient, atom_count, batches, input,
                            output, bank, owner, group, begin, end, split_begin, split_end, prefix,
                            windowed, coefficients, scaled_g, totals);
                }
            }
        } else {
            for (int base = 0; base < group_count; base += CHUNK) {
                int group = base + lane;
                bool matches = group < group_count && groups[4 * group + 1] == bank.level &&
                               groups[4 * group + 3] == owner;
                unsigned mask = __ballot_sync(0xffffffffu, matches);
                while (mask) {
                    int selected = base + __ffs(mask) - 1;
                    mask &= mask - 1;
                    int begin = offsets[selected], end = offsets[selected + 1];
                    if (pass == 0)
                        prefix += end - begin;
                    else
                        fused_grad_cin_range<atom_t, projection_t, FixedModes>(
                            atoms, rows, groups, h, zx, zy, g, gradient, atom_count, batches, input,
                            output, bank, owner, selected, begin, end, split_begin, split_end,
                            prefix, windowed, coefficients, scaled_g, totals);
                }
            }
        }
        if (pass == 0) {
            split_begin = static_cast<int>(static_cast<int64_t>(prefix) * blockIdx.z / splits);
            split_end = static_cast<int>(static_cast<int64_t>(prefix) * (blockIdx.z + 1) / splits);
        }
    }
    int local_split = blockIdx.z % pages.page_splits;
    int page = blockIdx.z / pages.page_splits;
    for (int work = lane; work < elements; work += CHUNK) {
        int batch = work / input.basis_count(), mode = work % input.basis_count();
        if (batch < batches) {
            int64_t index =
                ((static_cast<int64_t>(local_split) * batches + batch) * input.node_count + node) *
                    input.basis_count() +
                mode;
            pages.pointers[page][index] = totals[work];
        }
    }
}

// Reverse slots gather original-layout slots without copying atom values.
// A split owns a contiguous input-owner interval, even across group boundaries.
template <bool Fused, typename atom_t, typename projection_t, int FixedModes,
          bool Compensated = false, bool DotCompensated = false>
__global__ void
reverse_cin_kernel(const atom_t *atoms, const int64_t *rows, const int *groups,
                   const int64_t *offsets, const int *reverse_slots, const int64_t *reverse_offsets,
                   const double *zx, const double *zy, const projection_t *h, projection_t *g,
                   atom_t *gradient, SplitPages pages, int atom_count, int group_count, int batches,
                   Axis input, Axis output, int splits, int atoms_per_split, bool windowed,
                   CdotBudget budget, const int *preflight) {
    static_assert(!Fused || !DotCompensated, "fused gradients keep FP64 C dots");
    int node = blockIdx.x, lane = threadIdx.x;
    int bank_index = 0;
    while (bank_index + 1 < input.bank_count && node >= input.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = input.banks[bank_index];
    int owner = node - bank.node_offset;
    int begin = reverse_offsets[node], end = reverse_offsets[node + 1];
    int count = end - begin;
    int active = atoms_per_split
                     ? min(splits, max(1, (count + atoms_per_split - 1) / atoms_per_split))
                     : splits;
    int batch_begin = blockIdx.y * BATCH_TILE;
    int elements = BATCH_TILE * input.basis_count();
    int local_split = blockIdx.z % pages.page_splits, page = blockIdx.z / pages.page_splits;
    if (blockIdx.z >= active) {
        // Uniform for the whole block: all unused partials must be initialized.
        for (int work = lane; work < elements; work += CHUNK) {
            int batch = batch_begin + work / input.basis_count(), mode = work % input.basis_count();
            if (batch < batches) {
                int64_t index =
                    ((static_cast<int64_t>(local_split) * batches + batch) * input.node_count +
                     node) *
                        input.basis_count() +
                    mode;
                pages.pointers[page][index] = 0;
            }
        }
        return;
    }
    int first = begin + static_cast<int>(static_cast<int64_t>(count) * blockIdx.z / active);
    int last = begin + static_cast<int>(static_cast<int64_t>(count) * (blockIdx.z + 1) / active);
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_g = shared + SHARED_STRIDE * input.basis_count();
    double *totals = scaled_g + SHARED_STRIDE * BATCH_TILE;
    for (int work = lane; work < elements; work += CHUNK)
        totals[work] = 0;
    __syncthreads();
    for (int chunk = first; chunk < last; chunk += CHUNK) {
        int position = chunk + lane;
        if (position < last) {
            int slot = reverse_slots[position];
            if (Fused) {
                int group = group_for_slot(offsets, group_count, slot);
                fused_grad_cin_atom<atom_t, projection_t, FixedModes>(
                    atoms, rows, groups, h, zx, zy, g, gradient, atom_count, batches, input, output,
                    bank, owner, group, slot, lane, windowed, coefficients, scaled_g);
            } else {
                int original = rows[slot];
                Profile profile = make_profile<false, FixedModes, Compensated>(
                    input, bank, atom_sigma(atoms, original, input),
                    atom_center(atoms, original, input, false), owner);
                store_c_coefficients<FixedModes, Compensated>(profile, input.modes, lane,
                                                              coefficients);
                for (int b = 0; b < BATCH_TILE; ++b)
                    scaled_g[b * SHARED_STRIDE + lane] =
                        batch_begin + b < batches ? atom_amplitude(atoms, original, output) *
                                                        h[(batch_begin + b) * atom_count + slot]
                                                  : 0;
            }
        } else {
            for (int mode = 0; mode < input.basis_count(); ++mode)
                coefficients[mode * SHARED_STRIDE + lane] = 0;
            for (int b = 0; b < BATCH_TILE; ++b)
                scaled_g[b * SHARED_STRIDE + lane] = 0;
        }
        bool safe = false;
        if constexpr (DotCompensated) {
            if (*preflight == 0) {
                double maximum = 0;
                bool invalid = false;
                for (int work = lane; work < elements; work += CHUNK) {
                    invalid |= !isfinite(totals[work]);
                    maximum = fmax(maximum, fabs(totals[work]));
                }
                safe = prepare_cdot_chunk(coefficients, scaled_g, input.basis_count(), budget,
                                          maximum, invalid);
            }
        }
        __syncthreads();
        for (int work = lane; work < elements; work += CHUNK) {
            int batch = work / input.basis_count(), mode = work % input.basis_count();
            double total = totals[work];
            if constexpr (DotCompensated) {
                accumulate_cdot<true>(total, coefficients, scaled_g, batch, mode, safe);
            } else {
                for (int atom = 0; atom < CHUNK; ++atom)
                    total += scaled_g[batch * SHARED_STRIDE + atom] *
                             coefficients[mode * SHARED_STRIDE + atom];
            }
            totals[work] = total;
        }
        __syncthreads();
    }
    for (int work = lane; work < elements; work += CHUNK) {
        int batch = batch_begin + work / input.basis_count(), mode = work % input.basis_count();
        if (batch < batches) {
            int64_t index =
                ((static_cast<int64_t>(local_split) * batches + batch) * input.node_count + node) *
                    input.basis_count() +
                mode;
            pages.pointers[page][index] = totals[work];
        }
    }
}

// Build owner counts and an exclusive prefix of active splits in one CTA.
// Empty owners receive zero tasks; all metadata stays on the device.
template <bool Reverse>
__global__ void ragged_metadata_kernel(const int *groups, const int64_t *offsets,
                                       const int64_t *reverse_offsets, int *counts,
                                       int *task_offsets, Axis input, Axis output, bool cartesian,
                                       int group_count, int splits, int grain) {
    constexpr int THREADS_SCAN = 256, MAX_NODES = 2048, ITEMS = MAX_NODES / THREADS_SCAN;
    __shared__ int scan[MAX_NODES];
    int tid = threadIdx.x;
    const Axis &target = Reverse ? input : output;
    double maximum_sigma =
        static_cast<double>(static_cast<float>(fmin(input.period, output.period) / 3));
    for (int item = 0; item < ITEMS; ++item) {
        int node = tid + item * THREADS_SCAN;
        int count = 0;
        if (node < target.node_count) {
            if (Reverse) {
                count = static_cast<int>(reverse_offsets[node + 1] - reverse_offsets[node]);
            } else {
                int bank_index = 0;
                while (bank_index + 1 < output.bank_count &&
                       node >= output.banks[bank_index + 1].node_offset)
                    ++bank_index;
                Bank bank = output.banks[bank_index];
                int owner = node - bank.node_offset;
                for (int range = 0; range < input.bank_count; ++range) {
                    Bank other = input.banks[range];
                    if (cartesian &&
                        !compatible_sigma_banks(output, bank_index, input, range, maximum_sigma))
                        continue;
                    int first, last;
                    if (cartesian) {
                        first = bank.node_offset * input.node_count +
                                bank.tile_count * other.node_offset + owner * other.tile_count;
                        last = first + other.tile_count;
                    } else {
                        first = lower_group(groups, group_count, bank.level, other.level, owner);
                        last = lower_group(groups, group_count, bank.level, other.level, owner + 1);
                    }
                    count += static_cast<int>(offsets[last] - offsets[first]);
                }
            }
            counts[node] = count;
        }
        scan[node] = count ? min(splits, (count + grain - 1) / grain) : 0;
    }
    __syncthreads();
    for (int stride = 1; stride < MAX_NODES; stride *= 2) {
        int add[ITEMS];
        for (int item = 0; item < ITEMS; ++item) {
            int node = tid + item * THREADS_SCAN;
            add[item] = node >= stride ? scan[node - stride] : 0;
        }
        __syncthreads();
        for (int item = 0; item < ITEMS; ++item)
            scan[tid + item * THREADS_SCAN] += add[item];
        __syncthreads();
    }
    for (int item = 0; item < ITEMS; ++item) {
        int node = tid + item * THREADS_SCAN;
        if (node < target.node_count)
            task_offsets[node] = node ? scan[node - 1] : 0;
    }
    if (tid == 0)
        task_offsets[target.node_count] = scan[target.node_count - 1];
}

// upper_bound skips duplicate offsets from empty owners. The caller has
// already excluded task >= total, so the result is always a nonempty owner.
__device__ inline int ragged_task_owner(const int *task_offsets, int nodes, int task) {
    int first = 0, last = nodes + 1;
    while (first < last) {
        int middle = (first + last) / 2;
        if (task_offsets[middle] <= task)
            first = middle + 1;
        else
            last = middle;
    }
    return first - 1;
}

// One 32-lane CTA owns one active output-owner split and one batch tile.
// Fused H/C uses the existing atom helper; separate C uses the existing dot.
template <bool Fused, int ForwardBatchTile = BATCH_TILE,
          bool ReuseSigma = false, bool KnownInputBank = false>
__global__ void ragged_output_kernel(const float *atoms, const int64_t *rows, const int *groups,
                                     const int64_t *offsets, const double *zx, float *h_output,
                                     const float *h_input, const int *counts,
                                     const int *task_offsets, double *partial, int atom_count,
                                     int group_count, int batches, Axis input, Axis output,
                                     bool cartesian, bool windowed) {
    static_assert(Fused || ForwardBatchTile == BATCH_TILE,
                  "separate C retains the original batch tile");
    int task = blockIdx.x;
    if (task >= task_offsets[output.node_count])
        return;
    int node = ragged_task_owner(task_offsets, output.node_count, task);
    int active = task_offsets[node + 1] - task_offsets[node];
    int local_split = task - task_offsets[node];
    int count = counts[node];
    int split_begin = static_cast<int>(static_cast<int64_t>(count) * local_split / active);
    int split_end = static_cast<int>(static_cast<int64_t>(count) * (local_split + 1) / active);
    int bank_index = 0;
    while (bank_index + 1 < output.bank_count && node >= output.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = output.banks[bank_index];
    int owner = node - bank.node_offset, lane = threadIdx.x;
    int batch_begin = blockIdx.y * ForwardBatchTile;
    int elements = ForwardBatchTile * output.basis_count();
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_h = shared + SHARED_STRIDE * output.basis_count();
    double *totals = scaled_h + SHARED_STRIDE * ForwardBatchTile;
    double accum[MAX_ACCUMULATORS] = {};
    if constexpr (Fused) {
        for (int work = lane; work < elements; work += CHUNK)
            totals[work] = 0;
        __syncthreads();
    }
    int prefix = 0;
    double maximum_sigma =
        static_cast<double>(static_cast<float>(fmin(input.period, output.period) / 3));
    for (int range = 0; range < input.bank_count; ++range) {
        Bank other = input.banks[range];
        if (cartesian && !compatible_sigma_banks(output, bank_index, input, range, maximum_sigma))
            continue;
        int first, last;
        if (cartesian) {
            first = bank.node_offset * input.node_count + bank.tile_count * other.node_offset +
                    owner * other.tile_count;
            last = first + other.tile_count;
        } else {
            first = lower_group(groups, group_count, bank.level, other.level, owner);
            last = lower_group(groups, group_count, bank.level, other.level,
                               owner + 1);
        }
        int begin = offsets[first], end = offsets[last];
        if constexpr (Fused) {
          fused_forward_range<float, true, ForwardBatchTile, ReuseSigma, KnownInputBank>(
              atoms, rows, groups, offsets, zx, h_output, atom_count,
              group_count, batches, batch_begin, input, output, bank, owner,
              begin, end, split_begin, split_end, prefix, windowed,
              coefficients, scaled_h, totals, &other);
        } else {
          split_consume_range<float, float, 8, true, false>(
              atoms, rows, h_input, atom_count, batches, output, bank, owner,
              false, batch_begin, begin, end, split_begin, split_end, prefix,
              coefficients, scaled_h, accum, CdotBudget{}, nullptr);
        }
    }
    for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
        int batch = batch_begin + work / output.basis_count(), mode = work % output.basis_count();
        if (batch < batches) {
            int64_t i =
                (static_cast<int64_t>(task) * batches + batch) * output.basis_count() + mode;
            if constexpr (Fused)
                partial[i] = totals[work];
            else
                partial[i] = accum[a];
        }
    }
}

// Reverse slots already concatenate all groups for one input owner. One lane
// evaluates one original atom; no atom values or coefficients are copied globally.
__global__ void ragged_input_kernel(const float *atoms, const int64_t *rows,
                                    const int *reverse_slots, const int64_t *reverse_offsets,
                                    const float *g, const int *counts, const int *task_offsets,
                                    double *partial, int atom_count, int batches, Axis input) {
    int task = blockIdx.x;
    if (task >= task_offsets[input.node_count])
        return;
    int node = ragged_task_owner(task_offsets, input.node_count, task);
    int active = task_offsets[node + 1] - task_offsets[node];
    int local_split = task - task_offsets[node], count = counts[node];
    int first = static_cast<int>(reverse_offsets[node]) +
                static_cast<int>(static_cast<int64_t>(count) * local_split / active);
    int last = static_cast<int>(reverse_offsets[node]) +
               static_cast<int>(static_cast<int64_t>(count) * (local_split + 1) / active);
    int bank_index = 0;
    while (bank_index + 1 < input.bank_count && node >= input.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = input.banks[bank_index];
    int owner = node - bank.node_offset, lane = threadIdx.x;
    int batch_begin = blockIdx.y * BATCH_TILE;
    int elements = BATCH_TILE * input.basis_count();
    extern __shared__ double shared[];
    double *coefficients = shared;
    double *scaled_g = shared + SHARED_STRIDE * input.basis_count();
    double accum[MAX_ACCUMULATORS] = {};
    for (int chunk = first; chunk < last; chunk += CHUNK) {
        int position = chunk + lane;
        if (position < last) {
            int slot = reverse_slots[position], original = rows[slot];
            Profile profile = make_profile<false, 8, true>(
                input, bank, atom_sigma(atoms, original, input),
                atom_center(atoms, original, input, false), owner);
            store_c_coefficients<8, true>(profile, 8, lane, coefficients);
            for (int b = 0; b < BATCH_TILE; ++b)
                scaled_g[b * SHARED_STRIDE + lane] =
                    batch_begin + b < batches ? atom_amplitude(atoms, original, input) *
                                                    g[(batch_begin + b) * atom_count + slot]
                                              : 0;
        } else {
            for (int mode = 0; mode < input.basis_count(); ++mode)
                coefficients[mode * SHARED_STRIDE + lane] = 0;
            for (int b = 0; b < BATCH_TILE; ++b)
                scaled_g[b * SHARED_STRIDE + lane] = 0;
        }
        __syncthreads();
        for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
            int batch = work / input.basis_count(), mode = work % input.basis_count();
            for (int atom = 0; atom < CHUNK; ++atom)
                accum[a] += scaled_g[batch * SHARED_STRIDE + atom] *
                            coefficients[mode * SHARED_STRIDE + atom];
        }
        __syncthreads();
    }
    for (int work = lane, a = 0; work < elements; work += CHUNK, ++a) {
        int batch = batch_begin + work / input.basis_count(), mode = work % input.basis_count();
        if (batch < batches) {
            int64_t i = (static_cast<int64_t>(task) * batches + batch) * input.basis_count() + mode;
            partial[i] = accum[a];
        }
    }
}

// Each dense C element has one writer. Active tasks are summed in split order;
// no atom-by-mode table or dense weight/weight-gradient is created.
__global__ void ragged_reduce_kernel(const int *task_offsets, const double *partial, double *c,
                                     int64_t elements, int batches, int nodes, int basis) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements)
        return;
    int mode = i % basis, node = (i / basis) % nodes, batch = i / (basis * nodes);
    double value = 0;
    for (int task = task_offsets[node]; task < task_offsets[node + 1]; ++task)
        value += partial[(static_cast<int64_t>(task) * batches + batch) * basis + mode];
    c[i] = value;
}

// Unique output site/batch ownership: add contributions from each bank after
// evaluating C in that bank's basis. No Y atomic or mixed-basis C summation.
template <typename scalar_t>
__global__ void expand_c_kernel(const double *c, scalar_t *y, int batches, Axis axis, int width,
                                const int *bank_masks = nullptr, int mask_side = 0) {
    for (int work = threadIdx.x; work < width * BATCH_TILE; work += blockDim.x) {
        int site = blockIdx.x * width + work % width;
        int batch = blockIdx.y * BATCH_TILE + work / width;
        if (site >= axis.sites || batch >= batches)
            continue;
        double total = 0;
        for (int index = 0; index < axis.bank_count; ++index) {
            // Consumer bits only: raw ancestor banks need not contribute C.
            if (bank_masks && !(bank_masks[2 * mask_side] & (1 << index)))
                continue;
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
            double omega = bank.omega;
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
                double factor = integrated_sine_factor(bank, harmonic, half_sine);
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
                        bool output_side, bool float_storage, int threads, bool windowed,
                        bool compensated) {
    check_layout(atoms, rows, groups, offsets);
    Axis axis = parse_axis(config);
    TORCH_CHECK(!compensated || (axis.modes == 8 && axis.tail_bound == 1e-3 &&
                                 atoms.scalar_type() == torch::kFloat32 && float_storage),
                "compensated values require M8, tail 1e-3 and FP32 atoms/projections");
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
            auto launch = [&](auto tag, auto compensated_tag) {
                constexpr int fixed_modes = decltype(tag)::value;
                constexpr bool use_compensated = decltype(compensated_tag)::value;
                compute_h_kernel<atom_t, scalar_t, fixed_modes, use_compensated>
                    <<<grid, threads, 0, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                        offsets.data_ptr<int64_t>(), z.data_ptr<double>(), h.data_ptr<scalar_t>(),
                        atoms.size(0), groups.size(0), z.size(0), axis, output_side, windowed);
            };
            if (compensated)
                launch(std::integral_constant<int, 8>{}, std::true_type{});
            else if (axis.modes == 8)
                launch(std::integral_constant<int, 8>{}, std::false_type{});
            else
                launch(std::integral_constant<int, 0>{}, std::false_type{});
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
}

torch::Tensor compute_c(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                        torch::Tensor offsets, torch::Tensor h, std::vector<double> input_config,
                        std::vector<double> output_config, bool reverse, bool compensated,
                        bool compensated_cdot) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(!compensated ||
                    (output.modes == 8 && output.tail_bound == 1e-3 &&
                     atoms.scalar_type() == torch::kFloat32 && h.scalar_type() == torch::kFloat32),
                "compensated values require target M8, tail 1e-3 and FP32 atoms/projections");
    TORCH_CHECK(!compensated_cdot || (output.modes == 8 && atoms.scalar_type() == torch::kFloat32 &&
                                      h.scalar_type() == torch::kFloat32),
                "compensated C dot requires target M8 and FP32 atoms/projections");
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
    size_t shared = sizeof(double) * SHARED_STRIDE * (output.basis_count() + BATCH_TILE);
    CdotBudget budget = compensated_cdot
                            ? cdot_budget(atoms.size(0), 1, 0, output, input.bank_count, reverse)
                            : CdotBudget{};
    auto stream = c10::cuda::getCurrentCUDAStream();
    torch::Tensor preflight_flag;
    if (compensated_cdot)
        preflight_flag = make_cdot_preflight(atoms, rows, h, stream, output);
    const int *preflight = compensated_cdot ? preflight_flag.data_ptr<int>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_compute_c_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_compute_c_storage", [&] {
            auto launch = [&](auto tag, auto compensated_tag, auto dot_tag) {
                constexpr int fixed_modes = decltype(tag)::value;
                constexpr bool use_compensated = decltype(compensated_tag)::value;
                constexpr bool use_dot = decltype(dot_tag)::value;
                compute_c_kernel<atom_t, scalar_t, fixed_modes, use_compensated, use_dot>
                    <<<grid, CHUNK, shared, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                        offsets.data_ptr<int64_t>(), h.data_ptr<scalar_t>(), c.data_ptr<double>(),
                        atoms.size(0), groups.size(0), h.size(0), input, output, reverse, budget,
                        preflight);
            };
            if constexpr (std::is_same_v<atom_t, float> && std::is_same_v<scalar_t, float>) {
                if (compensated_cdot) {
                    if (compensated)
                        launch(std::integral_constant<int, 8>{}, std::true_type{},
                               std::true_type{});
                    else
                        launch(std::integral_constant<int, 8>{}, std::false_type{},
                               std::true_type{});
                    return;
                }
            }
            if (compensated)
                launch(std::integral_constant<int, 8>{}, std::true_type{}, std::false_type{});
            else if (output.modes == 8)
                launch(std::integral_constant<int, 8>{}, std::false_type{}, std::false_type{});
            else
                launch(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{});
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

// Research candidate: fixed owner/split partials, then one writer per C
// element. S=1 remains available through compute_c for direct baseline
// comparison.
torch::Tensor compute_c_split(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                              torch::Tensor offsets, torch::Tensor h,
                              std::vector<double> input_config, std::vector<double> output_config,
                              bool reverse, int splits, int atoms_per_split, bool compensated,
                              bool compensated_cdot) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    TORCH_CHECK(!compensated ||
                    (output.modes == 8 && output.tail_bound == 1e-3 &&
                     atoms.scalar_type() == torch::kFloat32 && h.scalar_type() == torch::kFloat32),
                "compensated values require target M8, tail 1e-3 and FP32 atoms/projections");
    TORCH_CHECK(splits >= 1 && splits <= 256, "splits must lie in [1,256]");
    TORCH_CHECK(atoms_per_split >= 0 && atoms_per_split <= 65536,
                "atoms_per_split must lie in [0,65536]");
    TORCH_CHECK(!compensated_cdot || (output.modes == 8 && atoms.scalar_type() == torch::kFloat32 &&
                                      h.scalar_type() == torch::kFloat32),
                "compensated C dot requires target M8 and FP32 atoms/projections");
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
    TORCH_CHECK((splits + page_splits - 1) / page_splits <= 64,
                "split scratch needs more than 64 pages; reduce splits or batch size");
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
    size_t shared = sizeof(double) * SHARED_STRIDE * (output.basis_count() + BATCH_TILE);
    // Counting layouts contain every Cartesian group, including empty ones;
    // a fully populated unique sorted layout has the same complete order.
    bool cartesian = groups.size(0) == input.node_count * output.node_count;
    CdotBudget budget = compensated_cdot ? cdot_budget(atoms.size(0), splits, atoms_per_split,
                                                       output, input.bank_count, reverse)
                                         : CdotBudget{};
    auto stream = c10::cuda::getCurrentCUDAStream();
    torch::Tensor preflight_flag;
    if (compensated_cdot)
        preflight_flag = make_cdot_preflight(atoms, rows, h, stream, output);
    const int *preflight = compensated_cdot ? preflight_flag.data_ptr<int>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_split_c_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_split_c_storage", [&] {
            auto launch = [&](auto tag, auto compensated_tag, auto dot_tag) {
                constexpr int fixed_modes = decltype(tag)::value;
                constexpr bool use_compensated = decltype(compensated_tag)::value;
                constexpr bool use_dot = decltype(dot_tag)::value;
                compute_c_split_kernel<atom_t, scalar_t, fixed_modes, use_compensated, use_dot>
                    <<<grid, CHUNK, shared, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                        offsets.data_ptr<int64_t>(), h.data_ptr<scalar_t>(), pages, atoms.size(0),
                        groups.size(0), h.size(0), input, output, reverse, cartesian, splits,
                        atoms_per_split, budget, preflight);
            };
            if constexpr (std::is_same_v<atom_t, float> && std::is_same_v<scalar_t, float>) {
                if (compensated_cdot) {
                    if (compensated)
                        launch(std::integral_constant<int, 8>{}, std::true_type{},
                               std::true_type{});
                    else
                        launch(std::integral_constant<int, 8>{}, std::false_type{},
                               std::true_type{});
                    return;
                }
            }
            if (compensated)
                launch(std::integral_constant<int, 8>{}, std::true_type{}, std::false_type{});
            else if (output.modes == 8)
                launch(std::integral_constant<int, 8>{}, std::false_type{}, std::false_type{});
            else
                launch(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{});
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t elements = c.numel();
    reduce_c_splits_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        pages, c.data_ptr<double>(), elements, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

// Opt-in research path: H and output-owner C share the same atom work.
// The first prototype is deliberately restricted to M8 on both axes.
std::vector<torch::Tensor> compute_h_c(torch::Tensor atoms, torch::Tensor rows,
                                       torch::Tensor groups, torch::Tensor offsets,
                                       torch::Tensor zx, std::vector<double> input_config,
                                       std::vector<double> output_config, int splits,
                                       int atoms_per_split, bool windowed, bool compensated) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_moments(zx, atoms, input);
    TORCH_CHECK(input.modes == 8 && output.modes == 8, "fused H/C currently requires M8 axes");
    TORCH_CHECK(!compensated || (atoms.scalar_type() == torch::kFloat32 &&
                                 input.tail_bound == 1e-3 && output.tail_bound == 1e-3),
                "compensated values require FP32 atoms and tail 1e-3 on both axes");
    TORCH_CHECK(splits >= 1 && splits <= 256, "splits must lie in [1,256]");
    TORCH_CHECK(atoms_per_split >= 0 && atoms_per_split <= 65536,
                "atoms_per_split must lie in [0,65536]");
    TORCH_CHECK(zx.size(0) * atoms.size(0) <= INT_MAX, "H exceeds index capacity");
    TORCH_CHECK(zx.size(0) <= INT_MAX / (output.node_count * output.basis_count()),
                "C exceeds index capacity");
    TORCH_CHECK(zx.size(0) <= static_cast<int64_t>(65535) * BATCH_TILE,
                "batch grid exceeds CUDA capacity");
    TORCH_CHECK(atoms.size(0) == 0 || groups.size(0) > 0, "nonempty atoms need groups");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto h = torch::empty({zx.size(0), atoms.size(0)}, atoms.options());
    auto c = torch::empty({zx.size(0), output.node_count, output.basis_count()}, zx.options());
    if (c.numel() == 0)
        return {h, c};
    constexpr int64_t PAGE_BYTES = 1024 * 1024;
    int64_t c_bytes = c.numel() * static_cast<int64_t>(sizeof(double));
    int page_splits =
        static_cast<int>(std::min<int64_t>(splits, std::max<int64_t>(1, PAGE_BYTES / c_bytes)));
    TORCH_CHECK((splits + page_splits - 1) / page_splits <= 64,
                "split scratch needs more than 64 pages; reduce splits or batch size");
    SplitPages pages{};
    pages.page_splits = page_splits;
    std::vector<torch::Tensor> partial_pages;
    partial_pages.reserve((splits + page_splits - 1) / page_splits);
    for (int begin = 0; begin < splits; begin += page_splits) {
        int count = std::min(page_splits, splits - begin);
        partial_pages.push_back(torch::empty(
            {count, zx.size(0), output.node_count, output.basis_count()}, c.options()));
        pages.pointers[partial_pages.size() - 1] = partial_pages.back().data_ptr<double>();
    }
    dim3 grid(output.node_count, (zx.size(0) + BATCH_TILE - 1) / BATCH_TILE, splits);
    size_t shared = sizeof(double) * (SHARED_STRIDE * (output.basis_count() + BATCH_TILE) +
                                      BATCH_TILE * output.basis_count());
    bool cartesian = groups.size(0) == input.node_count * output.node_count;
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_fused_forward", [&] {
        auto launch = [&](auto compensated_tag) {
            constexpr bool use_compensated = decltype(compensated_tag)::value;
            fused_forward_kernel<scalar_t, use_compensated><<<grid, CHUNK, shared, stream>>>(
                atoms.data_ptr<scalar_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                offsets.data_ptr<int64_t>(), zx.data_ptr<double>(), h.data_ptr<scalar_t>(), pages,
                atoms.size(0), groups.size(0), zx.size(0), input, output, cartesian, splits,
                atoms_per_split, windowed);
        };
        if (compensated)
            launch(std::true_type{});
        else
            launch(std::false_type{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t elements = c.numel();
    reduce_c_splits_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        pages, c.data_ptr<double>(), elements, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h, c};
}

// Joint backward experiment for a single batch tile. It preserves the saved
// H/G storage dtype and bounds all partial C scratch by owners and splits.
std::vector<torch::Tensor>
compute_g_parameter_grad_cin(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                             torch::Tensor offsets, torch::Tensor zx, torch::Tensor zy,
                             torch::Tensor h, std::vector<double> input_config,
                             std::vector<double> output_config, int splits, bool windowed) {
    check_layout(atoms, rows, groups, offsets);
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    check_moments(zx, atoms, input);
    check_moments(zy, atoms, output);
    TORCH_CHECK(zx.size(0) >= 1 && zx.size(0) <= BATCH_TILE && zy.size(0) == zx.size(0),
                "fused G/gradient/Cin requires 1 to 4 batches and matching moments");
    TORCH_CHECK(splits >= 1 && splits <= 256, "splits must lie in [1,256]");
    TORCH_CHECK(h.is_contiguous() && h.device() == atoms.device() && h.dim() == 2 &&
                    (h.scalar_type() == torch::kFloat32 || h.scalar_type() == torch::kFloat64) &&
                    h.size(0) == zx.size(0) && h.size(1) == atoms.size(0) && h.numel() <= INT_MAX,
                "invalid H");
    TORCH_CHECK(atoms.size(0) == 0 || groups.size(0) > 0, "nonempty atoms need groups");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto c = torch::empty({h.size(0), input.node_count, input.basis_count()},
                          h.options().dtype(torch::kFloat64));
    auto g = torch::empty_like(h);
    auto gradient = torch::empty_like(atoms);
    constexpr int64_t PAGE_BYTES = 1024 * 1024;
    int64_t c_bytes = c.numel() * static_cast<int64_t>(sizeof(double));
    int page_splits =
        static_cast<int>(std::min<int64_t>(splits, std::max<int64_t>(1, PAGE_BYTES / c_bytes)));
    TORCH_CHECK((splits + page_splits - 1) / page_splits <= 64,
                "split scratch needs more than 64 pages; reduce splits or batch size");
    SplitPages pages{};
    pages.page_splits = page_splits;
    std::vector<torch::Tensor> partial_pages;
    partial_pages.reserve((splits + page_splits - 1) / page_splits);
    for (int begin = 0; begin < splits; begin += page_splits) {
        int count = std::min(page_splits, splits - begin);
        partial_pages.push_back(
            torch::empty({count, h.size(0), input.node_count, input.basis_count()}, c.options()));
        pages.pointers[partial_pages.size() - 1] = partial_pages.back().data_ptr<double>();
    }
    bool cartesian = groups.size(0) == input.node_count * output.node_count;
    dim3 grid(input.node_count, 1, splits);
    size_t shared = sizeof(double) * (SHARED_STRIDE * (input.basis_count() + BATCH_TILE) +
                                      BATCH_TILE * input.basis_count());
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_fused_cin_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_fused_cin_projection", [&] {
            auto launch = [&](auto tag) {
                constexpr int fixed_modes = decltype(tag)::value;
                fused_grad_cin_kernel<atom_t, scalar_t, fixed_modes>
                    <<<grid, CHUNK, shared, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
                        offsets.data_ptr<int64_t>(), zx.data_ptr<double>(), zy.data_ptr<double>(),
                        h.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), gradient.data_ptr<atom_t>(),
                        pages, atoms.size(0), groups.size(0), h.size(0), input, output, cartesian,
                        splits, windowed);
            };
            if (input.modes == 8 && output.modes == 8)
                launch(std::integral_constant<int, 8>{});
            else
                launch(std::integral_constant<int, 0>{});
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t elements = c.numel();
    reduce_c_splits_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        pages, c.data_ptr<double>(), elements, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {c, g, gradient};
}

template <bool Fused>
static std::vector<torch::Tensor> compute_reverse_cin_impl(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor reverse_slots, torch::Tensor reverse_offsets, torch::Tensor zx, torch::Tensor zy,
    torch::Tensor h, std::vector<double> input_config, std::vector<double> output_config,
    int splits, bool windowed, int atoms_per_split, bool compensated = false,
    bool compensated_cdot = false) {
    check_cuda_float_matrix(atoms);
    TORCH_CHECK(atoms.size(1) == 4 && atoms.size(0) <= 65536, "invalid atoms");
    Axis input = parse_axis(input_config);
    Axis output = Fused ? parse_axis(output_config) : input;
    TORCH_CHECK(!compensated ||
                    (!Fused && input.modes == 8 && input.tail_bound == 1e-3 &&
                     atoms.scalar_type() == torch::kFloat32 && h.scalar_type() == torch::kFloat32),
                "compensated values require target M8, tail 1e-3 and FP32 atoms/projections");
    TORCH_CHECK(!compensated_cdot ||
                    (!Fused && input.modes == 8 && atoms.scalar_type() == torch::kFloat32 &&
                     h.scalar_type() == torch::kFloat32),
                "compensated C dot requires target M8 and FP32 atoms/projections");
    TORCH_CHECK(rows.is_contiguous() && rows.device() == atoms.device() && rows.dim() == 1 &&
                    rows.scalar_type() == torch::kInt64 && rows.numel() == atoms.size(0),
                "invalid rows");
    TORCH_CHECK(reverse_slots.is_contiguous() && reverse_slots.device() == atoms.device() &&
                    reverse_slots.dim() == 1 && reverse_slots.scalar_type() == torch::kInt32 &&
                    reverse_slots.numel() == atoms.size(0),
                "invalid reverse slots");
    TORCH_CHECK(reverse_offsets.is_contiguous() && reverse_offsets.device() == atoms.device() &&
                    reverse_offsets.dim() == 1 && reverse_offsets.scalar_type() == torch::kInt64 &&
                    reverse_offsets.numel() == input.node_count + 1,
                "invalid reverse offsets");
    TORCH_CHECK(h.is_contiguous() && h.device() == atoms.device() && h.dim() == 2 &&
                    (h.scalar_type() == torch::kFloat32 || h.scalar_type() == torch::kFloat64) &&
                    h.size(1) == atoms.size(0) && h.numel() <= INT_MAX,
                "invalid projection");
    TORCH_CHECK(h.size(0) <= INT_MAX / (input.node_count * input.basis_count()),
                "C exceeds index capacity");
    TORCH_CHECK(h.size(0) <= static_cast<int64_t>(65535) * BATCH_TILE,
                "batch grid exceeds CUDA capacity");
    TORCH_CHECK(splits >= 1 && splits <= 256, "splits must lie in [1,256]");
    TORCH_CHECK(atoms_per_split >= 0 && atoms_per_split <= 65536,
                "atoms_per_split must lie in [0,65536]");
    if (Fused) {
        check_layout(atoms, rows, groups, offsets);
        check_moments(zx, atoms, input);
        check_moments(zy, atoms, output);
        TORCH_CHECK(h.size(0) >= 1 && h.size(0) <= BATCH_TILE && zx.size(0) == h.size(0) &&
                        zy.size(0) == h.size(0),
                    "fused reverse Cin requires 1 to 4 matching batches");
        TORCH_CHECK(atoms.size(0) == 0 || groups.size(0) > 0, "nonempty atoms need groups");
    }
    c10::cuda::CUDAGuard guard(atoms.device());
    auto c = torch::empty({h.size(0), input.node_count, input.basis_count()},
                          h.options().dtype(torch::kFloat64));
    torch::Tensor g, gradient;
    if (Fused) {
        g = torch::empty_like(h);
        gradient = torch::empty_like(atoms);
    }
    if (c.numel() == 0)
        return {c, g, gradient};
    constexpr int64_t PAGE_BYTES = 1024 * 1024;
    int64_t c_bytes = c.numel() * static_cast<int64_t>(sizeof(double));
    int page_splits =
        static_cast<int>(std::min<int64_t>(splits, std::max<int64_t>(1, PAGE_BYTES / c_bytes)));
    TORCH_CHECK((splits + page_splits - 1) / page_splits <= 64,
                "split scratch needs more than 64 pages; reduce splits or batch size");
    SplitPages pages{};
    pages.page_splits = page_splits;
    std::vector<torch::Tensor> partial_pages;
    partial_pages.reserve((splits + page_splits - 1) / page_splits);
    for (int begin = 0; begin < splits; begin += page_splits) {
        int count = std::min(page_splits, splits - begin);
        partial_pages.push_back(
            torch::empty({count, h.size(0), input.node_count, input.basis_count()}, c.options()));
        pages.pointers[partial_pages.size() - 1] = partial_pages.back().data_ptr<double>();
    }
    dim3 grid(input.node_count, (h.size(0) + BATCH_TILE - 1) / BATCH_TILE, splits);
    size_t shared = sizeof(double) * (SHARED_STRIDE * (input.basis_count() + BATCH_TILE) +
                                      BATCH_TILE * input.basis_count());
    CdotBudget budget = compensated_cdot
                            ? cdot_budget(atoms.size(0), splits, atoms_per_split, input, 1, false)
                            : CdotBudget{};
    auto stream = c10::cuda::getCurrentCUDAStream();
    torch::Tensor preflight_flag;
    if (compensated_cdot)
        preflight_flag = make_cdot_preflight(atoms, rows, h, stream, input);
    const int *preflight = compensated_cdot ? preflight_flag.data_ptr<int>() : nullptr;
    AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_reverse_cin_atoms", [&] {
        using atom_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "hierarchical_reverse_cin_projection", [&] {
            auto launch = [&](auto tag, auto compensated_tag, auto dot_tag) {
                constexpr int fixed_modes = decltype(tag)::value;
                constexpr bool use_compensated = decltype(compensated_tag)::value;
                constexpr bool use_dot = decltype(dot_tag)::value;
                reverse_cin_kernel<Fused, atom_t, scalar_t, fixed_modes, use_compensated, use_dot>
                    <<<grid, CHUNK, shared, stream>>>(
                        atoms.data_ptr<atom_t>(), rows.data_ptr<int64_t>(),
                        Fused ? groups.data_ptr<int>() : nullptr,
                        Fused ? offsets.data_ptr<int64_t>() : nullptr,
                        reverse_slots.data_ptr<int>(), reverse_offsets.data_ptr<int64_t>(),
                        Fused ? zx.data_ptr<double>() : nullptr,
                        Fused ? zy.data_ptr<double>() : nullptr, h.data_ptr<scalar_t>(),
                        Fused ? g.data_ptr<scalar_t>() : nullptr,
                        Fused ? gradient.data_ptr<atom_t>() : nullptr, pages, atoms.size(0),
                        Fused ? groups.size(0) : 0, h.size(0), input, output, splits,
                        atoms_per_split, windowed, budget, preflight);
            };
            if constexpr (!Fused && std::is_same_v<atom_t, float> &&
                          std::is_same_v<scalar_t, float>) {
                if (compensated_cdot) {
                    if (compensated)
                        launch(std::integral_constant<int, 8>{}, std::true_type{},
                               std::true_type{});
                    else
                        launch(std::integral_constant<int, 8>{}, std::false_type{},
                               std::true_type{});
                    return;
                }
            }
            if constexpr (!Fused) {
                if (compensated)
                    launch(std::integral_constant<int, 8>{}, std::true_type{}, std::false_type{});
                else if (input.modes == 8)
                    launch(std::integral_constant<int, 8>{}, std::false_type{}, std::false_type{});
                else
                    launch(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{});
            } else if (input.modes == 8 && output.modes == 8) {
                launch(std::integral_constant<int, 8>{}, std::false_type{}, std::false_type{});
            } else {
                launch(std::integral_constant<int, 0>{}, std::false_type{}, std::false_type{});
            }
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t elements = c.numel();
    reduce_c_splits_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        pages, c.data_ptr<double>(), elements, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {c, g, gradient};
}

torch::Tensor compute_input_c_reverse(torch::Tensor atoms, torch::Tensor rows,
                                      torch::Tensor reverse_slots, torch::Tensor reverse_offsets,
                                      torch::Tensor g, std::vector<double> input_config, int splits,
                                      int atoms_per_split, bool compensated,
                                      bool compensated_cdot) {
    return compute_reverse_cin_impl<false>(atoms, rows, {}, {}, reverse_slots, reverse_offsets, {},
                                           {}, g, input_config, {}, splits, false, atoms_per_split,
                                           compensated, compensated_cdot)[0];
}

std::vector<torch::Tensor> compute_g_parameter_grad_cin_reverse(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor reverse_slots, torch::Tensor reverse_offsets, torch::Tensor zx, torch::Tensor zy,
    torch::Tensor h, std::vector<double> input_config, std::vector<double> output_config,
    int splits, bool windowed, int atoms_per_split) {
    return compute_reverse_cin_impl<true>(atoms, rows, groups, offsets, reverse_slots,
                                          reverse_offsets, zx, zy, h, input_config, output_config,
                                          splits, windowed, atoms_per_split);
}

// Route 0: Cout; route 1: fused H/Cout; route 2: reverse-slot Cin.
// Metadata, partials and outputs remain alive until all same-stream kernels
// have been enqueued. The device task count is never copied back to the host.
template <int Route, int ForwardBatchTile = BATCH_TILE,
          bool ReuseSigma = false, bool KnownInputBank = false>
static std::vector<torch::Tensor>
run_ragged_c(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
             torch::Tensor reverse_slots, torch::Tensor reverse_offsets, torch::Tensor values,
             std::vector<double> input_config, std::vector<double> output_config, int splits,
             int grain, bool windowed, int cin_batch_slab = 0) {
    TORCH_CHECK(cin_batch_slab >= 0 && cin_batch_slab <= 65536, "Cin batch slab must lie in [0,65536]");
    TORCH_CHECK(cin_batch_slab == 0 || Route == 2, "batch slab is only supported for reverse Cin");
    static_assert(Route >= 0 && Route <= 2, "unknown ragged route");
    static_assert(ForwardBatchTile == 4 || ForwardBatchTile == 8, "invalid forward batch tile");
    static_assert(Route == 1 || ForwardBatchTile == BATCH_TILE,
                  "Cout and Cin retain the original batch tile");
    check_cuda_float_matrix(atoms);
    TORCH_CHECK(atoms.scalar_type() == torch::kFloat32 && atoms.size(1) == 4 &&
                    atoms.size(0) <= 65536,
                "ragged C requires FP32 [A<=65536,4] atoms");
    Axis input = parse_axis(input_config);
    Axis output = Route == 2 ? input : parse_axis(output_config);
    TORCH_CHECK(input.modes == 8 && input.tail_bound == 1e-3 &&
                    (Route == 2 || (output.modes == 8 && output.tail_bound == 1e-3)),
                "ragged C requires M8, tail 1e-3 axes and compensated values");
    TORCH_CHECK(splits >= 1 && splits <= 256, "splits must lie in [1,256]");
    TORCH_CHECK(grain >= 1 && grain <= 65536, "grain must lie in [1,65536]");
    if constexpr (Route == 2) {
        TORCH_CHECK(rows.is_contiguous() && rows.device() == atoms.device() && rows.dim() == 1 &&
                        rows.scalar_type() == torch::kInt64 && rows.numel() == atoms.size(0),
                    "invalid rows");
        TORCH_CHECK(reverse_slots.is_contiguous() && reverse_slots.device() == atoms.device() &&
                        reverse_slots.dim() == 1 && reverse_slots.scalar_type() == torch::kInt32 &&
                        reverse_slots.numel() == atoms.size(0),
                    "invalid reverse slots");
        TORCH_CHECK(reverse_offsets.is_contiguous() && reverse_offsets.device() == atoms.device() &&
                        reverse_offsets.dim() == 1 &&
                        reverse_offsets.scalar_type() == torch::kInt64 &&
                        reverse_offsets.numel() == input.node_count + 1,
                    "invalid reverse offsets");
    } else {
        check_layout(atoms, rows, groups, offsets);
        TORCH_CHECK(atoms.size(0) == 0 || groups.size(0) > 0, "nonempty atoms need groups");
    }
    if constexpr (Route == 1) {
        check_moments(values, atoms, input);
        TORCH_CHECK(values.size(0) * atoms.size(0) <= INT_MAX, "H exceeds index capacity");
    } else {
        TORCH_CHECK(values.is_contiguous() && values.device() == atoms.device() &&
                        values.dim() == 2 && values.scalar_type() == torch::kFloat32 &&
                        values.size(1) == atoms.size(0) && values.numel() <= INT_MAX,
                    "ragged C requires FP32 [batch,atom] projection");
    }
    int64_t batches = values.size(0);
    const Axis &target = Route == 2 ? input : output;
    int64_t nodes = target.node_count, basis = target.basis_count();
    TORCH_CHECK(nodes <= 2048, "ragged metadata supports at most 2048 owners");
    TORCH_CHECK(batches <= INT_MAX / (nodes * basis), "C exceeds index capacity");
    TORCH_CHECK(batches <= static_cast<int64_t>(65535) * ForwardBatchTile,
                "batch grid exceeds CUDA capacity");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto c = torch::empty({batches, nodes, basis}, atoms.options().dtype(torch::kFloat64));
    torch::Tensor h;
    if constexpr (Route == 1)
        h = torch::empty({batches, atoms.size(0)}, atoms.options());
    if (c.numel() == 0)
        return {h, c};
    if (atoms.size(0) == 0) {
        c.zero_();
        return {h, c};
    }
    // Sum of ceil(owner_count/grain) is bounded by ceil(A/grain)+N.
    int64_t task_cap = std::min(nodes * splits, (atoms.size(0) + grain - 1) / grain + nodes);
    TORCH_CHECK(task_cap > 0 && task_cap <= INT_MAX, "ragged task grid exceeds capacity");
    TORCH_CHECK(task_cap <= INT_MAX / (batches * basis), "ragged partial exceeds index capacity");
    auto integer_options = atoms.options().dtype(torch::kInt32);
    auto counts = torch::empty({nodes}, integer_options);
    auto task_offsets = torch::empty({nodes + 1}, integer_options);
    int64_t scratch_batches = cin_batch_slab ? std::min<int64_t>(batches, cin_batch_slab) : batches;
    auto partial = torch::empty({task_cap, scratch_batches, basis}, c.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    bool cartesian = Route != 2 && groups.size(0) == input.node_count * output.node_count;
    if constexpr (Route == 2) {
        ragged_metadata_kernel<true><<<1, 256, 0, stream>>>(
            nullptr, nullptr, reverse_offsets.data_ptr<int64_t>(), counts.data_ptr<int>(),
            task_offsets.data_ptr<int>(), input, output, false, 0, splits, grain);
    } else {
        ragged_metadata_kernel<false><<<1, 256, 0, stream>>>(
            groups.data_ptr<int>(), offsets.data_ptr<int64_t>(), nullptr, counts.data_ptr<int>(),
            task_offsets.data_ptr<int>(), input, output, cartesian, groups.size(0), splits, grain);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr (Route == 2) {
        if (cin_batch_slab) {
            // G and C retain their full batch layout. Each slab is a contiguous
            // view; only partial scratch is reused after its same-stream reduce.
            size_t shared = sizeof(double) * SHARED_STRIDE * (basis + BATCH_TILE);
            for (int64_t begin = 0; begin < batches; begin += cin_batch_slab) {
                int slab_batches = static_cast<int>(std::min<int64_t>(batches - begin, cin_batch_slab));
                dim3 slab_grid(task_cap, (slab_batches + BATCH_TILE - 1) / BATCH_TILE);
                ragged_input_kernel<<<slab_grid, CHUNK, shared, stream>>>(
                    atoms.data_ptr<float>(), rows.data_ptr<int64_t>(), reverse_slots.data_ptr<int>(),
                    reverse_offsets.data_ptr<int64_t>(), values.data_ptr<float>() + begin * atoms.size(0),
                    counts.data_ptr<int>(), task_offsets.data_ptr<int>(), partial.data_ptr<double>(),
                    atoms.size(0), slab_batches, input);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                int64_t elements = static_cast<int64_t>(slab_batches) * nodes * basis;
                ragged_reduce_kernel<<<(elements + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                    task_offsets.data_ptr<int>(), partial.data_ptr<double>(),
                    c.data_ptr<double>() + begin * nodes * basis, elements, slab_batches, nodes, basis);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            return {h, c};
        }
    }
    dim3 grid(task_cap, (batches + ForwardBatchTile - 1) / ForwardBatchTile);
    size_t shared = sizeof(double) * SHARED_STRIDE * (basis + ForwardBatchTile);
    if constexpr (Route == 1)
        shared += sizeof(double) * ForwardBatchTile * basis;
    if constexpr (Route == 0) {
        ragged_output_kernel<false><<<grid, CHUNK, shared, stream>>>(
            atoms.data_ptr<float>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
            offsets.data_ptr<int64_t>(), nullptr, nullptr, values.data_ptr<float>(),
            counts.data_ptr<int>(), task_offsets.data_ptr<int>(), partial.data_ptr<double>(),
            atoms.size(0), groups.size(0), batches, input, output, cartesian, false);
    } else if constexpr (Route == 1) {
        ragged_output_kernel<true, ForwardBatchTile, ReuseSigma, KnownInputBank><<<grid, CHUNK, shared, stream>>>(
            atoms.data_ptr<float>(), rows.data_ptr<int64_t>(), groups.data_ptr<int>(),
            offsets.data_ptr<int64_t>(), values.data_ptr<double>(), h.data_ptr<float>(), nullptr,
            counts.data_ptr<int>(), task_offsets.data_ptr<int>(), partial.data_ptr<double>(),
            atoms.size(0), groups.size(0), batches, input, output, cartesian, windowed);
    } else {
        ragged_input_kernel<<<grid, CHUNK, shared, stream>>>(
            atoms.data_ptr<float>(), rows.data_ptr<int64_t>(), reverse_slots.data_ptr<int>(),
            reverse_offsets.data_ptr<int64_t>(), values.data_ptr<float>(), counts.data_ptr<int>(),
            task_offsets.data_ptr<int>(), partial.data_ptr<double>(), atoms.size(0), batches,
            input);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ragged_reduce_kernel<<<(c.numel() + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        task_offsets.data_ptr<int>(), partial.data_ptr<double>(), c.data_ptr<double>(), c.numel(),
        batches, nodes, basis);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h, c};
}

torch::Tensor compute_c_ragged(torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups,
                               torch::Tensor offsets, torch::Tensor h,
                               std::vector<double> input_config, std::vector<double> output_config,
                               int splits, int grain) {
    return run_ragged_c<0>(atoms, rows, groups, offsets, {}, {}, h, input_config, output_config,
                           splits, grain, false)[1];
}

std::vector<torch::Tensor> compute_h_c_ragged(torch::Tensor atoms, torch::Tensor rows,
                                              torch::Tensor groups, torch::Tensor offsets,
                                              torch::Tensor zx, std::vector<double> input_config,
                                              std::vector<double> output_config, int splits,
                                              int grain, bool windowed) {
  return run_ragged_c<1>(atoms, rows, groups, offsets, {}, {}, zx, input_config,
                         output_config, splits, grain, windowed);
}

// Forward-only batch reuse; legacy H/C, Cout and Cin entry points retain tile
// four.
std::vector<torch::Tensor>
compute_h_c_ragged_batched(torch::Tensor atoms, torch::Tensor rows,
                           torch::Tensor groups, torch::Tensor offsets,
                           torch::Tensor zx, std::vector<double> input_config,
                           std::vector<double> output_config, int splits,
                           int grain, bool windowed, int batch_tile) {
  TORCH_CHECK(batch_tile == 4 || batch_tile == 8,
              "forward batch tile must be 4 or 8");
  if (batch_tile == 8)
    return run_ragged_c<1, 8>(atoms, rows, groups, offsets, {}, {}, zx,
                              input_config, output_config, splits, grain,
                              windowed);
  return run_ragged_c<1>(atoms, rows, groups, offsets, {}, {}, zx, input_config,
                         output_config, splits, grain, windowed);
}

// Independent opt-in switches retain the old instantiations for baseline calls.
template <int BatchTile>
static std::vector<torch::Tensor> run_exact_hc(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor zx, std::vector<double> input_config, std::vector<double> output_config,
    int splits, int grain, bool windowed, bool reuse_sigma, bool known_bank) {
    if (reuse_sigma) {
        if (known_bank)
            return run_ragged_c<1, BatchTile, true, true>(atoms, rows, groups, offsets, {}, {},
                zx, input_config, output_config, splits, grain, windowed);
        return run_ragged_c<1, BatchTile, true, false>(atoms, rows, groups, offsets, {}, {},
            zx, input_config, output_config, splits, grain, windowed);
    }
    if (known_bank)
        return run_ragged_c<1, BatchTile, false, true>(atoms, rows, groups, offsets, {}, {},
            zx, input_config, output_config, splits, grain, windowed);
    return run_ragged_c<1, BatchTile>(atoms, rows, groups, offsets, {}, {},
        zx, input_config, output_config, splits, grain, windowed);
}

std::vector<torch::Tensor> compute_h_c_ragged_exact(
    torch::Tensor atoms, torch::Tensor rows, torch::Tensor groups, torch::Tensor offsets,
    torch::Tensor zx, std::vector<double> input_config, std::vector<double> output_config,
    int splits, int grain, bool windowed, int batch_tile, bool reuse_sigma, bool known_bank) {
    TORCH_CHECK(batch_tile == 4 || batch_tile == 8, "forward batch tile must be 4 or 8");
    if (batch_tile == 8)
        return run_exact_hc<8>(atoms, rows, groups, offsets, zx, input_config, output_config,
                              splits, grain, windowed, reuse_sigma, known_bank);
    return run_exact_hc<4>(atoms, rows, groups, offsets, zx, input_config, output_config,
                          splits, grain, windowed, reuse_sigma, known_bank);
}

// Untimed compiler/runtime diagnostics. Active blocks are an occupancy API
// prediction for this launch configuration, not measured achieved occupancy.
template <int ForwardBatchTile, bool ReuseSigma = false, bool KnownInputBank = false>
static std::vector<int64_t> hc_kernel_resources_impl() {
    constexpr int BASIS = 17;
    constexpr size_t dynamic_shared = sizeof(double) *
        (SHARED_STRIDE * (BASIS + ForwardBatchTile) + ForwardBatchTile * BASIS);
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                         ragged_output_kernel<true, ForwardBatchTile, ReuseSigma, KnownInputBank>));
    int active_blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, ragged_output_kernel<true, ForwardBatchTile, ReuseSigma, KnownInputBank>, CHUNK, dynamic_shared));
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return {attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes),
            static_cast<int64_t>(attributes.sharedSizeBytes), attributes.maxThreadsPerBlock,
            active_blocks, static_cast<int64_t>(dynamic_shared), properties.multiProcessorCount};
}

std::vector<int64_t> hc_kernel_resources(int batch_tile) {
    TORCH_CHECK(batch_tile == 4 || batch_tile == 8, "forward batch tile must be 4 or 8");
    if (batch_tile == 8)
        return hc_kernel_resources_impl<8>();
    return hc_kernel_resources_impl<4>();
}

// Report the actual exact-reuse specialization while preserving the legacy API.
template <int BatchTile>
static std::vector<int64_t> hc_reuse_resources_impl(bool reuse_sigma, bool known_bank) {
    if (reuse_sigma) {
        if (known_bank)
            return hc_kernel_resources_impl<BatchTile, true, true>();
        return hc_kernel_resources_impl<BatchTile, true, false>();
    }
    if (known_bank)
        return hc_kernel_resources_impl<BatchTile, false, true>();
    return hc_kernel_resources_impl<BatchTile>();
}

std::vector<int64_t> hc_reuse_kernel_resources(int batch_tile, bool reuse_sigma, bool known_bank) {
    TORCH_CHECK(batch_tile == 4 || batch_tile == 8, "forward batch tile must be 4 or 8");
    if (batch_tile == 8)
        return hc_reuse_resources_impl<8>(reuse_sigma, known_bank);
    return hc_reuse_resources_impl<4>(reuse_sigma, known_bank);
}

torch::Tensor compute_input_c_reverse_ragged(torch::Tensor atoms, torch::Tensor rows,
                                             torch::Tensor reverse_slots,
                                             torch::Tensor reverse_offsets, torch::Tensor g,
                                             std::vector<double> input_config, int splits,
                                             int grain) {
    return run_ragged_c<2>(atoms, rows, {}, {}, reverse_slots, reverse_offsets, g, input_config, {},
                           splits, grain, false)[1];
}

torch::Tensor compute_input_c_reverse_ragged_slab(torch::Tensor atoms, torch::Tensor rows,
                                                  torch::Tensor reverse_slots,
                                                  torch::Tensor reverse_offsets, torch::Tensor g,
                                                  std::vector<double> input_config, int splits,
                                                  int grain, int batch_slab) {
    TORCH_CHECK(batch_slab >= 1 && batch_slab <= 65536, "Cin batch slab must lie in [1,65536]");
    return run_ragged_c<2>(atoms, rows, {}, {}, reverse_slots, reverse_offsets, g, input_config, {},
                           splits, grain, false, batch_slab)[1];
}

static torch::Tensor expand_c_impl(torch::Tensor c, torch::Tensor like,
                                   std::vector<double> config, int width,
                                   torch::Tensor bank_masks, int mask_side) {
    check_cuda_float_matrix(like);
    Axis axis = parse_axis(config);
    check_moments(c, like, axis);
    TORCH_CHECK(c.size(0) == like.size(0), "batch mismatch");
    TORCH_CHECK(width >= 1 && width <= 256 && power_of_two(width), "invalid output block width");
    if (bank_masks.defined()) {
        TORCH_CHECK(bank_masks.device() == like.device() && bank_masks.is_contiguous() &&
                    bank_masks.scalar_type() == torch::kInt32 && bank_masks.dim() == 1 &&
                    bank_masks.numel() == 4, "bank masks must be device-local int32[4]");
        TORCH_CHECK(mask_side == 0 || mask_side == 1, "mask side must be input=0 or output=1");
    }
    c10::cuda::CUDAGuard guard(like.device());
    auto y = torch::empty({c.size(0), axis.sites}, like.options());
    if (y.numel() == 0)
        return y;
    dim3 grid((axis.sites + width - 1) / width, (c.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(like.scalar_type(), "hierarchical_expand_c", [&] {
        expand_c_kernel<<<grid, THREADS, 0, stream>>>(c.data_ptr<double>(), y.data_ptr<scalar_t>(),
                                                      c.size(0), axis, width,
                                                      bank_masks.defined() ? bank_masks.data_ptr<int>() : nullptr,
                                                      mask_side);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Existing entry keeps all banks; the optional route receives forward-scoped
// masks derived from every nonempty atom group, irrespective of amplitude.
torch::Tensor expand_c(torch::Tensor c, torch::Tensor like, std::vector<double> config, int width) {
    return expand_c_impl(c, like, config, width, {}, 0);
}

torch::Tensor expand_c_masked(torch::Tensor c, torch::Tensor like, std::vector<double> config,
                             int width, torch::Tensor bank_masks, int mask_side) {
    TORCH_CHECK(bank_masks.defined(), "masked expansion requires bank masks");
    return expand_c_impl(c, like, config, width, bank_masks, mask_side);
}
