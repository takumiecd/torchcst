// Layout indices and varying-period moments; no atom value copies.
#include "common.cuh"
using namespace cell_mass_hierarchy;

namespace {
// The packed integer preserves (h_out,h_in,q_out,q_in) lexicographic order.
// It is temporary sort scratch. The surviving descriptors are per group.
template <typename scalar_t>
__global__ void classify_atoms_kernel(const scalar_t *atoms, int atom_count, Axis input,
                                      Axis output, int64_t *keys, int *errors) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atom_count)
        return;
    int error = 0;
    for (int field = 0; field < 4; ++field)
        if (!isfinite(static_cast<double>(atoms[4 * atom + field])))
            error |= 1;
    double sigma = static_cast<double>(atoms[4 * atom + 1]);
    scalar_t maximum_sigma = static_cast<scalar_t>(std::min(input.period, output.period) / 3);
    if (!(sigma > 0) || atoms[4 * atom + 1] > maximum_sigma)
        error |= 2;
    if (error) {
        keys[atom] = 0;
        atomicOr(errors, error); // One error word, not an output reduction.
        return;
    }
    Membership in = classify_axis(sigma, static_cast<double>(atoms[4 * atom + 3]), input);
    Membership out = classify_axis(sigma, static_cast<double>(atoms[4 * atom + 2]), output);
    int64_t key = out.level;
    key = key * (input.full_level + 1) + in.level;
    key = key * output.max_owners() + out.owner;
    keys[atom] = key * input.max_owners() + in.owner;
}

__global__ void decode_groups_kernel(const int64_t *keys, int groups, Axis input, Axis output,
                                     int *descriptors) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups)
        return;
    int64_t key = keys[group];
    descriptors[4 * group + 3] = key % input.max_owners();
    key /= input.max_owners();
    descriptors[4 * group + 2] = key % output.max_owners();
    key /= output.max_owners();
    descriptors[4 * group + 1] = key % (input.full_level + 1);
    descriptors[4 * group] = key / (input.full_level + 1);
}

// Fixed Cartesian group IDs are ordered by (output level, input level,
// output owner, input owner), including empty groups. Only indices are sorted;
// atom values remain in their original Parameter array.
template <typename scalar_t>
__global__ void count_memberships_kernel(const scalar_t *atoms, int atom_count, Axis input,
                                         Axis output, const scalar_t *signal, int64_t signal_count,
                                         int *keys, int *counts, int *errors) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Check signal before the atom boundary: empty/small atom sets can still
    // have input entries to validate. Bit 4 is separate from atom error bits.
    if (index < signal_count && !isfinite(static_cast<double>(signal[index])))
        atomicOr(errors, 4);
    if (index >= atom_count)
        return;
    int atom = static_cast<int>(index);
    int error = 0;
    for (int field = 0; field < 4; ++field)
        if (!isfinite(static_cast<double>(atoms[4 * atom + field])))
            error |= 1;
    double sigma = static_cast<double>(atoms[4 * atom + 1]);
    scalar_t maximum_sigma = static_cast<scalar_t>(std::min(input.period, output.period) / 3);
    if (!(sigma > 0) || atoms[4 * atom + 1] > maximum_sigma)
        error |= 2;
    if (error) {
        keys[atom] = -1;
        atomicOr(errors, error);
        return;
    }
    Membership in = classify_axis(sigma, static_cast<double>(atoms[4 * atom + 3]), input);
    Membership out = classify_axis(sigma, static_cast<double>(atoms[4 * atom + 2]), output);
    Bank ib = input.banks[bank_for_level(input, in.level)];
    Bank ob = output.banks[bank_for_level(output, out.level)];
    int key = ob.node_offset * input.node_count + ob.tile_count * ib.node_offset +
              out.owner * ib.tile_count + in.owner;
    keys[atom] = key;
    unsigned active = __activemask();
    unsigned peers = __match_any_sync(active, key);
    int lane = threadIdx.x % 32;
    if (lane == __ffs(peers) - 1)
        atomicAdd(counts + key, __popc(peers));
}

// Each thread scans a contiguous chunk, then a block-wide scan computes the
// chunk prefixes. At most 16 entries per thread for the bounded 4096 groups.
__global__ void prefix_counting_groups_kernel(const int *counts, int64_t *offsets, int *groups,
                                              int group_count, Axis input, Axis output) {
    constexpr int SCAN_THREADS = 256;
    __shared__ int sums[SCAN_THREADS];
    int tid = threadIdx.x;
    int chunk_size = (group_count + SCAN_THREADS - 1) / SCAN_THREADS;
    int begin = tid * chunk_size;
    int end = min(begin + chunk_size, group_count);
    int sum = 0;
    for (int group = begin; group < end; ++group)
        sum += counts[group];
    sums[tid] = sum;
    __syncthreads();
    for (int stride = 1; stride < SCAN_THREADS; stride *= 2) {
        int add = tid >= stride ? sums[tid - stride] : 0;
        __syncthreads();
        sums[tid] += add;
        __syncthreads();
    }
    int prefix = tid ? sums[tid - 1] : 0;
    for (int group = begin; group < end; ++group) {
        offsets[group] = prefix;
        prefix += counts[group];
        int oi = 0;
        while (oi + 1 < output.bank_count &&
               group >= output.banks[oi + 1].node_offset * input.node_count)
            ++oi;
        Bank ob = output.banks[oi];
        int local = group - ob.node_offset * input.node_count;
        int ii = 0;
        while (ii + 1 < input.bank_count &&
               local >= ob.tile_count * input.banks[ii + 1].node_offset)
            ++ii;
        Bank ib = input.banks[ii];
        local -= ob.tile_count * ib.node_offset;
        groups[4 * group] = ob.level;
        groups[4 * group + 1] = ib.level;
        groups[4 * group + 2] = local / ib.tile_count;
        groups[4 * group + 3] = local % ib.tile_count;
    }
    if (tid == 0)
        offsets[group_count] = sums[SCAN_THREADS - 1];
}

// Every Cartesian group has exactly one input owner. Build its reverse base
// in the same output-bank/output-owner order used by the reverse traversal.
// The entire scan is block-local; the dyadic axis supports at most 2048 nodes.
__global__ void prefix_reverse_groups_kernel(const int *counts, int *reverse_group_base,
                                             int64_t *reverse_offsets, Axis input, Axis output) {
    constexpr int SCAN_THREADS = 256;
    constexpr int MAX_INPUT_NODES = 2048;
    constexpr int ITEMS = MAX_INPUT_NODES / SCAN_THREADS;
    __shared__ int node_prefix[MAX_INPUT_NODES];
    int tid = threadIdx.x;
    for (int item = 0; item < ITEMS; ++item) {
        int node = tid + item * SCAN_THREADS;
        int count = 0;
        if (node < input.node_count) {
            int bank_index = 0;
            while (bank_index + 1 < input.bank_count &&
                   node >= input.banks[bank_index + 1].node_offset)
                ++bank_index;
            Bank bank = input.banks[bank_index];
            int owner = node - bank.node_offset;
            for (int range = 0; range < output.bank_count; ++range) {
                Bank other = output.banks[range];
                int first = other.node_offset * input.node_count +
                            other.tile_count * bank.node_offset + owner;
                for (int other_owner = 0; other_owner < other.tile_count; ++other_owner) {
                    int group = first + other_owner * bank.tile_count;
                    reverse_group_base[group] = count;
                    count += counts[group];
                }
            }
        }
        node_prefix[node] = count;
    }
    __syncthreads();
    for (int stride = 1; stride < MAX_INPUT_NODES; stride *= 2) {
        int add[ITEMS];
        for (int item = 0; item < ITEMS; ++item) {
            int node = tid + item * SCAN_THREADS;
            add[item] = node >= stride ? node_prefix[node - stride] : 0;
        }
        __syncthreads();
        for (int item = 0; item < ITEMS; ++item)
            node_prefix[tid + item * SCAN_THREADS] += add[item];
        __syncthreads();
    }
    for (int item = 0; item < ITEMS; ++item) {
        int node = tid + item * SCAN_THREADS;
        if (node < input.node_count) {
            int prefix = node == 0 ? 0 : node_prefix[node - 1];
            reverse_offsets[node] = prefix;
            int bank_index = 0;
            while (bank_index + 1 < input.bank_count &&
                   node >= input.banks[bank_index + 1].node_offset)
                ++bank_index;
            Bank bank = input.banks[bank_index];
            int owner = node - bank.node_offset;
            for (int range = 0; range < output.bank_count; ++range) {
                Bank other = output.banks[range];
                int first = other.node_offset * input.node_count +
                            other.tile_count * bank.node_offset + owner;
                for (int other_owner = 0; other_owner < other.tile_count; ++other_owner) {
                    int group = first + other_owner * bank.tile_count;
                    reverse_group_base[group] += prefix;
                }
            }
        }
    }
    if (tid == 0)
        reverse_offsets[input.node_count] = node_prefix[input.node_count - 1];
}

template <bool Reverse>
__global__ void scatter_counting_rows_kernel(const int *keys, int atom_count,
                                             const int64_t *offsets, int *cursors, int64_t *rows,
                                             const int *reverse_group_base, int *reverse_slots) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atom_count || keys[atom] < 0)
        return;
    int key = keys[atom];
    unsigned peers = __match_any_sync(__activemask(), key);
    int lane = threadIdx.x % 32;
    int leader = __ffs(peers) - 1;
    int reservation = 0;
    if (lane == leader)
        reservation = atomicAdd(cursors + key, __popc(peers));
    reservation = __shfl_sync(peers, reservation, leader);
    unsigned earlier = peers & ((1u << lane) - 1u);
    int rank = reservation + __popc(earlier);
    int slot = static_cast<int>(offsets[key]) + rank;
    rows[slot] = atom;
    if (Reverse)
        reverse_slots[reverse_group_base[key] + rank] = slot;
}

// A block owns one spatial node and four batches. A tunable small thread team
// shares one (batch,mode) site sum; different teams work independently. Every
// thread reaches every block barrier, including unused teams and merge jobs.
template <typename scalar_t>
__global__ void build_direct_kernel(const scalar_t *signal, double *z, int batches, Axis axis,
                                    bool merge, int team_size) {
    __shared__ double partials[THREADS];
    int node = blockIdx.x;
    int bank_index = 0;
    while (bank_index + 1 < axis.bank_count && node >= axis.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = axis.banks[bank_index];
    int first = (node - bank.node_offset) * bank.tile_sites;
    int team_count = THREADS / team_size;
    int team = threadIdx.x / team_size;
    int lane = threadIdx.x % team_size;
    int count = BATCH_TILE * axis.basis_count();
    for (int wave = 0; wave * team_count < count; ++wave) {
        int work = wave * team_count + team;
        int batch = blockIdx.y * BATCH_TILE + work / axis.basis_count();
        int mode = work % axis.basis_count();
        int harmonic = (mode + 1) / 2;
        bool direct =
            !merge || bank_index == 0 || bank.full_chart || (mode != 0 && harmonic % 2 == 1);
        bool valid = work < count && batch < batches && direct;
        double total = 0;
        if (valid && lane < bank.tile_sites) {
            if (mode == 0) {
                for (int local = lane; local < bank.tile_sites; local += team_size) {
                    int site = first + local;
                    total += static_cast<double>(signal[batch * axis.sites + site]) * axis.spacing;
                }
            } else {
                // A lane visits sites spaced team_size apart. The integral
                // factor is site-independent; rotate its phase in FP64 after
                // each site instead of evaluating trigonometry per site.
                // The common chart origin and harmonic are unchanged.
                double omega = bank.omega * harmonic;
                double factor = cell_integral_factor(bank, harmonic, axis.spacing);
                double phase_sin, phase_cos, step_sin, step_cos;
                sincos(omega * (first + lane) * axis.spacing, &phase_sin, &phase_cos);
                sincos((omega * axis.spacing) * team_size, &step_sin, &step_cos);
                for (int local = lane; local < bank.tile_sites; local += team_size) {
                    int site = first + local;
                    double basis = factor * (mode % 2 == 1 ? phase_cos : phase_sin);
                    total += static_cast<double>(signal[batch * axis.sites + site]) * basis;
                    double next_cos = phase_cos * step_cos - phase_sin * step_sin;
                    phase_sin = phase_sin * step_cos + phase_cos * step_sin;
                    phase_cos = next_cos;
                }
            }
        }
        partials[threadIdx.x] = total;
        __syncthreads();
        for (int step = team_size / 2; step > 0; step /= 2) {
            if (lane < step)
                partials[threadIdx.x] += partials[threadIdx.x + step];
            __syncthreads();
        }
        if (valid && lane == 0)
            z[(batch * axis.node_count + node) * axis.basis_count() + mode] = partials[threadIdx.x];
        __syncthreads();
    }
}

// The same cell-integrated basis as build_direct_kernel, but one block owns a
// small site chunk for one bank. The bank period is never changed. The scratch
// is [batch, bank, chart chunk, mode], independent of the number of atoms.
template <typename scalar_t>
__global__ void build_direct_chunk_kernel(const scalar_t *signal, double *scratch, int batches,
                                          Axis axis, bool merge, int team_size, int chunk_sites,
                                          int chunks) {
    __shared__ double partials[THREADS];
    int bank_index = blockIdx.x / chunks;
    int chunk = blockIdx.x % chunks;
    Bank bank = axis.banks[bank_index];
    int first = chunk * chunk_sites;
    int team_count = THREADS / team_size;
    int team = threadIdx.x / team_size;
    int lane = threadIdx.x % team_size;
    int count = BATCH_TILE * axis.basis_count();
    for (int wave = 0; wave * team_count < count; ++wave) {
        int work = wave * team_count + team;
        int batch = blockIdx.y * BATCH_TILE + work / axis.basis_count();
        int mode = work % axis.basis_count();
        int harmonic = (mode + 1) / 2;
        bool direct =
            !merge || bank_index == 0 || bank.full_chart || (mode != 0 && harmonic % 2 == 1);
        bool valid = work < count && batch < batches && direct;
        double total = 0;
        if (valid && lane < chunk_sites) {
            if (mode == 0) {
                for (int local = lane; local < chunk_sites; local += team_size)
                    total += static_cast<double>(signal[batch * axis.sites + first + local]) *
                             axis.spacing;
            } else {
                double omega = bank.omega * harmonic;
                double factor = cell_integral_factor(bank, harmonic, axis.spacing);
                double phase_sin, phase_cos, step_sin, step_cos;
                sincos(omega * (first + lane) * axis.spacing, &phase_sin, &phase_cos);
                sincos((omega * axis.spacing) * team_size, &step_sin, &step_cos);
                for (int local = lane; local < chunk_sites; local += team_size) {
                    double basis = factor * (mode % 2 == 1 ? phase_cos : phase_sin);
                    total +=
                        static_cast<double>(signal[batch * axis.sites + first + local]) * basis;
                    double next_cos = phase_cos * step_cos - phase_sin * step_sin;
                    phase_sin = phase_sin * step_cos + phase_cos * step_sin;
                    phase_cos = next_cos;
                }
            }
        }
        partials[threadIdx.x] = total;
        __syncthreads();
        for (int step = team_size / 2; step > 0; step /= 2) {
            if (lane < step)
                partials[threadIdx.x] += partials[threadIdx.x + step];
            __syncthreads();
        }
        if (valid && lane == 0) {
            int64_t index =
                ((static_cast<int64_t>(batch) * axis.bank_count + bank_index) * chunks + chunk) *
                    axis.basis_count() +
                mode;
            scratch[index] = partials[threadIdx.x];
        }
        __syncthreads();
    }
}

// A unique writer reduces each node's chunks; inherited modes are left for
// the subsequent level merges and do not read uninitialized scratch.
__global__ void reduce_direct_chunks_kernel(const double *scratch, double *z, int batches,
                                            Axis axis, bool merge, int chunk_sites, int chunks) {
    int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t count = static_cast<int64_t>(batches) * axis.node_count * axis.basis_count();
    if (work >= count)
        return;
    int mode = work % axis.basis_count();
    int node = (work / axis.basis_count()) % axis.node_count;
    int batch = work / (axis.basis_count() * axis.node_count);
    int bank_index = 0;
    while (bank_index + 1 < axis.bank_count && node >= axis.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = axis.banks[bank_index];
    int harmonic = (mode + 1) / 2;
    bool direct = !merge || bank_index == 0 || bank.full_chart || (mode != 0 && harmonic % 2 == 1);
    if (!direct)
        return;
    int chunks_per_tile = bank.tile_sites / chunk_sites;
    int first_chunk = (node - bank.node_offset) * chunks_per_tile;
    double total = 0;
    for (int i = 0; i < chunks_per_tile; ++i) {
        int64_t index = ((static_cast<int64_t>(batch) * axis.bank_count + bank_index) * chunks +
                         first_chunk + i) *
                            axis.basis_count() +
                        mode;
        total += scratch[index];
    }
    z[work] = total;
}

// P_parent=2*P_child: parent harmonic 2*j matches child harmonic j.
// Odd parent harmonics were built directly, not inherited with the same index.
__global__ void merge_bank_kernel(double *z, int batches, Axis axis, int bank_index) {
    Bank parent_bank = axis.banks[bank_index];
    Bank child_bank = axis.banks[bank_index - 1];
    int work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= batches * parent_bank.tile_count * axis.basis_count())
        return;
    int mode = work % axis.basis_count();
    int harmonic = (mode + 1) / 2;
    if (mode != 0 && harmonic % 2 != 0)
        return;
    int child_mode = mode == 0 ? 0 : (mode % 2 == 1 ? harmonic - 1 : harmonic);
    int tile = (work / axis.basis_count()) % parent_bank.tile_count;
    int batch = work / (axis.basis_count() * parent_bank.tile_count);
    int left = child_bank.node_offset + 2 * tile;
    int parent = parent_bank.node_offset + tile;
    z[(batch * axis.node_count + parent) * axis.basis_count() + mode] =
        z[(batch * axis.node_count + left) * axis.basis_count() + child_mode] +
        z[(batch * axis.node_count + left + 1) * axis.basis_count() + child_mode];
}
// Collapse each owner's retained period after every raw hierarchy merge.
// This is independent of atom values. Never feed these windows into merge_bank.
__global__ void build_window_moments_kernel(const double *z, double *window, int elements,
                                            Axis axis) {
    int work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= elements)
        return;
    int mode = work % axis.basis_count();
    int node = (work / axis.basis_count()) % axis.node_count;
    int batch = work / (axis.basis_count() * axis.node_count);
    int bank_index = 0;
    while (bank_index + 1 < axis.bank_count && node >= axis.banks[bank_index + 1].node_offset)
        ++bank_index;
    Bank bank = axis.banks[bank_index];
    if (bank.full_chart) {
        window[work] = z[work];
        return;
    }
    int owner = node - bank.node_offset;
    int harmonic = (mode + 1) / 2;
    double total = 0;
    for (int offset = -1; offset <= 1; ++offset) {
        int raw_tile = owner + offset;
        int tile = positive_mod(raw_tile, bank.tile_count);
        int image = (raw_tile - tile) / bank.tile_count;
        int base = (batch * axis.node_count + bank.node_offset + tile) * axis.basis_count();
        if (mode == 0) {
            total += z[base];
        } else {
            double rs, rc;
            image_rotation(bank, harmonic, image, rs, rc);
            double zc = z[base + 2 * harmonic - 1], zs = z[base + 2 * harmonic];
            total += mode % 2 ? zc * rc - zs * rs : zs * rc + zc * rs;
        }
    }
    window[work] = total;
}
} // namespace

std::vector<torch::Tensor> classify_atoms(torch::Tensor atoms, std::vector<double> input_config,
                                          std::vector<double> output_config) {
    check_cuda_float_matrix(atoms);
    TORCH_CHECK(atoms.size(1) == 4 && atoms.size(0) <= 65536, "expected [A<=65536,4] atoms");
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    c10::cuda::CUDAGuard guard(atoms.device());
    auto keys = torch::empty({atoms.size(0)}, atoms.options().dtype(torch::kInt64));
    auto errors = torch::zeros({}, atoms.options().dtype(torch::kInt32));
    if (atoms.size(0) > 0) {
        int blocks = (atoms.size(0) + THREADS - 1) / THREADS;
        auto stream = c10::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_classify_atoms", [&] {
            classify_atoms_kernel<<<blocks, THREADS, 0, stream>>>(
                atoms.data_ptr<scalar_t>(), atoms.size(0), input, output, keys.data_ptr<int64_t>(),
                errors.data_ptr<int>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {keys, errors};
}

static std::vector<torch::Tensor> prepare_counting_layout_impl(torch::Tensor atoms,
                                                               const torch::Tensor &signal,
                                                               std::vector<double> input_config,
                                                               std::vector<double> output_config,
                                                               bool reverse) {
    check_cuda_float_matrix(atoms);
    TORCH_CHECK(atoms.size(1) == 4 && atoms.size(0) <= 65536, "expected [A<=65536,4] atoms");
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    if (signal.defined()) {
        check_cuda_float_matrix(signal);
        TORCH_CHECK(signal.device() == atoms.device() &&
                        signal.scalar_type() == atoms.scalar_type(),
                    "signal and atoms must share device and dtype");
        TORCH_CHECK(signal.size(1) == input.sites, "signal does not match input axis");
        TORCH_CHECK(signal.numel() <= INT_MAX, "signal exceeds index capacity");
    }
    int group_count = input.node_count * output.node_count;
    TORCH_CHECK(group_count <= 4096, "counting layout supports at most 4096 Cartesian groups");
    c10::cuda::CUDAGuard guard(atoms.device());
    auto integer_options = atoms.options().dtype(torch::kInt32);
    auto keys = torch::empty({atoms.size(0)}, integer_options);
    auto counts = torch::zeros({group_count}, integer_options);
    auto cursors = torch::zeros({group_count}, integer_options);
    auto errors = torch::zeros({}, integer_options);
    auto groups = torch::empty({group_count, 4}, integer_options);
    auto offsets = torch::empty({group_count + 1}, atoms.options().dtype(torch::kInt64));
    auto rows = torch::empty({atoms.size(0)}, atoms.options().dtype(torch::kInt64));
    torch::Tensor reverse_slots, reverse_offsets, reverse_group_base;
    if (reverse) {
        TORCH_CHECK(input.node_count <= 2048, "reverse layout supports at most 2048 input nodes");
        reverse_slots = torch::empty({atoms.size(0)}, integer_options);
        reverse_offsets =
            torch::empty({input.node_count + 1}, atoms.options().dtype(torch::kInt64));
        reverse_group_base = torch::empty({group_count}, integer_options);
    }
    auto stream = c10::cuda::getCurrentCUDAStream();
    int64_t signal_count = signal.defined() ? signal.numel() : 0;
    int64_t work = std::max(atoms.size(0), signal_count);
    if (work > 0) {
        int blocks = (work + THREADS - 1) / THREADS;
        AT_DISPATCH_FLOATING_TYPES(atoms.scalar_type(), "hierarchical_count_memberships", [&] {
            count_memberships_kernel<<<blocks, THREADS, 0, stream>>>(
                atoms.data_ptr<scalar_t>(), atoms.size(0), input, output,
                signal.defined() ? signal.data_ptr<scalar_t>() : nullptr, signal_count,
                keys.data_ptr<int>(), counts.data_ptr<int>(), errors.data_ptr<int>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    prefix_counting_groups_kernel<<<1, 256, 0, stream>>>(
        counts.data_ptr<int>(), offsets.data_ptr<int64_t>(), groups.data_ptr<int>(), group_count,
        input, output);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (reverse) {
        prefix_reverse_groups_kernel<<<1, 256, 0, stream>>>(
            counts.data_ptr<int>(), reverse_group_base.data_ptr<int>(),
            reverse_offsets.data_ptr<int64_t>(), input, output);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (atoms.size(0) > 0) {
        int blocks = (atoms.size(0) + THREADS - 1) / THREADS;
        if (reverse) {
            scatter_counting_rows_kernel<true><<<blocks, THREADS, 0, stream>>>(
                keys.data_ptr<int>(), atoms.size(0), offsets.data_ptr<int64_t>(),
                cursors.data_ptr<int>(), rows.data_ptr<int64_t>(),
                reverse_group_base.data_ptr<int>(), reverse_slots.data_ptr<int>());
        } else {
            scatter_counting_rows_kernel<false><<<blocks, THREADS, 0, stream>>>(
                keys.data_ptr<int>(), atoms.size(0), offsets.data_ptr<int64_t>(),
                cursors.data_ptr<int>(), rows.data_ptr<int64_t>(), nullptr, nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (reverse)
        return {rows, groups, offsets, errors, reverse_slots, reverse_offsets};
    return {rows, groups, offsets, errors};
}

std::vector<torch::Tensor> prepare_counting_layout(torch::Tensor atoms,
                                                   std::vector<double> input_config,
                                                   std::vector<double> output_config) {
    return prepare_counting_layout_impl(atoms, torch::Tensor(), input_config, output_config, false);
}

std::vector<torch::Tensor> prepare_counting_layout_signal(torch::Tensor atoms, torch::Tensor signal,
                                                          std::vector<double> input_config,
                                                          std::vector<double> output_config) {
    return prepare_counting_layout_impl(atoms, signal, input_config, output_config, false);
}

std::vector<torch::Tensor> prepare_counting_layout_reverse(torch::Tensor atoms,
                                                           std::vector<double> input_config,
                                                           std::vector<double> output_config) {
    return prepare_counting_layout_impl(atoms, torch::Tensor(), input_config, output_config, true);
}

std::vector<torch::Tensor>
prepare_counting_layout_reverse_signal(torch::Tensor atoms, torch::Tensor signal,
                                       std::vector<double> input_config,
                                       std::vector<double> output_config) {
    return prepare_counting_layout_impl(atoms, signal, input_config, output_config, true);
}

torch::Tensor decode_groups(torch::Tensor keys, std::vector<double> input_config,
                            std::vector<double> output_config) {
    TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && keys.dim() == 1 &&
                    keys.scalar_type() == torch::kInt64 && keys.numel() <= 65536,
                "expected internal sorted group keys");
    Axis input = parse_axis(input_config), output = parse_axis(output_config);
    c10::cuda::CUDAGuard guard(keys.device());
    auto groups = torch::empty({keys.numel(), 4}, keys.options().dtype(torch::kInt32));
    if (keys.numel() > 0) {
        int blocks = (keys.numel() + THREADS - 1) / THREADS;
        auto stream = c10::cuda::getCurrentCUDAStream();
        decode_groups_kernel<<<blocks, THREADS, 0, stream>>>(keys.data_ptr<int64_t>(), keys.numel(),
                                                             input, output, groups.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return groups;
}

torch::Tensor build_hierarchy(torch::Tensor signal, std::vector<double> config, bool merge,
                              int team_size) {
    check_cuda_float_matrix(signal);
    Axis axis = parse_axis(config);
    TORCH_CHECK(signal.size(1) == axis.sites, "signal does not match axis");
    TORCH_CHECK(signal.size(0) <= INT_MAX / (axis.node_count * axis.basis_count()),
                "hierarchy exceeds prototype index capacity");
    TORCH_CHECK(power_of_two(team_size) && team_size <= 32, "invalid Z team size");
    c10::cuda::CUDAGuard guard(signal.device());
    auto z = torch::empty({signal.size(0), axis.node_count, axis.basis_count()},
                          signal.options().dtype(torch::kFloat64));
    if (signal.size(0) == 0)
        return z;
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 grid(axis.node_count, (signal.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "hierarchical_build_z", [&] {
        build_direct_kernel<<<grid, THREADS, 0, stream>>>(signal.data_ptr<scalar_t>(),
                                                          z.data_ptr<double>(), signal.size(0),
                                                          axis, merge, team_size);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Kernel order on this stream is the inter-block dependency boundary.
    for (int bank = 1; merge && bank < axis.bank_count - 1; ++bank) {
        int work = signal.size(0) * axis.banks[bank].tile_count * axis.basis_count();
        merge_bank_kernel<<<(work + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            z.data_ptr<double>(), signal.size(0), axis, bank);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return z;
}

torch::Tensor build_hierarchy_chunked(torch::Tensor signal, std::vector<double> config, bool merge,
                                      int team_size, int requested_chunk_sites) {
    check_cuda_float_matrix(signal);
    Axis axis = parse_axis(config);
    TORCH_CHECK(signal.size(1) == axis.sites, "signal does not match axis");
    TORCH_CHECK(power_of_two(team_size) && team_size <= 32, "invalid Z team size");
    TORCH_CHECK(power_of_two(requested_chunk_sites), "chunk sites must be a positive power of two");
    int chunk_sites = std::min(requested_chunk_sites, axis.min_tile_sites);
    TORCH_CHECK(axis.sites % chunk_sites == 0, "chunk must divide chart sites");
    for (int bank = 0; bank < axis.bank_count; ++bank)
        TORCH_CHECK(axis.banks[bank].tile_sites % chunk_sites == 0,
                    "chunk must divide every bank tile");
    int chunks = axis.sites / chunk_sites;
    int64_t z_per_batch = static_cast<int64_t>(axis.node_count) * axis.basis_count();
    int64_t scratch_per_batch = static_cast<int64_t>(axis.bank_count) * chunks * axis.basis_count();
    TORCH_CHECK(signal.size(0) <= INT_MAX / z_per_batch, "hierarchy exceeds index capacity");
    TORCH_CHECK(signal.size(0) <= INT_MAX / scratch_per_batch,
                "chunk scratch exceeds index capacity");
    TORCH_CHECK(signal.size(0) <= INT_MAX / axis.sites, "signal exceeds index capacity");
    TORCH_CHECK(signal.size(0) <= static_cast<int64_t>(65535) * BATCH_TILE,
                "chunk grid exceeds CUDA grid.y capacity");
    c10::cuda::CUDAGuard guard(signal.device());
    auto z = torch::empty({signal.size(0), axis.node_count, axis.basis_count()},
                          signal.options().dtype(torch::kFloat64));
    if (signal.size(0) == 0)
        return z;
    auto scratch =
        torch::empty({signal.size(0), axis.bank_count, chunks, axis.basis_count()}, z.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    dim3 grid(axis.bank_count * chunks, (signal.size(0) + BATCH_TILE - 1) / BATCH_TILE);
    AT_DISPATCH_FLOATING_TYPES(signal.scalar_type(), "hierarchical_build_z_chunked", [&] {
        build_direct_chunk_kernel<<<grid, THREADS, 0, stream>>>(
            signal.data_ptr<scalar_t>(), scratch.data_ptr<double>(), signal.size(0), axis, merge,
            team_size, chunk_sites, chunks);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int64_t work = signal.size(0) * z_per_batch;
    reduce_direct_chunks_kernel<<<(work + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        scratch.data_ptr<double>(), z.data_ptr<double>(), signal.size(0), axis, merge, chunk_sites,
        chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int bank = 1; merge && bank < axis.bank_count - 1; ++bank) {
        int merge_work = signal.size(0) * axis.banks[bank].tile_count * axis.basis_count();
        merge_bank_kernel<<<(merge_work + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            z.data_ptr<double>(), signal.size(0), axis, bank);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return z;
}

// Same shape as raw Z, but each local owner now spans its full retained period.
// Consumers must explicitly request windowed projection; shape alone cannot
// distinguish the two mathematical layouts.
torch::Tensor build_window_moments(torch::Tensor z, std::vector<double> config) {
    Axis axis = parse_axis(config);
    TORCH_CHECK(z.is_cuda() && z.is_contiguous() && z.dim() == 3 &&
                    z.scalar_type() == torch::kFloat64 && z.size(1) == axis.node_count &&
                    z.size(2) == axis.basis_count() && z.numel() <= INT_MAX,
                "expected raw FP64 hierarchy moments for this axis");
    c10::cuda::CUDAGuard guard(z.device());
    auto window = torch::empty_like(z);
    if (z.numel() > 0) {
        auto stream = c10::cuda::getCurrentCUDAStream();
        build_window_moments_kernel<<<(z.numel() + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
            z.data_ptr<double>(), window.data_ptr<double>(), z.numel(), axis);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return window;
}
