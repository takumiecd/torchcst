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
        if (valid) {
            for (int local = lane; local < bank.tile_sites; local += team_size) {
                int site = first + local;
                total += static_cast<double>(signal[batch * axis.sites + site]) *
                         integrated_basis(axis, bank, mode, site);
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
