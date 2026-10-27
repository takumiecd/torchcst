// Read cell-mass-kernel-flow.ja.md for the new sigma/I and bank definitions.
// These kernels prepare indices and moments only. They never copy atom values
// into a second global Parameter array or pretend to implement H/C/Y.
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <torch/extension.h>
#include <vector>

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr int THREADS = 128;
constexpr int BATCH_TILE = 4;
constexpr int MAX_BANKS = 11;

struct Bank {
    int level, tile_sites, tile_count, node_offset;
    double period;
    bool full_chart;
};

struct Axis {
    int sites, min_tile_sites, modes, full_level, bank_count, node_count;
    double spacing, origin, period;
    Bank banks[MAX_BANKS];
    __host__ __device__ int basis_count() const {
        return 1 + 2 * modes;
    }
    __host__ __device__ int max_owners() const {
        return sites / min_tile_sites;
    }
};

bool power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

Axis parse_axis(const std::vector<double> &config) {
    TORCH_CHECK(config.size() == 5, "hierarchy axis needs five scalars");
    for (double value : config)
        TORCH_CHECK(std::isfinite(value), "axis values must be finite");
    TORCH_CHECK(config[0] == std::floor(config[0]) && config[0] >= 1 && config[0] <= 1024,
                "invalid site count");
    TORCH_CHECK(config[2] == std::floor(config[2]) && config[2] >= 1 && config[2] <= config[0],
                "invalid minimum tile size");
    TORCH_CHECK(config[4] == std::floor(config[4]) && config[4] >= 2 && config[4] <= 16,
                "invalid modes");
    Axis axis{};
    axis.sites = static_cast<int>(config[0]);
    axis.spacing = config[1];
    axis.min_tile_sites = static_cast<int>(config[2]);
    axis.origin = config[3];
    axis.modes = static_cast<int>(config[4]);
    TORCH_CHECK(power_of_two(axis.sites) && power_of_two(axis.min_tile_sites) &&
                    axis.modes % 2 == 0 && axis.spacing > 0,
                "invalid dyadic axis");
    axis.period = axis.sites * axis.spacing;
    TORCH_CHECK(std::isfinite(2 * axis.period) && axis.period > 0, "invalid chart period");
    for (int count = axis.max_owners(); count > 1; count /= 2)
        ++axis.full_level;

    int tile_sites = axis.min_tile_sites;
    int level = 0;
    while (3 * tile_sites <= axis.sites) {
        int count = axis.sites / tile_sites;
        axis.banks[axis.bank_count++] = {
            level, tile_sites, count, axis.node_count, 3 * tile_sites * axis.spacing, false};
        axis.node_count += count;
        tile_sites *= 2;
        ++level;
    }
    axis.banks[axis.bank_count++] = {axis.full_level, axis.sites,  1,
                                     axis.node_count, axis.period, true};
    ++axis.node_count;
    for (int bank = 0; bank < axis.bank_count; ++bank)
        TORCH_CHECK(std::isfinite(2 * PI * axis.modes / axis.banks[bank].period),
                    "bank frequency is not representable");
    return axis;
}

void check_cuda_float_matrix(const torch::Tensor &tensor) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() == 2,
                "expected a contiguous CUDA matrix");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64,
                "expected float32 or float64");
}

struct Membership {
    int level, owner;
};

__device__ Membership classify_axis(double sigma, double center, const Axis &axis) {
    int tile_sites = axis.min_tile_sites;
    int level = 0;
    while (sigma / 2 > tile_sites * axis.spacing) {
        tile_sites *= 2;
        ++level;
    }
    if (3 * tile_sites > axis.sites)
        return {axis.full_level, 0};

    double relative = fmod(center, axis.period) - fmod(axis.origin, axis.period);
    relative = fmod(relative, axis.period);
    if (relative < 0)
        relative += axis.period;
    int owner =
        static_cast<int>(floor((relative + axis.spacing / 2) / (tile_sites * axis.spacing)));
    return {level, owner % (axis.sites / tile_sites)};
}

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
    if (!(sigma > 0 && sigma < input.period && sigma < output.period))
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

__device__ double integrated_basis(const Axis &axis, const Bank &bank, int mode, int site) {
    if (mode == 0)
        return axis.spacing;
    int harmonic = (mode + 1) / 2;
    double omega = 2 * PI * harmonic / bank.period;
    double factor = 2 * sin(omega * axis.spacing / 2) / omega;
    // Coordinates are relative to the same chart origin at every level.
    double phase = omega * site * axis.spacing;
    return factor * (mode % 2 == 1 ? cos(phase) : sin(phase));
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
