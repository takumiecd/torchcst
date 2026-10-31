#pragma once
#include <ATen/ops/sort.h>

namespace radius_cell_mass {

// Keys encode output_owner * input_nodes + input_owner. Reverse changes the
// key orientation, never the Parameter rows themselves. Keys come from this
// forward's device-validated classification and remain read-only for backward.
__global__ void reverse_owner_keys(const int64_t *keys, int64_t *reversed,
    int atoms, int input_nodes, int output_nodes) {
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    int64_t key = keys[atom];
    reversed[atom] = (key % input_nodes) * output_nodes + key / input_nodes;
}

// All owners, including empty ones, receive a lower-bound offset. Generate
// each boundary in a register instead of allocating an arange Tensor and
// entering searchsorted through Python. Stable ATen sorting is unchanged.
__global__ void owner_range_offsets(const int64_t *sorted_keys, int64_t *offsets,
    int atoms, int owners, int stride) {
    int owner = blockIdx.x * blockDim.x + threadIdx.x;
    if (owner > owners) return;
    int64_t boundary = int64_t(owner) * stride;
    int begin = 0, end = atoms;
    while (begin < end) {
        int middle = begin + (end - begin) / 2;
        if (sorted_keys[middle] < boundary) begin = middle + 1;
        else end = middle;
    }
    offsets[owner] = begin;
}

inline void check_layout_keys(const Tensor &keys, int input_nodes, int output_nodes) {
    TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == torch::kLong &&
        keys.dim() == 1 && keys.is_contiguous() && keys.numel() <= MAX_ATOMS,
        "expected at most 65536 contiguous CUDA int64 owner keys");
    TORCH_CHECK(input_nodes > 0 && input_nodes <= MAX_OWNER_NODES &&
        output_nodes > 0 && output_nodes <= MAX_OWNER_NODES, "invalid owner node counts");
}

// Retain the general ATen path for larger atom tables and for diagnostic
// comparisons. No numerical state or sorting result persists between calls.
inline std::vector<Tensor> layout_ranges_aten(Tensor keys, int input_nodes,
    int output_nodes, bool reverse) {
    check_layout_keys(keys, input_nodes, output_nodes);
    c10::cuda::CUDAGuard guard(keys.device());
    int atoms = keys.numel();
    if (reverse && atoms) {
        auto reversed = torch::empty_like(keys);
        reverse_owner_keys<<<(atoms + THREADS - 1) / THREADS, THREADS, 0,
            at::cuda::getCurrentCUDAStream()>>>(keys.data_ptr<int64_t>(),
            reversed.data_ptr<int64_t>(), atoms, input_nodes, output_nodes);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        keys = reversed;
    }
    auto sorted = at::sort(keys, true, -1, false);
    Tensor sorted_keys = std::get<0>(sorted), rows = std::get<1>(sorted);
    int owners = reverse ? input_nodes : output_nodes;
    int stride = reverse ? output_nodes : input_nodes;
    auto offsets = torch::empty({owners + 1}, keys.options());
    owner_range_offsets<<<(owners + 1 + THREADS - 1) / THREADS, THREADS, 0,
        at::cuda::getCurrentCUDAStream()>>>(sorted_keys.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(), atoms, owners, stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rows, offsets};
}

constexpr int BLOCK_LAYOUT_THREADS = 256;

// One CTA performs orientation, stable sorting, and all owner boundaries.
// Only eligible owner products enter this 32-bit packing. Appending the
// original row in 12 bits preserves original atom order for equal owners.
// Padding is UINT_MAX, strictly above every valid key. Input keys stay intact
// for the saved forward/backward lifetime; only shared memory is sorted.
__global__ void block_layout_ranges(const int64_t *keys, int64_t *rows,
    int64_t *offsets, int atoms, int capacity, int input_nodes,
    int output_nodes, bool reverse) {
    extern __shared__ unsigned sorted[];
    unsigned key_count = unsigned(input_nodes * output_nodes);
    for (int i = threadIdx.x; i < capacity; i += blockDim.x) {
        unsigned packed = UINT_MAX;
        if (i < atoms) {
            int64_t original = keys[i];
            bool valid = original >= 0 && original < key_count;
            CUDA_KERNEL_ASSERT_MSG(valid, "invalid validated radius owner key");
            if (valid) {
                unsigned key = unsigned(original);
                if (reverse)
                    key = (key % unsigned(input_nodes)) * unsigned(output_nodes) +
                        key / unsigned(input_nodes);
                packed = (key << OWNER_ROW_BITS) | unsigned(i);
            }
        }
        sorted[i] = packed;
    }
    __syncthreads();

    // A stage's XOR pairs are disjoint. Only the lower index owns a pair, so
    // multiple pairs per thread cannot race with another thread's writes.
    // Every thread reaches every barrier, including padded/inactive lanes.
    for (int width = 2; width <= capacity; width *= 2) {
        for (int distance = width / 2; distance; distance /= 2) {
            for (int i = threadIdx.x; i < capacity; i += blockDim.x) {
                int partner = i ^ distance;
                if (partner > i) {
                    unsigned a = sorted[i], b = sorted[partner];
                    bool ascending = (i & width) == 0;
                    if (ascending ? a > b : a < b) {
                        sorted[i] = b;
                        sorted[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < atoms; i += blockDim.x)
        rows[i] = int64_t(sorted[i] & OWNER_ROW_MASK);

    int owners = reverse ? input_nodes : output_nodes;
    int stride = reverse ? output_nodes : input_nodes;
    for (int owner = threadIdx.x; owner <= owners; owner += blockDim.x) {
        unsigned boundary = unsigned(owner * stride) << OWNER_ROW_BITS;
        int begin = 0, end = atoms;
        while (begin < end) {
            int middle = begin + (end - begin) / 2;
            if (sorted[middle] < boundary) begin = middle + 1;
            else end = middle;
        }
        offsets[owner] = begin;
    }
}

inline std::vector<Tensor> layout_ranges(Tensor keys, int input_nodes,
    int output_nodes, bool reverse) {
    check_layout_keys(keys, input_nodes, output_nodes);
    int atoms = keys.numel();
    if (!can_use_block_layout(atoms, input_nodes, output_nodes))
        return layout_ranges_aten(keys, input_nodes, output_nodes, reverse);
    c10::cuda::CUDAGuard guard(keys.device());
    int owners = reverse ? input_nodes : output_nodes;
    auto rows = torch::empty_like(keys);
    auto offsets = torch::empty({owners + 1}, keys.options());
    int capacity = 1;
    while (capacity < atoms) capacity *= 2;
    block_layout_ranges<<<1, BLOCK_LAYOUT_THREADS, capacity * sizeof(unsigned),
        at::cuda::getCurrentCUDAStream()>>>(
        keys.data_ptr<int64_t>(), rows.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        atoms, capacity, input_nodes, output_nodes, reverse);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rows, offsets};
}

} // namespace radius_cell_mass
