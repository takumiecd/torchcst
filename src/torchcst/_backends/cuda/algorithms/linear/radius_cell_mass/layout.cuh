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

inline std::vector<Tensor> layout_ranges(Tensor keys, int input_nodes,
    int output_nodes, bool reverse) {
    TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == torch::kLong &&
        keys.dim() == 1 && keys.is_contiguous() && keys.numel() <= 4096,
        "expected at most 4096 contiguous CUDA int64 owner keys");
    TORCH_CHECK(input_nodes > 0 && input_nodes <= 510 &&
        output_nodes > 0 && output_nodes <= 510, "invalid owner node counts");
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

} // namespace radius_cell_mass
