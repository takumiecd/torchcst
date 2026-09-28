#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <type_traits>

template <int THREADS>
__global__ void materialize_bounded_kernel(
    const float* __restrict__ p,
    const float* __restrict__ circle,
    const float* __restrict__ section,
    const uint8_t* __restrict__ lists,
    const int32_t* __restrict__ counts,
    const int64_t* __restrict__ offsets,
    float* __restrict__ output,
    int k,
    int column_groups,
    int groups,
    int capacity,
    int station_start,
    int row_start) {
  const int station = station_start + blockIdx.x / 4;
  const int row_tile = blockIdx.x % 4;
  const int tile_index = station * 4 + row_tile;
  const int count = counts[tile_index];
  const int begin0 = groups == 1 ? offsets[0]
      : offsets[2 * ((station + groups - 1) % groups) + 1];
  const int length0 = groups == 1 ? offsets[1] - begin0
      : offsets[2 * ((station + groups - 1) % groups) + 2] - begin0;
  const int begin1 = groups == 1 ? 0 : offsets[2 * station];
  const int length1 = groups == 1 ? 0 : offsets[2 * station + 1] - begin1;
  const int begin2 = groups == 1 ? 0 : offsets[2 * station + 1];
  const int length2 = groups == 1 ? 0 : offsets[2 * station + 2] - begin2;
  const int limit = count < 0 ? length0 + length1 + length2 : count;

  constexpr int SITES_PER_THREAD = 1024 / THREADS;
  float xs[SITES_PER_THREAD];
  float ys[SITES_PER_THREAD];
  float zs[SITES_PER_THREAD];
  float ws[SITES_PER_THREAD];
  float sums[SITES_PER_THREAD];
#pragma unroll
  for (int j = 0; j < SITES_PER_THREAD; ++j) {
    const int site = threadIdx.x + j * THREADS;
    const int local_row = row_tile * 16 + site / 64;
    const int local_col = site % 64;
    const int site_row = station * 64 + local_row;
    const float cosine = circle[site_row * 2];
    const float sine = circle[site_row * 2 + 1];
    const float rho = section[local_col * 3];
    xs[j] = cosine * rho;
    ys[j] = sine * rho;
    zs[j] = section[local_col * 3 + 1];
    ws[j] = section[local_col * 3 + 2];
    sums[j] = 0.0f;
  }
  for (int i = 0; i < limit; ++i) {
    const int rank = count < 0 ? i : static_cast<int>(lists[tile_index * capacity + i]);
    const int atom = groups == 1 ? begin0 + rank
        : rank < length0 ? begin0 + rank
        : rank < length0 + length1 ? begin1 + rank - length0
        : begin2 + rank - length0 - length1;
    const float amplitude = p[atom * 6];
    const float precision = p[atom * 6 + 1];
    const float cx = p[atom * 6 + 2];
    const float cy = p[atom * 6 + 3];
    const float cz = p[atom * 6 + 4];
    const float cw = p[atom * 6 + 5];
#pragma unroll
    for (int j = 0; j < SITES_PER_THREAD; ++j) {
      const float dx = xs[j] - cx;
      const float dy = ys[j] - cy;
      const float dz = zs[j] - cz;
      const float dw = ws[j] - cw;
      const float distance2 = (dx * dx + dy * dy) + (dz * dz + dw * dw);
      const float gap = fmaxf(1.0f - distance2 * precision, 0.0f);
      sums[j] += (gap * gap * gap) * amplitude;
    }
  }
#pragma unroll
  for (int j = 0; j < SITES_PER_THREAD; ++j) {
    const int site = threadIdx.x + j * THREADS;
    const int local_row = row_tile * 16 + site / 64;
    const int local_col = site % 64;
    const int output_row = (station / column_groups) * 64 + local_row - row_start;
    const int output_col = (station % column_groups) * 64 + local_col;
    output[output_row * k + output_col] = sums[j];
  }
}

void materialize_bounded_cuda(
    torch::Tensor p,
    torch::Tensor circle,
    torch::Tensor section,
    torch::Tensor lists,
    torch::Tensor counts,
    torch::Tensor offsets,
    torch::Tensor output,
    int64_t column_groups,
    int64_t groups,
    int64_t capacity,
    int64_t station_start,
    int64_t row_start,
    int64_t threads) {
  TORCH_CHECK(p.is_cuda() && circle.is_cuda() && section.is_cuda()
      && lists.is_cuda() && counts.is_cuda() && offsets.is_cuda()
      && output.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(p.scalar_type() == torch::kFloat32
      && output.scalar_type() == torch::kFloat32
      && lists.scalar_type() == torch::kUInt8
      && counts.scalar_type() == torch::kInt32
      && offsets.scalar_type() == torch::kInt64, "unexpected dtype");
  TORCH_CHECK(threads == 32 || threads == 64 || threads == 128
      || threads == 256 || threads == 512, "threads must be 32, 64, 128, 256, or 512");
  TORCH_CHECK(output.size(0) % 64 == 0, "window rows must divide station rows");
  const int blocks = static_cast<int>(output.size(0) / 64 * column_groups * 4);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&](auto thread_tag) {
    constexpr int T = decltype(thread_tag)::value;
    materialize_bounded_kernel<T><<<blocks, T, 0, stream.stream()>>>(
        p.data_ptr<float>(), circle.data_ptr<float>(), section.data_ptr<float>(),
        lists.data_ptr<uint8_t>(), counts.data_ptr<int32_t>(),
        offsets.data_ptr<int64_t>(), output.data_ptr<float>(),
        static_cast<int>(output.size(1)), static_cast<int>(column_groups),
        static_cast<int>(groups), static_cast<int>(capacity),
        static_cast<int>(station_start), static_cast<int>(row_start));
  };
  if (threads == 32) launch(std::integral_constant<int, 32>{});
  else if (threads == 64) launch(std::integral_constant<int, 64>{});
  else if (threads == 128) launch(std::integral_constant<int, 128>{});
  else if (threads == 256) launch(std::integral_constant<int, 256>{});
  else launch(std::integral_constant<int, 512>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("materialize", &materialize_bounded_cuda);
}
