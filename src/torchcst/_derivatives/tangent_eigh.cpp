#include <torch/extension.h>
#include <ATen/record_function.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cusolverDn.h>

std::vector<torch::Tensor> eigh_unchecked(torch::Tensor input) {
  RECORD_FUNCTION("torchcst::cusolver_eigh", std::vector<c10::IValue>({input}));
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat64,
              "eigh_unchecked requires CUDA float64");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == input.size(1), "matrix must be square");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto a = input.contiguous().clone();
  int n = a.size(0), work_size = 0;
  auto values = torch::empty({n}, a.options());
  auto info = torch::empty({}, a.options().dtype(torch::kInt32));
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto status = cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER, n, a.data_ptr<double>(), n,
      values.data_ptr<double>(), &work_size);
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, "cuSOLVER workspace query failed");
  auto work = torch::empty({work_size}, a.options());
  status = cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, a.data_ptr<double>(), n, values.data_ptr<double>(), work.data_ptr<double>(),
      work_size, info.data_ptr<int>());
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, "cuSOLVER launch failed");
  // cuSOLVER writes column-major eigenvectors. No info read on the host.
  return {values, a.transpose(0, 1), info};
}
