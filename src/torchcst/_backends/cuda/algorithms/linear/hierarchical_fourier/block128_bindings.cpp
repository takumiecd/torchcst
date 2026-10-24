#include <torch/extension.h>

std::vector<at::Tensor> block128_forward_cuda(at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> block128_backward_cuda(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<int64_t> block128_resources_cuda(at::Tensor);

static void check(at::Tensor x, at::Tensor p, at::Tensor fit) {
    TORCH_CHECK(x.is_cuda() && p.device()==x.device() && fit.device()==x.device(),
                "block128 requires tensors on the same CUDA device");
    TORCH_CHECK(x.scalar_type()==at::kFloat && p.scalar_type()==at::kFloat &&
                fit.scalar_type()==at::kDouble, "block128 requires FP32 X/P and FP64 fit");
    TORCH_CHECK(x.is_contiguous() && p.is_contiguous() && fit.is_contiguous(),
                "block128 requires contiguous tensors");
    TORCH_CHECK(x.dim()==2 && x.size(1)==128 && x.size(0)>=1 && x.size(0)<=64 &&
                p.dim()==2 && p.size(0)==128 && p.size(1)==4 &&
                fit.dim()==2 && fit.size(0)==32 && fit.size(1)==9,
                "block128 requires X[B1..64,128], P[128,4], fit[32,9]");
}

static std::vector<at::Tensor> forward(at::Tensor x, at::Tensor p, at::Tensor fit) {
    check(x,p,fit);
    return block128_forward_cuda(x,p,fit);
}
static std::vector<at::Tensor> backward(at::Tensor x, at::Tensor p, at::Tensor dy,
                                      at::Tensor fit, at::Tensor norm) {
    check(x,p,fit);
    TORCH_CHECK(dy.device()==x.device() && dy.scalar_type()==at::kFloat &&
                dy.is_contiguous() && dy.sizes()==x.sizes(), "invalid block128 dY");
    TORCH_CHECK(norm.device()==x.device() && norm.scalar_type()==at::kDouble &&
                norm.is_contiguous() && norm.dim()==3 && norm.size(0)==128 &&
                norm.size(1)==2 && norm.size(2)==3, "invalid block128 saved norm");
    return block128_backward_cuda(x,p,dy,fit,norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
    m.def("resource_info", &block128_resources_cuda);
}
