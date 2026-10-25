#include <torch/extension.h>

std::vector<at::Tensor> block128_forward_cuda(at::Tensor, at::Tensor, at::Tensor, bool);
std::vector<at::Tensor> block128_backward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                               at::Tensor, bool);
std::vector<int64_t> block128_resources_cuda(at::Tensor);
std::vector<at::Tensor> block128_prepare_layout_cuda(at::Tensor);
std::vector<at::Tensor> block128_forward_sorted_cuda(at::Tensor, at::Tensor, at::Tensor, bool);
std::vector<at::Tensor> block128_backward_sorted_cuda(at::Tensor, at::Tensor, at::Tensor,
                                                      at::Tensor, at::Tensor, at::Tensor,
                                                      at::Tensor, at::Tensor, bool);
std::vector<int64_t> block128_sorted_resources_cuda(at::Tensor);

// Pythonから来たTensorのshape・dtype・deviceを検査する。
// 数値の分類やソートはここでは行わず、CUDAのprepare_atom_layoutへ任せる。
static void check(at::Tensor x, at::Tensor p, at::Tensor fit) {
    TORCH_CHECK(x.is_cuda() && p.device() == x.device() && fit.device() == x.device(),
                "block128 requires tensors on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && p.scalar_type() == at::kFloat &&
                    fit.scalar_type() == at::kDouble,
                "block128 requires FP32 X/P and FP64 fit");
    TORCH_CHECK(x.is_contiguous() && p.is_contiguous() && fit.is_contiguous(),
                "block128 requires contiguous tensors");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 128 && x.size(0) >= 1 && x.size(0) <= 64 &&
                    p.dim() == 2 && p.size(0) == 128 && p.size(1) == 4 && fit.dim() == 2 &&
                    fit.size(0) == 32 && fit.size(1) == 9,
                "block128 requires X[B1..64,128], P[128,4], fit[32,9]");
}

static std::vector<at::Tensor> prepare_layout(at::Tensor parameters) {
    TORCH_CHECK(parameters.is_cuda() && parameters.scalar_type() == at::kFloat &&
                    parameters.is_contiguous() && parameters.dim() == 2 &&
                    parameters.size(0) == 128 && parameters.size(1) == 4,
                "layout requires contiguous CUDA FP32 parameters[128,4]");
    return block128_prepare_layout_cuda(parameters);
}

static std::vector<at::Tensor> forward_sorted(at::Tensor x, at::Tensor parameters, at::Tensor fit,
                                              bool profile) {
    check(x, parameters, fit);
    return block128_forward_sorted_cuda(x, parameters, fit, profile);
}

static std::vector<at::Tensor> backward_sorted(at::Tensor x, at::Tensor sorted_parameters,
                                               at::Tensor dy, at::Tensor fit, at::Tensor norm,
                                               at::Tensor source_rows, at::Tensor atom_groups,
                                               at::Tensor ranges, bool profile) {
    check(x, sorted_parameters, fit);

    TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == at::kFloat && dy.is_contiguous() &&
                    dy.sizes() == x.sizes(),
                "invalid block128 dY");
    TORCH_CHECK(norm.device() == x.device() && norm.scalar_type() == at::kDouble &&
                    norm.is_contiguous() && norm.dim() == 3 && norm.size(0) == 128 &&
                    norm.size(1) == 2 && norm.size(2) == 3,
                "invalid block128 saved norm");

    // これらはforwardが作り、autogradが同じ呼出しのbackwardまで保存した配列。
    TORCH_CHECK(source_rows.device() == x.device() && source_rows.scalar_type() == at::kLong &&
                    source_rows.is_contiguous() && source_rows.dim() == 1 &&
                    source_rows.size(0) == 128,
                "invalid block128 source rows");
    TORCH_CHECK(atom_groups.device() == x.device() && atom_groups.scalar_type() == at::kInt &&
                    atom_groups.is_contiguous() && atom_groups.dim() == 1 &&
                    atom_groups.size(0) == 128,
                "invalid block128 atom groups");
    TORCH_CHECK(ranges.device() == x.device() && ranges.scalar_type() == at::kInt &&
                    ranges.is_contiguous() && ranges.dim() == 4 && ranges.size(0) == 2 &&
                    ranges.size(1) == 4 && ranges.size(2) == 12 && ranges.size(3) == 2,
                "invalid block128 ranges");

    return block128_backward_sorted_cuda(x, sorted_parameters, dy, fit, norm, source_rows,
                                         atom_groups, ranges, profile);
}

static std::vector<at::Tensor> forward(at::Tensor x, at::Tensor p, at::Tensor fit, bool profile) {
    check(x, p, fit);
    return block128_forward_cuda(x, p, fit, profile);
}

static std::vector<at::Tensor> backward(at::Tensor x, at::Tensor p, at::Tensor dy, at::Tensor fit,
                                        at::Tensor norm, bool profile) {
    check(x, p, fit);
    TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == at::kFloat && dy.is_contiguous() &&
                    dy.sizes() == x.sizes(),
                "invalid block128 dY");
    TORCH_CHECK(norm.device() == x.device() && norm.scalar_type() == at::kDouble &&
                    norm.is_contiguous() && norm.dim() == 3 && norm.size(0) == 128 &&
                    norm.size(1) == 2 && norm.size(2) == 3,
                "invalid block128 saved norm");
    return block128_backward_cuda(x, p, dy, fit, norm, profile);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, pybind11::arg("x"), pybind11::arg("p"), pybind11::arg("fit"),
          pybind11::arg("profile") = false);
    m.def("backward", &backward, pybind11::arg("x"), pybind11::arg("p"), pybind11::arg("dy"),
          pybind11::arg("fit"), pybind11::arg("norm"), pybind11::arg("profile") = false);
    m.def("resource_info", &block128_resources_cuda);
    m.def("prepare_layout", &prepare_layout, pybind11::arg("parameters"));
    m.def("forward_sorted", &forward_sorted, pybind11::arg("x"), pybind11::arg("parameters"),
          pybind11::arg("fit"), pybind11::arg("profile") = false);
    m.def("backward_sorted", &backward_sorted, pybind11::arg("x"),
          pybind11::arg("sorted_parameters"), pybind11::arg("dy"), pybind11::arg("fit"),
          pybind11::arg("norm"), pybind11::arg("source_rows"), pybind11::arg("atom_groups"),
          pybind11::arg("ranges"), pybind11::arg("profile") = false);
    m.def("sorted_resource_info", &block128_sorted_resources_cuda);
}
