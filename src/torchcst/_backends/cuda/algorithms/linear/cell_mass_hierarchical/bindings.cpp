// Hierarchical numeric stages used by the explicit forward/backward binding.
#include <torch/extension.h>
#include <vector>

using Tensor = torch::Tensor;
using Config = std::vector<double>;

std::vector<Tensor> classify_atoms(Tensor atoms, Config input, Config output);
std::vector<Tensor> prepare_counting_layout(Tensor atoms, Config input, Config output);
Tensor decode_groups(Tensor keys, Config input, Config output);
Tensor build_hierarchy(Tensor signal, Config axis, bool merge, int team_size);
Tensor build_hierarchy_chunked(Tensor signal, Config axis, bool merge, int team_size,
                               int chunk_sites);
Tensor compute_h(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor z, Config axis,
                 bool output_side, bool float_storage, int threads);
Tensor compute_c(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor h, Config input,
                 Config output, bool reverse);
Tensor compute_c_split(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor h,
                       Config input, Config output, bool reverse, int splits);
std::vector<Tensor> compute_g_and_parameter_grad(Tensor atoms, Tensor rows, Tensor groups,
                                                 Tensor offsets, Tensor zx, Tensor zy, Tensor h,
                                                 Config input, Config output, int threads);
Tensor parameter_grad(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor zx,
                      Tensor zy, Tensor h, Tensor g, Config input, Config output, int threads);
Tensor expand_c(Tensor c, Tensor like, Config axis, int width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("classify_atoms", &classify_atoms);
    m.def("decode_groups", &decode_groups);
    m.def("prepare_counting_layout", &prepare_counting_layout);
    m.def("build_hierarchy", &build_hierarchy);
    m.def("build_hierarchy_chunked", &build_hierarchy_chunked);
    m.def("compute_h", &compute_h);
    m.def("compute_c", &compute_c);
    m.def("expand_c", &expand_c);
    m.def("compute_c_split", &compute_c_split);
    m.def("compute_g_and_parameter_grad", &compute_g_and_parameter_grad);
    m.def("parameter_grad", &parameter_grad);
}
