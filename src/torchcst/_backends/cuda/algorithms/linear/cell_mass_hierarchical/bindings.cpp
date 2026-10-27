// Hierarchical preparation only; no old fixed-bank executor dependency.
#include <torch/extension.h>
#include <vector>

using Tensor = torch::Tensor;
using Config = std::vector<double>;

std::vector<Tensor> classify_atoms(Tensor atoms, Config input, Config output);
Tensor decode_groups(Tensor keys, Config input, Config output);
Tensor build_hierarchy(Tensor signal, Config axis, bool merge, int team_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("classify_atoms", &classify_atoms);
    m.def("decode_groups", &decode_groups);
    m.def("build_hierarchy", &build_hierarchy);
}
