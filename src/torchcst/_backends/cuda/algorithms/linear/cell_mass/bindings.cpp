// Numeric research entry points; no public dispatcher registration yet.
#include <torch/extension.h>
#include <vector>

using Tensor = torch::Tensor;
using Config = std::vector<double>;

Tensor build_z(Tensor signal, Config config, bool hierarchical, int team_size);
Tensor compute_h(Tensor x, Tensor atoms, Tensor rows, Tensor z, Config config);
Tensor compute_y(Tensor h, Tensor atoms, Tensor rows, Config config, int sites_per_block);
Tensor compute_g(Tensor dy, Tensor atoms, Tensor rows, Tensor z, Config config);
Tensor compute_dx(Tensor g, Tensor atoms, Tensor rows, Config config, int sites_per_block);
Tensor compute_parameter_grad(Tensor x, Tensor dy, Tensor atoms, Tensor rows, Tensor h, Tensor g,
                              Tensor zx, Tensor zy, Config input_config, Config output_config);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_z", &build_z, pybind11::arg("signal"), pybind11::arg("config"),
          pybind11::arg("hierarchical") = false, pybind11::arg("team_size") = 4);
    m.def("compute_h", &compute_h);
    m.def("compute_y", &compute_y, pybind11::arg("h"), pybind11::arg("atoms"),
          pybind11::arg("rows"), pybind11::arg("config"), pybind11::arg("sites_per_block") = 32);
    m.def("compute_g", &compute_g);
    m.def("compute_dx", &compute_dx, pybind11::arg("g"), pybind11::arg("atoms"),
          pybind11::arg("rows"), pybind11::arg("config"), pybind11::arg("sites_per_block") = 32);
    m.def("compute_parameter_grad", &compute_parameter_grad);
}
