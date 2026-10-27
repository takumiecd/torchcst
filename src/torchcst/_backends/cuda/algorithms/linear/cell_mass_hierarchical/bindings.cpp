// Hierarchical numeric stages used by the explicit forward/backward binding.
#include <torch/extension.h>
#include <vector>

using Tensor = torch::Tensor;
using Config = std::vector<double>;

std::vector<Tensor> classify_atoms(Tensor atoms, Config input, Config output);
std::vector<Tensor> prepare_counting_layout(Tensor atoms, Config input, Config output);
std::vector<Tensor> prepare_counting_layout_signal(Tensor atoms, Tensor signal, Config input,
                                                   Config output);
std::vector<Tensor> prepare_counting_layout_reverse(Tensor atoms, Config input, Config output);
std::vector<Tensor> prepare_counting_layout_reverse_signal(Tensor atoms, Tensor signal,
                                                           Config input, Config output);
Tensor decode_groups(Tensor keys, Config input, Config output);
Tensor build_hierarchy(Tensor signal, Config axis, bool merge, int team_size);
Tensor build_hierarchy_chunked(Tensor signal, Config axis, bool merge, int team_size,
                               int chunk_sites);
Tensor build_window_moments(Tensor z, Config axis);
std::vector<Tensor> prepare_compact_layout(Tensor atoms, Config input, Config output);
std::vector<Tensor> prepare_compact_layout_signal(Tensor atoms, Tensor signal, Config input,
                                                  Config output);
Tensor compute_h(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor z, Config axis,
                 bool output_side, bool float_storage, int threads, bool windowed,
                 bool compensated);
Tensor compute_c(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor h, Config input,
                 Config output, bool reverse, bool compensated, bool compensated_cdot);
Tensor compute_c_split(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor h,
                       Config input, Config output, bool reverse, int splits, int atoms_per_split,
                       bool compensated, bool compensated_cdot);
std::vector<Tensor> compute_h_c(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor zx,
                                Config input, Config output, int splits, int atoms_per_split,
                                bool windowed, bool compensated);
std::vector<Tensor> compute_g_and_parameter_grad(Tensor atoms, Tensor rows, Tensor groups,
                                                 Tensor offsets, Tensor zx, Tensor zy, Tensor h,
                                                 Config input, Config output, int threads,
                                                 bool windowed, bool specialize_modes);
Tensor parameter_grad(Tensor atoms, Tensor rows, Tensor groups, Tensor offsets, Tensor zx,
                      Tensor zy, Tensor h, Tensor g, Config input, Config output, int threads,
                      bool windowed);
std::vector<Tensor> compute_g_parameter_grad_cin(Tensor atoms, Tensor rows, Tensor groups,
                                                 Tensor offsets, Tensor zx, Tensor zy, Tensor h,
                                                 Config input, Config output, int splits,
                                                 bool windowed);
Tensor compute_input_c_reverse(Tensor atoms, Tensor rows, Tensor reverse_slots,
                               Tensor reverse_offsets, Tensor g, Config input, int splits,
                               int atoms_per_split, bool compensated, bool compensated_cdot);
std::vector<Tensor> compute_g_parameter_grad_cin_reverse(Tensor atoms, Tensor rows, Tensor groups,
                                                         Tensor offsets, Tensor reverse_slots,
                                                         Tensor reverse_offsets, Tensor zx,
                                                         Tensor zy, Tensor h, Config input,
                                                         Config output, int splits, bool windowed,
                                                         int atoms_per_split);
Tensor expand_c(Tensor c, Tensor like, Config axis, int width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("classify_atoms", &classify_atoms);
    m.def("decode_groups", &decode_groups);
    m.def("prepare_counting_layout", &prepare_counting_layout);
    m.def("prepare_counting_layout_signal", &prepare_counting_layout_signal);
    m.def("prepare_counting_layout_reverse", &prepare_counting_layout_reverse);
    m.def("prepare_counting_layout_reverse_signal", &prepare_counting_layout_reverse_signal);
    m.def("prepare_compact_layout", &prepare_compact_layout);
    m.def("prepare_compact_layout_signal", &prepare_compact_layout_signal);
    m.def("build_hierarchy", &build_hierarchy);
    m.def("build_hierarchy_chunked", &build_hierarchy_chunked);
    m.def("build_window_moments", &build_window_moments);
    m.def("compute_h", &compute_h);
    m.def("compute_c", &compute_c);
    m.def("expand_c", &expand_c);
    m.def("compute_c_split", &compute_c_split);
    m.def("compute_h_c", &compute_h_c);
    m.def("compute_g_and_parameter_grad", &compute_g_and_parameter_grad);
    m.def("parameter_grad", &parameter_grad);
    m.def("compute_g_parameter_grad_cin", &compute_g_parameter_grad_cin);
    m.def("compute_input_c_reverse", &compute_input_c_reverse);
    m.def("compute_g_parameter_grad_cin_reverse", &compute_g_parameter_grad_cin_reverse);
}
