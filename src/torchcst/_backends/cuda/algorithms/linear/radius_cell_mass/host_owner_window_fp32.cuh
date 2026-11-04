#pragma once
namespace radius_cell_mass {
namespace fp32_body {
std::vector<Tensor> owner_window_transform(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config,
                                           bool, bool, bool, bool, bool, Tensor, Tensor, Tensor, Tensor);
std::vector<Tensor> owner_window_transform_flat_c(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config,
                                           bool, bool, bool, bool, bool, Tensor, Tensor, Tensor, Tensor);
void check_owner_window_saved(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config);
void check_scope(Tensor, Config, Config);
void check_geometry_saved(Tensor, Tensor);
void check_phase_saved(Tensor, Tensor);
std::vector<Tensor> owner_window_transform_phase(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config,
    bool, bool, bool, bool, bool, Tensor, Tensor, Tensor, Tensor, Tensor);
std::vector<Tensor> diagnose(Tensor, Tensor, Tensor, Config, Config);
} // namespace fp32_body
std::vector<Tensor> classify_atoms_geometry_async(Tensor, Config, Config);
std::vector<Tensor> classify_atoms_phase_async(Tensor, Config, Config);
std::vector<Tensor> autograd_ranges32(Tensor, Config, Config, bool);
template<bool GeometryShared = false, bool Sort32 = false, bool FlatC = false, bool PhaseShared = false>
class OwnerWindowFloat : public torch::autograd::Function<OwnerWindowFloat<GeometryShared, Sort32, FlatC, PhaseShared>> {
  public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x, Tensor atoms,
                          Config input, Config output, int mode, bool need_derivative) {
        Tensor keys, geometry, delta;
        if constexpr (GeometryShared) {
            auto classified = (PhaseShared ? classify_atoms_phase_async : classify_atoms_geometry_async)(atoms, input, output);
            if constexpr (PhaseShared) delta = classified[2];
            keys = classified[0]; geometry = classified[1];
        } else keys = classify_atoms_async(atoms, input, output);
        auto ranges = (Sort32 ? autograd_ranges32 : autograd_ranges)(keys, input, output, false);
        auto transform = [](Tensor signal, Tensor p, Tensor keys, Tensor rows, Tensor offsets,
            Config input, Config output, bool reverse, bool need_derivative, bool fold, bool recompute,
            bool emit_x, Tensor h, Tensor dh, Tensor windows, Tensor geometry, Tensor delta) {
            if constexpr (PhaseShared) return fp32_body::owner_window_transform_phase(signal, p, keys, rows, offsets, input, output, reverse, need_derivative, fold, recompute, emit_x, h, dh, windows, geometry, delta);
            else return (FlatC ? fp32_body::owner_window_transform_flat_c : fp32_body::owner_window_transform)(signal, p, keys, rows, offsets, input, output, reverse, need_derivative, fold, recompute, emit_x, h, dh, windows, geometry);
        };
        auto values = transform(
            x, atoms, keys, ranges[0], ranges[1], input, output, false, need_derivative, mode != 0,
            mode == 2, true, Tensor(), Tensor(), Tensor(), geometry, delta);
        ctx->saved_data["input"] = input;
        ctx->saved_data["output"] = output;
        ctx->saved_data["mode"] = mode;
        torch::autograd::variable_list retained =
            {x, atoms, mode == 2 ? values[4] : values[1], mode == 2 ? Tensor() : values[2], keys};
        if constexpr (GeometryShared) retained.push_back(geometry);
        if constexpr (PhaseShared) retained.push_back(delta);
        ctx->save_for_backward(retained);
        return values[0];
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list gradients) {
        const bool delayed_error =
            torch::GradMode::is_enabled() && gradients[0].defined() && gradients[0].requires_grad();
        torch::autograd::variable_list result(6);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto x = saved[0], atoms = saved[1];
            Tensor geometry, delta;
            if constexpr (GeometryShared) {
                geometry = saved[5];
                fp32_body::check_geometry_saved(atoms, geometry);
            }
            if constexpr (PhaseShared) {
                delta = saved[6];
                fp32_body::check_phase_saved(atoms, delta);
            }
            auto dy = gradients[0].contiguous();
            auto input = ctx->saved_data["input"].toDoubleVector();
            auto output = ctx->saved_data["output"].toDoubleVector();
            int mode = ctx->saved_data["mode"].toInt();
            bool needs_x = ctx->needs_input_grad(0), needs_p = ctx->needs_input_grad(1);
            if (mode != 2)
                check_owner_derivative_saved(x, atoms, saved[2], saved[3], saved[4], dy, input,
                                             output, true, needs_p);
            else {
                fp32_body::check_owner_window_saved(x, atoms, saved[2], saved[4], dy, input,
                                                    output);
            }
            auto ranges = (Sort32 ? autograd_ranges32 : autograd_ranges)(saved[4], input, output, true);
            std::vector<Tensor> values;
            if constexpr (PhaseShared) {
            values = fp32_body::owner_window_transform_phase(
                dy, atoms, saved[4], ranges[0], ranges[1], input, output, true, needs_p, mode != 0,
                mode == 2, needs_x, mode == 2 ? Tensor() : saved[2],
                mode == 2 ? Tensor() : saved[3], mode == 2 ? saved[2] : Tensor(), geometry, delta);
            } else {
            values = (FlatC ? fp32_body::owner_window_transform_flat_c : fp32_body::owner_window_transform)(
                dy, atoms, saved[4], ranges[0], ranges[1], input, output, true, needs_p, mode != 0,
                mode == 2, needs_x, mode == 2 ? Tensor() : saved[2],
                mode == 2 ? Tensor() : saved[3], mode == 2 ? saved[2] : Tensor(), geometry);
            }
            if (needs_x)
                result[0] = values[0];
            if (needs_p)
                result[1] = values[3];
        }
        if (delayed_error) {
            using NodePointer = decltype(std::declval<torch::autograd::Edge>().function);
            auto error = first_order_error<NodePointer>(result.size());
            for (auto &gradient : result) {
                if (!gradient.defined())
                    continue;
                gradient = gradient.detach();
                gradient.set_requires_grad(true);
                torch::autograd::set_history(gradient, error);
            }
        }
        return result;
    }
};
void bind_owner_window_fp32_autograd(pybind11::module_ &module) {
    module.def("research_owner_window_phase_classify",
        [](Tensor p, Config input, Config output) {
            fp32_body::check_scope(p, input, output);
            return classify_atoms_phase_async(p, input, output);
        }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "research_owner_window_fp32_diagnose",
        [](Tensor x, Tensor dy, Tensor p, Config input, Config output) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6);
            check_autograd_entry(dy, p, output, input, 2, 2, 32, 6);
            TORCH_CHECK(x.size(0) == dy.size(0), "diagnostic batches differ");
            fp32_body::check_scope(p, input, output);
            return fp32_body::diagnose(x, dy, p, input, output);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "research_owner_window_fp32",
        [](Tensor x, Tensor p, Config input, Config output, bool graph) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6, graph);
            fp32_body::check_scope(p, input, output);
            bool derivative = torch::GradMode::is_enabled() && p.requires_grad();
            return OwnerWindowFloat<>::apply(x, p, input, output, 2, derivative);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "research_owner_window_fp32_geometry",
        [](Tensor x, Tensor p, Config input, Config output, bool graph) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6, graph);
            fp32_body::check_scope(p, input, output);
            bool derivative = torch::GradMode::is_enabled() && p.requires_grad();
            return OwnerWindowFloat<true>::apply(x, p, input, output, 2, derivative);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "research_owner_window_fp32_geometry_sort32",
        [](Tensor x, Tensor p, Config input, Config output, bool graph) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6, graph);
            fp32_body::check_scope(p, input, output);
            bool derivative = torch::GradMode::is_enabled() && p.requires_grad();
            return OwnerWindowFloat<true, true>::apply(x, p, input, output, 2, derivative);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "research_owner_window_fp32_geometry_sort32_flat_c",
        [](Tensor x, Tensor p, Config input, Config output, bool graph) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6, graph);
            fp32_body::check_scope(p, input, output);
            bool derivative = torch::GradMode::is_enabled() && p.requires_grad();
            return OwnerWindowFloat<true, true, true>::apply(x, p, input, output, 2, derivative);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());

    module.def(
        "research_owner_window_fp32_phase_delta",
        [](Tensor x, Tensor p, Config input, Config output, bool graph) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6, graph);
            fp32_body::check_scope(p, input, output);
            bool derivative = torch::GradMode::is_enabled() && p.requires_grad();
            return OwnerWindowFloat<true, true, true, true>::apply(x, p, input, output, 2, derivative);
        }, pybind11::call_guard<pybind11::gil_scoped_release>());

}
} // namespace radius_cell_mass
