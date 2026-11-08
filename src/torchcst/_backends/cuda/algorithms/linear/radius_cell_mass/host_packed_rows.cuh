#pragma once
namespace radius_cell_mass {
void packed_check_scope(Tensor, Tensor, Config, Config);
Tensor packed_classify(Tensor, Config, Config);
void packed_check_tensor(Tensor, Tensor, std::vector<int64_t>);
Tensor packed_make_geometry(Tensor, Config, Config);
std::vector<Tensor> packed_project(Tensor, Tensor, Config, bool, Tensor = Tensor(), bool = false);
std::vector<Tensor> packed_transform(Tensor, Tensor, Tensor, Tensor, Config, Config, bool, Tensor = Tensor(), bool = false);
Tensor packed_parameter_grad(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config, bool, Tensor = Tensor());

class PackedRows : public torch::autograd::Function<PackedRows> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x, Tensor p,
        Config ci, Config co, bool recompute, bool parallel, bool cache_geometry, bool level_launch) {
        auto keys = packed_classify(p, ci, co);
        auto ranges = autograd_ranges(keys, ci, co, false);
        Tensor geometry = cache_geometry ? packed_make_geometry(p, ci, co) : Tensor();
        auto values = packed_transform(x, p, ranges[0], ranges[1], ci, co, false, geometry, level_launch);
        ctx->saved_data["input"] = ci; ctx->saved_data["output"] = co;
        ctx->saved_data["levels"] = level_launch;
        ctx->saved_data["recompute"] = recompute; ctx->saved_data["parallel"] = parallel;
        ctx->save_for_backward({x, p,
            !recompute && p.requires_grad() ? values[1] : Tensor(),
            !recompute && p.requires_grad() ? values[2] : Tensor(),
            x.requires_grad() ? keys : Tensor(), p.requires_grad() || x.requires_grad() ? geometry : Tensor()});
        return values[0];
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        const bool delayed = torch::GradMode::is_enabled() && gradients[0].defined() && gradients[0].requires_grad();
        torch::autograd::variable_list out(8);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto x = saved[0], p = saved[1], dy = gradients[0].contiguous();
            auto ci = ctx->saved_data["input"].toDoubleVector(), co = ctx->saved_data["output"].toDoubleVector();
            packed_check_scope(x, p, ci, co);
            if (saved[5].defined()) packed_check_tensor(saved[5], p, {p.size(0), 25});
            TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == x.scalar_type() &&
                dy.dim() == 2 && dy.size(0) == x.size(0) && dy.size(1) == int(co[0]), "invalid packed dY");
            bool levels = ctx->saved_data["levels"].toBool();
            Tensor zy, g;
            if (ctx->needs_input_grad(0)) {
                auto ranges = autograd_ranges(saved[4], ci, co, true);
                auto reverse = packed_transform(dy, p, ranges[0], ranges[1], co, ci, true, saved[5], levels);
                out[0] = reverse[0]; zy = reverse[1]; g = reverse[2];
            } else if (ctx->needs_input_grad(1)) {
                auto reverse = packed_project(dy, p, co, true, saved[5], levels); zy = reverse[0]; g = reverse[1];
            }
            if (ctx->needs_input_grad(1)) {
                Tensor zx = saved[2], h = saved[3];
                if (ctx->saved_data["recompute"].toBool()) {
                    auto forward = packed_project(x, p, ci, false, saved[5], levels); zx = forward[0]; h = forward[1];
                }
                out[1] = packed_parameter_grad(p, x, dy, zx, zy, h, g, ci, co,
                    ctx->saved_data["parallel"].toBool(), saved[5]);
            }
        }
        if (delayed) {
            using NodePointer = decltype(std::declval<torch::autograd::Edge>().function);
            auto error = first_order_error<NodePointer>(out.size());
            for (auto &gradient : out) if (gradient.defined()) {
                gradient = gradient.detach(); gradient.set_requires_grad(true);
                torch::autograd::set_history(gradient, error);
            }
        }
        return out;
    }
};
void bind_packed_rows(pybind11::module_ &module) {
    module.def("packed_rows_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedRows::apply(x, p, ci, co, recompute, parallel, false, false);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_geometry_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedRows::apply(x, p, ci, co, recompute, parallel, true, false);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_levels_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedRows::apply(x, p, ci, co, recompute, parallel, true, true);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass
