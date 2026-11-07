#pragma once
// Recompute in 64-row windows; retain global eight-row gradient partials only.
// Reduction order is the same as packed_gradient_reduce over the original R.
namespace radius_cell_mass {
void packed_gradient_partial_into(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, int64_t, Config, Config);
Tensor packed_finish_gradient(Tensor, Tensor, Config);

class PackedStream64 : public torch::autograd::Function<PackedStream64> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x, Tensor p,
        Config ci, Config co, bool level_launch) {
        auto keys = packed_classify(p, ci, co);
        auto ranges = autograd_ranges(keys, ci, co, false);
        auto geometry = packed_make_geometry(p, ci, co);
        auto y = torch::empty({x.size(0), int64_t(co[0])}, x.options());
        for (int64_t first = 0; first < x.size(0); first += 64) {
            int64_t count = std::min(int64_t(64), x.size(0) - first);
            auto values = packed_transform(x.narrow(0, first, count), p,
                ranges[0], ranges[1], ci, co, false, geometry, level_launch);
            y.narrow(0, first, count).copy_(values[0]);
        }
        ctx->saved_data["levels"] = level_launch;
        ctx->saved_data["input"] = ci; ctx->saved_data["output"] = co;
        ctx->save_for_backward({x, p, keys, geometry});
        return y;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        const bool delayed = torch::GradMode::is_enabled() && gradients[0].defined() && gradients[0].requires_grad();
        torch::autograd::variable_list out(5);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto x = saved[0], p = saved[1], dy = gradients[0].contiguous(), geometry = saved[3];
            auto ci = ctx->saved_data["input"].toDoubleVector(), co = ctx->saved_data["output"].toDoubleVector();
            packed_check_scope(x, p, ci, co);
            TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == x.scalar_type() &&
                dy.dim() == 2 && dy.size(0) == x.size(0) && dy.size(1) == int(co[0]), "invalid streamed dY");
            bool levels = ctx->saved_data["levels"].toBool();
            std::vector<Tensor> ranges;
            if (ctx->needs_input_grad(0)) {
                out[0] = torch::empty_like(x);
                ranges = autograd_ranges(saved[2], ci, co, true);
            }
            Tensor partial;
            if (ctx->needs_input_grad(1))
                partial = torch::empty({(x.size(0) + 7) / 8, p.size(0), 3}, p.options());
            for (int64_t first = 0; first < x.size(0); first += 64) {
                int64_t count = std::min(int64_t(64), x.size(0) - first);
                auto xx = x.narrow(0, first, count), dd = dy.narrow(0, first, count);
                Tensor zy, g;
                if (ctx->needs_input_grad(0)) {
                    auto reverse = packed_transform(dd, p, ranges[0], ranges[1], co, ci, true, geometry, levels);
                    out[0].narrow(0, first, count).copy_(reverse[0]);
                    zy = reverse[1]; g = reverse[2];
                } else if (ctx->needs_input_grad(1)) {
                    auto reverse = packed_project(dd, p, co, true, geometry, levels);
                    zy = reverse[0]; g = reverse[1];
                }
                if (ctx->needs_input_grad(1)) {
                    auto forward = packed_project(xx, p, ci, false, geometry, levels);
                    packed_gradient_partial_into(p, xx, dd, forward[0], zy, forward[1], g,
                        geometry, partial, first / 8, ci, co);
                }
            }
            if (ctx->needs_input_grad(1)) out[1] = packed_finish_gradient(p, partial, ci);
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
void bind_packed_stream(pybind11::module_ &module) {
    module.def("packed_stream64_linear", [](Tensor x, Tensor p, Config ci, Config co) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedStream64::apply(x, p, ci, co, false);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_levels_stream64_linear", [](Tensor x, Tensor p, Config ci, Config co) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedStream64::apply(x, p, ci, co, true);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass
