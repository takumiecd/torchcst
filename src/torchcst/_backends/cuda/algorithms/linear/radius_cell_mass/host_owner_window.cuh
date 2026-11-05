#pragma once
namespace radius_cell_mass {
std::vector<Tensor> owner_window_transform(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config,
                                           bool, bool, bool, bool, bool, Tensor, Tensor, Tensor);
// mode 0: shared 17; 1: folded/saved H; 2: folded/saved windows.
void check_owner_window_saved(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config);
class OwnerWindow : public torch::autograd::Function<OwnerWindow> {
  public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x, Tensor atoms,
                          Config input, Config output, int mode, bool need_derivative) {
        auto keys = classify_atoms_async(atoms, input, output);
        auto ranges = autograd_ranges(keys, input, output, false);
        auto values = owner_window_transform(x, atoms, keys, ranges[0], ranges[1], input, output,
                                             false, need_derivative, mode != 0, mode == 2, true,
                                             Tensor(), Tensor(), Tensor());
        ctx->saved_data["input"] = input;
        ctx->saved_data["output"] = output;
        ctx->saved_data["mode"] = mode;
        ctx->save_for_backward(
            {x, atoms, mode == 2 ? values[4] : values[1], mode == 2 ? Tensor() : values[2], keys});
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
            auto x = saved[0], atoms = saved[1], dy = gradients[0].contiguous();
            auto input = ctx->saved_data["input"].toDoubleVector();
            auto output = ctx->saved_data["output"].toDoubleVector();
            int mode = ctx->saved_data["mode"].toInt();
            bool needs_x = ctx->needs_input_grad(0), needs_p = ctx->needs_input_grad(1);
            if (mode != 2)
                check_owner_derivative_saved(x, atoms, saved[2], saved[3], saved[4], dy, input,
                                             output, true, needs_p);
            else {
                check_owner_window_saved(x, atoms, saved[2], saved[4], dy, input, output);
            }
            auto ranges = autograd_ranges(saved[4], input, output, true);
            auto values = owner_window_transform(
                dy, atoms, saved[4], ranges[0], ranges[1], input, output, true, needs_p, mode != 0,
                mode == 2, needs_x, mode == 2 ? Tensor() : saved[2],
                mode == 2 ? Tensor() : saved[3], mode == 2 ? saved[2] : Tensor());
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
void bind_owner_window_autograd(pybind11::module_ &module) {
    // Research-only whole-step caller owns replay guards and host lifetimes.
    // The public dispatcher continues to use the eager-only entry below.
    module.def(
        "research_graph_owner_window",
        [](Tensor x, Tensor p, Config input, Config output) {
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6, true);
            return OwnerWindow::apply(x, p, input, output, 2, true);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def(
        "checked_autograd_linear_owner_window",
        [](Tensor x, Tensor p, Config input, Config output, int mode) {
            TORCH_CHECK(mode >= 0 && mode <= 2, "invalid owner window mode");
            check_autograd_entry(x, p, input, output, 2, 2, 32, 6);
            bool need_derivative = torch::GradMode::is_enabled() && p.requires_grad();
            return OwnerWindow::apply(x, p, input, output, mode, need_derivative);
        },
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass
