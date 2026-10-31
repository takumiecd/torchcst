#pragma once
// Appended mode9 Function, leaving mode7 and every legacy Function literal.
namespace radius_cell_mass {
std::vector<Tensor> owner_compact_partials_transform(Tensor, Tensor, Tensor, Tensor,
    Config, Config, bool, bool);
class OwnerCompactPartials : public torch::autograd::Function<OwnerCompactPartials> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x,
        Tensor atoms, Config input, Config output, bool need_derivative) {
        auto keys = classify_atoms_async(atoms, input, output);
        auto ranges = autograd_ranges(keys, input, output, false);
        auto values = owner_compact_partials_transform(x, atoms, ranges[0], ranges[1],
            input, output, false, need_derivative);
        ctx->saved_data["input"] = input;
        ctx->saved_data["output"] = output;
        // H/D_H replace the old Zx/H pair. Zx was an internal temporary and
        // has already finished its consumers on the current stream.
        ctx->save_for_backward({x, atoms, values[1], values[2],
            x.requires_grad() ? keys : Tensor()});
        return values[0];
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        const bool delayed_error = torch::GradMode::is_enabled() &&
            gradients[0].defined() && gradients[0].requires_grad();
        torch::autograd::variable_list result(5);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto x = saved[0], atoms = saved[1];
            auto input = ctx->saved_data["input"].toDoubleVector();
            auto output = ctx->saved_data["output"].toDoubleVector();
            auto dy = gradients[0].contiguous();
            bool needs_x = ctx->needs_input_grad(0), needs_p = ctx->needs_input_grad(1);
            check_owner_derivative_saved(x, atoms, saved[2], saved[3], saved[4],
                dy, input, output, needs_x, needs_p);
            Tensor g, dg;
            if (needs_x) {
                auto ranges = autograd_ranges(saved[4], input, output, true);
                auto reverse = owner_compact_partials_transform(dy, atoms, ranges[0], ranges[1],
                    output, input, true, needs_p);
                result[0] = reverse[0];
                g = reverse[1]; dg = reverse[2];
            } else if (needs_p) {
                auto reverse = owner_derivative_project(dy, atoms, output);
                g = reverse[0]; dg = reverse[1];
            }
            // The reverse transform completed full dX and G/D_G first. No
            // projection, phase work, atomics or radius task derivative here.
            if (needs_p) result[1] = owner_derivative_parameter_grad(atoms,
                saved[2], g, saved[3], dg, input, output);
        }
        if (delayed_error) {
            using NodePointer = decltype(std::declval<torch::autograd::Edge>().function);
            auto error = first_order_error<NodePointer>(result.size());
            for (auto &gradient : result) {
                if (!gradient.defined()) continue;
                gradient = gradient.detach();
                gradient.set_requires_grad(true);
                torch::autograd::set_history(gradient, error);
            }
        }
        return result;
    }
};


void bind_owner_compact_autograd(pybind11::module_ &module) {
    module.def("checked_autograd_linear_compact_partials", [](Tensor x, Tensor p,
        Config input, Config output) {
        check_autograd_entry(x, p, input, output, 2, 2, 32, 6);
        bool need_derivative = torch::GradMode::is_enabled() && p.requires_grad();
        return OwnerCompactPartials::apply(x, p, input, output, need_derivative);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass
