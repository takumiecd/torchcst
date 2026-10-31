#pragma once
// C++ only, included by host_autograd.cpp after its original implementation.
// This separate Function keeps every old mode's saved-state schema intact.
namespace radius_cell_mass {
std::vector<Tensor> owner_derivative_transform(Tensor, Tensor, Tensor, Tensor,
    Config, Config, bool, bool);
std::vector<Tensor> owner_derivative_project(Tensor, Tensor, Config);
void check_owner_derivative_saved(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Config, Config, bool, bool);
Tensor owner_derivative_parameter_grad(Tensor, Tensor, Tensor, Tensor, Tensor,
    Config, Config);

class OwnerResponseDerivatives : public torch::autograd::Function<OwnerResponseDerivatives> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x,
        Tensor atoms, Config input, Config output, bool need_derivative) {
        auto keys = classify_atoms_async(atoms, input, output);
        auto ranges = autograd_ranges(keys, input, output, false);
        auto values = owner_derivative_transform(x, atoms, ranges[0], ranges[1],
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
                auto reverse = owner_derivative_transform(dy, atoms, ranges[0], ranges[1],
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

Tensor response_derivative_autograd_linear(Tensor x, Tensor atoms, Config input,
    Config output, int input_start, int output_start, bool merge, int tile_sites) {
    // Mode7 inherits mode6's strict geometry, index and live eager validation.
    // Both raw and checked entries validate before any device classification.
    check_autograd_entry(x, atoms, input, output, input_start, output_start,
        tile_sites, 6);
    TORCH_CHECK(merge, "owner response derivatives require merge moments");
    // Function::forward runs with grad disabled, so inspect eligibility before
    // apply. Do not query needs_input_grad on a nonexecutable context.
    bool need_derivative = torch::GradMode::is_enabled() && atoms.requires_grad();
    return OwnerResponseDerivatives::apply(x, atoms, input, output, need_derivative);
}
} // namespace radius_cell_mass
