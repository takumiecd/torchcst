#pragma once
// Separate mode4 Function: all primary modes and their saved schemas stay literal.
namespace radius_cell_mass {
void check_local_tile_entry(Tensor, Tensor, Config, Config, int, int, int, int);
std::vector<Tensor> local_tile_forward(Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config);
std::vector<Tensor> local_tile_backward(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Config, Config, bool, bool);
class LocalTileLinear : public torch::autograd::Function<LocalTileLinear> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x,
        Tensor atoms, Config input, Config output) {
        auto keys = classify_atoms_async(atoms, input, output);
        auto ranges = autograd_ranges(keys, input, output, false);
        auto values = local_tile_forward(x, atoms, keys, ranges[0], ranges[1], input, output);
        ctx->saved_data["input"] = input; ctx->saved_data["output"] = output;
        // Unlike modes5..9, atom-only mode4 needs owner keys for its gradient
        // grouping. Ordinary SavedVariables preserve offload/version/retention.
        // No executable edges under no_grad/inference: the context is discarded.
        ctx->save_for_backward({x, atoms, values[1], values[2],
            (x.requires_grad() || atoms.requires_grad()) ? keys : Tensor()});
        return values[0];
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        const bool delayed_error = torch::GradMode::is_enabled() &&
            gradients[0].defined() && gradients[0].requires_grad();
        torch::autograd::variable_list result(4);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto input = ctx->saved_data["input"].toDoubleVector();
            auto output = ctx->saved_data["output"].toDoubleVector();
            auto values = local_tile_backward(saved[0], saved[1], saved[2], saved[3],
                saved[4], gradients[0].contiguous(), input, output,
                ctx->needs_input_grad(0), ctx->needs_input_grad(1));
            result[0] = values[0]; result[1] = values[1];
        }
        if (delayed_error) {
            using NodePointer = decltype(std::declval<torch::autograd::Edge>().function);
            auto error = first_order_error<NodePointer>(result.size());
            for (auto &gradient : result) {
                if (!gradient.defined()) continue;
                gradient = gradient.detach(); gradient.set_requires_grad(true);
                torch::autograd::set_history(gradient, error);
            }
        }
        return result;
    }
};
void bind_local_tile_autograd(pybind11::module_ &module) {
    module.def("checked_autograd_linear_local_tile", [](Tensor x, Tensor atoms,
        Config input, Config output, int input_start, int output_start,
        bool merge, int tile_sites) {
        // Original mode4 preflight, including the larger legal A4003..4096
        // local-owner range, before classification or any parameter work.
        check_local_tile_entry(x, atoms, input, output, input_start, output_start, tile_sites, 4);
        TORCH_CHECK(merge, "local tile moments require merge moments");
        return LocalTileLinear::apply(x, atoms, input, output);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass
