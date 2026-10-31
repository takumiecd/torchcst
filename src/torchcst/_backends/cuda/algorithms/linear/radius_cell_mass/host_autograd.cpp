// The host-side dependency graph for the existing CUDA stages. No arithmetic
// kernels live here: projection, routing, expansion and gradients are the same
// implementations used by the Python autograd reference.
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <type_traits>

namespace radius_cell_mass {
using Tensor = torch::Tensor;
using Config = std::vector<double>;

// PyTorch changed Node ownership from shared_ptr to intrusive_ptr. Select the
// matching constructor from the installed header instead of pinning a version.
template<class Pointer, class Error = torch::autograd::DelayedError>
auto first_order_error(int outputs) {
    const char *message = "trying to differentiate twice a function that was "
        "marked with @once_differentiable";
    if constexpr (std::is_same_v<Pointer,
        c10::intrusive_ptr<torch::autograd::Node>>) {
        return c10::make_intrusive<Error>(message, outputs);
    } else {
        return std::make_shared<Error>(message, outputs);
    }
}

// Defined in pipeline.cu so this C++ translation unit never includes CUDA
// kernel syntax or device headers. This also keeps NVCC out of autograd.h.
Tensor classify_atoms_async(Tensor, Config, Config);
void check_autograd_entry(Tensor, Tensor, Config, Config, int, int, int, int);
std::vector<Tensor> autograd_ranges(Tensor, Config, Config, bool);
std::vector<Tensor> autograd_transform(Tensor, Tensor, Tensor, Tensor,
    Config, Config, int, int, bool, int, int, bool);
std::vector<Tensor> autograd_project(Tensor, Tensor, Config, int, bool, int);
Tensor autograd_parameter_grad(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Config, Config, int, int, int);
void autograd_check_owner_saved(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Config, Config, bool);
Tensor response_derivative_autograd_linear(Tensor, Tensor, Config, Config,
    int, int, bool, int);

class RadiusLinear : public torch::autograd::Function<RadiusLinear> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x,
        Tensor atoms, Config input, Config output, int input_start,
        int output_start, bool merge, int tile_sites, int schedule) {
        // Classification follows the CURRENT atoms on the device. Only the
        // original-row owner keys are retained, and only when dX needs them.
        auto keys = classify_atoms_async(atoms, input, output);
        auto ranges = autograd_ranges(keys, input, output, false);
        auto values = autograd_transform(x, atoms, ranges[0], ranges[1],
            input, output, input_start, output_start, merge, tile_sites,
            schedule, false);
        ctx->saved_data["input"] = input;
        ctx->saved_data["output"] = output;
        ctx->saved_data["input_start"] = input_start;
        ctx->saved_data["output_start"] = output_start;
        ctx->saved_data["merge"] = merge;
        ctx->saved_data["tile_sites"] = tile_sites;
        ctx->saved_data["schedule"] = schedule;
        // Ordinary SavedVariables preserve offload hooks, version checking,
        // retain_graph and release on normal backward. No global tensor cache.
        // Function::apply has no next_edges under no_grad/inference or when
        // every input is frozen, so needs_input_grad(0) is not queryable here.
        // Input requires_grad is equivalent when executable; otherwise this
        // context is discarded without committing any SavedVariables.
        ctx->save_for_backward({x, atoms, values[1], values[2],
            x.requires_grad() ? keys : Tensor()});
        return values[0];
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        // Match the Python reference's @once_differentiable. The first-order
        // CUDA kernels do not implement a differentiable second derivative.
        const bool delayed_error = torch::GradMode::is_enabled() &&
            gradients[0].defined() && gradients[0].requires_grad();
        torch::autograd::variable_list result(9);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto x = saved[0], atoms = saved[1];
            auto input = ctx->saved_data["input"].toDoubleVector();
            auto output = ctx->saved_data["output"].toDoubleVector();
            const int input_start = ctx->saved_data["input_start"].toInt();
            const int output_start = ctx->saved_data["output_start"].toInt();
            const bool merge = ctx->saved_data["merge"].toBool();
            const int tile_sites = ctx->saved_data["tile_sites"].toInt();
            const int schedule = ctx->saved_data["schedule"].toInt();
            auto dy = gradients[0].contiguous();
            if (schedule == 6) autograd_check_owner_saved(x, atoms, saved[2], saved[3],
                saved[4], dy, input, output, ctx->needs_input_grad(0));
            Tensor output_moments, grad_response;
            if (ctx->needs_input_grad(0)) {
                auto ranges = autograd_ranges(saved[4], input, output, true);
                // dY -> output moments/G -> dX, once. Coefficients stay shared.
                auto reverse = autograd_transform(dy, atoms, ranges[0], ranges[1],
                    input, output, input_start, output_start, merge, tile_sites,
                    schedule, true);
                result[0] = reverse[0];
                output_moments = reverse[1];
                grad_response = reverse[2];
            } else if (ctx->needs_input_grad(1)) {
                // Atom-only backward needs G/moments but no dX/routing sort.
                auto reverse = autograd_project(dy, atoms, output, output_start,
                    merge, schedule);
                output_moments = reverse[0];
                grad_response = reverse[1];
            }
            if (ctx->needs_input_grad(1)) {
                result[1] = autograd_parameter_grad(atoms, x, dy, saved[2],
                    output_moments, saved[3], grad_response, input, output,
                    input_start, output_start, schedule);
            }
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

void bind_host_autograd(pybind11::module_ &module) {
    module.def("checked_autograd_linear", [](Tensor x, Tensor atoms, Config input,
        Config output, int input_start, int output_start, bool merge,
        int tile_sites, int schedule) {
        if (schedule == 7) return response_derivative_autograd_linear(x, atoms,
            input, output, input_start, output_start, merge, tile_sites);
        check_autograd_entry(x, atoms, input, output, input_start,
            output_start, tile_sites, schedule);
        TORCH_CHECK(schedule != 6 || merge, "owner response coefficients require merge moments");
        return RadiusLinear::apply(x, atoms, input, output, input_start,
            output_start, merge, tile_sites, schedule);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("autograd_linear", [](Tensor x, Tensor atoms, Config input,
        Config output, int input_start, int output_start, bool merge,
        int tile_sites, int schedule) {
        if (schedule == 7) return response_derivative_autograd_linear(x, atoms,
            input, output, input_start, output_start, merge, tile_sites);
        TORCH_CHECK((schedule >= 0 && schedule <= 3) || schedule == 5 || schedule == 6,
            "invalid native host schedule");
        // Unlike the old shared modes, the new raw global entry always checks
        // general launch/configuration limits before device classification.
        if (schedule == 5 || schedule == 6) check_autograd_entry(x, atoms, input, output,
            input_start, output_start, tile_sites, schedule);
        TORCH_CHECK(schedule != 6 || merge, "owner response coefficients require merge moments");
        return RadiusLinear::apply(x, atoms, input, output, input_start,
            output_start, merge, tile_sites, schedule);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass

#include "host_response_derivatives.cuh"

#include "host_compact_partials.cuh"
