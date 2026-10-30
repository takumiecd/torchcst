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
namespace py = pybind11;
using NodePointer = decltype(std::declval<torch::autograd::Edge>().function);
template<class Pointer> struct WeakNodeType {
    using type = std::weak_ptr<torch::autograd::Node>;
};
template<class T> struct WeakNodeType<c10::intrusive_ptr<T>> {
    using type = c10::weak_intrusive_ptr<T>;
};
using NodeWeak = typename WeakNodeType<NodePointer>::type;
template<class Base, class = void> struct HasResourceRelease : std::false_type {};
template<class Base> struct HasResourceRelease<Base,
    std::void_t<decltype(std::declval<Base &>().release_resources())>> : std::true_type {};
template<class Base> void release_base_resources(Base *node) {
    if constexpr (HasResourceRelease<Base>::value) node->Base::release_resources();
}

// A callable in AtomState's reader list. It owns no model, Tensor or Node.
// Its Python owner is weak; the graph is weak too, so abandoning a graph or
// owner cannot leave a cycle. Only Python/GIL-held paths own this object.
struct RadiusReadProbe {
    py::weakref owner;
    NodeWeak node;
    bool active = true;
    RadiusReadProbe(py::object owner_, NodePointer node_)
        : owner(owner_), node(node_) {}
    py::object check() const {
        return active && !node.expired() ? py::object(py::bool_(true)) : py::none();
    }
};

// Delegate computation and SavedVariables to the standard CppNode. The outer
// Node intercepts the actual release_variables boundary, not a backward hook:
// a later user-hook exception therefore cannot release the lease prematurely.
struct RadiusLeaseNode : torch::autograd::Node {
    NodePointer inner;
    std::weak_ptr<RadiusReadProbe> probe;
    RadiusLeaseNode(NodePointer inner_, std::weak_ptr<RadiusReadProbe> probe_)
        : inner(std::move(inner_)), probe(std::move(probe_)) {
        set_next_edges(torch::autograd::edge_list(inner->next_edges()));
    }
    torch::autograd::variable_list apply(
            torch::autograd::variable_list &&inputs) override {
        return (*inner)(std::move(inputs));
    }
    void will_release_variables() override {
        if (inner) inner->will_release_variables();
    }
    void release_variables() override {
        if (inner) inner->release_variables();
        // A weak_ptr lock may become the last probe owner if Python concurrently
        // discards its AtomState. Acquire the GIL BEFORE locking, including for
        // destruction of the probe's weak Python reference.
        py::gil_scoped_acquire acquire;
        if (auto reader = probe.lock()) reader->active = false;
    }
    // On intrusive_ptr PyTorch, weak observers may outlive the last strong
    // graph reference. Drop saved numerical resources at STRONG-count zero.
    // Older shared_ptr Node versions simply use the ordinary destructor.
    void release_resources() {
        release_base_resources<torch::autograd::Node>(this);
        inner.reset();
    }
    std::string name() const override { return "RadiusLeaseNode"; }
};

template<class Pointer, class Node = RadiusLeaseNode>
auto lease_node(NodePointer inner, std::weak_ptr<RadiusReadProbe> probe) {
    if constexpr (std::is_same_v<Pointer, c10::intrusive_ptr<torch::autograd::Node>>)
        return c10::make_intrusive<Node>(std::move(inner), std::move(probe));
    else return std::make_shared<Node>(std::move(inner), std::move(probe));
}

Tensor attach_radius_lease(Tensor result, const py::object &owner,
        const py::object &qualifies) {
    if (!result.requires_grad() || owner.is_none() || qualifies.is_none()) return result;
    py::gil_scoped_acquire acquire;
    // Keep custom callbacks/modes on the original identity protection path.
    if (qualifies(py::cast(result), owner, true).is_none()) return result;
    auto readers = owner.attr("_readers");
    if (!PyList_CheckExact(readers.ptr())) return result;
    auto reader = std::make_shared<RadiusReadProbe>(owner, result.grad_fn());
    auto wrapper = lease_node<NodePointer>(result.grad_fn(), reader);
    reader->node = NodeWeak(wrapper);
    readers.attr("append")(py::cast(reader));
    torch::autograd::set_history(result, wrapper);
    return result;
}

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
std::vector<Tensor> autograd_ranges(Tensor, Config, Config, bool);
std::vector<Tensor> autograd_transform(Tensor, Tensor, Tensor, Tensor,
    Config, Config, int, int, bool, int, int, bool);
std::vector<Tensor> autograd_project(Tensor, Tensor, Config, int, bool, int);
Tensor autograd_parameter_grad(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Config, Config, int, int, int);

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
        ctx->save_for_backward({x, atoms, values[1], values[2],
            ctx->needs_input_grad(0) ? keys : Tensor()});
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
    py::class_<RadiusReadProbe, std::shared_ptr<RadiusReadProbe>>(
        module, "RadiusReadProbe", py::module_local())
        .def("__call__", &RadiusReadProbe::check);
    module.def("radius_lease_guard", [](py::object qualifies) {
        return py::cpp_function([qualifies](Tensor result, py::object owner) {
            auto node = result.grad_fn();
            auto wrapper = dynamic_cast<RadiusLeaseNode *>(node.get());
            if (!wrapper) return false;
            auto probe = wrapper->probe.lock();
            return probe && probe->active && !probe->node.expired()
                && probe->owner().is(owner)
                && !qualifies(py::cast(result), owner, true).is_none();
        });
    });
    module.def("autograd_linear_leased", [](Tensor x, Tensor atoms, Config input,
        Config output, int input_start, int output_start, bool merge,
        int tile_sites, int schedule, py::object owner, py::object qualifies) {
        TORCH_CHECK(schedule >= 0 && schedule <= 2, "invalid native host schedule");
        Tensor result;
        {
            py::gil_scoped_release release;
            result = RadiusLinear::apply(x, atoms, input, output, input_start,
                output_start, merge, tile_sites, schedule);
        }
        return attach_radius_lease(result, owner, qualifies);
    });
    module.def("autograd_linear", [](Tensor x, Tensor atoms, Config input,
        Config output, int input_start, int output_start, bool merge,
        int tile_sites, int schedule) {
        TORCH_CHECK(schedule >= 0 && schedule <= 2, "invalid native host schedule");
        return RadiusLinear::apply(x, atoms, input, output, input_start,
            output_start, merge, tile_sites, schedule);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
} // namespace radius_cell_mass
