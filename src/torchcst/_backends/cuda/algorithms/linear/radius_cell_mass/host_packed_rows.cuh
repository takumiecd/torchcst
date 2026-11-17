#pragma once
namespace radius_cell_mass {
void packed_check_scope(Tensor, Tensor, Config, Config);
#ifdef RADIUS_LM_WIDTH
bool width_qkv_cohort(double, double, int64_t);
Tensor packed_initialization_energies(Tensor, Tensor, Config);

void check_dense_fourier_cohort(const Tensor &p, const Config &ci, const Config &co) {
    constexpr const char *message =
        "dense Fourier weights require an admitted FFN down or width-2048 QKV cohort";
    TORCH_CHECK(!ci.empty() && !co.empty() && p.dim() == 2, message);
    TORCH_CHECK(
        (ci[0] == 4096 && co[0] == 1024 && p.size(0) == 196608) ||
        width_qkv_cohort(ci[0], co[0], p.size(0)), message);
}
#endif
Tensor packed_classify(Tensor, Config, Config, float = 0);
void packed_check_tensor(Tensor, Tensor, std::vector<int64_t>);
Tensor packed_make_geometry(Tensor, Config, Config);
std::vector<Tensor> packed_project(Tensor, Tensor, Config, bool, Tensor = Tensor(), bool = false);
std::vector<Tensor> packed_transform(Tensor, Tensor, Tensor, Tensor, Config, Config, bool, Tensor = Tensor(), bool = false, bool = false);
Tensor packed_parameter_grad(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Config, Config, bool, Tensor = Tensor());

class PackedRows : public torch::autograd::Function<PackedRows> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x, Tensor p,
        Config ci, Config co, bool recompute, bool parallel, bool cache_geometry, bool level_launch, bool shared_wide) {
        auto keys = packed_classify(p, ci, co);
        auto ranges = autograd_ranges(keys, ci, co, false);
        Tensor geometry = cache_geometry ? packed_make_geometry(p, ci, co) : Tensor();
        auto values = packed_transform(x, p, ranges[0], ranges[1], ci, co, false, geometry, level_launch, shared_wide);
        ctx->saved_data["input"] = ci; ctx->saved_data["output"] = co;
        ctx->saved_data["levels"] = level_launch;
        ctx->saved_data["shared_wide"] = shared_wide;
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
        torch::autograd::variable_list out(9);
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
                auto reverse = packed_transform(dy, p, ranges[0], ranges[1], co, ci, true, saved[5], levels, ctx->saved_data["shared_wide"].toBool());
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
#ifdef RADIUS_LM_WIDTH
Tensor packed_dense_cotangent_grad(Tensor, Tensor, Tensor, Tensor, Config, Config);
#endif

#ifdef RADIUS_LM_WIDTH
Tensor packed_make_dense_weight(Tensor, Tensor, Config, Config);
Tensor packed_dense_weight_grad(Tensor, Tensor, Tensor, Config, Config);

class PackedDenseTileWeight : public torch::autograd::Function<PackedDenseTileWeight> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor p,
        Config ci, Config co) {
        packed_classify(p, ci, co, 512.f);
        auto geometry = packed_make_geometry(p, ci, co);
        ctx->saved_data["input"] = ci;
        ctx->saved_data["output"] = co;
        ctx->save_for_backward({p, geometry});
        return packed_make_dense_weight(p, geometry, ci, co);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        const bool delayed = torch::GradMode::is_enabled() && gradients[0].defined() &&
            gradients[0].requires_grad();
        torch::autograd::variable_list out(3);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto ci = ctx->saved_data["input"].toDoubleVector(), co = ctx->saved_data["output"].toDoubleVector();
            auto dense = gradients[0].contiguous();
            packed_check_tensor(dense, saved[0], {int64_t(co[0]), int64_t(ci[0])});
            out[0] = packed_dense_weight_grad(saved[0], dense, saved[1], ci, co);
        }
        if (delayed) {
            using NodePointer = decltype(std::declval<torch::autograd::Edge>().function);
            auto error = first_order_error<NodePointer>(out.size());
            out[0] = out[0].detach();
            out[0].set_requires_grad(true);
            torch::autograd::set_history(out[0], error);
        }
        return out;
    }
};

class PackedDenseTiles : public torch::autograd::Function<PackedDenseTiles> {
public:
    static Tensor forward(torch::autograd::AutogradContext *ctx, Tensor x, Tensor p,
        Config ci, Config co) {
        packed_classify(p, ci, co, 512.f);
        auto geometry = packed_make_geometry(p, ci, co);
        auto weight = packed_make_dense_weight(p, geometry, ci, co);
        ctx->saved_data["input"] = ci;
        ctx->saved_data["output"] = co;
        ctx->save_for_backward({x, p, p.requires_grad() ? geometry : Tensor(),
            x.requires_grad() ? weight : Tensor()});
        return at::mm(x, weight.t());
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list gradients) {
        const bool delayed = torch::GradMode::is_enabled() && gradients[0].defined() &&
            gradients[0].requires_grad();
        torch::autograd::variable_list out(4);
        {
            torch::NoGradGuard no_grad;
            auto saved = ctx->get_saved_variables();
            auto x = saved[0], p = saved[1], dy = gradients[0].contiguous();
            auto ci = ctx->saved_data["input"].toDoubleVector(), co = ctx->saved_data["output"].toDoubleVector();
            TORCH_CHECK(!at::globalContext().allowTF32CuBLAS(), "dense tiles require IEEE FP32 matmul");
            if (ctx->needs_input_grad(0)) out[0] = at::mm(dy, saved[3]);
            if (ctx->needs_input_grad(1)) out[1] = packed_dense_cotangent_grad(p, x, dy, saved[2], ci, co);
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
#endif

void bind_packed_rows(pybind11::module_ &module) {
    module.def("packed_rows_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedRows::apply(x, p, ci, co, recompute, parallel, false, false, false);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_geometry_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedRows::apply(x, p, ci, co, recompute, parallel, true, false, false);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_levels_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        return PackedRows::apply(x, p, ci, co, recompute, parallel, true, true, false);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
#ifdef RADIUS_LM_WIDTH
    module.def("packed_shared_wide_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool parallel) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        TORCH_CHECK(ci[0] == 4096 && co[0] == 1024 && p.size(0) == 196608,
            "shared wide tiles require the admitted 4096-to-1024 FFN down cohort");
        return PackedRows::apply(x, p, ci, co, recompute, parallel, true, true, true);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_dense_fourier_weight_linear", [](Tensor x, Tensor p, Config ci, Config co,
        bool recompute, bool) {
        check_autograd_entry(x, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(x, p, ci, co);
        check_dense_fourier_cohort(p, ci, co);
        TORCH_CHECK(recompute && !at::globalContext().allowTF32CuBLAS(),
            "dense tiles require IEEE FP32 matmul and no retained responses");
        return PackedDenseTiles::apply(x, p, ci, co);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_dense_fourier_weight_from_atoms", [](Tensor p, Config ci, Config co) {
        check_dense_fourier_cohort(p, ci, co);
        auto shape_check = torch::empty({1, int64_t(ci[0])}, p.options());
        check_autograd_entry(shape_check, p, ci, co, 2, 2, 32, 5);
        packed_check_scope(shape_check, p, ci, co);
        TORCH_CHECK(!at::globalContext().allowTF32CuBLAS(), "dense tiles require IEEE FP32 matmul");
        return PackedDenseTileWeight::apply(p, ci, co);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("packed_initialization_energies", [](Tensor p, Tensor banks, Config axis) {
        TORCH_CHECK(!axis.empty() && p.dim() == 2 && width_qkv_cohort(axis[0], axis[0], p.size(0)),
            "initial energies require a width-2048 QKV cohort");
        TORCH_CHECK(banks.device() == p.device() && banks.scalar_type() == torch::kInt32 &&
            banks.dim() == 1 && banks.size(0) == p.size(0) && banks.is_contiguous(),
            "initial energies require contiguous int32 banks on the Atom device");
        auto input = torch::empty({1, int64_t(axis[0])}, p.options());
        check_autograd_entry(input, p, axis, axis, 2, 2, 32, 5);
        packed_check_scope(input, p, axis, axis);
        return packed_initialization_energies(p, banks, axis);
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
#endif
}
} // namespace radius_cell_mass
