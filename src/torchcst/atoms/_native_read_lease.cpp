#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/core/InferenceMode.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <vector>

namespace py = pybind11;
namespace {

// This is the same identity and saved-marker lifetime as _ReadParameter. The
// PyTorch CppNode owns SavedVariables, including release_variables on a normal
// backward and their retention for retain_graph. No CUDA operation is added.
class ReadLease : public torch::autograd::Function<ReadLease> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
            const at::Tensor& input, const at::Tensor& marker) {
        ctx->save_for_backward({marker});
        return input;
    }
    static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list gradients) {
        auto saved = ctx->get_saved_variables(); // Enforce freed-graph semantics.
        return {gradients[0], at::Tensor()};
    }
};

PyObject* or_none(PyObject* value) { return value ? value : Py_None; }

struct Definition {
    py::object function, code, defaults, keywords, globals, closure, dictionary;
    py::dict keyword_values, dictionary_values;
    py::tuple bindings, cells;

    explicit Definition(py::tuple record)
        : function(record[0]), code(record[1]), defaults(record[2]),
          keywords(record[3]), globals(record[4]), closure(record[5]),
          dictionary(record[6]), keyword_values(record[7]),
          dictionary_values(record[8]), bindings(record[9]), cells(record[10]) {}

    static bool unchanged_dict(PyObject* current, const py::dict& expected) {
        if (!PyDict_CheckExact(current) || PyDict_Size(current) != PyDict_Size(expected.ptr()))
            return false;
        for (auto item : expected) {
            if (PyDict_GetItemWithError(current, item.first.ptr()) != item.second.ptr()) {
                if (PyErr_Occurred()) PyErr_Clear();
                return false;
            }
        }
        return true;
    }

    bool check() const {
        PyObject* fn = function.ptr();
        if (!PyFunction_Check(fn)
            || PyFunction_GET_CODE(fn) != code.ptr()
            || or_none(PyFunction_GET_DEFAULTS(fn)) != defaults.ptr()
            || or_none(PyFunction_GET_KW_DEFAULTS(fn)) != keywords.ptr()
            || PyFunction_GET_GLOBALS(fn) != globals.ptr()
            || or_none(PyFunction_GET_CLOSURE(fn)) != closure.ptr()
            || reinterpret_cast<PyFunctionObject*>(fn)->func_dict != dictionary.ptr())
            return false;
        if (!keywords.is_none() && !unchanged_dict(keywords.ptr(), keyword_values))
            return false;
        if (!unchanged_dict(dictionary.ptr(), dictionary_values)) return false;
        for (auto item : bindings) {
            auto pair = py::reinterpret_borrow<py::tuple>(item);
            PyObject* value = PyDict_GetItemWithError(globals.ptr(), pair[0].ptr());
            bool present = pair[1].ptr() == Py_True;
            if ((value != nullptr) != present || or_none(value) != pair[2].ptr()) {
                if (PyErr_Occurred()) PyErr_Clear();
                return false;
            }
        }
        if (!closure.is_none()) {
            if (PyTuple_GET_SIZE(closure.ptr()) != PyTuple_GET_SIZE(cells.ptr())) return false;
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(cells.ptr()); ++i) {
                if (or_none(PyCell_GET(PyTuple_GET_ITEM(closure.ptr(), i)))
                    != PyTuple_GET_ITEM(cells.ptr(), i)) return false;
            }
        }
        return true;
    }
};

struct Lookup {
    py::object owner, name, expected, owner_type, mro;
    explicit Lookup(py::tuple record)
        : owner(record[0]), name(record[1]), expected(record[2]),
          owner_type(py::type::of(owner)), mro(record[3]) {}
    bool check() const {
        if (reinterpret_cast<PyObject*>(Py_TYPE(owner.ptr())) != owner_type.ptr()) return false;
        PyObject* value = nullptr;
        if (PyType_Check(owner.ptr())) {
            if (reinterpret_cast<PyTypeObject*>(owner.ptr())->tp_mro != mro.ptr()) return false;
            value = _PyType_Lookup(reinterpret_cast<PyTypeObject*>(owner.ptr()), name.ptr());
        } else if (PyModule_CheckExact(owner.ptr()))
            value = PyDict_GetItemWithError(PyModule_GetDict(owner.ptr()), name.ptr());
        else return false;
        if (PyErr_Occurred()) { PyErr_Clear(); return false; }
        return or_none(value) == expected.ptr();
    }
};

py::object make_protector(py::object owner_type, py::object tensor_type,
        py::object parameter_type, py::tuple definitions, py::tuple lookups,
        py::object functorch_active, py::object wrapped_tensor,
        py::object tracing_state, py::object capturing) {
    auto functions = std::make_shared<std::vector<Definition>>();
    auto attributes = std::make_shared<std::vector<Lookup>>();
    for (auto record : definitions) functions->emplace_back(py::cast<py::tuple>(record));
    for (auto record : lookups) attributes->emplace_back(py::cast<py::tuple>(record));
    py::object mro = owner_type.attr("__mro__");
    py::object read_name = py::str("_readers");
    return py::cpp_function([=](py::object object, py::object state,
            bool eligible_only) -> py::object {
        // Check capabilities before invoking any replaced callback or unpacking
        // a Tensor subclass. Every rejection returns to the original Python
        // autograd Function in the same protect_tensor invocation.
        if (reinterpret_cast<PyObject*>(Py_TYPE(state.ptr())) != owner_type.ptr()
            || reinterpret_cast<PyTypeObject*>(owner_type.ptr())->tp_mro != mro.ptr())
            return py::none();
        if (reinterpret_cast<PyObject*>(Py_TYPE(object.ptr())) != tensor_type.ptr()
            && reinterpret_cast<PyObject*>(Py_TYPE(object.ptr())) != parameter_type.ptr())
            return py::none();
        if (!THPVariable_Check(object.ptr())) return py::none();
        for (const auto& record : *attributes) if (!record.check()) return py::none();
        for (const auto& record : *functions) if (!record.check()) return py::none();
        if (c10::InferenceMode::is_enabled()
            || at::impl::PythonTorchFunctionTLS::stack_len()
            || c10::impl::TorchDispatchModeTLS::any_modes_set()
            || py::cast<bool>(functorch_active())
            || py::cast<bool>(wrapped_tensor(object))
            || !tracing_state().is_none()) return py::none();
        const auto& input = THPVariable_Unpack(object.ptr());
        if (input.is_cuda() && py::cast<bool>(capturing())) return py::none();
        py::object readers = state.attr("_readers");
        if (!PyList_CheckExact(readers.ptr())) return py::none();
        if (eligible_only) return py::bool_(true);

        auto marker = at::empty({0}, at::TensorOptions().device(at::kCPU));
        // Keep this wrapper alive through apply/save_variables_to_ctx. The
        // caller's identity saved-tensor hooks then retain this same object.
        py::object marker_object = py::cast(marker);
        readers.attr("append")(py::weakref(marker_object));
        auto result = ReadLease::apply(input, marker);
        return py::cast(result);
    }, py::arg("tensor"), py::arg("state"), py::arg("eligible_only") = false);
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_protector", &make_protector);
}
