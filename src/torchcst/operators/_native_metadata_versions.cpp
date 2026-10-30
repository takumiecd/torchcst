// CPU-only epilogue for a fixed-configuration guard. Structural/type/storage
// metadata stays in the existing guard tree; this only batches numeric version
// reads that otherwise create one Python getattr result per registered buffer.
#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace py = pybind11;
template<class Pointer> struct WeakOf;
template<class T, class Null> struct WeakOf<c10::intrusive_ptr<T, Null>> {
    using type = c10::weak_intrusive_ptr<T, Null>;
};
using ImplPointer = std::decay_t<decltype(std::declval<at::Tensor>().getIntrusivePtr())>;
using ImplWeak = typename WeakOf<ImplPointer>::type;

struct VersionRecord {
    py::weakref object;
    ImplWeak impl;
    uint32_t version;
    VersionRecord(py::handle example, const at::Tensor &tensor)
        : object(example), impl(tensor.getIntrusivePtr()),
          version(tensor.unsafeGetTensorImpl()->version_counter().current_version()) {}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("make_guard", [](py::tuple examples) -> py::object {
        // Only canonical Tensor buffers with the ordinary C++ descriptor are
        // eligible. Custom accessors retain the Python version reads.
        auto tensor_type = reinterpret_cast<PyTypeObject *>(THPVariableClass);
        if (!tensor_type || !tensor_type->tp_base) return py::none();
        py::str version_name("_version"), getter_name("__getattribute__");
        PyObject *descriptor = _PyType_Lookup(tensor_type->tp_base, version_name.ptr());
        PyObject *getter = _PyType_Lookup(&PyBaseObject_Type, getter_name.ptr());
        if (!descriptor || Py_TYPE(descriptor) != &PyGetSetDescr_Type
            || _PyType_Lookup(tensor_type, version_name.ptr()) != descriptor
            || _PyType_Lookup(tensor_type, getter_name.ptr()) != getter) return py::none();
        auto records = std::make_shared<std::vector<VersionRecord>>();
        for (auto example : examples) {
            if (reinterpret_cast<PyObject *>(Py_TYPE(example.ptr())) != THPVariableClass)
                return py::none();
            const auto &tensor = THPVariable_Unpack(example.ptr());
            if (!tensor.unsafeGetTensorImpl()->version_counter().enabled()) return py::none();
            records->emplace_back(example, tensor);
        }
        // Neither examples nor Tensor storage are retained. Class accessors
        // are checked again even if the surrounding native API changes later.
        return py::cpp_function([records, version_name, getter_name,
                descriptor, getter, tensor_type](py::object) {
            if (at::impl::PythonTorchFunctionTLS::stack_len()
                || c10::impl::TorchDispatchModeTLS::any_modes_set()) return false;
            if (_PyType_Lookup(tensor_type, version_name.ptr()) != descriptor
                || _PyType_Lookup(tensor_type, getter_name.ptr()) != getter) return false;
            for (const auto &record : *records) {
                // A parent dictionary fast path may skip TensorMatch children
                // on older PyTorch. Changing __class__ does not bump the tensor
                // version, so check the live wrapper independently as well.
                py::object object = record.object();
                if (reinterpret_cast<PyObject *>(Py_TYPE(object.ptr()))
                        != THPVariableClass) return false;
                auto impl = record.impl.lock();
                if (!impl || !impl->version_counter().enabled()
                    || impl->version_counter().current_version() != record.version) return false;
            }
            return true;
        });
    });
}
