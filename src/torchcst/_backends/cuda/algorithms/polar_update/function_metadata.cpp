#include <pybind11/pybind11.h>
#include <memory>
#include <vector>
namespace py = pybind11;
namespace {

// Optional, non-limited CPython API. Build against the running interpreter's
// headers. Python qualification keeps unsupported/free-threaded builds on the
// existing metadata guard; no device state is accessed by this helper.

struct Definition {
    py::object function;
    py::object code;
    py::object defaults;
    py::object keywords;
    py::object globals;
    py::object closure;
    py::object dictionary;
    std::vector<py::object> cell_values;
};

static PyObject* none_if_null(PyObject* value) {
    return value == nullptr ? Py_None : value;
}

static py::object make_guard(py::tuple functions) {
    auto records = std::make_shared<std::vector<Definition>>();
    records->reserve(functions.size());
    for (py::handle value : functions) {
        PyObject* fn = value.ptr();
        if (!PyFunction_Check(fn)) throw py::type_error("expected Python functions");
        Definition r;
        r.function = py::reinterpret_borrow<py::object>(fn);
        r.dictionary = py::getattr(value, "__dict__");
        r.code = py::reinterpret_borrow<py::object>(PyFunction_GET_CODE(fn));
        r.defaults = py::reinterpret_borrow<py::object>(
            none_if_null(PyFunction_GET_DEFAULTS(fn)));
        r.keywords = py::reinterpret_borrow<py::object>(
            none_if_null(PyFunction_GET_KW_DEFAULTS(fn)));
        r.globals = py::reinterpret_borrow<py::object>(PyFunction_GET_GLOBALS(fn));
        PyObject* closure = PyFunction_GET_CLOSURE(fn);
        r.closure = py::reinterpret_borrow<py::object>(none_if_null(closure));
        if (closure) {
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(closure); ++i) {
                PyObject* cell = PyTuple_GET_ITEM(closure, i);
                PyObject* contents = PyCell_GET(cell);
                if (!contents) throw py::value_error("empty closure cell");
                r.cell_values.push_back(py::reinterpret_borrow<py::object>(contents));
            }
        }
        records->push_back(std::move(r));
    }
    return py::cpp_function([records](py::object) {
        // Called with the GIL. Read CPython metadata only: no Python callbacks,
        // tensor values, CUDA calls, or modification of the guarded functions.
        for (const Definition& r : *records) {
            PyObject* fn = r.function.ptr();
            if (!PyFunction_Check(fn)
                || reinterpret_cast<PyFunctionObject*>(fn)->func_dict != r.dictionary.ptr()
                || PyFunction_GET_CODE(fn) != r.code.ptr()
                || none_if_null(PyFunction_GET_DEFAULTS(fn)) != r.defaults.ptr()
                || none_if_null(PyFunction_GET_KW_DEFAULTS(fn)) != r.keywords.ptr()
                || PyFunction_GET_GLOBALS(fn) != r.globals.ptr()
                || none_if_null(PyFunction_GET_CLOSURE(fn)) != r.closure.ptr()) {
                return false;
            }
            if (!r.closure.is_none()) {
                PyObject* closure = r.closure.ptr();
                for (size_t i = 0; i < r.cell_values.size(); ++i) {
                    PyObject* current = PyCell_GET(PyTuple_GET_ITEM(closure, i));
                    if (current != r.cell_values[i].ptr()) {
                        return false;
                    }
                }
            }
        }
        return true;
    });
}
} // namespace

namespace {
struct Storage {
    py::object reference;  // Weak: a guard must not keep the example Tensor alive.
    py::object address;
    py::object descriptor;
    getattrofunc getter;
    PyCFunction method;
};

static py::object make_storage_guard(py::tuple references, py::tuple methods) {
    if (references.size() != methods.size()) throw py::value_error("storage arity");
    auto records = std::make_shared<std::vector<Storage>>();
    py::str name("data_ptr");
    for (size_t i = 0; i < references.size(); ++i) {
        PyObject* ref = references[i].ptr();
        PyObject* method = methods[i].ptr();
        if (!PyWeakref_CheckRef(ref) || !PyCFunction_Check(method)
            || PyCFunction_GetFlags(method) != METH_NOARGS
            || PyCFunction_GetSelf(method) != PyWeakref_GetObject(ref)) {
            throw py::type_error("expected weak Tensor references and builtin data_ptr");
        }
        // Only the qualified builtin is called. No custom Python accessor runs
        // inside this checker. data_ptr reads host metadata, never Tensor values.
        PyCFunction fn = PyCFunction_GetFunction(method);
        py::object address = py::reinterpret_steal<py::object>(
            fn(PyCFunction_GetSelf(method), nullptr));
        if (!address) throw py::error_already_set();
        if (!PyLong_CheckExact(address.ptr())) throw py::type_error("expected address");
        PyObject* value = PyWeakref_GetObject(ref);
        PyObject* descriptor = _PyType_Lookup(Py_TYPE(value), name.ptr());
        if (!descriptor) throw py::type_error("missing data_ptr descriptor");
        records->push_back({py::reinterpret_borrow<py::object>(ref), address,
            py::reinterpret_borrow<py::object>(descriptor), Py_TYPE(value)->tp_getattro, fn});
    }
    return py::cpp_function([records, name](py::object) {
        for (const Storage& r : *records) {
            PyObject* value = PyWeakref_GetObject(r.reference.ptr());
            if (value == Py_None) return false;
            // Look up the descriptor without invoking it. A newly installed
            // property/custom __getattribute__ must invalidate, never execute.
            if (Py_TYPE(value)->tp_getattro != r.getter
                || _PyType_Lookup(Py_TYPE(value), name.ptr()) != r.descriptor.ptr()) return false;
            py::object dictionary = py::reinterpret_steal<py::object>(
                PyObject_GenericGetDict(value, nullptr));
            if (!dictionary) { PyErr_Clear(); return false; }
            int overridden = PyDict_Contains(dictionary.ptr(), name.ptr());
            if (overridden < 0) { PyErr_Clear(); return false; }
            if (overridden) return false;
            py::object address = py::reinterpret_steal<py::object>(r.method(value, nullptr));
            if (!address) { PyErr_Clear(); return false; }
            int same = PyLong_CheckExact(address.ptr())
                ? PyObject_RichCompareBool(address.ptr(), r.address.ptr(), Py_EQ) : 0;
            if (same < 0) { PyErr_Clear(); return false; }
            if (!same) return false;
        }
        return true;
    });
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_guard", &make_guard);
    m.def("make_storage_guard", &make_storage_guard);
}
