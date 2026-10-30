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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("make_guard", &make_guard); }
