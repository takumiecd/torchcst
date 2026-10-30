#include <pybind11/pybind11.h>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
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

struct ClassDefinition {
    py::object type;
    py::object mro;
};

static PyObject* none_if_null(PyObject* value) {
    return value == nullptr ? Py_None : value;
}

static py::object make_guard(py::tuple functions, py::tuple classes) {
    auto records = std::make_shared<std::vector<Definition>>();
    auto types = std::make_shared<std::vector<ClassDefinition>>();
    for (py::handle value : classes) {
        if (!PyType_Check(value.ptr())) throw py::type_error("expected classes");
        PyObject* mro = reinterpret_cast<PyTypeObject*>(value.ptr())->tp_mro;
        if (!mro || !PyTuple_CheckExact(mro)) throw py::type_error("expected MRO tuple");
        // Keep the immutable tuple alive: an identity comparison must not be
        // fooled by address reuse after __bases__ replaces a class's MRO.
        types->push_back({py::reinterpret_borrow<py::object>(value),
            py::reinterpret_borrow<py::object>(mro)});
    }
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
    return py::cpp_function([records, types](py::object) {
        // Called with the GIL. Read CPython metadata only: no Python callbacks,
        // tensor values, CUDA calls, or modification of the guarded functions.
        for (const ClassDefinition& r : *types) {
            if (reinterpret_cast<PyTypeObject*>(r.type.ptr())->tp_mro != r.mro.ptr()) {
                return false;
            }
        }
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

namespace {
#if PY_VERSION_HEX >= 0x030C0000 && !defined(Py_GIL_DISABLED)
struct DictionaryEpoch;
struct DictionaryRegistration {
    bool watched = false;
    std::vector<std::weak_ptr<DictionaryEpoch>> epochs;
};
struct DictionaryWatchState {
    bool attempted = false;
    int watcher = -1;
    PyInterpreterState* interpreter = nullptr;
    std::unordered_map<PyObject*, DictionaryRegistration> dictionaries;
};

static DictionaryWatchState& dictionary_watch_state() {
    // No Python objects are owned by this process-wide registration table.
    static DictionaryWatchState state;
    return state;
}

struct DictionaryEpoch {
    bool valid = true;
    DictionaryWatchState* state;
    std::vector<PyObject*> dictionaries;  // Addresses only; no owning references.
    explicit DictionaryEpoch(DictionaryWatchState* value) : state(value) {}
    ~DictionaryEpoch();
};

static void prune_epochs(DictionaryRegistration& registration) {
    auto& epochs = registration.epochs;
    epochs.erase(std::remove_if(epochs.begin(), epochs.end(),
        [](const std::weak_ptr<DictionaryEpoch>& epoch) { return epoch.expired(); }),
        epochs.end());
}

DictionaryEpoch::~DictionaryEpoch() {
    // cpp_function's capsule is released with the GIL. During interpreter exit
    // or unsupported ownership contexts, do not call into a dead interpreter.
    if (!Py_IsInitialized() || !PyGILState_Check()
        || PyInterpreterState_Get() != state->interpreter) return;
    PyObject *error_type, *error_value, *error_traceback;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    for (PyObject* dictionary : dictionaries) {
        auto found = state->dictionaries.find(dictionary);
        if (found == state->dictionaries.end()) continue;
        prune_epochs(found->second);
        if (!found->second.epochs.empty()) continue;
        // DEALLOCATED removes the table entry before an address can be reused.
        // An entry still present here therefore denotes a live watched dict.
        if (found->second.watched
            && PyDict_Unwatch(state->watcher, dictionary) < 0) PyErr_Clear();
        state->dictionaries.erase(found);
    }
    PyErr_Restore(error_type, error_value, error_traceback);
}

static int dictionary_changed(PyDict_WatchEvent event, PyObject* dictionary,
    PyObject*, PyObject*) {
    auto& state = dictionary_watch_state();
    auto found = state.dictionaries.find(dictionary);
    if (found == state.dictionaries.end()) return 0;
    // Watchers run before modification and may enter with a pending exception.
    // Only native bookkeeping runs here: no Python API, allocation, callback,
    // dictionary write, or reference that could resurrect a dying dictionary.
    for (const auto& reference : found->second.epochs) {
        if (auto epoch = reference.lock()) epoch->valid = false;
    }
    if (event == PyDict_EVENT_DEALLOCATED) state.dictionaries.erase(found);
    return 0;
}
#endif

static py::object make_dictionary_guard(py::tuple dictionaries) {
#if PY_VERSION_HEX >= 0x030C0000 && !defined(Py_GIL_DISABLED)
    std::unordered_set<PyObject*> unique;
    for (py::handle value : dictionaries) {
        if (!PyDict_CheckExact(value.ptr())) throw py::type_error("expected exact dictionaries");
        unique.insert(value.ptr());
    }
    if (unique.empty()) return py::cpp_function([](py::object) { return true; });
    auto& state = dictionary_watch_state();
    auto interpreter = PyInterpreterState_Get();
    if (!state.attempted) {
        state.attempted = true;
        state.interpreter = interpreter;
        state.watcher = PyDict_AddWatcher(dictionary_changed);
        // Other extensions (including Dynamo) may exhaust the finite slots.
        // Reserve at most one ID; failed registration stays a cheap fallback.
        if (state.watcher < 0) PyErr_Clear();
    }
    if (state.watcher < 0 || state.interpreter != interpreter) return py::none();
    auto epoch = std::make_shared<DictionaryEpoch>(&state);
    epoch->dictionaries.reserve(unique.size());
    for (PyObject* dictionary : unique) {
        auto& registration = state.dictionaries[dictionary];
        prune_epochs(registration);
        // Record before any allocation/watch failure so epoch destruction also
        // rolls back partially prepared registrations without retaining dicts.
        epoch->dictionaries.push_back(dictionary);
        if (!registration.watched) {
            if (PyDict_Watch(state.watcher, dictionary) < 0) {
                PyErr_Clear();
                return py::none();
            }
            registration.watched = true;
        }
        registration.epochs.push_back(epoch);
    }
    return py::cpp_function([epoch](py::object) {
        return epoch->valid
            && PyInterpreterState_Get() == epoch->state->interpreter;
    });
#else
    return py::none();
#endif
}

static py::dict dictionary_guard_stats() {
    py::dict result;
#if PY_VERSION_HEX >= 0x030C0000 && !defined(Py_GIL_DISABLED)
    auto& state = dictionary_watch_state();
    size_t registrations = 0;
    for (const auto& item : state.dictionaries) registrations += item.second.epochs.size();
    result["supported"] = true;
    result["watcher"] = state.watcher;
    result["dictionaries"] = state.dictionaries.size();
    result["registrations"] = registrations;
#else
    result["supported"] = false;
    result["watcher"] = -1;
    result["dictionaries"] = 0;
    result["registrations"] = 0;
#endif
    return result;
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_guard", &make_guard, py::arg("functions"), py::arg("classes") = py::tuple());
    m.def("make_storage_guard", &make_storage_guard);
    m.def("make_dictionary_guard", &make_dictionary_guard);
    m.def("dictionary_guard_stats", &dictionary_guard_stats);
}
