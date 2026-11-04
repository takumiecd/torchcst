"""Finite stock qualification for the fixed whole-step research runner.

Import this module in the trusted worker bootstrap, before constructing models,
optimizers, or accepting user callbacks.  The fixed Torch wheel/source and clean
startup are the trust boundary; this cannot authenticate an earlier monkeypatch.
This is shared safety checking for G0/G1, not a public Graph execution API.

Only the explicit slots below and their declared decorator chains are examined.
Native Torch/autograd implementations and Dynamo's private disable machinery are
runtime boundaries, not a recursively authenticated Python namespace. Existing
state/group/hook/TLS/native CST eligibility checks remain required separately.
"""

from __future__ import annotations

import builtins
import dis
import gc
import inspect
import operator
import weakref
from time import perf_counter
from types import CodeType, FunctionType, MethodType

_BOOTSTRAP_START = perf_counter()

import torch
from torch import nn
from torch.nn import _reduction
from torch.nn import functional as F
from torch.optim import Adam, AdamW, Optimizer

from torchcst.atoms import Atoms, AtomState
from torchcst.atoms.optimizer_state import AtomOptimizerState
from torchcst.charts.regular_grid import RegularGridChartState
from torchcst.geometry.state import GeometryState
from torchcst.kernels.state import KernelState, ProfileState
from torchcst.nn import CSTLinear
from torchcst.optim import CSTAdamW
from torchcst.patterns.state import PatternState


class QualificationError(ValueError):
    """The fixed stock cohort cannot be established without running callbacks."""


_MISSING = object()
_GET_DICT = object.__getattribute__
_GET_STATIC = inspect.getattr_static
_PRIMITIVES = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    type(NotImplemented),
    type(Ellipsis),
)


class _Reference:
    """Remember identity without owning a runtime object or bound method."""

    def __init__(self, value):
        self.strong = (
            value if type(value) in _PRIMITIVES or value is _MISSING else _MISSING
        )
        self.identity = id(value)
        self.weak = None
        if self.strong is _MISSING and value is not _MISSING:
            try:
                self.weak = weakref.ref(value)
            except TypeError:
                # Only immutable definition metadata is permitted here.  In
                # particular never hold a registration dict or arbitrary list.
                definition = type(value) in (staticmethod, classmethod, property) or (
                    inspect.ismethoddescriptor(value)
                    or inspect.isgetsetdescriptor(value)
                )
                if not definition and (
                    type(value) is not tuple
                    or any(type(v) not in _PRIMITIVES for v in value)
                ):
                    raise QualificationError("unsupported stock metadata")
                self.strong = value

    def matches(self, value):
        if self.weak is not None:
            return self.weak() is value
        return self.strong is value


class _Function:
    """Metadata of a bootstrap-certified definition, never a cold user closure."""

    def __init__(self, fn):
        self.fn = fn  # Only clean-start library definitions are admitted.
        self.code = fn.__code__
        self.defaults = _Reference(fn.__defaults__)
        self.keywords_id = id(fn.__kwdefaults__)
        self.keywords = (
            None
            if fn.__kwdefaults__ is None
            else tuple(
                (key, _Reference(value)) for key, value in fn.__kwdefaults__.items()
            )
        )
        self.wrapped = _Reference(getattr(fn, "__wrapped__", _MISSING))
        self.cells = tuple(
            _Reference(cell.cell_contents) for cell in fn.__closure__ or ()
        )
        names = _global_names(fn.__code__)
        globals_ = []
        for name in names:
            if name in fn.__globals__:
                try:
                    value = fn.__globals__[name]
                    if fn.__module__ == "torchcst.nn.linear" and name == "_PLANS":
                        # One explicit library-definition map, not a namespace
                        # walker. Never retain the map or its later contents.
                        reference = _PlanMap(value)
                    else:
                        reference = _Reference(value)
                    globals_.append((name, reference))
                except QualificationError as error:
                    raise QualificationError(
                        f"unsupported global {fn.__qualname__}.{name}"
                    ) from error
        self.globals = tuple(globals_)
        self.builtins = tuple(
            (name, _Reference(getattr(builtins, name)))
            for name in names
            if name not in fn.__globals__ and hasattr(builtins, name)
        )

    def check(self):
        fn = self.fn
        if (
            fn.__code__ is not self.code
            or not self.defaults.matches(fn.__defaults__)
            or id(fn.__kwdefaults__) != self.keywords_id
            or not self.wrapped.matches(getattr(fn, "__wrapped__", _MISSING))
            or len(fn.__closure__ or ()) != len(self.cells)
            or any(
                not ref.matches(cell.cell_contents)
                for ref, cell in zip(self.cells, fn.__closure__ or ())
            )
            or any(
                not ref.matches(fn.__globals__.get(name, _MISSING))
                for name, ref in self.globals
            )
            or any(
                name in fn.__globals__
                or not ref.matches(getattr(builtins, name, _MISSING))
                for name, ref in self.builtins
            )
        ):
            return False
        return self.keywords is None or (
            tuple(fn.__kwdefaults__) == tuple(key for key, _ in self.keywords)
            and all(ref.matches(fn.__kwdefaults__[key]) for key, ref in self.keywords)
        )


_FUNCTIONS = {}


class _PlanMap:
    def __init__(self, value):
        if type(value) is not dict or any(type(key) is not str for key in value):
            raise QualificationError("canonical CST plan definitions required")
        self.identity = id(value)
        self.entries = tuple((key, _Reference(item)) for key, item in value.items())

    def matches(self, value):
        return (
            type(value) is dict
            and id(value) == self.identity
            and tuple(value) == tuple(key for key, _ in self.entries)
            and all(ref.matches(value[key]) for key, ref in self.entries)
        )


def _global_names(code):
    # co_names also contains attribute names (e.g. self.__dict__). Inspect only
    # actual global loads, including lexically nested code in this definition.
    names = {
        op.argval
        for op in dis.get_instructions(code)
        if op.opname in ("LOAD_GLOBAL", "LOAD_NAME")
    }
    for constant in code.co_consts:
        if type(constant) is CodeType:
            names.update(_global_names(constant))
    return tuple(sorted(names))


def _definition(fn):
    """Follow only the finite catalog's declared wrapper/closure definitions."""
    if type(fn) is not FunctionType or fn in _FUNCTIONS:
        return
    _FUNCTIONS[fn] = _Function(fn)
    _definition(getattr(fn, "__wrapped__", None))
    for cell in fn.__closure__ or ():
        value = cell.cell_contents
        if type(value) is FunctionType:
            _definition(value)
        elif type(value) is MethodType:
            # Do not retain the method or its context-manager owner.
            _definition(value.__func__)


class _Slot:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name  # Library modules/classes only.
        value = _GET_STATIC(owner, name, _MISSING)
        self.expected = _Reference(value)
        # These are library definition namespaces, never a model registration
        # dictionary. Class MROs are checked before this hot path. Ordinary
        # class/module namespaces cannot be rebound by attribute assignment.
        self.namespaces = (
            tuple(type.__getattribute__(base, "__dict__") for base in owner.__mro__)
            if isinstance(owner, type)
            else (vars(owner),)
        )
        if type(value) in (staticmethod, classmethod):
            value = value.__func__
        _definition(value)

    def check(self):
        for namespace in self.namespaces:
            value = namespace.get(self.name, _MISSING)
            if value is not _MISSING:
                return self.expected.matches(value)
        return self.expected.matches(_MISSING)


# Materialize exactly the stock lazy caches that our runner uses. This is the
# normal function-only factory operation at trusted bootstrap, before user code.
# A preexisting cache has no such provenance: fail qualification, never bless or
# replace it. Do not parse arbitrary Dynamo contexts/closures at prepare time.
_BOOTSTRAP_SUPPORTED = True
_DYNAMO_CACHES = []
for _outer in (Optimizer.zero_grad,):
    _cells = dict(zip(_outer.__code__.co_freevars, _outer.__closure__ or ()))
    _inner = getattr(_outer, "__wrapped__", None)
    if (
        type(_inner) is not FunctionType
        or tuple(_cells) != ("fn", "recursive")
        or _cells["fn"].cell_contents is not _inner
        or type(_cells["recursive"].cell_contents) is not bool
        or hasattr(_inner, "__dynamo_disable")
    ):
        _BOOTSTRAP_SUPPORTED = False
        continue
    import torch._dynamo

    _disabled = torch._dynamo.disable(
        _inner, _cells["recursive"].cell_contents, wrapping=False
    )
    _inner.__dynamo_disable = _disabled
    _DYNAMO_CACHES.append((_inner, _disabled))
    _definition(_disabled)
del _outer, _inner, _cells


# The only Module types admitted by this radius/regular-grid research runner.
_MODULE_TYPES = (
    nn.Sequential,
    nn.ModuleList,
    nn.Linear,
    nn.ReLU,
    CSTLinear,
    Atoms,
    AtomState,
    AtomOptimizerState,
    RegularGridChartState,
    GeometryState,
    PatternState,
    KernelState,
    ProfileState,
)
_MODULE_NAMES = (
    "__getattribute__",
    "__getattr__",
    "__dict__",
    "__call__",
    "_wrapped_call_impl",
    "_call_impl",
    "_slow_forward",
    "forward",
    "_compiled_call_impl",
    "named_parameters",
    "named_buffers",
    "modules",
    "named_modules",
    "_named_members",
    "children",
    "named_children",
    "_modules",
    "_parameters",
    "_buffers",
    "training",
)
_OPTIMIZER_NAMES = (
    "__getattribute__",
    "__getattr__",
    "__dict__",
    "zero_grad",
    "_init_group",
    "_adamw_proposal",
    "_optimizer_step_code",
    "_patch_step_function",
    "profile_hook_step",
)
_OPTIMIZER_TYPES = (Optimizer, Adam, AdamW, CSTAdamW)
_MROS = tuple((cls, cls.__mro__) for cls in (*_MODULE_TYPES, *_OPTIMIZER_TYPES))
_SLOTS = tuple(
    _Slot(cls, name)
    for cls, names in (
        *((cls, _MODULE_NAMES) for cls in _MODULE_TYPES),
        *((cls, _OPTIMIZER_NAMES) for cls in _OPTIMIZER_TYPES),
        (nn.Sequential, ("__iter__", "__getitem__", "_get_item_by_idx")),
        (nn.ModuleList, ("__iter__", "__getitem__", "_get_abs_string_index")),
        (operator, ("index",)),
        (
            F,
            (
                "linear",
                "relu",
                "cross_entropy",
                "has_torch_function_unary",
                "has_torch_function_variadic",
                "handle_torch_function",
                "_Reduction",
                "torch",
            ),
        ),
        (_reduction, ("get_enum", "legacy_get_string", "legacy_get_enum")),
        (torch, ("relu", "relu_", "_C")),
        (torch._C, ("_nn", "_get_tracing_state")),
        (torch._C._nn, ("linear", "cross_entropy_loss")),
    )
    for name in names
)

# A function-only template: no Optimizer or Tensor examples, class mutation,
# sealing, or extension build.  Each class may resolve an inherited raw anchor.
_RAW_STEPS = {cls: _GET_STATIC(cls, "step") for cls in _OPTIMIZER_TYPES}
if any(getattr(_raw, "hooked", False) for _raw in _RAW_STEPS.values()):
    # Already initialized optimizer classes are outside clean bootstrap.
    _BOOTSTRAP_SUPPORTED = False
for _raw in _RAW_STEPS.values():
    _definition(_raw)
_PROFILE_FACTORY = Optimizer.profile_hook_step
_PROFILE_TEMPLATE = _PROFILE_FACTORY(_RAW_STEPS[AdamW])
_PROFILE_CODE = _PROFILE_TEMPLATE.__code__
_PROFILE_GLOBALS = _PROFILE_TEMPLATE.__globals__
_definition(_PROFILE_FACTORY)
del _PROFILE_TEMPLATE, _raw


def _stock_step(cls):
    value = _GET_STATIC(cls, "step", _MISSING)
    raw = _RAW_STEPS[cls]
    if value is raw:
        hooked = getattr(raw, "hooked", _MISSING)
        return hooked is _MISSING or hooked is False
    return (
        type(value) is FunctionType
        and value.__code__ is _PROFILE_CODE
        and value.__globals__ is _PROFILE_GLOBALS
        and value.__defaults__ is None
        and value.__kwdefaults__ is None
        and value.__code__.co_freevars == ("func",)
        and len(value.__closure__ or ()) == 1
        and value.__closure__[0].cell_contents is raw
        and getattr(value, "__wrapped__", None) is raw
        and getattr(value, "hooked", None) is True
    )


def _catalog_check():
    try:
        return (
            _BOOTSTRAP_SUPPORTED
            and all(cls.__mro__ == mro for cls, mro in _MROS)
            and all(slot.check() for slot in _SLOTS)
            and all(record.check() for record in _FUNCTIONS.values())
            and all(_stock_step(cls) for cls in _OPTIMIZER_TYPES)
            and all(
                getattr(fn, "__dynamo_disable", _MISSING) is cached
                for fn, cached in _DYNAMO_CACHES
            )
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def _raw_dict(value):
    result = _GET_DICT(value, "__dict__")
    if type(result) is not dict:
        raise QualificationError("stock instance dictionary required")
    return result


def _module_check(node):
    if type(node) not in _MODULE_TYPES:
        return False
    data = _raw_dict(node)
    return (
        type(node) in _MODULE_TYPES
        and type(data.get("_parameters")) is dict
        and type(data.get("_buffers")) is dict
        and all(
            type(key) is str and (value is None or type(value) is nn.Parameter)
            for key, value in data["_parameters"].items()
        )
        and all(
            type(key) is str and (value is None or type(value) is torch.Tensor)
            for key, value in data["_buffers"].items()
        )
        and not any(
            name in data
            for name in _MODULE_NAMES
            if name
            not in (
                "_modules",
                "_parameters",
                "_buffers",
                "training",
                "_compiled_call_impl",
            )
        )
        and data.get("_compiled_call_impl") is None
        and (type(node) is not nn.ReLU or data.get("inplace") is False)
    )


def _prepare_native_catalog():
    """Batch this exact finite predicate with existing optional host helpers.

    No model/optimizer/registration dictionary is passed into a native guard.
    Unsupported private APIs return the identical Python fallback. This does
    not authenticate new functions: the complete bootstrap predicate is checked
    before snapshotting and after construction.
    """
    from torchcst._backends.cuda import _host_function_metadata as metadata
    from torchcst.operators._native_metadata_guard import guard_api

    factory_start = perf_counter()
    factory = metadata.get_factory()
    factory_seconds = perf_counter() - factory_start
    api = guard_api()
    if factory is None or api is None:
        return None, factory_seconds
    if not _catalog_check():
        raise QualificationError("stock definitions changed during helper preparation")
    try:
        functions = tuple(_FUNCTIONS)
        classes = tuple(cls for cls, _ in _MROS)
        function_check, _ = metadata.prepare_checks(
            factory, functions, classes, with_class_mro=True
        )
        if function_check is None:
            return None, factory_seconds
        cls, kind = api
        guard = cls()
        verbose = ["finite whole-step stock qualification"]
        owners = []
        seen = set()
        for slot in _SLOTS:
            values = (
                slot.owner.__mro__ if isinstance(slot.owner, type) else (slot.owner,)
            )
            for owner in values:
                if id(owner) not in seen:
                    owners.append(owner)
                    seen.add(id(owner))
        owners.append(builtins)
        roots = tuple(owners) + functions
        guard.add_length_check_guard(len(roots), verbose, None)
        managers = {}
        for index, owner in enumerate(roots):
            node = guard.tuple_getitem_manager(index, "library definition", owner, kind)
            node.add_id_match_guard(id(owner), verbose, None)
            managers[id(owner)] = node

        def attr(node, name, value):
            return node.getattr_manager(name, name, value, kind)

        def item(node, key, value):
            return node.dict_getitem_manager(key, str(key), value, kind)

        def identity(node, value):
            node.add_id_match_guard(id(value), verbose, None)

        def field(namespace, name, value):
            if value is _MISSING:
                namespace.add_dict_contains_guard(False, name, verbose, None)
            else:
                identity(item(namespace, name, value), value)

        # Class definition dictionaries contain library declarations only.
        # Version guards are conservative about unrelated class mutations, but
        # do not retain any old model registration dictionaries.
        for owner in owners:
            if not isinstance(owner, type) or owner.__flags__ & (1 << 8):
                continue
            raw = next(
                value
                for value in gc.get_referents(owner.__dict__)
                if type(value) is dict
            )
            namespace = managers[id(owner)].type_dict_manager(
                "stock class namespace", raw, kind
            )
            namespace.add_dict_version_guard(raw, verbose, None)
        for slot in _SLOTS:
            if not isinstance(slot.owner, type):
                namespace = attr(managers[id(slot.owner)], "__dict__", vars(slot.owner))
                field(namespace, slot.name, vars(slot.owner).get(slot.name, _MISSING))

        builtin_namespace = attr(managers[id(builtins)], "__dict__", vars(builtins))
        for fn, record in _FUNCTIONS.items():
            node = managers[id(fn)]
            namespace = attr(node, "__globals__", fn.__globals__)
            for name, reference in record.globals:
                value = fn.__globals__[name]
                child = item(namespace, name, value)
                identity(child, value)
                if isinstance(reference, _PlanMap):
                    child.add_dict_version_guard(value, verbose, None)
            for name, _ in record.builtins:
                namespace.add_dict_contains_guard(False, name, verbose, None)
                field(builtin_namespace, name, getattr(builtins, name))
            dictionary = attr(node, "__dict__", fn.__dict__)
            field(dictionary, "__wrapped__", getattr(fn, "__wrapped__", _MISSING))
            if fn.__kwdefaults__ is not None:
                keywords = attr(node, "__kwdefaults__", fn.__kwdefaults__)
                keywords.add_dict_version_guard(fn.__kwdefaults__, verbose, None)
        for fn, cached in _DYNAMO_CACHES:
            dictionary = attr(managers[id(fn)], "__dict__", fn.__dict__)
            field(dictionary, "__dynamo_disable", cached)

        # Profiling wrappers are constructed by Optimizer.__init__, not ours.
        # Never pass them into the strong function checker. Access exact fields
        # through their library class slot, preserving the constructor's stock
        # certification without retaining a runtime function example.
        for owner, raw in _RAW_STEPS.items():
            value = _GET_STATIC(owner, "step")
            node = attr(managers[id(owner)], "step", value)
            identity(node, value)
            dictionary = attr(node, "__dict__", value.__dict__)
            field(dictionary, "hooked", getattr(value, "hooked", _MISSING))
            if value is raw:
                continue
            for name, expected in (
                ("__code__", _PROFILE_CODE),
                ("__globals__", _PROFILE_GLOBALS),
                ("__defaults__", None),
                ("__kwdefaults__", None),
            ):
                identity(attr(node, name, expected), expected)
            field(dictionary, "__wrapped__", raw)
            closure = attr(node, "__closure__", value.__closure__)
            closure.add_length_check_guard(1, verbose, None)
            cell = closure.tuple_getitem_manager(
                0, "profile inner cell", value.__closure__[0], kind
            )
            identity(attr(cell, "cell_contents", raw), raw)

        if not _catalog_check():
            raise QualificationError(
                "stock definitions changed during guard construction"
            )
        if not function_check(None) or not guard.check(roots):
            return None, factory_seconds
        return (function_check, guard, roots), factory_seconds
    except (AttributeError, TypeError, RuntimeError, StopIteration):
        return None, factory_seconds


class StockQualification:
    """Weak, read-only qualification shared by both stock guard variants.

    Does not replace topology/Tensor/state/optimizer-option/hook/TLS guards. Its
    owner list is weak; it owns no model, optimizer, registration dict, grad or
    bound method. A failure is sticky for this certificate.
    """

    def __init__(self, model, optimizer, kind):
        self._invalid = True
        self._native = None
        self.native_factory_seconds = 0.0
        self.native_prepare_seconds = 0.0
        self._nodes = ()
        self._edges = ()
        self._optimizer_type = CSTAdamW if kind == "cst" else AdamW
        if kind not in ("cst", "dense") or not _catalog_check():
            raise QualificationError("stock bootstrap definitions changed")
        if (
            type(model) is not nn.Sequential
            or type(optimizer) is not self._optimizer_type
        ):
            raise QualificationError("exact stock model/optimizer required")
        self._optimizer = weakref.ref(optimizer)
        self.model_ref = weakref.ref(model)
        self.optimizer_ref = self._optimizer
        root = _raw_dict(model)
        children = root.get("_modules")
        if type(children) is not dict or tuple(children) != ("0", "1", "2", "3", "4"):
            raise QualificationError("canonical five-element Sequential required")
        expected = (
            nn.Linear,
            nn.ReLU,
            CSTLinear if kind == "cst" else nn.Linear,
            nn.ReLU,
            nn.Linear,
        )
        if tuple(type(value) for value in children.values()) != expected:
            raise QualificationError("canonical stock layer types required")
        # Raw discovery precedes any user-accessible named traversal/callback.
        todo, seen, nodes, edges = [model], set(), [], []
        while todo:
            node = todo.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if not _module_check(node):
                raise QualificationError("custom Module callback/configuration")
            nodes.append(weakref.ref(node))
            registered = _raw_dict(node).get("_modules")
            if type(registered) is not dict or any(
                type(key) is not str for key in registered
            ):
                raise QualificationError("canonical module registration required")
            if any(
                value is not None and type(value) not in _MODULE_TYPES
                for value in registered.values()
            ):
                raise QualificationError("unqualified registered Module")
            edges.append(
                (
                    weakref.ref(node),
                    id(registered),
                    tuple(
                        (key, None if value is None else weakref.ref(value))
                        for key, value in registered.items()
                    ),
                )
            )
            todo.extend(value for value in registered.values() if value is not None)
        self._nodes = tuple(nodes)
        self._edges = tuple(edges)
        self._invalid = False
        if not self.check():
            raise QualificationError("custom optimizer callback/configuration")
        started = perf_counter()
        self._native, self.native_factory_seconds = _prepare_native_catalog()
        self.native_prepare_seconds = perf_counter() - started
        if not self.check():
            raise QualificationError("stock qualification changed during preparation")

    @property
    def native_backend(self):
        return "native" if self._native is not None else "python"

    def _owner_check(self):
        optimizer = self._optimizer()
        if optimizer is None or type(optimizer) is not self._optimizer_type:
            return False
        data = _raw_dict(optimizer)
        return (
            not any(name in data for name in (*_OPTIMIZER_NAMES, "step"))
            and self._module_edges_check()
        )

    def _check_python(self):
        """Clock-external CPU shadow oracle; not called by the native hot path."""
        try:
            return _catalog_check() and self._owner_check()
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return False

    def check(self):
        if self._invalid:
            return False
        try:
            if self._native is None:
                definitions = _catalog_check()
            else:
                functions, guard, roots = self._native
                definitions = functions(None) and guard.check(roots)
            if not definitions or not self._owner_check():
                self._invalid = True
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            self._invalid = True
        if self._invalid:
            self._native = None  # Release native metadata; never reanchor.
        return not self._invalid

    def _module_edges_check(self):
        # Reject replacement/new nodes before any ordinary named traversal or
        # hooks walk can reach an uncertified callback, even if old nodes live.
        for ref, dictionary_id, expected in self._edges:
            owner = ref()
            if owner is None or not _module_check(owner):
                return False
            current = _raw_dict(owner).get("_modules")
            if (
                type(current) is not dict
                or id(current) != dictionary_id
                or len(current) != len(expected)
            ):
                return False
            for (key, value), (name, child_ref) in zip(current.items(), expected):
                if type(key) is not str or key != name:
                    return False
                if child_ref is None:
                    if value is not None:
                        return False
                elif child_ref() is None or child_ref() is not value:
                    return False
        return True


BOOTSTRAP_SECONDS = perf_counter() - _BOOTSTRAP_START
