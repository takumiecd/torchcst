"""One binding-owned eager entrance for an already-selected native radius Plan.

The guard replaces repeated *metadata* work, not numerical preparation. Current
atoms are still classified on every call. The ordinary execute method still
borrows parameters, checks their declaration, and invokes the same CUDA graph;
the output keeps the ordinary placement read lease. Any changed/custom state
returns to Dispatcher before a Tensor is borrowed or numerical work begins.
"""

import gc
import inspect
from dataclasses import is_dataclass

import torch

from torchcst._builtin_callbacks import original_attribute, original_functions


def _callbacks():
    from torchcst._backends import algorithm, registry, state
    from torchcst._backends.dispatch import base, prepared, runtime, select
    from torchcst.atoms import Atoms
    from torchcst.atoms.optimizer_state import AtomOptimizerState
    from torchcst.atoms.state import AtomState
    from torchcst.charts.base import ChartSpec
    from torchcst.nn.linear import CSTLinear
    from torchcst.nn.module import CSTModule
    from torchcst.operators import context, execution, metadata
    from torchcst.operators.spec import ChartPairSpec, OperatorSpec

    from .algorithm import RadiusCellMassAlgorithm

    return (
        (
            CSTLinear,
            (
                "forward",
                "in_features",
                "out_features",
                "validate_inputs",
                "build_context",
                "execution_parameters",
                "cst_charts",
                "atoms",
                "execution_declaration",
                "prepared_dispatch_signature",
            ),
        ),
        (CSTModule, ("atoms", "prepared_dispatch_signature", "execution_declaration")),
        (runtime.Dispatcher, ("_prepare", "run", "select")),
        (runtime, ("select_plan",)),
        (select.FixedSelector, ("_match", "registry", "revision", "fallback_plan")),
        (base.Selector, ("registry", "revision", "fallback_plan")),
        (registry.Registry, ("validate_plan", "get")),
        (algorithm.Algorithm, ("run", "validate_inputs")),
        (state.AlgorithmState, ("_signature", "is_current", "atom_state")),
        (state, ("tensor_signature",)),
        (
            RadiusCellMassAlgorithm,
            ("run", "validate_inputs", "validate_recipe", "execute", "prepare"),
        ),
        (AtomState, ("parameters_for_execution", "protect_tensor", "guard_result")),
        (Atoms, ()),
        (AtomOptimizerState, ("fields",)),
        (execution, ("validate_linear_inputs", "linear_execution")),
        (execution.LinearInputs, ("__init__",)),
        (context, ("context_from_tensors", "context_from_metadata")),
        (metadata, ("operator_declaration", "fixed_state_signature")),
        (metadata._FixedSignatureCache, ("current",)),
        (prepared, ("placement_signature",)),
        (OperatorSpec, ("in_features", "out_features", "shape", "charts")),
        (ChartPairSpec, ("shape", "charts")),
        (ChartSpec, ("features",)),
    )


def _definitions(callbacks):
    functions, seen = [], set()
    for owner, names in callbacks:
        for name in names:
            if not original_attribute(owner, name):
                return None
            fn = inspect.getattr_static(owner, name)
            if isinstance(fn, property):
                fn = fn.fget
            records = original_functions(fn)
            if records is None:
                return None
            for record in records:
                if id(record) not in seen:
                    seen.add(id(record))
                    functions.append(record)
    return tuple(functions)


def _metadata_snapshot(*values):
    """Guard frozen declarations, including explicit object.__setattr__ edits."""
    classes, dictionaries, seen = {}, [], set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            raise TypeError("execution metadata cannot own a Tensor")
        if type(value) is tuple:
            for child in value:
                visit(child)
        elif type(value) is dict:
            dictionaries.append(value)
            for child in value.values():
                visit(child)
        elif is_dataclass(value):
            cls = type(value)
            if (
                cls.__getattribute__ is not object.__getattribute__
                or not cls.__dataclass_params__.frozen
            ):
                raise TypeError("only plain frozen declarations are supported")
            classes[cls] = None
            visit(value.__dict__)
        elif type(value) is list:
            raise TypeError("mutable lists are not execution declarations")

    try:
        for value in values:
            visit(value)
    except (AttributeError, TypeError):
        return None
    return tuple(classes), tuple(dictionaries)


def _guard(dispatcher, cached, inputs, callbacks, functions, factory, metadata):
    """Check current inputs, ownership, fixed buffers, callbacks and global TLS.

    This uses the same optional PyTorch guard accessors as the existing optimizer
    preparation. No Tensor values, CUDA operation or input snapshot is cached.
    The two operation families share only CPU function/storage-check helpers.
    """
    from torchcst._backends.cuda._host_function_metadata import (
        prepare_checks,
        prepare_dictionary_checks,
        prepare_storage_checks,
    )
    from torchcst.operators._native_metadata_guard import (
        _build_guard,
        guard_api,
        tensor_metadata_guard,
    )

    api = guard_api()
    if api is None:
        return None
    cls, kind = api
    guard = cls()
    callback_guard = cls()
    binding, state = cached.state.binding, cached.state
    owners = tuple(owner for owner, _ in callbacks)
    fixed = (*binding.cst_charts(), binding.kernel)
    checker, dictionaries = prepare_checks(
        factory, functions, owners, with_class_mro=True
    )
    if checker is None:
        return None
    # Inputs is deliberately absent from the retained template. Its Tensor is
    # supplied afresh to check() and may have different storage on every call.
    namespaces = metadata[1]
    roots = (
        binding,
        dispatcher,
        inputs.x,
        cached,
        owners,
        functions,
        fixed,
        dictionaries,
        namespaces,
    )
    verbose = ["prepared native radius Linear metadata"]
    storage = []

    def attr(node, name, value):
        return node.getattr_manager(name, name, value, kind)

    def item(node, key, value):
        return node.dict_getitem_manager(key, str(key), value, kind)

    def same(node, value):
        node.add_type_match_guard(id(type(value)), verbose, None)
        node.add_id_match_guard(id(value), verbose, None)

    def field(node, obj, name):
        value = getattr(obj, name)
        same(attr(node, name, value), value)

    def root(index, value, *, identity=True, manager=guard):
        node = manager.tuple_getitem_manager(index, str(index), value, kind)
        if identity:
            same(node, value)
        else:
            node.add_type_match_guard(id(type(value)), verbose, None)
        return node

    def no_overrides(node, obj, names):
        namespace = attr(node, "__dict__", obj.__dict__)
        for name in names:
            namespace.add_dict_contains_guard(False, name, verbose, None)

    def namespace_version(node, obj):
        data = obj.__dict__
        attr(node, "__dict__", data).add_dict_version_guard(data, verbose, None)

    def registered(node, obj, name, table="_modules"):
        namespace = attr(node, "__dict__", obj.__dict__)
        namespace.add_dict_contains_guard(False, name, verbose, None)
        values = obj.__dict__[table]
        result = item(item(namespace, table, values), name, values[name])
        same(result, values[name])
        return result

    def tensor(node, value, *, owned=False):
        # Older TensorMatch APIs alone do not detect every Python __class__
        # change. Keep an explicit exact-type check on every Tensor node.
        node.add_type_match_guard(id(type(value)), verbose, None)
        if owned:
            node.add_id_match_guard(id(value), verbose, None)
        if not tensor_metadata_guard(node, value, "Linear tensor", verbose):
            raise TypeError("native TensorMatch is unavailable")
        field(node, value, "requires_grad")
        # Version counters of atom/optimizer values change normally each step.
        # Storage changes do not; these are weak checks, not retained examples.
        if owned:
            storage.append(value)

    try:
        bm = root(0, binding)
        no_overrides(
            bm,
            binding,
            (
                "forward",
                "in_features",
                "out_features",
                "validate_inputs",
                "build_context",
                "execution_parameters",
                "cst_charts",
                "prepared_dispatch_signature",
                "execution_declaration",
                "state_signature",
                "atoms",
                "atom_state",
                "input_type",
            ),
        )
        field(bm, binding, "input_type")
        owner = binding.atom_state
        om = registered(bm, binding, "atom_state")
        field(om, owner, "layout_version")
        no_overrides(
            om, owner, ("parameters_for_execution", "protect_tensor", "guard_result")
        )
        atoms = owner.atoms
        am = registered(om, owner, "atoms")
        tensor(registered(am, atoms, "p", "_parameters"), atoms.p, owned=True)
        optimizer_state = owner.optimizer_state
        os = registered(om, owner, "optimizer_state")
        field(os, optimizer_state, "_fields")
        fields = attr(os, "_fields", optimizer_state.fields)
        fields.add_dict_version_guard(optimizer_state.fields, verbose, None)
        for name, value in optimizer_state.fields.items():
            if type(value) is not torch.Tensor:
                return None
            tensor(item(fields, name, value), value, owned=True)
        for name, value in (
            ("kernel", binding.kernel),
            ("input_chart", binding.input_chart),
            ("output_chart", binding.output_chart),
        ):
            registered(bm, binding, name)
        namespace = attr(bm, "__dict__", binding.__dict__)
        same(item(namespace, "_prepared_dispatch", cached), cached)
        # The direct Module entrance reuses a dispatcher only while its selector
        # is still the exact live selector registered on this Module.
        same(item(namespace, "selector", dispatcher.selector), dispatcher.selector)

        dm = root(1, dispatcher, identity=False)
        no_overrides(dm, dispatcher, ("run", "_prepare", "select"))
        field(dm, dispatcher, "registry")
        field(dm, dispatcher, "selector")
        selector = attr(dm, "selector", dispatcher.selector)
        namespace_version(selector, dispatcher.selector)
        registry = attr(dm, "registry", dispatcher.registry)
        no_overrides(registry, dispatcher.registry, ("validate_plan", "get"))
        entries = dispatcher.registry._entries
        attr(registry, "_entries", entries).add_dict_version_guard(
            entries, verbose, None
        )

        input_node = root(2, inputs.x, identity=False)
        tensor(input_node, inputs.x)
        no_overrides(input_node, inputs.x, ("reshape", "contiguous"))
        cache = root(3, cached)
        namespace_version(cache, cached)
        state_node = attr(cache, "state", state)
        same(state_node, state)
        no_overrides(state_node, state, ("_signature", "is_current", "atom_state"))
        for name in ("binding", "algorithm", "recipe", "context", "_storage", "layout"):
            field(state_node, state, name)
        field(state_node, state, "source_layout_version")
        algorithm = attr(state_node, "algorithm", state.algorithm)
        namespace_version(algorithm, state.algorithm)
        plan = cached.decision.plan
        plan_node = attr(selector, "_plan", plan)
        namespace_version(plan_node, plan)
        # Frozen declaration dictionaries are already checked by callback_guard
        # below. The identity fields above connect those snapshots to live state.

        own = root(4, owners, manager=callback_guard)
        seen_bases = set()
        raw_dictionaries = list(dictionaries)
        seen_dicts = {id(value) for value in raw_dictionaries}

        def dictionary(value):
            if id(value) not in seen_dicts:
                seen_dicts.add(id(value))
                raw_dictionaries.append(value)

        for index, value in enumerate(owners):
            node = own.tuple_getitem_manager(index, "callback owner", value, kind)
            same(node, value)
            if isinstance(value, type):
                # The CPython checker already checks each complete MRO pointer.
                # Class/module namespace identities are read-only in CPython;
                # guard their dictionaries directly rather than retraverse MROs
                # and descriptors on every check. Every mutable version remains.
                for base in value.__mro__:
                    if base in seen_bases or base.__flags__ & (1 << 8):
                        continue
                    seen_bases.add(base)
                    raw = next(
                        d for d in gc.get_referents(base.__dict__) if type(d) is dict
                    )
                    dictionary(raw)
            else:
                dictionary(value.__dict__)
        dictionaries = tuple(raw_dictionaries)
        roots = (*roots[:7], dictionaries, roots[8])
        root(5, functions, manager=callback_guard)
        callback_guard.add_lambda_guard(checker, verbose, None)
        dicts = root(7, dictionaries, manager=callback_guard)
        declared = root(8, namespaces, manager=callback_guard)
        epoch = prepare_dictionary_checks(factory, (*dictionaries, *namespaces))
        if epoch is not None:
            callback_guard.add_lambda_guard(epoch, verbose, None)
        else:
            for i, dictionary in enumerate(dictionaries):
                node = dicts.tuple_getitem_manager(
                    i, "function dictionary", dictionary, kind
                )
                node.add_dict_version_guard(dictionary, verbose, None)
            for i, dictionary in enumerate(namespaces):
                node = declared.tuple_getitem_manager(
                    i, "immutable declaration", dictionary, kind
                )
                node.add_dict_version_guard(dictionary, verbose, None)
        fixed_node = guard.tuple_getitem_manager(6, "fixed state", fixed, kind)
        _build_guard(
            fixed_node,
            fixed,
            kind,
            epilogue_root=guard,
            guarded_classes=tuple(v for v in owners if isinstance(v, type)),
            guarded_class_dicts=seen_bases,
        )
        storage_check = prepare_storage_checks(factory, tuple(storage))
        if storage_check is None:
            return None
        guard.add_lambda_guard(storage_check, verbose, None)
        from torch._C._dynamo.guards import GlobalStateGuard

        guard.add_global_state_guard(GlobalStateGuard(), verbose, None)
        # This template owns fixed metadata/model state only, not Inputs/X.
        template = (
            binding,
            None,
            None,
            cached,
            owners,
            functions,
            fixed,
            dictionaries,
            namespaces,
        )
        if callback_guard.check(roots) and guard.check(roots):
            return guard, callback_guard, template
        return None
    except (AttributeError, RuntimeError, TypeError, ValueError, StopIteration):
        return None


class PreparedRadiusLinear:
    def __init__(self, dispatcher, cached, checked):
        self.dispatcher = dispatcher
        self.cached = cached
        self.guard, self.callback_guard, self.template = checked

    def _checked_roots(self, dispatcher, binding, value, *, plan=None):
        if (
            plan is not None
            or binding is not self.template[0]
            or self.cached is not self.template[3]
        ):
            return None
        if (
            torch._C._is_torch_function_mode_enabled()
            or torch._C._len_torch_dispatch_stack()
            or torch._C._are_functorch_transforms_active()
            or torch.autograd.forward_ad._current_level != -1
            or torch._C._get_tracing_state() is not None
            or torch.cuda.is_current_stream_capturing()
            or torch.is_autocast_enabled()
        ):
            return None
        roots = (binding, dispatcher, value, *self.template[3:])
        # Authenticate accessors before inspecting live data or constructing an
        # invocation wrapper. The native checks hold the GIL throughout.
        if not self.callback_guard.check(roots):
            return None
        return roots

    def _execute(self, inputs):
        cached = self.cached
        # Numerical execution and both placement protections are unchanged.
        result = cached.algorithm.execute(cached.state, inputs)
        result = cached.state.atom_state.protect_tensor(result)
        cached.state.binding.__dict__["_prepared_dispatch_stats"]["hits"] += 1
        return result

    def run(self, dispatcher, binding, inputs, *, plan=None):
        roots = self._checked_roots(dispatcher, binding, None, plan=plan)
        if roots is None:
            return None
        from torchcst.operators.execution import LinearInputs

        if type(inputs) is not LinearInputs:
            return None
        roots = (*roots[:2], inputs.x, *roots[3:])
        if not self.guard.check(roots):
            return None
        return self._execute(inputs)

    def forward(self, binding, value):
        """Narrow two-dimensional Module entrance; other inputs use forward."""
        roots = self._checked_roots(self.dispatcher, binding, value)
        if roots is None or not self.guard.check(roots):
            return None
        from torchcst.operators.execution import LinearInputs

        # A fresh wrapper owns this invocation only. No input/graph is cached.
        result = self._execute(LinearInputs(value))
        return binding.atom_state.guard_result(result)


def prepare_program(dispatcher, cached, inputs, *, plan=None):
    from torchcst._backends.cuda._host_function_metadata import (
        get_factory,
    )
    from torchcst._backends.dispatch.prepared import PreparedDispatch
    from torchcst._backends.dispatch.runtime import Dispatcher
    from torchcst._backends.dispatch.select import FixedSelector
    from torchcst._backends.registry import Registry
    from torchcst._backends.state import AlgorithmState
    from torchcst.atoms import Atoms
    from torchcst.atoms.optimizer_state import AtomOptimizerState
    from torchcst.atoms.state import AtomState
    from torchcst.nn.linear import CSTLinear
    from torchcst.operators import metadata
    from torchcst.operators._native_metadata_guard import _plain_owner, guard_api
    from torchcst.operators.context import LinearContext
    from torchcst.operators.execution import LinearInputs

    from .algorithm import RadiusCellMassAlgorithm
    from .recipe import RadiusCellMassRecipe

    # Authenticate the small recipe header before reading any instance value.
    # In particular, the ordinary non-native recipe remains a cheap rejection.
    for owner, names in (
        (PreparedDispatch, ("state", "__getattribute__", "__dict__")),
        (AlgorithmState, ("recipe", "__getattribute__", "__dict__")),
        (RadiusCellMassRecipe, ("native_entry", "__getattribute__", "__dict__")),
    ):
        if not all(original_attribute(owner, name) for name in names):
            return None
    if type(cached) is not PreparedDispatch or type(cached.state) is not AlgorithmState:
        return None
    state = cached.state
    if type(state.recipe) is not RadiusCellMassRecipe or not state.recipe.native_entry:
        return None
    if guard_api() is None:
        return None
    factory = get_factory()
    if factory is None:
        return None

    # Every registration/dict lookup below substitutes an ordinary attribute
    # read. Authenticate those descriptors AND resolved inherited callbacks
    # before calling state.is_current, atoms, cst_charts or any other getter.
    for owner, names in (
        (
            CSTLinear,
            (
                "input_type",
                "selector",
                "__getattribute__",
                "__getattr__",
                "__dict__",
                "atom_state",
                "kernel",
                "input_chart",
                "output_chart",
                "chart",
                "state_signature",
                "_modules",
                "_buffers",
                "_parameters",
            ),
        ),
        (AlgorithmState, ("binding", "algorithm", "context", "layout")),
        (
            AtomState,
            (
                "layout_version",
                "atoms",
                "optimizer_state",
                "__getattribute__",
                "__getattr__",
                "__dict__",
                "_modules",
                "_parameters",
            ),
        ),
        (Atoms, ("p", "__getattribute__", "__getattr__", "__dict__", "_parameters")),
        (
            AtomOptimizerState,
            ("_fields", "__getattribute__", "__getattr__", "__dict__"),
        ),
        (Dispatcher, ("registry", "selector", "__getattribute__", "__dict__")),
        (Registry, ("_entries", "__getattribute__", "__dict__")),
        (LinearContext, ("workspace_limit_bytes", "__getattribute__", "__dict__")),
        (LinearInputs, ("x", "__getattribute__", "__dict__")),
    ):
        if not all(original_attribute(owner, name) for name in names):
            return None
    # The Module entrance constructs a fresh dataclass after guarding raw X.
    # A custom allocator could return a different object/Tensor instead.
    if inspect.getattr_static(LinearInputs, "__new__") is not object.__new__:
        return None
    # TensorBase is immutable in CPython. A Python override on Tensor must use
    # the ordinary Module path where reshape/contiguous are actually invoked.
    for name in ("reshape", "contiguous", "shape", "ndim", "__getattribute__"):
        if inspect.getattr_static(torch.Tensor, name) is not inspect.getattr_static(
            torch._C.TensorBase, name
        ):
            return None
    callbacks = _callbacks()
    functions = _definitions(callbacks)
    if functions is None:
        return None

    binding = state.binding
    if (
        type(binding) is not CSTLinear
        or type(dispatcher) is not Dispatcher
        or type(cached.algorithm) is not RadiusCellMassAlgorithm
        or type(inputs) is not LinearInputs
    ):
        return None
    for obj, names in (
        (
            binding,
            (
                "forward",
                "in_features",
                "out_features",
                "validate_inputs",
                "build_context",
                "execution_parameters",
                "execution_declaration",
                "cst_charts",
                "state_signature",
                "prepared_dispatch_signature",
                "atoms",
                "atom_state",
                "kernel",
                "input_chart",
                "output_chart",
                "chart",
                "input_type",
            ),
        ),
        (state, ("_signature", "is_current", "atom_state")),
        (
            cached.algorithm,
            ("run", "validate_inputs", "validate_recipe", "execute", "prepare"),
        ),
        (dispatcher, ("run", "_prepare", "select")),
    ):
        if any(name in obj.__dict__ for name in names):
            return None
    if not metadata._standard_walk(binding):
        return None
    if (
        plan is not None
        or not isinstance(dispatcher.registry, Registry)
        or type(type(dispatcher.registry)) is not type
        or type(dispatcher.registry).__getattribute__ is not object.__getattribute__
        or type(dispatcher.registry).validate_plan is not Registry.validate_plan
        or type(dispatcher.registry).get is not Registry.get
        or inspect.getattr_static(type(dispatcher.registry), "_entries", None)
        is not None
        or type(dispatcher.selector) is not FixedSelector
        or dispatcher.selector.fallback_plan is not None
        or dispatcher.selector._plan is not cached.decision.plan
        or type(binding.atom_state) is not AtomState
        or type(binding.atoms) is not Atoms
        or type(binding.atom_state.optimizer_state) is not AtomOptimizerState
        or type(inputs.x) is not torch.Tensor
        or type(binding.atoms.p) is not torch.nn.Parameter
        or hasattr(binding, "state_signature")
        or "chart" in binding._modules
        or state.context.workspace_limit_bytes is not None
        or not state.is_current()
    ):
        return None

    fixed = (*binding.cst_charts(), binding.kernel)
    # Walk registrations without calling an as-yet-unqualified custom walker or
    # descriptor. The ordinary metadata path remains responsible for such states.
    pending, modules, seen = list(fixed), [], set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        kind = type(module)
        if (
            kind.__getattribute__ is not object.__getattribute__
            or inspect.getattr_static(kind, "__dict__", None)
            is not torch.nn.Module.__dict__["__dict__"]
            or any(
                inspect.getattr_static(kind, name, None) is not None
                for name in ("_modules", "_buffers", "_parameters")
            )
            or not _plain_owner(module, metadata._standard_walk)
        ):
            return None
        modules.append(module)
        pending.extend(child for child in module._modules.values() if child is not None)
    for axis in (state.layout.input, state.layout.output):
        # Finish lazy immutable Python metadata before capturing namespaces.
        _ = axis.banks
    declared = _metadata_snapshot(state.context, state.layout, cached.decision.plan)
    if declared is None:
        return None
    classes = dict.fromkeys(
        [type(dispatcher.registry), type(cached), torch.Tensor, torch.nn.Parameter]
        + [type(module) for module in modules]
        + list(declared[0])
    )
    callbacks = (*callbacks, *((kind, ()) for kind in classes))
    checked = _guard(
        dispatcher, cached, inputs, callbacks, functions, factory, declared
    )
    return (
        None if checked is None else PreparedRadiusLinear(dispatcher, cached, checked)
    )
