"""One binding-owned eager entrance for an already-selected native radius Plan.

The guard replaces repeated *metadata* work, not numerical preparation. Current
atoms are still classified on every call. The ordinary execute method still
borrows parameters, checks their declaration, and invokes the same CUDA graph;
the output keeps the ordinary placement read lease. Any changed/custom state
returns to Dispatcher before a Tensor is borrowed or numerical work begins.
"""

import gc
import inspect

import torch

from torchcst._builtin_callbacks import original_functions


def _callbacks():
    from torchcst._backends import algorithm, registry, state
    from torchcst._backends.dispatch import base, prepared, runtime, select
    from torchcst.atoms import Atoms
    from torchcst.atoms.optimizer_state import AtomOptimizerState
    from torchcst.atoms.state import AtomState
    from torchcst.nn.linear import CSTLinear
    from torchcst.nn.module import CSTModule
    from torchcst.operators import context, execution, metadata

    from .algorithm import RadiusCellMassAlgorithm

    return (
        (
            CSTLinear,
            ("validate_inputs", "build_context", "execution_parameters", "cst_charts"),
        ),
        (CSTModule, ("atoms", "prepared_dispatch_signature", "execution_declaration")),
        (runtime.Dispatcher, ("_prepare", "run", "select")),
        (runtime, ("select_plan",)),
        (select.FixedSelector, ("_match",)),
        (base.Selector, ("registry", "revision", "fallback_plan")),
        (registry.Registry, ("validate_plan", "get")),
        (algorithm.Algorithm, ("run", "validate_inputs")),
        (state.AlgorithmState, ("_signature", "is_current", "atom_state")),
        (state, ("tensor_signature",)),
        (RadiusCellMassAlgorithm, ("validate_recipe", "execute", "prepare")),
        (AtomState, ("parameters_for_execution", "protect_tensor")),
        (Atoms, ()),
        (AtomOptimizerState, ("fields",)),
        (execution, ("validate_linear_inputs", "linear_execution")),
        (execution.LinearInputs, ()),
        (context, ("context_from_tensors", "context_from_metadata")),
        (metadata, ("operator_declaration", "fixed_state_signature")),
        (metadata._FixedSignatureCache, ("current",)),
        (prepared, ("placement_signature",)),
    )


def _definitions(callbacks):
    functions, seen = [], set()
    for owner, names in callbacks:
        for name in names:
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


def _guard(dispatcher, cached, inputs, callbacks, functions, factory):
    """Check current inputs, ownership, fixed buffers, callbacks and global TLS.

    This uses the same optional PyTorch guard accessors as the existing optimizer
    preparation. No Tensor values, CUDA operation or input snapshot is cached.
    The two operation families share only CPU function/storage-check helpers.
    """
    from torchcst._backends.cuda.algorithms.polar_update.function_metadata import (
        prepare_checks,
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
    roots = (
        binding,
        dispatcher,
        inputs,
        cached,
        owners,
        functions,
        fixed,
        dictionaries,
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
        no_overrides(om, owner, ("parameters_for_execution", "protect_tensor"))
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

        ins = root(2, inputs, identity=False)
        tensor(attr(ins, "x", inputs.x), inputs.x)
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
        recipe = attr(state_node, "recipe", state.recipe)
        namespace_version(recipe, state.recipe)
        context = attr(state_node, "context", state.context)
        namespace_version(context, state.context)
        layout = attr(state_node, "layout", state.layout)
        namespace_version(layout, state.layout)
        for name in ("input", "output"):
            axis = getattr(state.layout, name)
            namespace_version(attr(layout, name, axis), axis)

        own = root(4, owners, manager=callback_guard)
        seen_bases = set()
        for index, value in enumerate(owners):
            node = own.tuple_getitem_manager(index, "callback owner", value, kind)
            same(node, value)
            if isinstance(value, type):
                bases = node.type_mro_manager("callback bases", value.__mro__, kind)
                for i, base in enumerate(value.__mro__):
                    if base in seen_bases or base.__flags__ & (1 << 8):
                        continue
                    seen_bases.add(base)
                    bn = bases.tuple_getitem_manager(i, "base", base, kind)
                    raw = next(
                        d for d in gc.get_referents(base.__dict__) if type(d) is dict
                    )
                    bn.type_dict_manager(
                        "callback namespace", raw, kind
                    ).add_dict_version_guard(raw, verbose, None)
            else:
                namespace_version(node, value)
        root(5, functions, manager=callback_guard)
        callback_guard.add_lambda_guard(checker, verbose, None)
        dicts = root(7, dictionaries, manager=callback_guard)
        for i, dictionary in enumerate(dictionaries):
            node = dicts.tuple_getitem_manager(
                i, "function dictionary", dictionary, kind
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
        template = (binding, None, None, cached, owners, functions, fixed, dictionaries)
        if callback_guard.check(roots) and guard.check(roots):
            return guard, callback_guard, template
        return None
    except (AttributeError, RuntimeError, TypeError, ValueError, StopIteration):
        return None


class PreparedRadiusLinear:
    def __init__(self, cached, checked):
        self.cached = cached
        self.guard, self.callback_guard, self.template = checked

    def run(self, dispatcher, inputs, *, plan=None):
        if plan is not None:
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
        roots = (self.template[0], dispatcher, inputs, *self.template[3:])
        # Reject modified accessors/callbacks before inspecting model attributes.
        # The two checks hold the GIL and run no intervening user callbacks.
        if not self.callback_guard.check(roots) or not self.guard.check(roots):
            return None
        cached = self.cached
        # Keep numerical execution, parameter borrowing, its final declaration
        # check, and result protection identical to Algorithm.run. Only the
        # now-certified metadata prologue is replaced by the native guard.
        result = cached.algorithm.execute(cached.state, inputs)
        result = cached.state.atom_state.protect_tensor(result)
        cached.state.binding.__dict__["_prepared_dispatch_stats"]["hits"] += 1
        return result


def prepare_program(dispatcher, cached, inputs, *, plan=None):
    from torchcst._backends.dispatch.runtime import Dispatcher
    from torchcst._backends.dispatch.select import FixedSelector
    from torchcst._backends.registry import Registry
    from torchcst._backends.state import AlgorithmState
    from torchcst.atoms import Atoms
    from torchcst.atoms.optimizer_state import AtomOptimizerState
    from torchcst.atoms.state import AtomState
    from torchcst.nn.linear import CSTLinear
    from torchcst.operators.execution import LinearInputs

    from .algorithm import RadiusCellMassAlgorithm

    state = cached.state
    binding = state.binding
    if (
        plan is not None
        or type(dispatcher) is not Dispatcher
        or not isinstance(dispatcher.registry, Registry)
        or type(type(dispatcher.registry)) is not type
        or type(dispatcher.registry).__getattribute__ is not object.__getattribute__
        or type(dispatcher.registry).validate_plan is not Registry.validate_plan
        or type(dispatcher.registry).get is not Registry.get
        or type(dispatcher.selector) is not FixedSelector
        or dispatcher.selector.fallback_plan is not None
        or dispatcher.selector._plan is not cached.decision.plan
        or type(binding) is not CSTLinear
        or type(binding.atom_state) is not AtomState
        or type(binding.atoms) is not Atoms
        or type(binding.atom_state.optimizer_state) is not AtomOptimizerState
        or type(state) is not AlgorithmState
        or type(cached.algorithm) is not RadiusCellMassAlgorithm
        or type(inputs) is not LinearInputs
        or LinearInputs.__getattribute__ is not object.__getattribute__
        or "x" in LinearInputs.__dict__
        or type(inputs.x) is not torch.Tensor
        or type(binding.atoms.p) is not torch.nn.Parameter
        or hasattr(binding, "state_signature")
        or "chart" in binding._modules
        or state.context.workspace_limit_bytes is not None
        or not state.recipe.native_entry
        or not state.is_current()
    ):
        return None
    from torchcst.operators import metadata
    from torchcst.operators._native_metadata_guard import _plain_owner

    fixed = (*binding.cst_charts(), binding.kernel)
    modules = tuple(module for root in fixed for module in root.modules())
    if not all(_plain_owner(module, metadata._standard_walk) for module in modules):
        return None
    classes = dict.fromkeys(
        [type(dispatcher.registry), torch.Tensor, torch.nn.Parameter]
        + [type(module) for module in modules]
    )
    callbacks = (*_callbacks(), *((kind, ()) for kind in classes))
    functions = _definitions(callbacks)
    if functions is None:
        return None
    from torchcst._backends.cuda.algorithms.polar_update.function_metadata import (
        get_factory,
    )

    factory = get_factory()
    if factory is None:
        return None
    checked = _guard(dispatcher, cached, inputs, callbacks, functions, factory)
    return None if checked is None else PreparedRadiusLinear(cached, checked)
