"""A binding-owned hot path for the standard radius/FlatTorus update.

This is the same AdamW proposal and Polar CUDA executor. Only the surrounding
Python validation/identity coordinate maps are specialized. Any changed or
custom callback returns to the ordinary Dispatcher before doing numerical work.
"""

import gc
import inspect
import math

import torch

from torchcst._backends.torch import finite as _finite
from torchcst._builtin_callbacks import original_functions
from torchcst.kernels import parameterizations as _parameterizations
from torchcst.optim import state as _optimizer_state

_MISSING = object()


def _snapshot_metadata_guard(point):
    """Own shape/stride/type metadata, never an example snapshot or storage."""
    from torchcst.operators._native_metadata_guard import (
        guard_api,
        tensor_metadata_guard,
    )

    api = guard_api()
    if api is None:
        return None
    # A custom detach must remain on the ordinary path. Read its definition and
    # instance namespace without invoking an unqualified Python accessor. The
    # actual example uses the immutable TensorBase descriptor only after this
    # qualification; an override cannot supply unrelated example metadata.
    point_type = type(point)
    tensor_base = torch._C.TensorBase
    if point_type not in (torch.Tensor, torch.nn.Parameter):
        return None
    namespace = inspect.getattr_static(point_type, "__dict__", None)
    if (
        not inspect.isgetsetdescriptor(namespace)
        or namespace.__objclass__ is not torch.Tensor
        or inspect.getattr_static(point_type, "__getattribute__", None)
        is not object.__getattribute__
        or inspect.getattr_static(point_type, "detach", None) is not tensor_base.detach
        or inspect.getattr_static(torch.Tensor, "detach", None)
        is not tensor_base.detach
        or inspect.getattr_static(torch.Tensor, "requires_grad", None)
        is not tensor_base.requires_grad
        or "detach" in namespace.__get__(point, point_type)
    ):
        return None
    cls, _kind = api
    guard = cls()
    verbose = ["detached atom snapshot metadata"]
    try:
        if not tensor_base.is_contiguous(point):
            return None
        example = tensor_base.detach(point)
        guard.add_type_match_guard(id(torch.Tensor), verbose, None)
        if not tensor_metadata_guard(guard, example, "previous point", verbose):
            return None
        # requires_grad is part of TensorMatch on supported versions. Keep an
        # explicit check as well: this is a required snapshot invariant.
        node = guard.getattr_manager("requires_grad", "requires_grad", False, _kind)
        node.add_false_match_guard(verbose, None)
        return guard if guard.check(example) else None
    except (AttributeError, TypeError, RuntimeError):
        return None


def _callbacks():
    from torchcst._backends import algorithm, registry, state
    from torchcst._backends.dispatch import runtime
    from torchcst._backends.torch import finite
    from torchcst._backends.torch.algorithms.polar_update import executor as polar
    from torchcst._backends.torch.geometry import base, euclidean
    from torchcst._backends.torch.geometry import execution as geometry
    from torchcst._backends.torch.kernels import execution as kernel
    from torchcst._backends.torch.parameterizations import polar_amp_width as parameters
    from torchcst._backends.torch.profiles import compact
    from torchcst._backends.torch.profiles import execution as profiles
    from torchcst.atoms.state import AtomState
    from torchcst.charts.base import ChartState
    from torchcst.kernels.state import KernelState
    from torchcst.nn.module import CSTModule
    from torchcst.operators.atom_update import AtomUpdateBinding
    from torchcst.operators.binding import Operator
    from torchcst.optim.state import OptimizerStateAdapter

    from . import executor as cuda_executor

    return (
        (algorithm.Algorithm, ("run", "validate_inputs", "prepare")),
        (state.AlgorithmState, ("_signature", "is_current", "mark_current")),
        (registry.Registry, ("validate_plan", "get")),
        (runtime.Dispatcher, ("_prepare", "prepare", "run")),
        (
            AtomUpdateBinding,
            (
                "validate_inputs",
                "build_context",
                "_scalar_metadata",
                "execution_declaration",
                "execution_parameters",
                "cst_charts",
                "state_signature",
                "prepared_dispatch_signature",
                "atom_state",
                "kernel",
            ),
        ),
        (Operator, ("p",)),
        (AtomState, ("protect_tensor",)),
        (KernelState, ("scalar", "setting")),
        (cuda_executor, ("fused_update_",)),
        (ChartState, ("center_parameter_dim",)),
        (OptimizerStateAdapter, ("transport",)),
        (CSTModule, ("atoms",)),
        (
            kernel,
            (
                "family",
                "_operation",
                "parameter_dim",
                "validate_layout",
                "project_parameter_gradient",
                "transport_parameter_state",
            ),
        ),
        (parameters, ("_split_many",)),
        (polar, ("project_parameter_gradient", "transport_parameter_state")),
        (
            profiles,
            (
                "_module",
                "invoke",
                "parameter_dim",
                "project_gradient",
                "transport_state",
            ),
        ),
        (compact, ("project_gradient", "transport_state")),
        (geometry, ("_evaluate", "project_tangent", "transport")),
        (euclidean, ("project_tangent", "transport")),
        (base, ("_validate_structure",)),
        (finite, ("checks_enabled", "first_nonfinite")),
    )


def _definitions(callbacks):
    result = []
    seen = set()
    for owner, names in callbacks:
        for name in names:
            fn = inspect.getattr_static(owner, name)
            if isinstance(fn, property):
                fn = fn.fget
            functions = original_functions(fn)
            if functions is None:
                return None
            for function in functions:
                if id(function) not in seen:
                    seen.add(id(function))
                    result.append(function)
    return tuple(result)


def _native_guard(
    binding,
    dispatcher,
    cached,
    callbacks,
    functions,
    site=None,
    *,
    function_factory=None,
):
    from torchcst.operators._native_metadata_guard import (
        _build_guard,
        guard_api,
        tensor_metadata_guard,
    )

    api = guard_api()
    if api is None:
        return None
    from torchcst._backends.cuda._host_function_metadata import (
        prepare_checks,
        prepare_storage_checks,
    )

    cls, kind = api
    guard = cls()
    owners = tuple(owner for owner, _ in callbacks)
    fixed_roots = (*binding.cst_charts(), binding.kernel)
    roots = (binding, dispatcher, cached.state, owners, functions, site, fixed_roots)
    checker, dictionaries = prepare_checks(
        function_factory, functions, owners, with_class_mro=True
    )
    if checker is not None:
        roots = (*roots, dictionaries)
    verbose = ["prepared atom optimizer metadata"]
    storage_nodes = []

    def attr(node, name, value):
        return node.getattr_manager(name, name, value, kind)

    def item(node, key, value):
        return node.dict_getitem_manager(key, str(key), value, kind)

    def registered(node, owner, name, table="_modules"):
        # Standard nn.Module attributes otherwise call Python __getattr__ from
        # inside every native guard. Check their actual registration directly.
        namespace = attr(node, "__dict__", owner.__dict__)
        namespace.add_dict_contains_guard(False, name, verbose, None)
        values = owner.__dict__[table]
        return item(item(namespace, table, values), name, values[name])

    def same(node, value):
        node.add_type_match_guard(id(type(value)), verbose, None)
        node.add_id_match_guard(id(value), verbose, None)

    def equal(node, value):
        node.add_type_match_guard(id(type(value)), verbose, None)
        node.add_equals_match_guard(value, verbose, None)

    def field(node, value, name):
        equal(attr(node, name, getattr(value, name)), getattr(value, name))

    def no_overrides(node, value, names):
        namespace = attr(node, "__dict__", value.__dict__)
        for name in names:
            namespace.add_dict_contains_guard(False, name, verbose, None)

    def tensor(node, value):
        same(node, value)
        matched = tensor_metadata_guard(node, value, "optimizer Tensor", verbose)
        for name in () if matched else ("shape", "dtype", "device", "layout"):
            field(node, value, name)
        # Numerical versions change on every AdamW update and are excluded.
        for name in () if matched else ("stride",):
            method = getattr(value, name)
            call = attr(node, name, method).call_function_no_args_manager(
                name, method(), kind
            )
            equal(call, method())
        storage_nodes.append((node, value))

    def root(index, value):
        node = guard.tuple_getitem_manager(index, str(index), value, kind)
        same(node, value)
        return node

    try:
        bm = root(0, binding)
        namespace = attr(bm, "__dict__", binding.__dict__)
        for name in (
            "execution_declaration",
            "validate_inputs",
            "build_context",
            "execution_parameters",
            "cst_charts",
            "_scalar_metadata",
            "state_signature",
            "prepared_dispatch_signature",
        ):
            namespace.add_dict_contains_guard(False, name, verbose, None)
        field(bm, binding, "workspace_limit_bytes")
        om = attr(bm, "operator", binding.operator)
        same(om, binding.operator)
        same(attr(om, "kernel", binding.kernel), binding.kernel)
        am = attr(om, "atoms", binding.operator.atoms)
        same(am, binding.operator.atoms)
        tensor(
            registered(am, binding.operator.atoms, "p", "_parameters"),
            binding.operator.p,
        )
        charts = binding.cst_charts()
        chart_nodes = attr(om, "charts", charts)
        same(chart_nodes, charts)
        for i, chart in enumerate(charts):
            cn = chart_nodes.tuple_getitem_manager(i, "chart", chart, kind)
            same(cn, chart)
            geometry = chart.geometry
            gn = registered(cn, chart, "geometry")
            same(gn, geometry)
            field(gn, geometry, "embedding_dim")
            field(gn, geometry, "center_parameter_dim")
            field(attr(gn, "spec", geometry.spec), geometry.spec, "revision")
        dm = root(1, dispatcher)
        no_overrides(dm, dispatcher, ("_prepare", "prepare", "run", "select"))
        for name in ("registry", "selector"):
            same(attr(dm, name, getattr(dispatcher, name)), getattr(dispatcher, name))
        sm = attr(dm, "selector", dispatcher.selector)
        for name in ("_plan", "fallback_plan"):
            same(
                attr(sm, name, getattr(dispatcher.selector, name)),
                getattr(dispatcher.selector, name),
            )
        field(sm, dispatcher.selector, "revision")
        registry = dispatcher.registry
        rm = attr(dm, "registry", registry)
        no_overrides(rm, registry, ("validate_plan", "get"))
        entries = attr(rm, "_entries", registry._entries)
        entries.add_dict_version_guard(registry._entries, verbose, None)
        state = cached.state
        stm = root(2, state)
        no_overrides(
            stm, state, ("_signature", "is_current", "mark_current", "invalidate")
        )
        algo_node = attr(stm, "algorithm", state.algorithm)
        no_overrides(
            algo_node,
            state.algorithm,
            (
                "run",
                "execute",
                "prepare",
                "supports",
                "validate_inputs",
                "validate_recipe",
            ),
        )
        for name in ("algorithm", "recipe", "context", "_storage"):
            same(attr(stm, name, getattr(state, name)), getattr(state, name))
        field(stm, state, "source_layout_version")
        owner = binding.atom_state
        owner_node = attr(am, "_atom_state_owner", owner)
        # _atom_state_owner is stored directly in Atoms.__dict__.
        same(owner_node, owner)
        field(owner_node, owner, "layout_version")
        optimizer_state = owner.optimizer_state
        owned = registered(owner_node, owner, "optimizer_state")
        fields = optimizer_state.fields
        fm = attr(owned, "fields", fields)
        fm.add_dict_version_guard(fields, verbose, None)
        for name, value in fields.items():
            if type(value) is not torch.Tensor:
                return None
            tensor(item(fm, name, value), value)
        # Registry Plan fields are checked even though ExecutionPlan is frozen.
        plan = cached.decision.plan
        pm = attr(sm, "_plan", plan)
        for name in ("algorithm_id", "algorithm_revision", "schema_version"):
            field(pm, plan, name)
        same(attr(pm, "recipe", plan.recipe), plan.recipe)
        own = root(3, owners)
        seen_bases = set()
        raw_dictionaries = list(dictionaries)
        seen_dictionaries = {id(value) for value in raw_dictionaries}

        def dictionary(value):
            if id(value) not in seen_dictionaries:
                seen_dictionaries.add(id(value))
                raw_dictionaries.append(value)

        for index, value in enumerate(owners):
            node = own.tuple_getitem_manager(index, str(index), value, kind)
            same(node, value)
            if isinstance(value, type) and checker is not None:
                # The CPython checker already checks the full MRO pointer.
                # Class namespace identity is immutable; retain/version-check
                # its raw dictionary rather than traverse each MRO per call.
                for base in value.__mro__:
                    if base in seen_bases or base.__flags__ & (1 << 8):
                        continue
                    seen_bases.add(base)
                    dictionary(
                        next(
                            d
                            for d in gc.get_referents(base.__dict__)
                            if type(d) is dict
                        )
                    )
            elif isinstance(value, type):
                bases = node.type_mro_manager("callback bases", value.__mro__, kind)
                if checker is None:
                    bases.add_length_check_guard(len(value.__mro__), verbose, None)
                for i, base in enumerate(value.__mro__):
                    # The CPython checker owns/checks the complete immutable
                    # MRO tuple. Repeated base IDs/types need no child guards;
                    # each mutable namespace still gets its version check.
                    if checker is not None and (
                        base in seen_bases or base.__flags__ & (1 << 8)
                    ):
                        continue
                    bn = bases.tuple_getitem_manager(i, "base", base, kind)
                    if checker is None:
                        same(bn, base)
                    if base not in seen_bases and not base.__flags__ & (1 << 8):
                        seen_bases.add(base)
                        raw = next(
                            d
                            for d in gc.get_referents(base.__dict__)
                            if type(d) is dict
                        )
                        bn.type_dict_manager(
                            "callback class", raw, kind
                        ).add_dict_version_guard(raw, verbose, None)
            elif checker is not None:
                dictionary(value if type(value) is dict else value.__dict__)
            elif type(value) is dict:
                node.add_dict_version_guard(value, verbose, None)
            else:
                raw = value.__dict__
                data = attr(node, "__dict__", raw)
                data.add_dict_version_guard(raw, verbose, None)
        if checker is not None:
            dictionaries = tuple(raw_dictionaries)
            roots = (*roots[:7], dictionaries)
            # Function slots are checked together through CPython's C API.
            # The immutable function tuple and every mutable dictionary remain
            # guarded. No callback, tensor value, or CUDA API is executed here.
            root(4, functions)
            guard.add_lambda_guard(checker, verbose, None)
            dictionaries_node = root(7, dictionaries)
            for index, value in enumerate(dictionaries):
                node = dictionaries_node.tuple_getitem_manager(
                    index, "function dictionary", value, kind
                )
                node.add_dict_version_guard(value, verbose, None)
        else:
            funcs = root(4, functions)
            globals_seen = set()
            for index, function in enumerate(functions):
                # Function guards are roots, outside dictionary fast paths: code
                # changes need not change the module/class namespace version.
                fnm = funcs.tuple_getitem_manager(index, str(index), function, kind)
                same(fnm, function)
                for name in ("__code__", "__defaults__", "__kwdefaults__"):
                    value = getattr(function, name)
                    accessor = {
                        "__code__": "code_manager",
                        "__defaults__": "func_defaults_manager",
                        "__kwdefaults__": "func_kwdefaults_manager",
                    }[name]
                    method = getattr(fnm, accessor, None)
                    # Some versions treat a null defaults slot as a failed accessor
                    # rather than Python None. Its ordinary attribute stays checked.
                    node = (
                        method(name, value, kind)
                        if method and value is not None
                        else attr(fnm, name, value)
                    )
                    same(node, value)
                    if type(value) is dict:
                        node.add_dict_version_guard(value, verbose, None)
                values = function.__dict__
                attr(fnm, "__dict__", values).add_dict_version_guard(
                    values, verbose, None
                )
                if id(function.__globals__) not in globals_seen:
                    globals_seen.add(id(function.__globals__))
                    values = function.__globals__
                    attr(fnm, "__globals__", values).add_dict_version_guard(
                        values, verbose, None
                    )
                closure = function.__closure__
                if closure:
                    cm = attr(fnm, "__closure__", closure)
                    for i, cell in enumerate(closure):
                        value = cell.cell_contents
                        cn = cm.tuple_getitem_manager(i, "cell", cell, kind)
                        same(attr(cn, "cell_contents", value), value)
        if site is not None:
            node = root(5, site)
            no_overrides(node, site, ("cst_charts", "atoms"))
            for name, value in (
                ("kernel", binding.kernel),
                ("input_chart", binding.operator.charts[0]),
                ("output_chart", binding.operator.charts[1]),
            ):
                same(registered(node, site, name), value)
            state_node = registered(node, site, "atom_state")
            same(state_node, owner)
            same(registered(state_node, owner, "atoms"), binding.operator.atoms)
        # Reuse the fixed-state guard builder, rather than checking the same
        # tree through operator_declaration and a second native entrance.
        # Every callback boundary still checks all versions and registrations.
        fixed_node = guard.tuple_getitem_manager(6, "fixed state", fixed_roots, kind)
        _build_guard(
            fixed_node,
            fixed_roots,
            kind,
            epilogue_root=guard,
            guarded_classes=tuple(value for value in owners if isinstance(value, type)),
            guarded_class_dicts=seen_bases,
        )
        storage_checker = prepare_storage_checks(
            function_factory, tuple(value for _, value in storage_nodes)
        )
        if storage_checker is not None:
            guard.add_lambda_guard(storage_checker, verbose, None)
        else:
            for node, value in storage_nodes:
                method = value.data_ptr
                call = attr(node, "data_ptr", method).call_function_no_args_manager(
                    "data_ptr", method(), kind
                )
                equal(call, method())
        return (guard, roots) if guard.check(roots) else None
    except (AttributeError, RuntimeError, TypeError, StopIteration, ValueError):
        return None


class PreparedPolarOptimizerUpdate:
    """Own one certified execution path; never retain an old-point snapshot."""

    def __init__(self, binding, dispatcher, cached, guard, launch=None):
        self.binding = binding
        self.dispatcher = dispatcher
        self.state = cached.state
        self.plan = cached.decision.plan
        self.context = cached.context
        self.declaration = binding.__dict__["_execution_declaration"]
        self.fixed_guard = binding.kernel.__dict__["_fixed_signature_cache"].guard
        self.guard, self.roots = guard
        self.native_function_metadata = len(self.roots) == 8
        self.point = binding.execution_parameters()
        self.snapshot_guard = _snapshot_metadata_guard(self.point)
        self.owner = binding.atom_state
        self.adapter_keys = ("exp_avg",)
        self.launch = launch
        self.transport_required = False

    def validate_inputs(self, inputs):
        """Use native snapshot metadata after this program's live guard passes.

        The program already guards the current Parameter and Polar declaration.
        Scalar validity, detachment and non-aliasing still apply at both call
        boundaries. Unsupported layouts/types use the original validator and
        request ordinary dispatch rather than a mismatched compiled launch.
        """
        if self.snapshot_guard is None:
            self.binding.validate_inputs(inputs)
            return True
        if type(inputs) is self.binding.input_type:
            rate, decay = inputs.step_size, inputs.amplitude_weight_decay
            valid_scalars = (
                type(rate) in (int, float)
                and math.isfinite(rate)
                and rate > 0
                and type(decay) in (int, float)
                and math.isfinite(decay)
                and decay >= 0
                and rate * decay <= 1
            )
            if valid_scalars and decay:
                valid_scalars = type(self.binding.kernel.spec.parameterization) in (
                    _parameterizations.PolarAmpWidthSpec,
                    _parameterizations.PolarAmpRadiusSpec,
                )
            if (
                valid_scalars
                and self.snapshot_guard.check(inputs.previous)
                and not torch._C._is_alias_of(inputs.previous, self.point)
            ):
                return True
        # Preserve the ordinary validation errors, including custom or invalid
        # inputs. A valid noncontiguous input is unsupported by this fast path.
        self.binding.validate_inputs(inputs)
        return False

    def _matches(self, dispatcher, adapter, point, decay, *, checked=False):
        if (
            dispatcher is not self.dispatcher
            or point is not self.point
            or type(adapter) is not _optimizer_state.OptimizerStateAdapter
        ):
            return False
        if not checked and not self.guard.check(self.roots):
            return False
        if (
            "transport" in adapter.__dict__
            or type(adapter.vector_keys) is not tuple
            or adapter.vector_keys != self.adapter_keys
            or _finite.checks_enabled()
            or (decay != 0) != self.context.amplitude_decay
            or torch._C._is_torch_function_mode_enabled()
            or torch._C._len_torch_dispatch_stack()
            or (self.launch is not None and not self.launch.enabled())
        ):
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        # The combined native guard already checked this live fixed tree.
        # Changed metadata returns to ordinary declaration rebuilding.
        fixed = self.binding.kernel.__dict__.get("_fixed_signature_cache")
        return (
            self.binding.__dict__.get("_execution_declaration") is self.declaration
            and fixed is not None
            and fixed.guard is self.fixed_guard
            and fixed.guard is not None
        )

    def preflight(self, dispatcher, adapter, point, gradient, decay):
        """The certified FlatTorus gradient projection is an identity map."""
        if self.snapshot_guard is not None:
            # Ordinary gradients have the same detached, contiguous metadata as
            # old-point snapshots. This guard owns no gradient or example data.
            if not self.snapshot_guard.check(gradient):
                return False
        elif (
            type(point) is not torch.nn.Parameter
            or type(gradient) is not torch.Tensor
            or gradient.layout != torch.strided
            or gradient.shape != point.shape
            or gradient.dtype != point.dtype
            or gradient.device != point.device
        ):
            return False
        return self._matches(dispatcher, adapter, point, decay)

    def run(self, dispatcher, adapter, inputs, state):
        """Execute the same selected CUDA update; None requests generic fallback."""
        if not self.guard.check(self.roots):
            return None
        if (
            type(inputs) is not self.binding.input_type
            or type(inputs.previous) is not torch.Tensor
        ):
            return None
        if (
            torch._C._is_torch_function_mode_enabled()
            or torch._C._len_torch_dispatch_stack()
        ):
            return None
        # With canonical callbacks and ordinary Tensors, input validation only
        # reads metadata; it cannot change a previously checked callback table.
        if self.validate_inputs(inputs) is False:
            return None
        if state is not self.owner.optimizer_state.fields or not self._matches(
            dispatcher, adapter, self.point, inputs.amplitude_weight_decay, checked=True
        ):
            return None
        if self.launch is not None:
            self.transport_required = self.launch.hooks_active()
            if not self.launch.run(inputs.previous, self.point, inputs):
                return None
            result = self.point.detach()
        else:
            # The generic Triton entrance may run user pre-run/launch hooks.
            # Recheck transport after those hooks, just as for a hooked runner.
            self.transport_required = True
            result = self.state.algorithm.execute(self.state, inputs)
        result = self.binding.atom_state.protect_tensor(result)
        # The certified standard FlatTorus exp_avg transport is also identity.
        return result

    def transport_is_identity(self, dispatcher, adapter, point, inputs, state):
        """Recheck after launch hooks, which may change callbacks or metadata."""
        return state is self.owner.optimizer_state.fields and self._matches(
            dispatcher, adapter, point, inputs.amplitude_weight_decay
        )


def prepare_program(
    dispatcher, cached, adapter, site=None, *, _function_factory=_MISSING
):
    """Compile only a supported, already-prepared standard radius update."""
    from torchcst._backends.dispatch.runtime import Dispatcher
    from torchcst._backends.dispatch.select import FixedSelector
    from torchcst._backends.registry import Registry
    from torchcst._backends.schema import DefaultRecipe
    from torchcst._backends.state import AlgorithmState
    from torchcst._backends.torch import finite
    from torchcst.charts import RegularGridChartState
    from torchcst.geometry.state import GeometryState
    from torchcst.nn import CSTLinear
    from torchcst.operators.atom_update import AtomUpdateBinding
    from torchcst.operators.binding import Operator
    from torchcst.optim.state import OptimizerStateAdapter

    from .algorithm import RadiusFlatTorusPolarUpdateAlgorithm

    binding, state = cached.state.binding, cached.state
    if (
        type(state) is not AlgorithmState
        or type(dispatcher) is not Dispatcher
        or type(dispatcher.registry) is not Registry
        or any(
            name in state.__dict__
            for name in ("_signature", "is_current", "mark_current", "invalidate")
        )
        or any(
            name in cached.algorithm.__dict__
            for name in (
                "run",
                "execute",
                "prepare",
                "supports",
                "validate_inputs",
                "validate_recipe",
            )
        )
        or any(
            name in dispatcher.__dict__
            for name in ("_prepare", "prepare", "run", "select")
        )
        or any(
            name in dispatcher.registry.__dict__ for name in ("validate_plan", "get")
        )
        or type(binding) is not AtomUpdateBinding
        or type(binding).__getattribute__ is not object.__getattribute__
        or type(binding.operator) is not Operator
        or type(site) is not CSTLinear
        or type(site).__getattribute__ is not object.__getattribute__
        or type(site).__getattr__ is not torch.nn.Module.__getattr__
        or type(cached.algorithm) is not RadiusFlatTorusPolarUpdateAlgorithm
        or state.algorithm is not cached.algorithm
        or type(adapter) is not OptimizerStateAdapter
        or "transport" in adapter.__dict__
        or type(adapter.vector_keys) is not tuple
        or adapter.vector_keys != ("exp_avg",)
        or type(dispatcher.selector) is not FixedSelector
        or dispatcher.selector._plan is not cached.decision.plan
        or type(state.recipe) is not DefaultRecipe
        or any(
            name in binding.__dict__
            for name in (
                "execution_declaration",
                "validate_inputs",
                "build_context",
                "execution_parameters",
                "cst_charts",
                "_scalar_metadata",
                "state_signature",
                "prepared_dispatch_signature",
            )
        )
    ):
        return None
    callbacks = (
        *_callbacks(),
        (
            type(cached.algorithm),
            (
                "execute",
                "validate_recipe",
                "validate_inputs",
                "prepare",
                "supports",
                "workspace_bound",
            ),
        ),
        (type(binding.operator.atoms), ()),
        (type(binding.atom_state), ()),
        (type(binding.atom_state.optimizer_state), ()),
        (type(dispatcher.selector), ()),
        (GeometryState, ()),
        (CSTLinear, ("cst_charts",)),
        (binding.input_type, ()),
        (torch.nn.Parameter, ()),
    )
    # Do not execute a custom validator merely to decide whether to specialize.
    functions = _definitions(callbacks)
    if functions is None:
        return None
    if finite.checks_enabled() or not all(
        hasattr(torch._C, name)
        for name in ("_is_torch_function_mode_enabled", "_len_torch_dispatch_stack")
    ):
        return None
    if (
        torch._C._is_torch_function_mode_enabled()
        or torch._C._len_torch_dispatch_stack()
    ):
        return None
    point = binding.execution_parameters()
    fixed = binding.kernel.__dict__.get("_fixed_signature_cache")
    if (
        fixed is None
        or site.kernel is not binding.kernel
        or any(x is not y for x, y in zip(site.cst_charts(), binding.cst_charts()))
        or any(
            inspect.getattr_static(GeometryState, name, _MISSING) is not _MISSING
            for name in ("embedding_dim", "center_parameter_dim")
        )
        or fixed.guard is None
        or not point.is_cuda
        or point.dtype != torch.float32
        or any(
            type(chart) is not RegularGridChartState
            or type(chart.geometry) is not GeometryState
            for chart in binding.cst_charts()
        )
        or not state.is_current()
        or not cached.algorithm.supports(
            cached.context, cached.decision.plan.recipe
        ).supported
    ):
        return None
    if _function_factory is _MISSING:
        from torchcst._backends.cuda._host_function_metadata import get_factory

        # Import/compilation is cold and may execute external build machinery.
        # Re-enter all eligibility/state checks afterwards rather than trusting
        # metadata observed before that work. The factory is process-cached.
        factory = get_factory()
        return prepare_program(
            dispatcher, cached, adapter, site, _function_factory=factory
        )
    from .compiled_launch import prepare_launch

    launch = prepare_launch(binding, state) if point.numel() else None
    callbacks = (
        *callbacks,
        *((type(chart), ()) for chart in binding.cst_charts()),
        *((launch.guarded_owners) if launch is not None else ()),
    )
    guard = _native_guard(
        binding,
        dispatcher,
        cached,
        callbacks,
        functions,
        site,
        function_factory=_function_factory,
    )
    if guard is None:
        return None
    return PreparedPolarOptimizerUpdate(binding, dispatcher, cached, guard, launch)
