"""Reuse group validation only while its complete CPU metadata contract holds.

This cache contains parameter IDs and existing group references, never tensor
values or invocation buffers. Every call checks registrations, group contents
and callback definitions. Changed or custom configurations use fresh validation.
"""

import gc
import inspect

import torch

# Capture the standard walkers when the optimizer module is first imported.
_WALKERS = tuple(
    (fn, fn.__code__, fn.__defaults__, fn.__kwdefaults__)
    for fn in (
        torch.nn.Module.modules,
        torch.nn.Module.named_modules,
        torch.nn.Module.named_parameters,
        torch.nn.Module._named_members,
        torch.nn.Module.__getattr__,
        torch.Tensor.__hash__,
    )
)


class GroupValidation:
    def __init__(self, guard, optimizer, groups, mapping):
        self.guard = guard
        self.optimizer = optimizer
        self.groups = groups
        self.mapping = mapping

    def current(self, optimizer, groups):
        return (
            optimizer is self.optimizer
            and groups is self.groups
            and not torch._C._is_torch_function_mode_enabled()
            and not torch._C._len_torch_dispatch_stack()
            and self.guard.check(optimizer)
        )

    def __getstate__(self):
        # Optional metadata is reconstructed after serialization/deepcopy.
        return {}

    def __setstate__(self, state):
        self.guard = self.optimizer = self.groups = self.mapping = None


def _prepare(optimizer, groups, mapping):
    """Build after successful fresh validation; unsupported APIs return None."""
    from torchcst._builtin_callbacks import original_functions
    from torchcst.charts.base import ChartSpec, ChartState
    from torchcst.charts.regular_grid import RegularGridChartSpec, RegularGridChartState
    from torchcst.nn.linear import CSTLinear
    from torchcst.nn.module import CSTModule
    from torchcst.operators._native_metadata_guard import (
        _plain_owner,
        guard_api,
        guard_callable_metadata,
    )
    from torchcst.operators.metadata import _standard_walk

    from .optimizer import CSTOptimizer

    api = guard_api()
    if (
        api is None
        or type(optimizer) is not CSTOptimizer
        or optimizer.check_finite is not False
        or type(optimizer._allows_amplitude_decay) is not bool
        or groups is not optimizer.param_groups
        or type(groups) is not list
        or "_validate_groups" in optimizer.__dict__
        or type(optimizer._sites) is not tuple
        or not all(type(site) is CSTLinear for site in optimizer._sites)
        or any(
            not hasattr(torch._C, name)
            for name in ("_is_torch_function_mode_enabled", "_len_torch_dispatch_stack")
        )
        or torch._C._is_torch_function_mode_enabled()
        or torch._C._len_torch_dispatch_stack()
    ):
        return None
    callbacks = (
        CSTOptimizer._validate_groups,
        CSTLinear.cst_charts,
        CSTModule.atoms.fget,
        ChartState.trainable.fget,
        ChartSpec.trainable.fget,
    )
    if any(original_functions(fn) is None for fn in callbacks):
        return None
    if any(
        fn.__code__ is not code
        or fn.__defaults__ is not defaults
        or fn.__kwdefaults__ is not keywords
        for fn, code, defaults, keywords in _WALKERS
    ):
        return None
    for group in groups:
        if (
            type(group) is not dict
            or any(type(key) is not str for key in group)
            or type(group.get("params")) is not list
            or type(group.get("differentiable", False)) is not bool
        ):
            return None
    model = optimizer.model
    if (
        type(model).named_parameters is not _WALKERS[2][0]
        or type(model)._named_members is not _WALKERS[3][0]
        or "named_parameters" in model.__dict__
        or "_named_members" in model.__dict__
    ):
        return None
    # Inspect registrations without calling an unknown root/child walker a
    # second time merely to decide whether caching is supported.
    modules, visited = [], set()

    def collect(module):
        if id(module) in visited:
            return True
        if not _plain_owner(module, _standard_walk):
            return False
        visited.add(id(module))
        modules.append(module)
        return all(
            child is None or collect(child) for child in module._modules.values()
        )

    if not collect(model):
        return None
    module_ids = {id(module) for module in modules}
    if any(id(site) not in module_ids for site in optimizer._sites):
        return None
    if any(
        "chart" in site.__dict__ or "chart" in site._modules
        for site in optimizer._sites
    ):
        return None
    if any(
        key in module.__dict__
        for module in modules
        for table in ("_parameters", "_modules")
        for key in module.__dict__[table]
    ):
        return None
    if any(
        name in site.__dict__
        for site in optimizer._sites
        for name in ("cst_charts", "atoms")
    ):
        return None
    if any(
        p is not None and type(p) is not torch.nn.Parameter
        for module in modules
        for p in module._parameters.values()
    ):
        return None
    charts = tuple(chart for site in optimizer._sites for chart in site.cst_charts())
    if any(
        type(chart) is not RegularGridChartState
        or type(chart.spec) is not RegularGridChartSpec
        or inspect.getattr_static(type(chart), "trainable") is not ChartState.trainable
        for chart in charts
    ):
        return None

    root_type, kind = api
    guard = root_type()
    verbose = ["optimizer groups and parameter ownership"]

    def attr(node, name, value):
        return node.getattr_manager(name, name, value, kind)

    def item(node, name, value):
        return node.dict_getitem_manager(name, str(name), value, kind)

    def same(node, value):
        node.add_id_match_guard(id(value), verbose, None)

    seen_classes, seen_bases = set(), set()

    def class_guard(node, value):
        cls = type(value)
        node.add_type_match_guard(id(cls), verbose, None)
        cm = node.type_manager("type", cls, kind)
        if cls in seen_classes:
            return
        seen_classes.add(cls)
        mro = cm.type_mro_manager("bases", cls.__mro__, kind)
        mro.add_length_check_guard(len(cls.__mro__), verbose, None)
        for index, base in enumerate(cls.__mro__):
            bn = mro.tuple_getitem_manager(index, "base", base, kind)
            same(bn, base)
            if base in seen_bases:
                continue
            seen_bases.add(base)
            if base.__flags__ & (1 << 8):
                continue
            raw = next(v for v in gc.get_referents(base.__dict__) if type(v) is dict)
            bn.type_dict_manager("namespace", raw, kind).add_dict_version_guard(
                raw, verbose, None
            )

    seen_modules = set()
    chart_ids = {id(chart) for chart in charts}
    if any(id(chart) not in module_ids for chart in charts):
        return None

    def walk(node, module):
        same(node, module)
        class_guard(node, module)
        if id(module) in seen_modules:
            return
        seen_modules.add(id(module))
        if id(module) in chart_ids:
            spec = module.spec
            spec_node = attr(node, "spec", spec)
            same(spec_node, spec)
            class_guard(spec_node, spec)
            # Exact, certified ChartSpec.trainable is the constant False. No
            # Python property call or tensor-buffer checks are needed here.
        ns = attr(node, "__dict__", module.__dict__)
        for name in ("modules", "named_modules", "named_parameters", "_named_members"):
            ns.add_dict_contains_guard(False, name, verbose, None)
        for name in ("_parameters", "_modules"):
            values = module.__dict__[name]
            dn = item(ns, name, values)
            same(dn, values)
            dn.add_dict_version_guard(values, verbose, None)
            for key in values:
                ns.add_dict_contains_guard(False, key, verbose, None)
            if name == "_parameters":
                for key, parameter in values.items():
                    if parameter is not None:
                        pn = item(dn, key, parameter)
                        same(pn, parameter)
                        class_guard(pn, parameter)
            if name == "_modules":
                for key, child in values.items():
                    if child is not None:
                        walk(item(dn, key, child), child)

    try:
        class_guard(guard, optimizer)
        ns = attr(guard, "__dict__", optimizer.__dict__)
        ns.add_dict_contains_guard(False, "_validate_groups", verbose, None)
        same(attr(guard, "check_finite", False), False)
        same(
            attr(guard, "_allows_amplitude_decay", optimizer._allows_amplitude_decay),
            optimizer._allows_amplitude_decay,
        )
        sites = attr(guard, "_sites", optimizer._sites)
        same(sites, optimizer._sites)
        walk(attr(guard, "model", model), model)
        gn = attr(guard, "param_groups", groups)
        same(gn, groups)
        gn.add_length_check_guard(len(groups), verbose, None)
        for index, group in enumerate(groups):
            node = gn.list_getitem_manager(index, "group", group, kind)
            same(node, group)
            node.add_dict_version_guard(group, verbose, None)
            params = group["params"]
            pn = item(node, "params", params)
            same(pn, params)
            pn.add_length_check_guard(len(params), verbose, None)
            for j, p in enumerate(params):
                same(pn.list_getitem_manager(j, "parameter", p, kind), p)
        for index, site in enumerate(optimizer._sites):
            sn = sites.tuple_getitem_manager(index, "site", site, kind)
            same(sn, site)
            for name in ("cst_charts", "atoms", "chart"):
                attr(sn, "__dict__", site.__dict__).add_dict_contains_guard(
                    False, name, verbose, None
                )
            # Existing module registration checks locate each chart. Check its
            # live trainable property as well; geometry is irrelevant when False.
            for chart in site.cst_charts():
                if chart.trainable is not False:
                    return None
        # The function tuple is constant, outside dictionary-version shortcuts.
        behavior = (
            *_WALKERS,
            *(
                (fn, fn.__code__, fn.__defaults__, fn.__kwdefaults__)
                for fn in callbacks
            ),
        )
        guard_callable_metadata(guard, behavior, kind, verbose)
        globals_seen = set()
        for fn in (*[row[0] for row in _WALKERS], *callbacks):
            namespace = fn.__globals__
            if id(namespace) in globals_seen:
                continue
            globals_seen.add(id(namespace))
            node = guard.lambda_manager(
                lambda _, ns=namespace: ns, "callback globals", namespace, kind
            )
            node.add_dict_version_guard(namespace, verbose, None)
        # Validation calls math.isfinite through this module. Its binding can
        # change without replacing the wrapper's global `math` module itself.
        namespace = callbacks[0].__globals__["math"].__dict__
        guard.lambda_manager(
            lambda _: namespace, "math callbacks", namespace, kind
        ).add_dict_version_guard(namespace, verbose, None)
        return (
            GroupValidation(guard, optimizer, groups, mapping)
            if guard.check(optimizer)
            else None
        )
    except (AttributeError, RuntimeError, TypeError, StopIteration, ValueError):
        return None


def prepare(optimizer, groups, mapping):
    """Unsupported metadata/private APIs keep the fresh Python validator."""
    try:
        return _prepare(optimizer, groups, mapping)
    except (AttributeError, RuntimeError, TypeError, StopIteration, ValueError):
        return None
