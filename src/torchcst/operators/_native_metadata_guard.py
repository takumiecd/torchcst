"""Optional CPU-only PyTorch guards for fixed configuration metadata.

No tracing, compilation, Tensor values, or CUDA API is used. Private guard APIs
are feature-detected; unsupported runtimes keep the Python validation path.
"""

import gc
import inspect
import sys
from abc import ABCMeta
from functools import cache

import torch

_MISSING = object()


@cache
def _guard_type():
    try:
        from torch._C._dynamo.guards import RootGuardManager
    except (ImportError, AttributeError):
        return None
    return RootGuardManager


def _plain_owner(module, standard_walk):
    cls = type(module)
    if (
        not standard_walk(module)
        or type(cls) not in (type, ABCMeta)
        or cls.__getattribute__ is not object.__getattribute__
        or cls.__getattr__ is not torch.nn.Module.__getattr__
        or type(module._parameters) is not dict
        or inspect.getattr_static(cls, "__dict__", None)
        is not torch.nn.Module.__dict__["__dict__"]
    ):
        return False
    # Attribute providers must keep their callbacks on each Python guard.
    if any(
        inspect.getattr_static(cls, name, _MISSING) is not _MISSING
        for name in ("spec", "binding", "_buffers", "_modules", "_parameters")
    ):
        return False
    return all(
        type(t) is torch.Tensor for t in module._buffers.values() if t is not None
    )


def guard_api():
    """Use guards once PyTorch loaded them; never import its tracing stack."""
    module = sys.modules.get("torch._dynamo.guards")
    kind = getattr(getattr(module, "GuardManagerType", None), "GUARD_MANAGER", None)
    cls = _guard_type()
    return (cls, kind) if cls is not None and kind is not None else None


def make_guard(roots, modules, standard_walk, api):
    """Build metadata checks, retaining neither examples nor input snapshots."""
    if api is None or not all(_plain_owner(m, standard_walk) for m in modules):
        return None
    try:
        cls, kind = api
        guard = _build_guard(cls(), roots, kind)
        # Construction alone does not qualify version-dependent accessors.
        return guard if guard.check(roots) else None
    except (AttributeError, TypeError, RuntimeError, StopIteration):
        # The optional private interface varies across supported PyTorch/Python
        # versions. No cached result is used if construction is unsupported.
        return None


def _build_guard(guard, roots, kind):
    verbose = ["CST fixed metadata"]
    seen_modules, seen_classes, seen_class_dicts = set(), set(), set()
    guard.add_length_check_guard(len(roots), verbose, None)
    behavior = tuple(
        (fn, fn.__code__, fn.__defaults__, fn.__kwdefaults__)
        for fn in (
            torch.nn.Module.modules,
            torch.nn.Module.named_modules,
            torch.nn.Module.__getattr__,
        )
    )

    def callbacks_unchanged(_):
        # Function objects can change code/defaults without changing their
        # containing dictionary. Keep this outside dictionary fast checks.
        return all(
            fn.__code__ is code
            and fn.__defaults__ is defaults
            and fn.__kwdefaults__ is keywords
            for fn, code, defaults, keywords in behavior
        )

    guard.add_epilogue_lambda_guard(callbacks_unchanged, verbose, None)

    def attribute(manager, name, value):
        return manager.getattr_manager(name, name, value, kind)

    def item(manager, name, value):
        return manager.dict_getitem_manager(name, str(name), value, kind)

    def identity(manager, value):
        manager.add_id_match_guard(id(value), verbose, None)

    def class_guards(manager, value):
        current = type(value)
        cm = manager.type_manager("type", current, kind)
        identity(cm, current)
        if current in seen_classes:
            return
        seen_classes.add(current)
        mro = cm.type_mro_manager("mro", current.__mro__, kind)
        mro.add_length_check_guard(len(current.__mro__), verbose, None)
        for index, base in enumerate(current.__mro__):
            bm = mro.tuple_getitem_manager(index, "base", base, kind)
            identity(bm, base)
            if base in seen_class_dicts:
                continue
            seen_class_dicts.add(base)
            # CPython's Py_TPFLAGS_IMMUTABLETYPE: these namespaces cannot be
            # monkeypatched. Some Python/PyTorch pairs cannot expose the
            # dictionary of static builtins through TypeDictGuardAccessor.
            if base.__flags__ & (1 << 8):
                continue
            # CPython exposes a mappingproxy for a class namespace. Its GC
            # referent is the actual dict needed by PyTorch's version guard.
            raw = next(v for v in gc.get_referents(base.__dict__) if type(v) is dict)
            dm = bm.type_dict_manager("class namespace", raw, kind)
            dm.add_dict_version_guard(raw, verbose, None)
            if base is torch.nn.Module:
                # Changing a function's code/defaults does not change its
                # class dictionary. These callbacks must still fall back.
                for name in ("modules", "named_modules", "__getattr__"):
                    function = base.__dict__[name]
                    fm = item(dm, name, function)
                    for field in ("__code__", "__defaults__", "__kwdefaults__"):
                        value = getattr(function, field)
                        child = attribute(fm, field, value)
                        identity(child, value)
                        if type(value) is dict:
                            child.add_dict_version_guard(value, verbose, None)

    def walk(manager, module):
        identity(manager, module)
        class_guards(manager, module)
        if id(module) in seen_modules:
            return
        seen_modules.add(id(module))
        namespace = attribute(manager, "__dict__", module.__dict__)
        for name in ("spec", "binding", "modules", "named_modules"):
            if name in module.__dict__:
                identity(
                    item(namespace, name, module.__dict__[name]), module.__dict__[name]
                )
            else:
                namespace.add_dict_contains_guard(False, name, verbose, None)
        for name in ("_buffers", "_modules", "_parameters"):
            values = getattr(module, name)
            dm = attribute(manager, name, values)
            dm.add_dict_version_guard(values, verbose, None)
            if name == "_buffers":
                for key, tensor in values.items():
                    if tensor is None:
                        continue
                    tm = item(dm, key, tensor)
                    identity(tm, tensor)
                    class_guards(tm, tensor)
                    for field in ("_version", "dtype", "device", "shape", "layout"):
                        metadata = getattr(tensor, field)
                        attribute(tm, field, metadata).add_equals_match_guard(
                            metadata, verbose, None
                        )
            elif name == "_modules":
                for key, child in values.items():
                    if child is not None:
                        walk(item(dm, key, child), child)

    for index, root in enumerate(roots):
        walk(guard.tuple_getitem_manager(index, "root", root, kind), root)
    return guard
