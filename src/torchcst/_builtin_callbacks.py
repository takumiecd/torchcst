"""Identify library definitions eligible for an optional prepared implementation.

An unchanged function is a capability, not a claim based on its name. A wrapper
which copies attributes with functools.wraps still has a different identity.
This metadata holds functions/code only, never model or invocation tensors.
"""

from types import FunctionType

_MISSING = object()


def _cell_value(cell):
    try:
        return cell.cell_contents
    except ValueError:
        return _MISSING


def _snapshot(fn):
    keywords = fn.__kwdefaults__
    return (
        fn,
        fn.__code__,
        fn.__defaults__,
        keywords,
        None if keywords is None else tuple(keywords.items()),
        tuple(
            (name, fn.__globals__.get(name, _MISSING)) for name in fn.__code__.co_names
        ),
        tuple(_cell_value(cell) for cell in fn.__closure__ or ()),
        getattr(fn, "__wrapped__", _MISSING),
    )


def _functions(function):
    seen, result = set(), []

    def visit(fn):
        if type(fn) is not FunctionType or id(fn) in seen:
            return
        seen.add(id(fn))
        result.append(fn)
        visit(getattr(fn, "__wrapped__", None))
        for name in fn.__code__.co_names:
            dependency = fn.__globals__.get(name)
            if type(dependency) is FunctionType and dependency.__module__.startswith(
                "torchcst."
            ):
                visit(dependency)
        for cell in fn.__closure__ or ():
            try:
                visit(cell.cell_contents)
            except ValueError:  # An uninitialized closure cell.
                pass

    visit(function)
    return tuple(result)


def seal_functions(namespace):
    """Mark original module functions/classes at their definition boundary."""
    module = namespace["__name__"]
    for value in tuple(namespace.values()):
        if type(value) is FunctionType and value.__module__ == module:
            functions = (value,)
        elif isinstance(value, type) and value.__module__ == module:
            functions = tuple(
                fn.fget if isinstance(fn, property) else fn
                for fn in vars(value).values()
                if type(fn) is FunctionType or isinstance(fn, property)
            )
        else:
            continue
        for function in functions:
            snapshots = tuple(_snapshot(fn) for fn in _functions(function))
            function.__dict__["_cst_builtin"] = (function, snapshots)


def original_functions(function):
    """Return certified definitions, or None for custom/modified callbacks."""
    if type(function) is not FunctionType:
        return None
    marker = function.__dict__.get("_cst_builtin")
    if marker is None or marker[0] is not function:
        return None
    for fn, code, defaults, keywords, keyword_items, bindings, cells, wrapped in marker[
        1
    ]:
        if (
            fn.__code__ is not code
            or fn.__defaults__ is not defaults
            or fn.__kwdefaults__ is not keywords
            or getattr(fn, "__wrapped__", _MISSING) is not wrapped
            or any(
                fn.__globals__.get(name, _MISSING) is not value
                for name, value in bindings
            )
            or len(fn.__closure__ or ()) != len(cells)
            or any(
                _cell_value(cell) is not value
                for cell, value in zip(fn.__closure__ or (), cells)
            )
            or (
                keywords is not None
                and (
                    tuple(keywords) != tuple(name for name, _ in keyword_items)
                    or any(keywords[name] is not value for name, value in keyword_items)
                )
            )
        ):
            return None
    return tuple(row[0] for row in marker[1])
