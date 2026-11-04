"""Weak, cold-built native checks for three stock whole-step signature columns.

Call only after the common stock/callback qualification. This certificate does
not replace optimizer, grad, input, period, TLS or hook checks. Its owner must
also ensure its current model is ``model_ref()``; check() cannot observe an
unrelated owner's model attribute being rebound.
"""

import importlib
import inspect
import weakref

import torch

from torchcst._backends.cuda import _host_function_metadata as host
from torchcst.operators._native_metadata_guard import guard_api, tensor_metadata_guard


class LeavesUnavailable(ValueError):
    """The installed private CPU guard interface cannot certify these leaves."""


class GraphSignatureLeaves:
    def __init__(self, model):
        try:
            self._build(model)
        except LeavesUnavailable:
            raise
        except (AttributeError, TypeError, RuntimeError, ValueError, ImportError) as e:
            raise LeavesUnavailable("native signature leaves unavailable") from e

    def _build(self, model):
        importlib.import_module("torch._dynamo.guards")
        api = guard_api()
        if api is None:
            raise LeavesUnavailable("RootGuardManager is unavailable")
        factory = host.get_factory()
        if factory is None:
            raise LeavesUnavailable("qualified native storage checker is unavailable")
        cls, kind = api
        guard, refs, nodes = cls(), [], {}
        verbose = ["whole-step signature leaves"]
        modules, dictionaries, tensors = [], [], []
        parameter_names, buffer_names = [], []
        seen_modules, seen_parameters, seen_buffers = set(), set(), set()
        missing = object()

        def collect(module, path):
            if id(module) in seen_modules:
                return
            seen_modules.add(id(module))
            if (
                not isinstance(module, torch.nn.Module)
                or type(module).__getattribute__ is not object.__getattribute__
                or type(module).__getattr__ is not torch.nn.Module.__getattr__
                or type(module.__dict__) is not dict
            ):
                raise LeavesUnavailable("plain qualified Modules required")
            modules.append((module, path))
            for field, seen, names in (
                ("_parameters", seen_parameters, parameter_names),
                ("_buffers", seen_buffers, buffer_names),
                ("_modules", None, None),
            ):
                if inspect.getattr_static(type(module), field, missing) is not missing:
                    raise LeavesUnavailable(
                        "registration attribute provider unsupported"
                    )
                table = module.__dict__[field]
                if type(table) is not dict or any(type(k) is not str for k in table):
                    raise LeavesUnavailable("exact ordered registration dict required")
                dictionaries.append(table)
                for key, value in table.items():
                    if value is None:
                        continue
                    if field == "_modules":
                        if not isinstance(value, torch.nn.Module):
                            raise LeavesUnavailable("invalid registered Module")
                        continue
                    allowed = (
                        (torch.nn.Parameter,)
                        if field == "_parameters"
                        else (torch.Tensor, torch.nn.Parameter)
                    )
                    if type(value) not in allowed:
                        raise LeavesUnavailable("Tensor subclasses are unsupported")
                    if id(value) in seen:
                        continue
                    seen.add(id(value))
                    name = path + "." + key if path else key
                    # named_buffers deduplicates BEFORE the old name filter.
                    if field == "_buffers" and ".optimizer_state." in name:
                        continue
                    names.append(name)
                    tensors.append((value, field == "_buffers"))
            for key, child in module.__dict__["_modules"].items():
                if child is not None:
                    collect(child, path + "." + key if path else key)

        collect(model, "")
        epoch = host.prepare_dictionary_checks(factory, tuple(dictionaries))

        def same(node, value):
            node.add_type_match_guard(id(type(value)), verbose, None)
            node.add_id_match_guard(id(value), verbose, None)
            if value is not None:
                pin(value)  # Live weak witness makes object-id checks ABA safe.

        def pin(value):
            if id(value) in nodes:
                return nodes[id(value)]
            reference = weakref.ref(value)
            index = len(refs)
            refs.append(reference)
            node = guard.tuple_getitem_manager(index, "weak leaf", reference, kind)
            node = node.weakref_call_manager("live leaf", value, kind)
            nodes[id(value)] = node
            node.add_not_none_guard(verbose, None)
            node.add_id_match_guard(id(value), verbose, None)
            node.add_type_match_guard(id(type(value)), verbose, None)
            # A dynamic original class must not be replaceable via id reuse.
            pin(type(value))
            return node

        for module, _ in modules:
            node = pin(module)
            namespace = node.getattr_manager(
                "__dict__", "Module dict", module.__dict__, kind
            )
            for field in ("_modules", "_parameters", "_buffers"):
                table = module.__dict__[field]
                dm = namespace.dict_getitem_manager(field, field, table, kind)
                dm.add_type_match_guard(id(dict), verbose, None)
                dm.add_id_match_guard(id(table), verbose, None)
                if epoch is None:
                    # Version plus owner->dict identity; never id-only for dicts.
                    dm.add_dict_version_guard(table, verbose, None)
                for key, value in table.items():
                    same(dm.dict_getitem_manager(key, key, value, kind), value)
            training = module.training
            if type(training) is not bool:
                raise LeavesUnavailable("boolean training state required")
            node.getattr_manager(
                "training", "training", training, kind
            ).add_equals_match_guard(training, verbose, None)
            if "forward" in module.__dict__:
                value = module.__dict__["forward"]
                same(
                    namespace.dict_getitem_manager(
                        "forward", "instance forward", value, kind
                    ),
                    value,
                )
            else:
                namespace.add_dict_contains_guard(False, "forward", verbose, None)
            cls_node = node.type_manager("Module type", type(module), kind)
            value = type(module).forward
            same(
                cls_node.getattr_manager("forward", "class forward", value, kind), value
            )

        storage = []
        for value, versioned in tensors:
            node = pin(
                value
            )  # Outside dict fast paths: metadata changes independently.
            if not tensor_metadata_guard(node, value, "registered Tensor", verbose):
                raise LeavesUnavailable("TensorMatch is unavailable")
            for field in (
                ("requires_grad", "_version") if versioned else ("requires_grad",)
            ):
                expected = getattr(value, field)
                node.getattr_manager(
                    field, field, expected, kind
                ).add_equals_match_guard(expected, verbose, None)
            storage.append(value)
        storage_check = host.prepare_storage_checks(factory, tuple(storage))
        if storage_check is None:
            raise LeavesUnavailable("native weak storage qualification failed")
        if epoch is not None:
            guard.add_lambda_guard(epoch, verbose, None)
        guard.add_length_check_guard(len(refs), verbose, None)
        self.model_ref = weakref.ref(model)
        self.parameter_names = tuple(parameter_names)
        self.buffer_names = tuple(buffer_names)
        self.module_names = tuple(path for _, path in modules)
        self._guard, self._refs, self._storage = guard, tuple(refs), storage_check
        self._valid = True
        if not self.check():
            raise LeavesUnavailable("constructed certificate did not qualify")

    def check(self):
        if not self._valid:
            return False
        try:
            self._valid = bool(self._storage(None) and self._guard.check(self._refs))
        except (AttributeError, TypeError, RuntimeError, ValueError, LookupError):
            self._valid = False
        return self._valid
