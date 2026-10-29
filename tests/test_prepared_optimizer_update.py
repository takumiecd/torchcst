"""Prepared updates retain live metadata/callback boundaries and own no inputs."""

import gc
import importlib
import weakref
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
from test_atom_update_metadata_cache import inputs
from test_radius_polar_cuda_update import radius_model

from torchcst import CSTOptimizer, FixedSelector
from torchcst._backends.catalog import get_registry
from torchcst._backends.cuda.algorithms.polar_update.optimizer_program import (
    _callbacks,
    _definitions,
    _native_guard,
)
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)
from torchcst._backends.dispatch.runtime import Dispatcher
from torchcst._backends.state import AlgorithmState
from torchcst._builtin_callbacks import original_functions


def example(*, function_factory=None):
    importlib.import_module("torch._dynamo.guards")
    model = radius_model()
    opt = CSTOptimizer(
        torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0),
        model=model,
        check_finite=False,
    )
    model.atoms.p.grad = torch.ones_like(model.atoms.p) * 0.01
    opt.step()
    binding = opt._update_bindings[model]
    registry = get_registry()
    plan = replace(RADIUS_FLAT_TORUS_FUSED)
    dispatcher = Dispatcher(selector=FixedSelector(plan, registry=registry))
    algorithm = registry.get(plan.algorithm_id, revision=plan.algorithm_revision)
    state = AlgorithmState(binding, algorithm, recipe=plan.recipe)
    state.context = binding.build_context(inputs(model, decay=0.01))
    state.mark_current()
    cached = SimpleNamespace(
        state=state,
        algorithm=algorithm,
        context=state.context,
        decision=SimpleNamespace(plan=plan),
    )
    callbacks = (
        *_callbacks(),
        (type(algorithm), ("execute", "validate_recipe", "validate_inputs", "prepare")),
    )
    functions = _definitions(callbacks)
    assert functions is not None
    native = _native_guard(
        binding,
        dispatcher,
        cached,
        callbacks,
        functions,
        model,
        function_factory=function_factory,
    )
    if native is None:
        pytest.skip("optional native optimizer guard unavailable")
    return model, opt, dispatcher, cached, native


def test_native_storage_guard_accepts_numeric_updates_but_not_data_replacement():
    model, _opt, _dispatcher, _cached, (guard, roots) = example()
    assert guard.check(roots)
    with torch.no_grad():
        model.atoms.p.add_(0.01)
        for value in model.atom_state.optimizer_state.fields.values():
            value.add_(0.01)
    assert guard.check(roots)
    p = model.atoms.p
    with torch.no_grad():
        p.data = p.data.clone()
    assert not guard.check(roots)


def test_function_wrapper_copying_annotations_is_not_a_builtin():
    from torchcst._backends.torch.profiles import execution

    original = execution.project_gradient
    assert original_functions(original) is not None

    @wraps(original)
    def wrapped(*a, **kw):
        return original(*a, **kw)

    assert original_functions(wrapped) is None


@pytest.mark.parametrize(
    "target",
    [
        "module",
        "class",
        "instance",
        "code",
        "moment_shape",
        "recipe",
        "registry",
        "buffer_write",
        "registration",
        "geometry",
        "operator_kernel",
        "site_config",
    ],
)
def test_native_guard_rejects_callback_and_storage_changes(monkeypatch, target):
    model, _opt, dispatcher, cached, (guard, roots) = example()
    assert guard.check(roots)
    if target == "module":
        from torchcst._backends.torch.profiles import execution

        monkeypatch.setattr(execution, "project_gradient", lambda *a: a[-1])
    elif target == "class":
        from torchcst._backends.algorithm import Algorithm

        monkeypatch.setattr(Algorithm, "validate_inputs", lambda *a: None)
    elif target == "instance":
        cached.state._signature = lambda: ()
    elif target == "code":
        from torchcst._backends.torch.profiles import execution

        def custom(*args):
            return args[-1]

        monkeypatch.setattr(execution.project_gradient, "__code__", custom.__code__)
    elif target == "moment_shape":
        model.atom_state.optimizer_state.fields["exp_avg"].resize_(1)
    elif target == "recipe":
        object.__setattr__(cached.decision.plan, "schema_version", 0)
    elif target == "registry":
        dispatcher.registry._entries = dict(dispatcher.registry._entries)
    elif target == "buffer_write":
        model.kernel.scalar("w_c").add_(0)
    elif target == "registration":
        model.kernel._buffers["w_c"] = model.kernel.scalar("w_c").clone()
    elif target == "geometry":
        model.input_chart.geometry.embedding_dim = 2
    elif target == "operator_kernel":
        import copy

        object.__setattr__(
            cached.state.binding.operator, "kernel", copy.deepcopy(model.kernel)
        )
    else:
        import copy

        model.input_chart = copy.deepcopy(model.input_chart)
    assert not guard.check(roots)


def test_native_guard_itself_does_not_retain_model_or_input_examples():
    model, opt, dispatcher, cached, native = example()
    guard, roots = native
    refs = [
        weakref.ref(model),
        weakref.ref(model.atoms.p),
        weakref.ref(cached.state.binding),
    ]
    del roots, native, model, opt, dispatcher, cached
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert guard is not None


def test_binding_serialization_drops_prepared_program():
    model, opt, _dispatcher, _cached, _native = example()
    binding = opt._update_bindings[model]
    binding.__dict__["_optimizer_program"] = object()
    assert "_optimizer_program" not in binding.__getstate__()


@pytest.mark.parametrize("generic", [False, True])
def test_empty_standard_hooks_skip_only_the_extra_post_launch_guard(
    monkeypatch, generic
):
    from typing import Generic, TypeVar

    from torchcst._backends.cuda.algorithms.polar_update.compiled_launch import (
        CompiledPolarLaunch,
        _empty_hook_certificate,
        _hook_loop,
    )

    item = TypeVar("item")

    class GenericChain(Generic[item]):
        __call__ = _hook_loop

    chain_type = (
        GenericChain if generic else type("HookChain", (), {"__call__": _hook_loop})
    )
    enter, exit = chain_type(), chain_type()
    enter.calls, exit.calls = [], []
    enter.reversed, exit.reversed = False, True
    knobs = SimpleNamespace(
        HookChain=chain_type,
        runtime=SimpleNamespace(launch_enter_hook=enter, launch_exit_hook=exit),
    )
    launch = CompiledPolarLaunch.__new__(CompiledPolarLaunch)
    launch.knobs = knobs
    launch.empty_hook = _empty_hook_certificate(knobs)
    assert launch.empty_hook is not None
    assert not launch.hooks_active()
    exit.calls.append(lambda: None)
    assert launch.hooks_active()
    exit.calls.clear()
    assert not launch.hooks_active()
    monkeypatch.setattr(chain_type, "__call__", lambda *a, **kw: None)
    assert launch.hooks_active()
    assert _empty_hook_certificate(knobs) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["finite_chord", "time_energy"])
@pytest.mark.parametrize(
    "change",
    [
        "none",
        "decay",
        "regularization",
        "pre_hook",
        "post_hook",
        "custom_projection",
        "custom_validator",
        "input_constructor_post_hook",
        "custom_transport",
        "custom_adapter",
        "replaced_moment",
        "zero_decay",
        "unit_decay",
        "checked",
        "jit_hook",
        "jit_hook_transport",
        "launch_hook",
        "grad_none",
        "zero_rate",
        "noncontiguous_grad",
    ],
)
def test_cuda_prepared_optimizer_matches_generic_and_live_changes(
    monkeypatch, change, mode
):
    import copy

    from torchcst._backends.algorithm import Algorithm
    from torchcst._backends.cuda.algorithms.polar_update.algorithm import (
        RadiusFlatTorusPolarUpdateAlgorithm,
    )
    from torchcst._backends.cuda.algorithms.polar_update.optimizer_program import (
        _definitions,
    )
    from torchcst._backends.torch import finite
    from torchcst._backends.torch.profiles import execution as profiles

    actual = radius_model(mode).cuda()
    baseline = copy.deepcopy(actual)
    registry = get_registry()

    def make(model):
        return CSTOptimizer(
            torch.optim.AdamW(
                [
                    {
                        "params": model.parameters(),
                        "weight_decay": 0.0,
                        "amplitude_weight_decay": 0.01,
                    }
                ],
                lr=0.001,
                fused=True,
            ),
            model=model,
            check_finite=False,
            update_selector=FixedSelector(RADIUS_FLAT_TORUS_FUSED, registry=registry),
        )

    oa, ob = make(actual), make(baseline)
    monkeypatch.setattr(
        ob._update_dispatcher, "prepare_optimizer_update", lambda *a: None
    )
    generator = torch.Generator().manual_seed(8221)
    jit_calls = []
    for i in range(6):
        gradient = torch.randn(actual.atoms.p.shape, generator=generator).cuda() * 0.01
        actual.atoms.p.grad, baseline.atoms.p.grad = gradient.clone(), gradient.clone()
        if i == 2:
            program = oa._update_bindings[actual].__dict__.get("_optimizer_program")
            if program is None:
                cb = (
                    *_callbacks(),
                    (
                        RadiusFlatTorusPolarUpdateAlgorithm,
                        (
                            "execute",
                            "validate_recipe",
                            "validate_inputs",
                            "prepare",
                            "supports",
                            "workspace_bound",
                        ),
                    ),
                )
                raise AssertionError(
                    "prepared program missing; callback definitions="
                    + str(_definitions(cb) is not None)
                )
            assert program.launch is not None
            # This test runs without profiler/launch callbacks until i == 2.
            assert program.launch.empty_hook is not None
            assert not program.launch.hooks_active()
            with finite.checking(False):
                assert program.preflight(
                    oa._update_dispatcher,
                    oa.state_adapter,
                    actual.atoms.p,
                    actual.atoms.p.grad,
                    0.01,
                )
            if change == "decay":
                oa.param_groups[0]["amplitude_weight_decay"] = 0.02
                ob.param_groups[0]["amplitude_weight_decay"] = 0.02
            elif change == "regularization":
                actual.kernel.scalar("radial_regularization").mul_(1.1)
                baseline.kernel.scalar("radial_regularization").mul_(1.1)
            elif change in ("pre_hook", "post_hook"):

                def hook(opt, args, kwargs):
                    opt.param_groups[0]["lr"] = 0.0008

                for optimizer in (oa, ob):
                    (
                        optimizer.base_optimizer.register_step_pre_hook
                        if change == "pre_hook"
                        else optimizer.base_optimizer.register_step_post_hook
                    )(hook)
            elif change == "custom_projection":
                original = profiles.project_gradient
                monkeypatch.setattr(
                    profiles,
                    "project_gradient",
                    lambda *a, _original=original: _original(*a) * 0.9,
                )
            elif change == "custom_validator":
                original = Algorithm.validate_inputs

                def validate(self, state, inputs, _original=original):
                    return _original(self, state, inputs)

                monkeypatch.setattr(Algorithm, "validate_inputs", validate)
            elif change == "input_constructor_post_hook":
                from torchcst.operators.atom_update import AtomUpdateInputs

                original = AtomUpdateInputs.__init__

                def construct(
                    self,
                    previous,
                    step_size,
                    amplitude_weight_decay=0.0,
                    _original=original,
                ):
                    _original(self, previous, step_size * 0.9, amplitude_weight_decay)

                def change_inputs(*args, _construct=construct):
                    monkeypatch.setattr(AtomUpdateInputs, "__init__", _construct)

                for optimizer in (oa, ob):
                    optimizer.base_optimizer.register_step_post_hook(change_inputs)
            elif change == "custom_transport":
                original = profiles.transport_state
                monkeypatch.setattr(
                    profiles,
                    "transport_state",
                    lambda *a, _original=original: _original(*a) * 0.9,
                )
            elif change == "custom_adapter":
                from torchcst.optim.state import OptimizerStateAdapter

                class CountingAdapter(OptimizerStateAdapter):
                    def transport(self, site, old, new, state):
                        super().transport(site, old, new, state)
                        state["exp_avg"].mul_(0.9)

                oa.state_adapter = ob.state_adapter = CountingAdapter(("exp_avg",))
            elif change == "replaced_moment":
                for optimizer, model in ((oa, actual), (ob, baseline)):
                    state = optimizer.state[model.atoms.p]
                    state["exp_avg"] = state["exp_avg"].clone()
            elif change in ("zero_decay", "unit_decay"):
                for optimizer in (oa, ob):
                    optimizer.param_groups[0]["amplitude_weight_decay"] = (
                        0.0 if change == "zero_decay" else 1000.0
                    )
            elif change == "checked":
                oa.check_finite = ob.check_finite = True
            elif change in ("jit_hook", "jit_hook_transport"):
                from torchcst._backends.cuda.algorithms.polar_update.kernels import (
                    apply_polar,
                )

                original_transport = profiles.transport_state

                def pre_run(*args, _transport=original_transport, **kwargs):
                    # First let the generic path prepare a program whose launch
                    # uses Algorithm.execute, then mutate inside that entrance.
                    if change == "jit_hook_transport" and len(jit_calls) == 2:
                        monkeypatch.setattr(
                            profiles, "transport_state", lambda *a: _transport(*a) * 0.9
                        )
                    jit_calls.append(1)

                monkeypatch.setattr(
                    apply_polar, "pre_run_hooks", [*apply_polar.pre_run_hooks, pre_run]
                )
            elif change == "launch_hook":
                from triton import knobs

                original = profiles.transport_state

                def change_transport(*args, _original=original, **kwargs):
                    if not jit_calls:
                        monkeypatch.setattr(
                            profiles,
                            "transport_state",
                            lambda *a: _original(*a) * 0.9,
                        )
                    jit_calls.append(1)

                hooks = knobs.runtime.launch_exit_hook
                monkeypatch.setattr(hooks, "calls", [*hooks.calls, change_transport])
        if i == 3 and change == "jit_hook_transport":
            current = oa._update_bindings[actual].__dict__.get("_optimizer_program")
            assert current is not None and current.launch is None
        if i >= 2:
            if change == "grad_none":
                actual.atoms.p.grad = baseline.atoms.p.grad = None
            elif change == "zero_rate":
                oa.param_groups[0]["lr"] = ob.param_groups[0]["lr"] = 0.0
            elif change == "noncontiguous_grad":
                for model in (actual, baseline):
                    model.atoms.p.grad = gradient.T.contiguous().T
        if i >= 2 and change == "noncontiguous_grad":
            # PyTorch fused AdamW rejects this layout before Polar executes.
            # Preserve its error and any state changes made before that error.
            for optimizer in (oa, ob):
                with pytest.raises(
                    RuntimeError, match="same dtype, device, and layout"
                ):
                    optimizer.step()
        else:
            oa.step()
            ob.step()
        torch.testing.assert_close(actual.atoms.p, baseline.atoms.p, atol=0, rtol=0)
        sa, sb = oa.state[actual.atoms.p], ob.state[baseline.atoms.p]
        assert sa.keys() == sb.keys()
        for key in sa:
            torch.testing.assert_close(sa[key], sb[key], atol=0, rtol=0)
    if change in ("jit_hook", "jit_hook_transport", "launch_hook"):
        assert len(jit_calls) == 8
