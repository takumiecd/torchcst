"""CUDA-graph-capturing training step: capture once, replay every step.

A ``torch.cuda.CUDAGraph`` records the exact CUDA kernel launches issued by
one Python call and replays them with a single ``cudaGraphLaunch`` -- this
collapses a training step with O(1000s) of small kernel launches (one CST
conv layer alone assembles a dense filter from atoms, several such layers
per model) down to one launch, which matters when the CPU-side dispatch
overhead of those launches, not the GPU compute itself, is what is keeping
the GPU under-utilized.

Two things make a training step *uncapturable* as a single region:

1. A Python float learning rate baked into ``optimizer.step()``'s traced
   kernels can never change on replay -- graph capture bakes in whatever
   values were live at trace time. ``capturable_sgd_step`` below is
   ``torch.optim.SGD``'s own multi-tensor formula, but always taking the
   "Tensor lr" branch that plain eager ``SGD.step()`` only takes under
   ``torch.compiler.is_compiling()`` (see ``torch/optim/sgd.py``'s
   ``_multi_tensor_sgd``): ``torch._foreach_mul(bufs, -lr_tensor)`` then
   ``torch._foreach_add_(params, update)``, reading a Tensor lr's *current*
   value on every replay via ``lr_tensor.fill_()`` before each launch.
2. Host-side, data-dependent control flow -- e.g. a structural policy's
   top-k candidate selection, whose op count and shapes depend on tensor
   values only knowable at run time -- cannot be captured at all; CUDA
   graph capture requires a completely static kernel sequence.
   ``GraphedCellRunner`` handles this by capturing only the "quiet" steps
   between structural events (an event step always executes eagerly), which
   is a fundamentally different reuse question from (1): it requires that a
   quiet-step graph captured once stays correct to replay across many
   events, i.e. that events mutate storage *in place* rather than
   reallocating it. See the class docstring below.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor


def make_momentum_buffers(
    optimizer: torch.optim.Optimizer, *, register: bool = True
) -> dict[int, Tensor]:
    """One zero-initialized SGD momentum buffer per grad-requiring parameter.

    When ``register`` (the default), each buffer is also installed as
    ``optimizer.state[param]["momentum_buffer"]`` -- the *same* tensor
    object, not a copy. This is load-bearing for any cell with a
    structural engine attached: a birth/death event's store commit fires its
    ``FollowerHub``, which notifies the subscribed
    ``OptimizerStateFollower`` (see ``torchcst.optim``), which zeroes
    ``optimizer.state[param]["momentum_buffer"]`` at the
    born/dead physical slots -- this is what gives a newly-born atom fresh
    (zero) momentum instead of inheriting whatever the slot's previous,
    now-dead occupant had accumulated, and it is exactly the behavior real
    ``torch.optim.SGD.step()`` gets for free the first time it lazily
    initializes a parameter's buffer. A ``GraphedCellRunner`` never calls
    ``optimizer.step()`` (it needs a Tensor lr for graph capture, see
    ``capturable_sgd_step``), so without this registration
    ``optimizer.state`` would stay permanently empty and that reset would
    silently never fire -- a dead atom's momentum would leak into its
    replacement's slot forever. Registering the *same* tensor object means
    the engine's in-place ``index_fill_(0, reset_slots, 0)`` and this
    module's own ``capturable_sgd_step`` reads/writes are automatically
    kept consistent with no separate bookkeeping.
    """

    buffers: dict[int, Tensor] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            buf = torch.zeros_like(p)
            buffers[id(p)] = buf
            if register:
                state = optimizer.state.get(p)
                if state is None:
                    state = {}
                    optimizer.state[p] = state
                state["momentum_buffer"] = buf
    return buffers


@torch.no_grad()
def capturable_sgd_step(
    param_groups: Sequence[Mapping[str, object]],
    momentum_buffers: Mapping[int, Tensor],
    lr_tensor: Tensor,
    *,
    momentum: float,
    dampening: float = 0.0,
) -> None:
    """One SGD update, capture-safe because ``lr_tensor`` is read by value
    on every call rather than baked in as a Python float.

    Mirrors ``torch.optim.sgd._multi_tensor_sgd``'s momentum + weight-decay
    formula exactly, restricted to the dense (non-sparse-grad), non-maximize,
    non-nesterov case every caller in this repo uses, taking that function's
    ``isinstance(lr, Tensor) and torch.compiler.is_compiling()`` branch
    (``torch._foreach_mul`` then a separate ``torch._foreach_add_``)
    unconditionally, since that is the only formula that is both capturable
    and reads the lr by value on every replay.

    *** This is NOT bit-identical to plain eager ``torch.optim.SGD.step()``
    with a Python float lr. *** That path issues a single fused
    ``torch._foreach_add_(params, grads, alpha=-lr)``. A Tensor ``alpha``
    reproduces that exact fused kernel and *is* bit-identical to it
    (verified directly -- see
    ``tests/test_framework_rebase_cuda_graphs.py::
    test_capturable_sgd_step_matches_optimizer_step``'s git history / the
    module's own test suite), but is not capture-safe: ATen's Tensor-alpha
    foreach overload raises ``cudaErrorStreamCaptureUnsupported`` inside
    ``torch.cuda.graph()`` (confirmed empirically, not assumed -- it is not
    a documented restriction). The two-step mul-then-add form is the only
    formula found that both replays correctly and never touches a host
    scalar, at the cost of a small ULP-level rounding difference from the
    eager reference's single fused op every step. This is a real, measured
    limitation, not an oversight: a graphed run is bit-for-bit
    *self-consistent* (replay vs replay, same seed and tape) but not
    bit-for-bit identical to a fully-eager run of the same seed and tape.
    See the wrapper's report for the measured size of the resulting
    divergence.

    Momentum buffers must already exist and be zero-initialized (see
    ``make_momentum_buffers``) before the first call: SGD's own "clone the
    grad on first use" branch is never taken here, but
    ``0 * momentum + grad == grad`` exactly, so this is equivalent to it
    starting cold.
    """

    for group in param_groups:
        wd = group.get("weight_decay", 0.0)
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            continue
        grads = [p.grad for p in params]
        bufs = [momentum_buffers[id(p)] for p in params]
        if wd:
            grads = torch._foreach_add(grads, params, alpha=wd)
        torch._foreach_mul_(bufs, momentum)
        torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        update = torch._foreach_mul(bufs, -lr_tensor)
        torch._foreach_add_(params, update)


class GraphedCellRunner:
    """Capture one training step as a CUDA graph and replay it every step.

    This class is deliberately agnostic of what a "step" computes: the
    caller supplies ``quiet_step_fn``, a zero-argument closure that performs
    exactly one step's core compute (zero grads in place, forward, loss,
    backward, an optimizer update, and any post-step correction) reading
    from and writing to tensors the caller owns and keeps at fixed
    addresses. This module only handles the *mechanics* of graph capture:
    side-stream warmup, one-shot capture, and replay -- the CST-specific
    composition (forward/backward/``capturable_sgd_step``/gauge
    fixing/``StructuralEngine`` hooks) lives in the caller
    (``experiments/framework_rebase_main.py``'s cell runner).

    For cells with no structural engine, ``event_interval``/``event_step_fn``
    are left ``None`` and every step replays the one captured graph forever.

    For cells with a ``StructuralEngine`` attached, pass ``event_interval``
    (the engine's structural cadence) and ``event_step_fn`` (a ``(step,
    total_steps) -> Tensor`` closure that runs one full step *eagerly*,
    including the actual structural surgery, whenever ``step %
    event_interval == 0``). Every other step replays the graph captured
    from ``quiet_step_fn``.

    *** The load-bearing assumption of this design ***: a quiet-step graph
    is captured exactly once (during ``build()``) and then replayed across
    every event for the rest of the run -- it is never recaptured. CUDA
    graph capture bakes in fixed *memory addresses*; replaying it after an
    event is only correct if that event mutated tensor *contents* in place
    at those same addresses (``index_copy_``/``index_fill_``, never a
    reallocation such as ``Parameter.set_()`` to a new, larger tensor).
    This holds for the ``S:syn`` structural cells in this repo specifically
    because every structural policy here composes
    ``EvenBudgetDistributor(replacement_only=True)``, which caps a site's
    realized births at its realized death count, so live-atom count (and
    therefore ``SynapseStore`` capacity) never changes across an event --
    ``SynapseStore.commit`` only calls ``self._grow(...)`` (which does
    reallocate, via ``Parameter.set_()``) when capacity must change. This
    is an assumption about the *policy composition* the caller builds, not
    something this class can verify on its own; the caller is responsible
    for proving it empirically for the given policy set (see
    ``tests/test_framework_rebase_cuda_graphs.py``'s
    ``test_graphed_structural_cell_matches_eager_across_events``, which
    replays a graphed structural cell across multiple real events and
    checks its loss sequence against a fully-eager run of the same seed and
    tape) and for using ``recapture_after_event=True`` as a fallback if it
    does not hold for some other policy composition (correct in that case,
    at the cost of paying capture overhead again after every event).
    """

    def __init__(
        self,
        *,
        device: torch.device,
        quiet_step_fn: Callable[[], Tensor],
        event_step_fn: Callable[[int, int], Tensor] | None = None,
        event_interval: int | None = None,
        recapture_after_event: bool = False,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("GraphedCellRunner requires a CUDA device")
        if (event_interval is None) != (event_step_fn is None):
            raise ValueError(
                "event_interval and event_step_fn must both be given or both be None"
            )
        self.device = device
        self._quiet_step_fn = quiet_step_fn
        self._event_step_fn = event_step_fn
        self.event_interval = event_interval
        self.recapture_after_event = recapture_after_event
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_loss: Tensor | None = None
        self.n_captures = 0
        self.n_event_steps = 0
        self.n_replay_steps = 0

    def is_event_step(self, step: int) -> bool:
        return self.event_interval is not None and step % self.event_interval == 0

    def warmup(self, n_steps: int, prepare_step: Callable[[int], None]) -> None:
        """Run ``n_steps`` real iterations on a side stream before capture.

        Standard CUDA graph recipe: cuDNN algorithm selection, lazy
        allocation, and (for this repo) first-touch of momentum buffers and
        ``optimizer.state`` all need to happen on a *non-default* stream
        before capture, not during it -- capturing a first-time allocation
        is unsafe. ``prepare_step(step)`` must fill every static input
        tensor ``quiet_step_fn`` reads (batch, labels, lr) for step ``1..
        n_steps``; callers with a structural engine must choose
        ``n_steps`` so none of these land on an event boundary (warmup
        always exercises the quiet path only).
        """

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for step in range(1, n_steps + 1):
                prepare_step(step)
                self._quiet_step_fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize(self.device)

    def capture(self) -> None:
        """Capture ``quiet_step_fn`` as this runner's replayable graph."""

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_loss = self._quiet_step_fn()
        self.n_captures += 1

    def replay(self, step: int, prepare_step: Callable[[int], None]) -> Tensor:
        """Fill static inputs for ``step`` and replay the captured graph."""

        if self.graph is None:
            raise RuntimeError("capture() must run before replay()")
        prepare_step(step)
        self.graph.replay()
        self.n_replay_steps += 1
        assert self.static_loss is not None
        return self.static_loss

    def step(
        self, step: int, total_steps: int, prepare_step: Callable[[int], None]
    ) -> Tensor:
        """One training step: eager at an event boundary, else replayed.

        Returns the step's loss tensor -- the *static* graph-owned tensor
        on a replay (read it before the next ``replay()``/``step()`` call
        overwrites it in place) or a fresh eager tensor on an event step.
        """

        if self.is_event_step(step):
            assert self._event_step_fn is not None
            loss = self._event_step_fn(step, total_steps)
            self.n_event_steps += 1
            if self.recapture_after_event:
                self.capture()
            return loss
        return self.replay(step, prepare_step)
