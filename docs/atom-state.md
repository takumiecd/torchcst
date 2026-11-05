# Atom and Algorithm state ownership

**English** | [日本語](atom-state.ja.md)

`CSTModule` owns `atom_state: AtomState`. `layer.atoms` references the existing
`Atoms` within it, preserving the fixed-shape, opaque-Parameter contract in
`atoms.py`. Modules sharing Atoms or constructed from the same Parameter use the
same AtomState. Atom IDs form a fixed set `[0, count)` within an AtomState and are
independent of physical rows. Do not compare IDs across different AtomStates.

## Initializing and sharing an atom table

`CSTLinear(atoms=...)` uses this ownership contract:

| `atoms=` | Behavior |
| --- | --- |
| Positive integer | Create new Atoms using the kernel's initialization policy |
| Tensor | Detach and clone into newly owned Atoms; conversion to the requested device/dtype is allowed |
| `nn.Parameter` | Register the same Parameter, preserving identity, gradients and existing optimizer references |
| `Atoms` | Share the owner and Parameter unchanged |

The coordinate-column count must match the kernel contract. An explicitly empty
table represents the zero operator. If a shared Parameter/Atoms has a different
device/dtype from the requested one, the constructor rejects it rather than
silently replacing it. Finalize model placement before constructing the
optimizer, and rebuild the optimizer if the atom Parameter is replaced. Access
the Parameter through `layer.atoms.p`.

## Layout and training state

| Class | Location | Owns |
| --- | --- | --- |
| `Atoms` | `atoms/atoms.py` | A fixed-shape Parameter |
| `AtomState` | `atoms/state.py` | Atoms, ID-to-row mapping, layout version and AtomOptimizerState |
| `AtomOptimizerState` | `atoms/optimizer_state.py` | Optimizer contract, named training state and atom-axis declarations |
| `OptimizerFieldSpec` | Same file | `atom_axis`; `None` denotes state independent of layout |
| `Algorithm` | `_backends/algorithm.py` | Shared interface for state creation, preparation and execution |
| `AlgorithmState` | `_backends/state.py` | AtomState reference, dedicated layout, reusable buffers and referenced generation |

The optimizer performs numerical state computation and updates; the Atom side
manages state lifetime and layout. `CSTOptimizer` connects lazy initialization in
a Torch optimizer to the AtomOptimizerState dictionary. Existing trained Torch
state can also be connected without copying. The wrapper and base share
state/groups, and the per-atom dictionary is the same object as
`layer.atom_state.optimizer_state.fields`. Empty state is not registered in the
Torch state map until the first step. Ordinary non-CST Parameters retain normal
Torch state. Only one training-state update source may connect to an AtomState.

Built-in SGD, Adam, AdamW, RMSprop, Adamax, Adagrad, Adadelta and Rprop declare the
atom axis of each state field. Second moments and AMSGrad maximum moments follow
the layout; scalar step counters do not move. Custom optimizers can add
`OptimizerFieldSpec` entries through `state_specs=`. Undeclared state causes
relayout and model conversion to be rejected. `OptimizerStateAdapter`, which
transports vector state during coordinate updates, is a separate contract from
layout-axis declarations.

## Changing the authoritative layout

```python
ids = torch.arange(layer.atoms.count, device=layer.atoms.p.device).flip(0)
layer.atom_state.relayout(ids)
```

The argument is an int64 Tensor specifying the atom ID for each row after the
change. Every ID must appear exactly once; additions, deletions and shape changes
are unsupported. Move IDs and their inverse mapping, Parameters, existing
gradients and declared training state with the same mapping, then advance
`layout_version` when complete. An unchanged layout does not advance the version.
Do not replace Parameter, gradient or moment objects. Allocate and validate all
destinations before writing. Overlapping storage ranges are rejected, but
non-overlapping moment regions within one slab are allowed. If a copy raises an
exception, mark the state unusable rather than publishing a partial move as a new
generation.

Relayout is restricted to eager step boundaries and cannot run during CUDA Graph
capture. On CUDA, serialize all work on the same current stream. Concurrent
changes to authoritative state from different streams/threads are unsupported.
Module forward and Algorithm.run protect autograd's saved-state lifetime,
rejecting relayout, conversion and loading while a backward is pending or
retained. The zero-element CPU marker used for protection is independent of
outer saved-tensor offload hooks. Even frozen atoms are protected when their
state is saved for dX.

Implementations using Parameters directly must use `parameters_for_execution()`
or `protect_tensor()`; the caller manages lifetimes and step boundaries for
inference and custom CUDA Graph execution. State in a raw Torch optimizer cannot
be discovered automatically. Connect it through `CSTOptimizer` if it must move as
well. The CSTOptimizer coordinate policy still does not support CUDA Graph
capture; this change does not add capture support.

Keep Algorithm-specific slot permutations inside that AlgorithmState. Updating a
copy without changing authoritative state does not advance AtomState's version.
Do not overwrite call-specific saved backward values in reusable workspace.

## Algorithm execution and state

The shared Algorithm interface provides
`create_state(binding, recipe=..., **configuration)`, `prepare(state)` and
`run(state, inputs)`. A run checks Algorithm identity and the prepared generation,
prepares if needed, and calls execute. It also protects the autograd lifetime of
Tensor results. Implementations returning composite non-Tensor results must
protect each contained Tensor's lifetime themselves.

Implement concrete layouts, buffers, preparation and computation inside each
Algorithm's directory. The backend-wide ABC defines immutable Algorithm
ID/revision/operation/semantics, recipe validation, support, workspace upper bounds
and execution/state lifecycles. The CUDA-only ABC was removed. Public CSTLinear
passes Inputs and the binding interface to Dispatcher, which retrieves and runs
Module-owned AlgorithmState. Forced Plans use the same validation in
Dispatcher.run. State retains the binding and Recipe; atom-based methods
reference the binding's AtomState. Methods without atoms may use the same State
with no retained memory.

AlgorithmState checks the identity/device/dtype/shape/stride/storage of
Parameters and optimizer-state Tensors, as well as the layout version. It does not
reuse stale views after checkpoint restoration, model conversion or lazy moment
initialization. Changes to Context shape/dtype and other metadata, or to
binding.state_signature, also require preparation. Layout versions do not
establish numerical freshness: operations such as Polar decoding prepare current
values during execute. Concrete Polar or optimizer-proposal Algorithms can define
their operation contract and join the same Registry. Dispatcher does not require
Linear or CSTOperation input shapes. Production registration of concrete
Polar/optimizer Algorithms is outside this change.

## Checkpoint

The Module format is version 2; the CSTOptimizer manifest is version 3. There is
no compatibility loader for older formats. Module checkpoints include ID mappings
and training state. Optimizer checkpoints record Torch groups/state, mathematical
contracts, field specs and atom row order. Restore the model first and the
optimizer second, as usual. Loading only an optimizer checkpoint into a model with
a different row order is rejected. A model-only restoration can also connect to a
new optimizer with the same contract.

AlgorithmState is reconstructible execution state and is excluded from
checkpoints and Module deepcopy. Reloading a model advances the layout generation
and invalidates existing AlgorithmState. If a single checkpoint stores both model
and optimizer, references to training state appear in both state dicts. At
runtime, this does not create duplicate update sources or moment copies.
