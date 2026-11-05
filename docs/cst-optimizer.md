# CST optimizers

**English** | [日本語](cst-optimizer.ja.md)

## CSTAdamW: one PyTorch lifecycle

`CSTAdamW(params, model=model, ...)` (PyTorch 2.6+) subclasses AdamW. One
optimizer owns `param_groups`, `state`, hooks and scheduler integration. A step
performs gradient projection and an old-coordinate snapshot, PyTorch functional
AdamW, then the existing Kernel update and vector transport. It does not call a
wrapped parent's `step()`. Standard step pre/post-hooks run once each, and
post-hooks observe values after the CST update. This differs from an existing
wrapper's base post-hook, which observes the Adam proposal, so migration must
be explicit.

Managed atom groups default to raw `weight_decay=0`; specify amplitude-only
decay through `amplitude_weight_decay`. Ordinary parameter groups retain AdamW's
default raw decay of .01. Explicit nonzero raw decay for atoms is rejected. The
Kernel equations are shared with the wrapper below. `grad=None` skips updates;
`lr=0` still preserves ordinary Adam clock/moment updates. `check_finite=False`
is the default; set it to True for additional strict checks. lr/decay,
active-atom metadata and callback-change validation remain. Full-model ownership
and structure scans run during construction, add_param_group and checkpoint
load, rather than every step.

`zero_grad`, schedulers, closures and state-dict hooks use ordinary PyTorch
mechanisms. GradScaler skips the entire step through its standard unscale/skip
path; Polar is not left running when only fused Adam's internal overflow check
skips its update. Managed atoms do not support capturable/differentiable modes.
Checkpoint post-hooks can observe the cst manifest and reconnected atom-owned
state. Standard `load_state_dict` accepts native checkpoints. To migrate from
the current exact AdamW wrapper, load the model checkpoint into a new model,
construct a native optimizer and explicitly call `load_legacy_state_dict`.
Do not share a model already bound to another optimizer.

### Saving and restoring checkpoints

Choose the model's final device/dtype before constructing the optimizer. Save
model and optimizer state dicts together, rebuild the model with the same
declaration, then restore model followed by optimizer. When using a scheduler,
construct and restore it through the standard PyTorch procedure. Rebuild the
optimizer after replacing an atom Parameter.
Fixed grid/geometry values are saved in model extra state and validated against
the rebuilt declaration. Algorithm workspace is reconstructed. For atom row
order and training-state ownership, see the [Atom state checkpoint contract](atom-state.md#checkpoint).
Removal of the old optimizer family and validation at that time are documented
in the [migration record](cst-optimizer-migration.ja.md).

## CSTOptimizer: a compatibility wrapper for arbitrary optimizers

`CSTOptimizer(base_optimizer, model=model, state_adapter=None)` wraps an existing
PyTorch optimizer. `check_finite=True` is the default. The old optimizer,
config, N/D solver, moment system and AtomGrad have been removed. Reproduce old
experiments with the preceding Git revision `33504c6`.

## Ownership and updates

| Item | Owner / handling |
| --- | --- |
| Parameter | Model |
| Learning rate, ordinary weight decay, optimizer state | Base optimizer |
| Amplitude-only weight decay | Apply the group's amplitude_weight_decay during the Kernel update |
| Gradient | Ordinary backend autograd |
| Gradient projection, update law, vector transport | Parameterization / Geometry specified by KernelSpec; computation belongs to the backend |
| Update order, CST site discovery | CSTOptimizer |
| Scheduler | Attach the PyTorch scheduler to the wrapper |

If a closure is supplied, evaluate it once at the current point. When checks
are enabled, validate gradient finiteness for every group. Project atom
gradients and save old coordinates for atom tables being updated before the
base update. Pass the actual difference `proposal - old` and the group's
current `lr` to the Kernel, transport vector state to the update result and
write it back to the Parameter.

Atoms with `grad is None` do not advance their update laws. Zero gradients are
valid steps and advance activity decay and similar rules. `lr=0` stops the CST
update law, while the base optimizer's own clock follows its normal semantics.
The wrapper has no independent schedule.

Ordinary Parameters use the base update unchanged. Each base updates only its
own atoms, allowing model parameters to be split across multiple optimizers.
Assigning update laws from multiple CST sites to the same atom is rejected.

The wrapper does not interpret atom columns. The time unit of Polar activity
is the group's lr. Different optimizers use the same rule, which does not imply
identical training trajectories. Weight decay on raw atom coordinates is
included in the base proposal, then projected/adjusted by the Kernel law.
This is separate from Kernel regularization; set `weight_decay=0` for CST groups
when needed.

### Applying weight decay only to Polar amplitude

Set `weight_decay=0` and `amplitude_weight_decay=λ` on the AdamW atom group.
Ordinary Dense parameters use ordinary `weight_decay` in a separate group.
Groups are not automatically split, and base group settings are not rewritten
during a step.

```python
base = torch.optim.AdamW([
    {"params": cst_layer.parameters(), "weight_decay": 0.0,
     "amplitude_weight_decay": 0.01},
    {"params": dense_layer.parameters(), "weight_decay": 0.01},
], lr=1e-3)
optimizer = CSTOptimizer(base, model=model)
```

λ is a finite, nonnegative Python scalar satisfying `0 <= lr*λ <= 1`. This
option is for exact `torch.optim.AdamW`; it does not infer closure semantics for
subclasses or other optimizers. Combining it with raw decay is rejected before
AdamW modifies Parameters/state. Groups without the option retain earlier
behavior for compatibility; raw decay is not automatically converted into
amplitude decay.

For Polar, let `q=s²+t²` and `amp=Amax*s/sqrt(q)`. First apply the existing Polar
task/activity/radial update to the AdamW proposal without raw decay, obtaining
`(sf,tf)`. With `rf=sqrt(sf²+tf²)` and `u_old=amp_old/Amax`,

\[
u'=\operatorname{clamp}\left(\frac{s_f}{r_f}-\mathrm{lr}\lambda u_{old},-1,1\right),
\quad s'=r_f u',\quad
t'=\operatorname{copysign}\left(r_f\sqrt{(1-|u'|)(1+|u'|)},t_f\right).
\]

This decays the old-value term as
`amp_target = amp_after_Polar - lr*λ*amp_old`, without further decaying the task
increment. Values exceeding the bound are projected into the amplitude domain.
Without a task update, `amp_new=(1-lr*λ)*amp_old`. When λ=0, reconstruction itself
is skipped. The sign side is preserved even for `t=±0`. This decay does not
apply to center coordinates, Adam moments or step counters.
Over real numbers, `q_new=q_after_Polar`; the existing activity update law is
unchanged. If the profile radius/bandwidth upper bound depends on amplitude,
ordinary decoding can change the profile radius. Preserving `q` and fixing
profile radius are different properties.

After the base's existing pre-hooks, a temporary final pre-hook performs
validation, gradient projection, snapshotting and Plan preparation, capturing
the lr/λ/grad actually used by AdamW. Even if a post-hook changes settings, the
current step uses captured values. The closure is evaluated once at wrapper
entry and not reevaluated when presented to the base. Replacing the closure
through a pre-hook is rejected in this mode. Temporary hooks are removed even
on exceptions. No GPU-value reads to CPU are added to steady-state steps.

The amplitude-decay-capable general Torch / Polar Torch / Euclidean, FlatTorus
and radius fused Plans are v2. Old Sphere-fused and intrinsic-Torus-specific v1
Plans reject positive amplitude decay; use their corresponding Torch reference
Plans. λ is passed through per-step Inputs; changing its coefficient does not
keep extra Fourier coefficients or atom copies resident. The CUDA fused version
performs reconstruction inside the existing Polar kernel.

## State adapters

| Base optimizer | State transported as vectors |
| --- | --- |
| SGD | momentum_buffer |
| Adam / AdamW / Adamax | exp_avg |
| RMSprop | momentum_buffer, centered grad_avg |
| Rprop | prev (previous gradient) |
| Adagrad / Adadelta | None |

Squared accumulators and step clocks remain in their original stored coordinate
system. This combines a coordinate-based optimizer with vector transport; it
does not implement intrinsic Riemannian Adam or covariance transport.

Custom optimizers / subclasses must explicitly declare state semantics.

```python
from torchcst import CSTOptimizer, OptimizerStateAdapter

optimizer = CSTOptimizer(
    custom_base,
    model=model,
    state_adapter=OptimizerStateAdapter(vector_keys=("velocity",)),
)
```

The backend transports the specified parameter-shaped state through the common
KernelState. Whether preserving other state is correct belongs to the custom
optimizer's contract. Subclass names alone do not imply known state semantics.
ASGD averaged parameters and LBFGS multiple proposals / closure reevaluations
are not handled by the default path.

## PyTorch integration

The wrapper is a `torch.optim.Optimizer` and shares `param_groups` / `state` with
the base. Use the wrapper for schedulers, GradScaler and step hooks. Evaluate a
closure once and pass the base a closure returning that result.
`add_param_group()` also delegates to the base. Do not directly step/load the
base while also using the wrapper.

Checkpoints add a `cst` manifest to standard state / param_groups. Restoration
validates optimizer type, vector keys, parameter names/shapes, KernelSpec and
CST policy ownership. Kernel/Geometry settings are restored from the model
checkpoint. The base and wrapper still share state/groups after load. Old
optimizer checkpoints and raw base checkpoints are not accepted. Move the model
to its target device before constructing the wrapper, and rebuild the wrapper
after replacing atom Parameters.

## Following ordinary PyTorch without additional finite checks

`CSTOptimizer(..., check_finite=False)` adds no optimizer value-finiteness checks.
It follows ordinary AdamW behavior without automatic NaN/Inf rejection or step
skipping. Shape/dtype/device, group/Plan and fixed-setting validation always
remain. The default True preserves existing checks. This mode is scoped to the
call and does not suppress geometry validation in closures or outside the call.
It changes neither checkpoint mathematical contracts nor Adam state; callers
choose the mode.

For configurations selecting the FlatTorus Plan in the next section, add
`check_finite=False` at the end. Under fixed FP32/fused AdamW/FlatTorus settings,
steady-state optimizer steps were measured with 0 synchronization API calls and
0 DtoH transfers. This does not guarantee zero synchronization during initial
preparation, revalidation after changes, other Plans/optimizers, closures,
AMP/GradScaler or the entire training process. CUDA Graphs remain unsupported.

## Current scope

For CUDA FP32 Polar with 1D FlatTorus Charts, explicitly selecting an update
Plan fuses the Polar update and periodic center wrapping in Triton.

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import FLAT_TORUS_FUSED

optimizer = CSTOptimizer(
    base_optimizer,
    model=model,
    update_selector=FixedSelector(FLAT_TORUS_FUSED, registry=make_algorithm_registry()),
)
```

This selects the `atom_update` implementation independently of the Linear Plan.
It targets contiguous FP32 `[atom_count, 4]` with two center coordinates. Do not
select this Plan for other geometries or dtypes. The default remains the Torch
reference path. The common wrapper handles Adam proposals/moments/steps,
gradient projection, state adapters and invalid-value checks. Fixed periods are
prepared as execution Tensor caches from Python geometry metadata; Polar
scalars use live buffers. Setting, placement and version changes trigger
revalidation/preparation. This is separate from the existing Euclidean `FUSED`
Plan, and the periodic version also requires eager execution.

It is independent of the Linear / Conv2d forward backend. Euclidean trainable
Charts are supported. Trainable non-Euclidean Charts require a different update
contract from atom centers and are rejected until a dedicated policy is added.

CUDA real-value finiteness checks are grouped by device/dtype and return only a
small decision vector to the host. Raw gradients for all Parameters are checked
before Adam. Euclidean/FlatTorus transport similarly groups old points, new
points and vectors. Update-law and moment arithmetic are unchanged, and
synchronization to reject invalid values immediately remains. Complex/CPU/
unsupported layouts retain the earlier checks, without extending CUDA Graph
support.

Snapshot and update workspace scale with the atom parameter count. Dense
weights, weight gradients and Hessians are not constructed. Finiteness checks
synchronize the device. CUDA Graph capture including CST policies, Tensor
learning rates and differentiable optimizer steps are unsupported.
`capturable=True` is rejected. Ordinary eager CUDA and foreach / fused base
optimizers are separate execution settings.

`check_finite=True` rejects nonfinite raw gradients before the base update. It
does not transactionally roll back all optimizer state on custom-policy failure
after the base update. Custom policies must meet the shape and finiteness
contracts.
