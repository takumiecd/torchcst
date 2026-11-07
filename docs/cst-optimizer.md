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

## Field learning rates

Both `CSTAdamW(..., field_lrs=...)` and
`CSTOptimizer(base, model=model, field_lrs=...)` accept absolute **initial**
proposal rates. A group's `field_lrs` overrides the constructor mapping;
`field_lrs={}` disables the constructor overrides for that group. Field mappings
are copied at construction/add-group boundaries. Unknown field names and
invalid rates are rejected before parameter or moment updates.

```python
optimizer = CSTAdamW(
    cst_layer.parameters(), model=cst_layer, lr=4.8e-3,
    field_lrs={"amplitude": 2.4e-3, "input_center": 4.8e-3,
               "output_center": 4.8e-3},
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
```

For each group, let `lr_ref` be its initial `lr`. The current effective rate is

\[
\eta_f(t)=\mathrm{field\_lrs}[f]\frac{\mathrm{lr}(t)}{\mathrm{lr}_{ref}}.
\]

Thus standard PyTorch schedulers supply a common factor without needing to
modify nested dictionaries. Initial field values are rates, not multipliers.
When updating an already configured group's mapping, values remain expressed
at its saved initial-group anchor. The stored proposal-minus-old difference is
scaled before the coordinate update, following that proposal's floating-point rounding.
`get_field_lrs()` returns a list of current override dictionaries in group order;
unspecified fields use the current group `lr`, and groups with no overrides
return `{}`. The initial group anchor must be finite and positive; field rates
may be zero. A later group `lr=0` still follows ordinary Adam moment/counter
behavior and skips the coordinate policy. Distinct per-field schedules require
explicit field-setting updates; a stock scheduler supplies the common factor.

Ordinary Parameters retain their group `lr`. For a mixed model, give ordinary
and Atom Parameters separate groups when they need different base rates:

```python
atom_ids = {id(m.atoms.p) for m in model.modules() if isinstance(m, CSTLinear)}
atom_parameters = [p for p in model.parameters() if id(p) in atom_ids]
ordinary_parameters = [p for p in model.parameters() if id(p) not in atom_ids]
optimizer = CSTAdamW([
    {"params": ordinary_parameters, "lr": 3e-4},
    {"params": atom_parameters, "lr": 4.8e-3,
     "field_lrs": {"amplitude": 2.4e-3, "input_center": 4.8e-3,
                   "output_center": 4.8e-3}},
], model=model, weight_decay=0.0)
```

Rates scale the base optimizer's proposal after Adam normalization when using
Adam; they do not scale its input gradients or moment accumulators. There is
one Parameter, moment table and counter per Atom table. Unit-ratio blocks are
untouched. Group `lr` remains the coordinate update's outer time and the time
used for `amplitude_weight_decay`; field overrides change task motion. Raw
weight decay must be zero on Atom groups with field overrides. Polar activity
still depends on amplitude/motion, so equal outer time does not freeze q/radii.
A zero amplitude rate freezes its task direction, while the activity law can
still change the `(s,t)` norm. `grad=None` skips both proposal and policy.

Field names come from the parameterization, with center block widths taken
from the Geometry's stored center representation:

| Parameterization | Non-center fields |
| --- | --- |
| Fixed width | none |
| Amplitude-dependent width | `amplitude` |
| Log width | `amplitude`, `log_bandwidth` |
| Physical amplitude/radius | `amplitude`, `radius` |
| Direct activity width | `amplitude` (activity is policy-owned) |
| Polar width/radius | `amplitude`, the joint `(s,t)` block |

All built-in two-chart declarations additionally expose `output_center` and
`input_center`, including multi-coordinate centers. `q`, `s` and `t` are not
separate Polar learning-rate fields. An unsupported composition/layout rejects
overrides instead of guessing columns. Extension authors can declare blocks
through `ParameterizationSpec.optimizer_fields(center_dimensions)` returning
immutable `ParameterFieldSpec(name, start, stop)` values from `torchcst.kernels`.
The built-in `optimizer_prefix` declaration reserves unnamed policy coordinates
when needed. The optimizer consumes declarations rather than branching on kernels.

Checkpoint loading restores the saved field mappings and scheduler anchors,
just as it restores group learning rates. Rebuild the same model and optimizer
type, restore model followed by optimizer/scheduler, and resume the normal
PyTorch lifecycle. Saved field-layout metadata is checked before mutation.
A default constructor can load a field-rate checkpoint; loading an older
default checkpoint disables constructor field overrides for its saved groups.
Checkpoints without field overrides retain the earlier format. Field-rate
qualification currently covers eager training; the stock complete-step Graph benchmark rejects these
overrides and falls back to the full eager step. No Graph qualification or
universal recommended rate follows from this API.

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

The [native/actual-LM validation record](research-history/optimizer-field-lrs-20261107.md) covers the tested eager field-rate scope.
