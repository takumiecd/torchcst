# CSTPullbackAdam

`CSTPullbackAdam` owns every trainable parameter of the continuous CST
representation and nothing else. Pair it with an ordinary PyTorch optimizer
for dense layers, norms, gates, and other non-CST parameters:

```python
from torchcst import CSTPullbackAdam

cst_optimizer = CSTPullbackAdam(
    model,
    metric="block",
    lr=1e-3,
    damping=1e-2,
    max_step_sigma=0.1,
)
dense_optimizer = torch.optim.AdamW(
    cst_optimizer.non_cst_parameters(), lr=1e-3,
)
cst_optimizer.validate_dense_optimizer(dense_optimizer)
```

The CST owner includes atom amplitudes and coordinates, factor-declared
per-atom columns, learnable neuron charts, and learnable factor bandwidths.
`ownership()` reports the partition. A trainable factor parameter without a
pullback block raises at construction instead of falling through silently.

## Metric tiers

For atom `k`, let `theta_k` contain its amplitude and all of its trainable
per-atom representation parameters, and let

```text
J_k = d vec(W_k) / d theta_k.
```

`metric="diag"` uses `diag(J_k.T @ J_k)`. `metric="block"` uses the complete
same-atom `J_k.T @ J_k`. Neither tier forms `J_k.T @ J_l` for `k != l`, so
cost grows linearly rather than quadratically in the atom count.

A gauge declares exact algebraic zeros through `PullbackStructure`. The
conservative default declares none. `L2NormalizedColumns` guarantees that the
amplitude derivative is orthogonal to every same-atom tangent derivative and
that input- and output-side tangents are orthogonal. The optimizer uses only
declared identities; an unnormalised gauge receives the full local coupling.

Learnable charts use per-neuron blocks. A bandwidth shared by several atoms or
sites has one shared block whose metric sums same-atom self contributions;
cross-atom terms remain excluded by the same approximation rule.

## Step scale and safety

Direct pullback Adam is the default:

```python
CSTPullbackAdam(model, lr=1e-3, target_map_step=None)
```

For first-step calibration instead, set exactly one scale source:

```python
CSTPullbackAdam(model, lr=None, target_map_step=0.01)
```

The calibrated mode chooses one scale so the median first-order represented-map
movement is `target_map_step`. Its optimizer parameter-group `lr` starts at
`1.0` and remains a multiplicative scheduler scale. Calibration is optional
and is not part of Adam itself.

`max_step_sigma` is an independent trust-region safety mechanism. It caps
joint atom-coordinate movement, chart movement, and relative bandwidth
movement. `None` disables it:

```python
# Pure pullback Adam, with neither calibration nor sigma-relative clipping.
CSTPullbackAdam(
    model,
    metric="block",
    lr=1e-3,
    target_map_step=None,
    max_step_sigma=None,
)
```

`damping` regularises each local metric before applying its inverse square
root. It is relative to the local diagonal scale and prevents nearly invisible
directions from producing unbounded coordinate steps. `eps` remains the
separate Adam moment-denominator safeguard.

## CUDA metric compilation

`compile_metric=True` enables the pure-tensor Gaussian +
`L2NormalizedColumns` fast path on CUDA:

```python
CSTPullbackAdam(
    model,
    metric="block",
    lr=1e-3,
    compile_metric=True,
)
```

That path fuses normalized-column construction, coordinate Grams, and the
learnable-bandwidth tangent calculation. It also uses the gauge-declared
block structure directly: one amplitude scalar plus independent input- and
output-coordinate blocks, rather than padding them into one larger eigensolve.
Other factors and gauges keep the general eager implementation.

`metric_chunk_elements` bounds the temporary tensor budget used by each
metric chunk (default `1 << 24`). Increasing it reduces chunk launches at the
cost of peak memory; it does not change the metric.

## Kernel-coherence regularization

`SampledKernelCoherence` discourages different atoms from delivering the same
normalized map direction. For atom factors `u_k` and `v_k`, it penalizes

```text
mean_{k < l} ((u_k.T @ u_l) * (v_k.T @ v_l))**2.
```

This is not the position-only `PairRepulsion`: coherence is evaluated on the
actual finite neuron charts, so boundaries and irregular chart sampling are
included. It is disabled unless explicitly supplied:

```python
from torchcst import CSTPullbackAdam, SampledKernelCoherence

# Exact all-pairs diagnostic or small experiment.
exact = CSTPullbackAdam(
    model,
    coherence=SampledKernelCoherence(1e-3, pairs=None),
)

# Unbiased fixed-cost approximation for a large atom population.
sampled = CSTPullbackAdam(
    model,
    coherence=SampledKernelCoherence(1e-3, pairs=4096),
    coherence_seed=17,
)

# Independent post-Adam interaction step; it does not enter Adam's moments.
decoupled = CSTPullbackAdam(
    model,
    decoupled_coherence=SampledKernelCoherence(1.0, pairs=4096),
    coherence_lr=1e-3,
    coherence_seed=17,
)
```

The gradient enters before pullback whitening and Adam's moments.
The decoupled form instead applies ``-coherence_lr * grad`` after Adam's
ordinary task step and leaves both moments unchanged. These two forms are
mutually exclusive. `optimizer.last_coherence` reports the latest unweighted
mean squared cosine.
Exact mode stores vectors proportional to the number of pairs and is therefore
intended only for small populations; sampled mode never constructs a `K x K`
Gram.

## Structural lifecycle

Atom and chart moments are physical-slot indexed followers. Capacity growth
pads state; birth, death, and refit reset affected rows; remap carries surviving
rows. Shared factor state is not row indexed. The optimizer is a standard
`torch.optim.Optimizer`, so its parameter groups and state dictionary follow
PyTorch conventions.

`OffsetCSTConv2d` currently raises at construction: its trailing displacement
coordinates act through a bilinear stencil rather than a factor column, and
that stencil Jacobian does not yet expose a pullback block. Ordinary
`CSTLinear`, `CSTConv2d`, and depthwise sites use the factor path above.
