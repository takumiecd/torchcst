# Cholesky training peak: allocation attribution and causal interventions

The dominant fixed allocation at K64 is cuBLAS workspace retention, not the
Gram matrix itself. This conclusion combines allocation stack traces,
byte-exact reconstruction of the live peak, and interventions that change
workspace retention while retaining the Cholesky equations.

## Scope and protocol

A100 80GB PCIe MIG3g.40gb, PyTorch2.6.0+cu126, batch128, K64/P4, square
width128 and512, FP32 model, highest matmul precision, factored DeviceRay1,
explicit damping1e-4. Same model seed17, eight resident synthetic batches,
8 warm updates and3 timed blocks of8 updates. Each case is a fresh process.
Final parameters are saved after32 updates. A separate33rd update supplies the
allocation audit. All measured loops pass synchronization-error mode and the
CST final error latch. These are not long-run learning-accuracy experiments.

Numbers are PyTorch allocator **peak allocated** bytes after warmup, including
retained buffers and input data, not cold compilation peaks or total process
GPU use. Reserved memory is recorded separately. Memory outside PyTorch's
allocator is not attributed here, consistent with its
[snapshot documentation](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html).

`memory_attribution.py` records allocator history before warmup, inventories
CUDA graph input/output storage addresses and pool IDs, and takes snapshots
before/after the diagnostic step. `analyze_memory_snapshot.py` replays alloc
and free-requested events from the initial live allocation set and saves the
largest live set. Trace requests are rounded to512 bytes for this default
native allocator configuration. It checks the reconstructed maximum against
the actual allocator peak counter. Timestamp/symbolization differences between
snapshot exports are excluded from trace-prefix identity checks; action,
address, size, and stream must still match. Truncated histories fail analysis.
This is a diagnostic for this allocator configuration, not a universal parser
for every allocator backend or rounding policy.

## Exact accounting at width128

The reconstructed and measured peaks are both80,505,344 bytes (76.775879MiB).

| Live allocation category at peak | MiB |
| --- | ---: |
| cuBLAS workspaces:8 allocations of8.125MiB | 65.000000 |
| Retained graph input copies | 6.957031 |
| Active allocations in private graph pools (excluding preceding categories) | 1.580566 |
| Two sets of returned factor value/Jacobian/Hessian copies | 2.625000 |
| Remaining data, metric, parameters/state and small temporaries | 0.613281 |
| Total | 76.775879 |

The65MiB is84.7% of the peak. Each large allocation's C++ stack passes through
`at::cuda::getCurrentCUDABlasHandle()` and the allocator workspace request.
Python contexts identify forward, backward, factor observations, two Ray
captures, transport, Gram construction, and Cholesky. There are seven retained
graph entries, including two Ray entries. No claim is made that every workspace
belongs to a distinct CUDA stream: handles and streams jointly determine reuse.

This matches the deployed
[PyTorch2.6 source](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CublasHandlePool.cpp):
it caches a workspace for each handle/stream pair; the A100 default is
`4096*1024*2 + 16*1024*8` bytes (8.125MiB).
`CapturedCall` creates a stream for every new cache entry in
`src/torchcst/_runtime/graphs.py:21`; warmup initializes these library workspaces,
which remain allocated during later training. The static input clones and
private graph pools are additional allocations, not included in the65MiB.

A 256x256 Gram is only0.25MiB in FP32 (0.5MiB in FP64). It cannot explain the
observed65MiB. The operation that actually crosses the maximum is also not
Gram-matrix allocation: it is the returned factor-derivative clone in
`graphs.py:38`, called from `FactoredFrameGeometry.gram`. The old cached factors
are still live while the new result is being built. `frame()` clones the point,
and factor caching compares point identity, so the compression frame triggers
another factor evaluation/copy. This explains the peak trigger, but its2.625MiB
is much smaller than the persistent library workspace contribution.

## Interventions

Setting `CUBLAS_WORKSPACE_CONFIG=:16:8` **before process launch** changes each
workspace from8.125MiB to0.125MiB in this PyTorch version. The same Cholesky
algorithm and damping remain selected.

| Method / width | Default peak MiB | Small-workspace peak MiB | Reduction MiB |
| --- | ---: | ---: | ---: |
| CST128 | 76.775879 | 12.775879 | 64.000000 |
| CST512 | 111.110352 | 47.110352 | 64.000000 |
| Dense+Adam128 | 17.134277 | 1.134277 | 16.000000 |

The snapshot classifier finds8 default workspaces totaling65MiB versus8 small
ones totaling1MiB for both CST widths. Dense has2, totaling16.25 versus0.25MiB.
The observed reductions exactly equal the predicted workspace-size differences.
This is causal evidence, beyond attributing a large allocation by its stack.
**Comparing small-workspace CST against default-workspace Dense would be unfair.**
CST still uses more memory when both receive the same setting.

Additional interventions bypass graph capture for a single selected function,
while executing that function's same mathematical operations:

| Width128 configuration | Peak MiB | Change vs baseline MiB |
| --- | ---: | ---: |
| Normal captured Cholesky | 76.775879 | — |
| Gram construction uncaptured | 73.899902 | -2.875977 |
| Cholesky solve uncaptured | 68.597168 | -8.178711 |

Both bypass variants reproduce all saved parameters bit-for-bit after32 updates.
Each has7 workspaces (56.875MiB). Uncaptured Gram construction uses more transient
workspace, so its total peak drop is smaller than the one removed cuBLAS buffer.
One-at-a-time peak differences are not an additive memory budget.

Reducing workspace size can alter library kernel choices and rounding. After32
updates, small-workspace versus baseline CST has relative parameter L2 difference
0.003137 at width128 (maximum absolute0.023973), and0.0002339 at width512
(maximum absolute0.002110). Final synthetic losses are close, but this is not
bitwise equivalence or an accuracy guarantee. Dense parameters match exactly
in this experiment. No environment setting or runtime default was changed.

### Reuse only the capture/warmup stream

A diagnostic `--shared-stream` variant preserves separate graph pools, input
and output isolation, events, the same compiled functions, damping and default
workspace size. Only the cold capture/warmup stream is reused across graph
entries in this single-thread benchmark. This is not a production concurrency
implementation.

| Width | Original peak MiB | Shared-stream peak MiB | Reduction MiB |
| --- | ---: | ---: | ---: |
| 128 | 76.775879 | 44.275879 | 32.500000 |
| 512 | 111.110352 | 78.610352 | 32.500000 |

Snapshot attribution shows4 workspaces totaling32.5MiB instead of8 totaling65MiB.
The full peak drops by exactly the removed32.5MiB. Both saved parameter sets
match their original baseline bit-for-bit after32 updates. This isolates
stream-related workspace multiplicity without changing the cuBLAS workspace
size or replacing the Gram solver. It is stronger evidence than simply
turning off a mathematical operation. It also identifies a more conservative
next implementation target than changing workspace configuration.

### Timing without allocation-history recording

Separate fresh-process runs omit stack/history recording but retain the small
diagnostic wrapper. Three8-step blocks produce these medians:

| Width | Default workspace ms/step | Small workspace ms/step |
| --- | ---: | ---: |
| 128 | 15.76 | 16.78 |
| 512 | 16.35 | 18.07 |

Individual blocks span12.69–20.05ms across these runs. This small, host-sensitive
sample does not establish performance equivalence; small workspaces can affect
kernel selection. It does establish that the64MiB reduction does not require
the multi-second PCG iterations in this experiment. The shared-stream variant
was measured with allocation history and has no separate clean timing claim.


## Implications

At width512 the same65MiB is58.5% of the111.1MiB peak; the remaining46.1MiB
already matters. Factor caches and graph input copies grow with K times width;
Gram storage grows with K squared. Thus this finding is about the measured K64
regime, not a proof that Gram never becomes a bottleneck at large K.

The first target is workspace/stream retention in graph setup. The second is
redundant factor caches and static input copies, including the two retained Ray
entries. Replacing Cholesky with many PCG iterations was an expensive way of
removing some of these fixed allocations; it did not isolate their real cause.
Production stream sharing still requires concurrency/lifetime testing. A
multi-layer checkpointed memory advantage over Dense remains unestablished.

## Reproduction

```bash
PYTHONPATH=src python -m experiments.memory_attribution \
  --history --output output/memory_audit/base128
CUBLAS_WORKSPACE_CONFIG=:16:8 PYTHONPATH=src \
  python -m experiments.memory_attribution --history \
  --output output/memory_audit/small128
PYTHONPATH=src python -m experiments.memory_attribution \
  --history --bypass _cholesky --output output/memory_audit/no_cholesky_capture128
python -m experiments.analyze_memory_snapshot output/memory_audit/base128
```

Use `--width 512` for width scaling; `--dense` for Dense+Adam. Omit `--history`
for the separate timing check. Snapshot instrumentation materially slows CPU
submission, so history-enabled wall times must not be compared with ordinary
uninstrumented benchmark times. The diagnostic wrapper still performs memory
counter reads in all variants.


## Validation and artifacts

All ten history-enabled cases reconstruct the exact measured diagnostic peak,
with no unmatched allocations/frees or truncated traces. Four additional
history-disabled timing cases reproduce the corresponding memory peaks. All
case subprocesses exit successfully. Every complete measured loop passes the
CUDA synchronization-error guard, and CST passes its final validation latch.
This is not a full CPU/CUDA profiler synchronization audit.

Ruff and diff checks pass. Runtime source and defaults are unchanged. Raw
snapshots, attribution JSON, graph inventory, final parameters and logs are
retained in `output/memory_a100/results.tgz` and extracted alongside it.
`parameter_comparisons.json` beside the extracted cases records the local
post-run parameter comparisons. The snapshot archive contains trusted local
pickle data; do not load arbitrary third-party pickle files.

Initial attribution checkpoint: `724d65a`.
