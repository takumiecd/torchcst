# Persistent local-product execution slots

Research branch `codex/local-product-hybrid`, parent `af67ce05`.

The user approved keeping execution placement across steps and moving only the
atoms whose tile or width-band membership changes. This candidate preserves the
normalized product-kernel contract and the existing local/saved-H hybrid. It
uses the existing Linear benchmark, not an outer GEMM implementation.

## Placement and numerical state

Canonical trainable atom rows and optimizer moments keep their IDs. Two
model-owned ID/slot mappings serve forward and dX. The bucket classification is
unchanged from `hybrid_packed`: full-domain live singleton pairs go into their
output/input owner tiles; other active atoms enter rho<1, 1<=rho<mid, rho>=mid
bands; locally inactive atoms enter a tail. Singleton status is tested from
actual support and normalization, never inferred solely from rho<1.

Each bucket starts with round-up-to16(count)+16 slots. An unchanged membership
keeps its slot. Departing atoms leave holes; incoming atoms fill holes in their
destination buckets. Only a destination count exceeding capacity redistributes
all segment capacities and sorts/rebuilds IDs. A conservative fixed upper bound
on total slots supports all possible distributions with unchanged Graph shapes.
High-water endpoints avoid traversing unused trailing capacity in forward/dX.
All-hole parameter blocks skip contractions. This is a bounded small-operator
prototype with two layout CTAs, not a scalable sort for arbitrary large matrices.

Every step still computes current polar coefficients, full-domain normalization,
support intervals and membership keys. Current numerical metadata is written to
per-forward execution views; placement reuse does not freeze values or sigma.
Keys, IDs and slot positions are changed only when membership changes. Immutable
per-forward metadata/order/segment snapshots preserve delayed backward correctness
even when a later forward repairs mutable topology. Saved H is indexed by
canonical atom ID, independently of spare execution slots. The backward writes
all atom gradients to canonical rows. Production-equivalent capturable AdamW and
polar update remain unchanged.

`PersistentLayout` belongs to the research model, is absent from state_dict and
is reconstructed at model setup. Setup reads counts on the host; refresh/repair
has no device-to-host synchronization. Calls must be serialized on the model's
CUDA stream. Spare capacity and snapshots add memory; no explicit L1/L2 residency
or cache-hit improvement is asserted. All atoms' membership keys are checked
every step; only the actual placement mutations are incremental. This first
version does not fuse maintenance with the optimizer.

## Validation protocol

Use `plans-local-persistent.json` with
`cases/local-128-persistent-{early,middle,late,narrow-only}.json`. Controls are
support-saved and the validated full per-step `hybrid_packed` route. N128/B32/A819
is5% of dense weight count, FP32 with TF32 off, shared spacing1 and bounds
(minimum/birth=.25,maximum16), live sigma with the same optimizer contract.
Initial singleton shares10/50/95/100% are synthetic fixtures, not evidence of
training convergence. All main timing includes preparation, layout maintenance,
forward/loss, dX, all atom gradients, AdamW and polar update, with Graph capture
and replay included in memory measurements.

Tests check stable slots under coefficient/centre changes; a one-atom tile
migration changes only that atom's reverse mappings; capacity overflow rebuilds
correctly; support growth moves atoms to wide buckets; multiple outstanding
forwards preserve their own backward views; empty operators work. Independent
scalar FP64 comparisons include full/sliced domains and spacing1/.5, mixed bands,
floor-active and two-site guards, all gradients, and captured CSTOptimizer state.
Full N128 mixture oracles and runner correctness gates precede accepted timing.

First L4 job `l4job-fa2a95ffbfec4aaaaf9a5a7a9267dd66` failed compiling refresh:
vector capacity endpoints and scalar loop endpoints shared a variable name,
causing Triton's branch type join to fail. Names were separated, including
sorted vector bucket IDs and scalar loop indices. No accuracy/performance claim
comes from that run. Failed raw output is preserved under ignored
`benchmarks/cuda/linear/evidence/local-persistent-20261004/failed-compile/`.

Revalidation job: `l4job-6967e2fe5ce94b25904dcabab175b154`.

Separate fixed-state component graphs compare normalized metadata refresh with
full sorting/packing. Separate repeated1/8/32-atom centre flips measure genuine
small tile migration, including flip+normalization+maintenance. Those diagnostics
are explicitly not complete training-step timing; counter reports prove which
placement path ran. Stable synthetic fixtures may change all sigma values while
crossing no membership boundary, so their timing alone cannot establish the
cost of actual migrations.

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-persistent.json \
  --case benchmarks/cuda/linear/cases/local-128-persistent-late.json \
  --phase-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-persistent-late.json
```

## Verified L4 result

The revalidation job succeeded:254 checks passed in386.73s, all12 full N128
scalar FP64 Y/dX/dP comparisons and all12 runner correctness gates passed.
Largest full-shape absolute errors were2.180e-6 /2.204e-6 /2.491e-6.
CPU checks:141 passed,113 CUDA-only skipped. Targeted Ruff format/lint and
whitespace checks pass.

Source archive SHA256:
`cc346bd11b122af3c89c54985c8b71777c89828973e258422aac8f3e1c099a6b`.
Verified result archive SHA256:
`d04cc7d70e46163218504fd07b5585b4d45f2a86f0cdbcac3d93646e21f23a4e`.
Actual hardware: NVIDIA L4, driver580.82.07, Torch2.11.0+cu130, CUDA13.0,
Triton3.6.0, Python3.13.15. Pool/runtime were verified stopped after retrieval.
Raw output, summary, driver and receipts remain in the ignored evidence directory
and the pool job directory. Source/tests/catalogs match the measured snapshot;
post-measurement edits only document the result.

### Complete training step

| Initial live singleton share | Support saved ms | Full per-step pack ms | Persistent ms | Median reduction vs pack |
| --- | ---: | ---: | ---: | ---: |
| 10% | .333434 | .300114 | .296987 | 1.04% |
| 50% | .327887 | .240219 | .232877 | 3.06% |
| 95% | .305832 | .176170 | .167530 | 4.90% |
| 100% | .297422 | .153366 | .145894 | 4.87% |

All819 sigma values changed in each timed route. Each persistent view reports
52 refreshes, zero moved atoms, zero incremental repairs, zero overflow rebuilds
through primary warmup/capture/replay. Thus these are live-width training steps
with stable membership, not actual-migration full-step measurements. The fixtures
retain their initial singleton counts. Spare capacity is1200 slots per view for
819 canonical atoms.

Medians use21 synchronized Graph replays in separate fresh workers. Sequential
workers are not paired randomized timing. The p10–p90 ranges overlap at10% and
100%; at50% and95% they do not. Do not treat the1.04% early reduction as a robust
speed improvement. The same-job dense reference is .045492ms, still3.21x faster
than the100% singleton persistent step. Only the early case enables dense; all
four use the same shape/dtype/dense optimizer contract.

Every persistent case peaks at **.397949MiB allocated** through capture/replay,
versus .324219MiB for full-pack and .265137MiB for support-saved. The persistent
increase over full-pack is77,312bytes (.073730MiB). Each custom route reserves
6MiB. Dense peaks at32.814941MiB allocated /46MiB reserved in the same warmed
isolated-worker measurement scope. Total GPU process usage is unmeasured. These
are measured allocator peaks, not tensor budgets or cache-residency measurements.

Separate phase diagnostics (forward+loss / backward / optimizer, ms):

| Share | Full-pack phases | Persistent phases |
| --- | --- | --- |
| 10% | .106496 / .100352 / .091136 | .103424 / .099328 / .091136 |
| 50% | .076800 / .069632 / .091136 | .071680 / .066560 / .091136 |
| 95% | .047104 / .035840 / .091136 | .039936 / .033792 / .091136 |
| 100% | .040960 / .019456 / .091136 | .032768 / .019456 / .091136 |

The optimizer contract remains a substantial cost. Diagnostic phases are from
a separate instrumented graph after primary timing/memory; they are not an exact
additive decomposition of the primary wall-time median.

### Placement maintenance diagnostics

Fixed-state metadata preparation costs9.339–9.564us; full two-view sorting/packing
costs13.804–14.428us; persistent key-check+value-view refresh costs5.693–6.164us.
There were2104 refreshes per view and no moves/rebuilds in each fixed-state graph.
These exclude the rest of the training step and do not isolate cache effects.

Narrow-only1/8/32-atom centre-flip diagnostics include the same flip operation,
current normalization, and placement maintenance. Each repeat moves the selected
atoms to opposite owner tiles; their sigma is fixed only in this maintenance
fixture. The number of movers is small, but the centre jumps themselves are
large. This is not a dynamic-width complete training step.

| Movers per repeat | Full-pack us | Incremental us | Result |
| --- | ---: | ---: | --- |
| 1 | 25.599 | 25.948 | Approximately tied, no speed gain established |
| 8 | 25.887 | 33.556 | Incremental29.6% slower |
| 32 | 25.876 | 38.973 | Incremental50.6% slower |

Each incremental view reports2104 repairs,0 rebuilds and respectively
2104/16,832/67,328 moved atoms. No full pack occurs on these paths. Unit tests
independently assert that a one-atom migration changes only its reverse-map
entries; overflowing96 atoms and widening support trigger the intended rebuild
and preserve Y/dX/all gradients. The bucket-by-bucket hole scan incurs repeated
prefix scans when movers target several buckets. The measurement establishes
this implementation's movement penalty, not that incremental layouts in general
must lose to full sorting. Improving hole allocation is an outstanding performance
opportunity; reverting to a full per-step sort would violate the intended design.

Persistent refresh uses168 registers/8192 shared bytes with zero reported spill
slots. Forward48/3648, dX40/17408 and parameter VJP105/24576 likewise report no
spills. Compiler reports do not measure actual cache hit rates or memory traffic.

## Decision

Retain `hybrid_persistent` as a validated research alternative. It implements the
requested placement lifetime and selective mutation, and lowers stable-membership
maintenance cost. The complete-step gains are modest, memory increases, and
frequent multi-bucket migrations remain a measured negative result. Keep the
full-pack control available in the same runner. No main-tree promotion or outer
GEMM change is made. Future optimization can replace repeated per-bucket hole
scans and reduce spare/snapshot traversal while preserving mutable sigma,
normalization, canonical IDs and independent backward snapshots.
