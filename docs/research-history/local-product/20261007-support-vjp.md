# Local support tiles for normalized parameter VJP

Branch `kernel/local-support-vjp`, based on merged main `eff5c277`. Existing parameter VJP falls back from a serial support loop to full-domain matrix contractions when any atom in its block has support wider than8 sites. The new explicit `_pv16` / `_pv32` variants gather at most16/32 actual local sites per atom and batch row, then reduce value and normalized center derivatives. This preserves the full-domain normalizer coefficients; no position gradient is removed. The actual input/output interval lengths select the branch, not initial rho. Larger spans use the existing exact matrix path.

Forward, dX, preparation/order/cache repair, canonical gradient mapping, decoded-width VJP contract and production AdamW/Polar stay unchanged. Parameter tiles are compiler-local tensors; no global H/G or additional autograd storage is added. Physical traffic/on-chip residency and complete-step allocated/reserved memory require measurement.

Six variants compare matched uncached parameter-atom16 and counter-free cached repair4/repair8 parameter-atom32 controls. They keep batch splitting, warps, owner divisions and all fixture bytes. The new GPU suite covers independent FP64 Y/dX/all source gradients, positions, sliced geometry, batch1/32/64, retained/outstanding backwards, empty owner neighbors, single-axis singleton support, actual16/32-site threshold cases, wider fallback and20 captured optimizer updates atN64/N128. Validation gates performance; no tolerance is relaxed.

Host CPU validation, changed-file Ruff, eight fresh prepare/check pairs and wheel/sdist build pass. GPU validation and current/candidate/dense complete-step measurements are pending. Performance fixtures N64/A204 andN128/A819 B32 have initial decoded rho>1; widths evolve with production fused AdamW/Polar. First target is N128 rho8/mixed; follow-up will retain negative results and repeat promising cases with reverse order.

Reproduction: `tools.kernel_dev prepare/check` with `benchmarks/cuda/linear/plans-local-supportvjp.json` and `cases/local-supportvjp-{64,128}-rho*.json`, then the existing runner with `--polar-update fused --phase-diagnostics --kernel-diagnostics`. Uninstrumented complete-step time is primary; component phases are separate diagnostics. Frozen scripts/logs/source archives remain ignored under `benchmarks/cuda/linear/evidence/support-vjp-20261007/` and shared pool jobs. PR#55 compact-dX work remains on its original branch; this experiment does not depend on it.

A local preparation error reused an existing parameter-atom catalog filename. CPU regression tests caught it before GPU submission; existing files were restored and this comparison renamed `supportvjp`. The failed CPU log is retained and excluded from completed validation.


## Owner routing candidate added during GPU validation

Three `_spanranges` variants retain the existing exact position order and cached repair. For each non-singleton band, current integer support max span M implies every contributor to owner[j0,j1) has j0-M+1 <=lo<j1. Two lower-bound searches on sorted lo produce a conservative candidate envelope. Final factor/interval masks retain exact contributions. No prefix-maximum array or histogram is added; only compiler-local reductions/searches. Fresh span maxima are required even when cached order is unchanged, because support ends can change independently.

The same parallel physical-copy launch runs without its full atom scan per owner. This tests reduction of owner exploration, not changing geometry. Empty bands, uneven slices, repeated snapshots, severe cached inversions and end-only changes are gated by the independent support-coverage and gradient suites. This route is separate from `_pv16/32`; the first frozen source `efe982f3` is the parameter-only screen. Timing across those source revisions is not treated as a matched algorithmic effect.


## Whole-support gather: validated negative screen

First source `efe982f3`, L4 job `l4job-e8470a312c0d4cb38a75a6a2b5fabbeb`: all77 checks pass (75 CUDA +2 declarations), with197 runtime files and source/result archive hashes verified. Full-shape Y/dX/all source gradients and nonzero center gradients pass before timing. AtN128/rho8 matched repair4 changes65.77us to77.39us (pv16) /83.91us (pv32); mixed68.73us to80.66us /86.90us. Repair8 shows the same regression direction. Allocated remains306,688 B; reserved remains6,291,456 B. Dense45.40/45.48us. This is one validated screen, not an independent-repeat claim.

Compiler report shows current parameter kernel80 registers /2 spill slots versus255 registers /368 spills(pv16) or216 spills(pv32), all40,960 shared bytes. These are compiler reports, not measured physical traffic. The large regression and compiler spilling reject the whole-tile variants for adoption; no GPU repeat or G4 comparison is justified. Raw artifacts plus byte-identical/idempotent DB2 records are preserved; concise hashes/deltas are in [the negative record](20261007-support-vjp-negative.json).

A new `_pvc16` / `_pvc32` variant processes the support in4-site chunks to reduce compiler live tensors. Same exact span threshold and normalized derivatives, matrix fallback for wider support, unchanged global allocation and captured update contracts. The original variants remain reproducible. Chunked VJP and owner-span searches are not yet GPU-validated; their next source must pass independent gates before timing.


## Chunked gather and span lower bounds: N128 negative screen

Source `352a04a49d57e32fa595fa78a7153f75719d7455`, job `l4job-86dded7899884b32b1d8458c146656b8`:111 checks pass (108 CUDA +3 declarations),312.10s;197 committed/submitted/worker runtime files and both archives hash-verified. AtN128/rho8 repair4 current66.11us vs span69.16us, pvc16 69.92us, pvc32 76.28us; mixed69.67us vs72.10/73.20/79.23us. Repair8 has matching regression directions. Dense45.86/45.69us. Allocated306688 B and reserved6291456 B unchanged. One screen only, neither family adopted.

Chunks remove spilling but still206 registers versus80/2 spill slots in current parameter kernel. Layout lower bounds change ordered_layout registers47→48 (repair4), copy/range56→40, all0 spills. Separate diagnostic layout10.24→13.31us: moving searches into only two sort CTAs adds latency even when per-owner scans disappear. These are compiler/event diagnostics, not physical traffic measurements. Concise timing/hash/compiler records are `20261007-support-vjp-chunk-span.json`; raw negatives are retained and DB import/export verification recorded separately. N64 screen at the same frozen source is in flight before changing adoption conclusions.

## Scalar support-limit candidate

Six `_ps16` / `_ps32` recipes change only the parameter VJP's exact span threshold for its existing dynamic scalar-site loop. Forward/dX retain support_limit8; wider parameter support keeps the original matrix contraction. This avoids rank3 gathers, extra global arrays and fixed-cap chunk iterations. The existing normalized factor and singleton derivative remain intact. CPU/prepare/build and independent GPU gates must pass before timing; scalar candidate is not yet GPU-validated.
