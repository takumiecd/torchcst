# IEEE split-K profile-product candidate

The thin forward/input-gradient FP32 IEEE matrix products can expose few CUDA
blocks. This candidate partitions their reduction dimension across1/4/8/16
splits, stores FP32 partials and reduces them in a fixed order. Effective splits
are capped at the next power of two of available BK32 blocks. K<=32 uses the
unchanged unsplit kernel. Canonical-parameter dW retains the unsplit contraction
to avoid quadratic S*NO*NI partial scratch when batch K is at most64.

Only reduction scheduling changes; normalization, complete support, grouped
atom preparation/assembly/VJP, live widths/pitches, precision and optimizer
contracts stay fixed. Small matrix revision v4 and large matrix revision v2
have strict SplitMatrixProduct/StripRecipe types; previous revisions and public
dispatch stay unchanged. Split1 is the same native control path.

Selected comparisons are Product/Strip1024 and2048, B32,5% atoms,seed41,rho3/8,
G8P8, support preparationG16/sites16, split1/4/8/16 plus Torch control and dense.
Catalogs and eight cases preserve the existing runner/adapter workflow. Complete
Graph time and capture/replay allocated/reserved peaks will determine disposition;
no improvement is claimed before GPU measurement. Gate includes independent
FP64 tail/stride contractions and unchanged full-site/all-atom, floor/support,
retained-forward and20 captured optimizer-update scenarios. Raw evidence lives
in ignored benchmarks/cuda/linear/evidence/profile-product-split-k-20261008/.

CPU gate:1162 passed/2069 skipped/18 warnings. Four catalogs/eight snapshots,
two prepare/check workflows, Ruff and wheel/sdist build passed. GPU tests and
complete-step timing remain pending; no speedup claim.

## Correct first-gate setup before measurement

Job l4job-484582e01b334140acf029658404172d at18e0b3421e2825ddcfbf7287c854bf87b7127a9a
returned1, with209 passed/10 failed in the219-test split suite. Every failure
was in the large fixture's pitch or gradient-branch scenario: the layer used
8191-input Strip or2049-input Product while reused tests still constructed65
input features (and33 output cotangents). The failures occurred in input binding
or independent-oracle shape multiplication before candidate contraction. No
performance measurement was submitted from this failed gate. Raw logs and source
are preserved under jobs/ and kernel/profile-product-split-k-first-gate.

Correct shared test inputs/cotangents to use layer.in_features/out_features. The
old small scenarios retain their original65x33 values, while the large fixture
now exercises its declared complete chart. No scenario, numerical check or
optimizer gate is removed or relaxed. All candidate CUDA runtime file bytes
remain identical to18e0b342; a new full GPU gate is required for corrected source.

Carry forward the large branch's bounded exact CPU support reports/Strip width
export. Add the prepared factor control to all split catalogs alongside split1,
4,8,16,Torch and dense. All twelve cases (1024/2048/8192,Product/Strip,rho3/8)
round-trip. First selected split measurements remain1024/2048 for both families;
add one first Strip8192 rho3 comparison to test its slow thin native contraction.
No8192 Product split speedup is inferred from the Strip case. All complete-step
budgets remain predeclared600s per selected child (1300s for two-rho families;
700s for the single8192 Strip case), with unchanged full-site oracles and21
samples. Actual timing/memory and independent order checks are still pending.

Corrected full CPU gate:1198 passed/2069 skipped/18 warnings. Twelve case
snapshots and all catalogs pass. Installed-wheel metadata/build evidence remains
from the byte-identical CUDA runtime in the first source checkpoint.

## Corrected L4 gate and first complete-step measurements

Frozen source `0864a7524a18ba1ea0432b8677a991e1ad105321` passed job
`l4job-52b753308c7b450db247962bc5eea551`: split223, matrix596 and large48,
867 tests in total. All 1,148 implementation/benchmark/test/kernel-dev file
sets and bytes match its Git archive. The source snapshot SHA256 is
`d66d262e6c958adebf517576c1b561e92f0b5ab13078f145448d4f9627ba3551`;
downloaded result archive SHA256 is
`a9494026a0aea55eb3ebc5aa38eef267064693375ac73c72e8963ce8386d9bfb`.
The immutable source is retained on `kernel/profile-product-split-k-measured`.

Five independent queued jobs completed nine primary artifacts on NVIDIA L4.
Each artifact includes all six CST plans and dense, with 21 uninstrumented
complete Graph-step samples per route, production Polar and fused capturable
AdamW, live widths, full-site FP64 Y/dX/all-atom VJP and update checks. Allocated
and reserved peaks include capture/replay and optimizer state; total GPU process
usage is unmeasured. Results and verified snapshots are preserved in ignored
evidence; the tracked summary retains every route, timing sample, peak and hash.

Graph medians in milliseconds, within each artifact:

| Chart and size | rho | Prepared | Split1 | Split4 | Split8 | Split16 | Torch | Dense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Product1024 | 3 | .558629 | .310314 | .281564 | .278250 | .281317 | .242282 | .082410 |
| Product1024 | 8 | .655723 | .457521 | .428029 | .423897 | .426203 | .386829 | .082942 |
| Product2048 | 3 | 2.512362 | 1.155153 | 1.102783 | 1.096303 | 1.113491 | .970947 | .587311 |
| Product2048 | 8 | 3.170825 | 1.909975 | 1.866445 | 1.812922 | 1.824589 | 1.744107 | .592252 |
| Strip64x1024 | 3 | .112950 | .092971 | .066735 | .062669 | .062152 | .063103 | .054962 |
| Strip64x1024 | 8 | .123655 | .104995 | .078230 | .073969 | .074143 | .074798 | .054986 |
| Strip64x2048 | 3 | .186808 | .147873 | .092526 | .083644 | .083153 | .080133 | .057045 |
| Strip64x2048 | 8 | .209570 | .168948 | .113801 | .103886 | .103608 | .099597 | .056985 |
| Strip64x8192 | 3 | .434674 | .454855 | .230807 | .195165 | .194694 | .159994 | .065654 |

The declared split8 route improves rho3 relative to split1 by about10.3% and
5.1% for Product1024/2048, and32.6%,43.4%,57.1% for the three Strip sizes.
This is a first-batch observation, not a final selection. Split8/16 differ by
less than one microsecond on these Strip rho3 cases, and Strip1024 split8/Torch
are also close. Retain both alternatives and confirm independently; do not
pick the smallest median from this batch. Product remains slower than Torch
and dense, while Strip8192 remains slower than Torch and dense. Product8192
complete-step split performance has not been measured.

Measured allocated peak is identical across split1/4/8/16 for each case:
Product1024/2048 use16,688,640/66,493,952 bytes, Strip1024/2048/8192 use
1,305,088/2,575,360/10,205,184 bytes. This does not imply zero split scratch:
temporaries fit under the existing complete-step peak. Product1024 split8
reserved peak is58,720,256 bytes versus56,623,104 for split1/4/16. Product2048
reserved peak is163,577,856 across splits; Strip1024/2048 is6,291,456 and
Strip8192 is52,428,800 across splits. All prepared/Torch/dense controls and
their actual allocated/reserved peaks remain in the summary.

Primary jobs: Product1024 `l4job-dfe241a84b9540df87e2ee7f0822efc9`,
Strip1024 `l4job-0990cae35ac64a99a78c42a76ff3f0d2`, Product2048
`l4job-a04b36b20e394aa2a563c3cf5546edc0`, Strip2048
`l4job-dd404ab6a79c47e0b7ca80d674011726`, Strip8192
`l4job-464a307ef8d94db19cfc755d276dcdb7`.

## Independent confirmation and large-phase diagnostics

The same source/cases/six plans/dense are queued as five independent rho3
jobs with plan order reversed,700s outer/600s child budgets and21 samples.
No earlier failed gate is re-budgeted or used as a paired performance result.
Jobs: Product1024 `l4job-4b86075b3eb047d78970101394cce28c`, Product2048
`l4job-09bd6343a49e4f88b639835d0bdf0e88`, Strip1024
`l4job-8175c801fa3746c79b21d252f4e2d711`, Strip2048
`l4job-1d68e32b81044d0aa76e3ed4cf1a9ad2`, Strip8192
`l4job-94aeab61bb9d4f8299af0ee3c7648e90`.

Warm external-event diagnostics are separately queued at900s for Product
2048/8192 split1/split8 and Strip8192 split8, job
`l4job-71b16bb0c91649418b6044d22e57bae1`. Each case performs its full
correctness check, an uninstrumented learning Graph, then a separate evolving
parameter Graph with preparation/assembly/matrix/VJP and phase events.
Diagnostic timings are not primary complete-step comparisons, and phase
medians must not be summed into a replacement step time. Spatial atom sorting
is a hypothesis until these measurements identify an actual bottleneck.

The first reverse job (Product1024) completed: prepared .565126, split1
.311725, split4 .283991, split8 .280713, split16 .282311, Torch .245924 and
dense .083404ms. Split8 improves on split1 by9.95%, consistent with the primary
10.3% observation, with unchanged allocated peaks. Four other confirmations
and the diagnostics remain pending at this checkpoint. Public
dispatch and literal Torus mathematics remain unchanged. Merge disposition for
the split implementation awaits independent confirmation and final-head CI.
