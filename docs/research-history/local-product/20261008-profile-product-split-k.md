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

All five reverse jobs completed successfully, retaining all seven routes and
their full-site/all-atom correctness gates. Their Graph medians (milliseconds):

| Chart and size | Prepared | Split1 | Split4 | Split8 | Split16 | Torch | Dense |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Product1024 | .565126 | .311725 | .283991 | .280713 | .282311 | .245924 | .083404 |
| Product2048 | 2.427544 | 1.142683 | 1.085915 | 1.083115 | 1.100422 | .963934 | .591985 |
| Strip64x1024 | .113618 | .093679 | .067010 | .062471 | .062580 | .063334 | .054978 |
| Strip64x2048 | .187700 | .148262 | .093415 | .084097 | .084473 | .080785 | .056958 |
| Strip64x8192 | .436559 | .455160 | .231797 | .196154 | .194970 | .160389 | .066072 |

Split8 relative to split1 improves Product1024/2048 by9.95%/5.21%, and
Strip1024/2048/8192 by33.31%/43.28%/56.90%. The primary and reversed orders
agree on improvement, while the tiny split8/16 ordering at Strip1024/2048
changes between runs. Retain both explicit alternatives rather than declare a
universal split-count winner. All split allocated peaks reproduce exactly;
reserved peaks reproduce too, including the2MiB Product1024 split8 increase.
Product/Strip rho3 has two independent jobs per size, rho8 one, and Product8192
has no primary split timing. The14 artifacts' maximum absolute Y/dX/dp/update
errors are7.82474e-5/4.23136e-5/3.77818e-6/0.

Disposition: integrate strict research matrix revisions v4/v2, their split1
controls, explicit split4/8/16 alternatives, complete cases and verified
negative/positive evidence. Keep recipe default split4 and public dispatch
unchanged. The results support a scheduling improvement with these size/width
conditions, not dense parity or a Torus speed claim. Single-chart work remains
necessary: the confirmed2048 improvement is only about5%, and both native and
Torch trail dense. The large-phase job remains in flight at this checkpoint.

PR#69 head `0f68fae90773090c8f423a6c0ff9ac6d0d9e43c4` passed exact-head
CPU validation run37699282958: prepare/check, declarations, CPU tests and
wheel/sdist build all succeed. Final evidence-only changes require their own
final-head Actions before GitHub merge. Implementation/benchmark/test file sets
and bytes remain exactly those of measured0864a752; no CUDA change follows the
gate or measurements. Literal Torus mathematics remains an unanswered user
choice; the separate CPU proposal is not an integrated declaration or kernel.

## Completed large-phase study and integration

PR#69 merged through GitHub at30775a547ff10175791e59b2dbec0709a5403a5d after
final head a12ab8fee0cc464810ee0c12720d2ffea268e10f passed Actions37700168659.
Local main was fast-forwarded afterwards. The separate phase job completed
five full-site/all-atom FP64 and production Polar gates, including the actual
8192x8192/3,355,443-atom Product fixture for split1 and split8. Source file sets
and bytes, downloaded receipt, all21-sample event arrays and compiler metadata
are verified and retained in the summary/raw evidence. Runtime: NVIDIA L4,
Torch2.11.0+cu130/CUDA13.0/Triton3.6.0. Supervisor31002 exited0 and every pool
slot is stopped; no GPU process memory claim is made.

Warm kernel-event medians in milliseconds (external events, sequential plans,
continued parameter updates; diagnostic scope only):

| Product size / split | Preparation | Assembly | Forward matrix | dX matrix | dW matrix | Atom VJP |
| --- | --- | --- | --- | --- | --- | --- |
|2048 /1|.275456|.129024|.256000|.063488|.065536|.281600|
|2048 /8|.280576|.130048|.207872+.006144|.053248+.005120|.067584|.288768|
|8192 /1|5.384192|17.207296|4.121600|1.297408|1.189888|9.718784|
|8192 /8|5.571584|17.347584|4.128768+.032768|1.178624+.030720|1.199104|9.779200|

Split8 matrix entries show partial contraction plus its separate reduction;
the enclosing matrix wrappers and phase events are also preserved but must not
be summed with these nested events. The independent learning-Graph medians in
this diagnostic process are43.213354/43.416781ms for Product8192 split1/8:
no improvement is observed there. This is not a primary isolated complete-step
comparison or memory measurement. It limits extrapolation from the confirmed
1024/2048 gains and identifies assembly and atom VJP as larger measured costs
at8192. The corresponding FP64 gates pass; split1 max Y/dX errors are
1.80482e-4/1.79605e-4, split8 4.26656e-5/3.80911e-5, dp4.81145e-6 and
same-cotangent Polar update exactly equal for both.

Assembly/VJP compiled registers remain52/72 with zero reported spills at both
sizes. Increased working set and scattered W/dW accesses are plausible causes
of their growth, but no cache-counter or causal ordering experiment has yet
proved that explanation. The next isolated candidate is physical atom ordering
by two-dimensional support tiles with canonical IDs retained. Its sort/copy
cost, additional scratch, floating accumulation order and complete-step peaks
must be measured before any adoption. Continue on kernel/profile-product-spatial-order;
no ordering implementation or speedup is claimed by this completed study.
