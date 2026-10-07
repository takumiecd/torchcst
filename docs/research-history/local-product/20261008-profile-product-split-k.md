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
