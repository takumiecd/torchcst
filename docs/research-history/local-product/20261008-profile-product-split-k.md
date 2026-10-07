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
