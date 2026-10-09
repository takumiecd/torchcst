# Sphere direct profile recomputation study

The preceding CUDA Core direct cohort was correct but failed the compact-W
time/memory adoption criterion. This separate study tests whether deleting
profile storage at allocation time improves that tradeoff. It preserves the
existing pair of intrinsic S² charts, Polar/Triweight chord profiles, complete
per-chart discrete L2 floors, live widths and all six atom gradients.

## Execution change and constraints

The new research Algorithm owns the storage choice; its Recipe has the same
CAP64/T16/int16/group1-or4/separate-or-merged fields as stored direct.
The existing decoder and site normalization are reused. A dedicated packer
stores complete Norm and Count plus ordered support IDs, never allocating Phi.
The contraction recomputes gap³/max(savedNorm,floor) from saved sites, centres
and precision. Backward does not reread mutable live kernel/chart buffers.
The original stored Algorithm retains its arithmetic and execution contract.

This removes2*A*64*4 bytes of tensor capacity:26,843,136 bytes at1024 and
107,374,080 bytes at2048. These estimates are not measured peak reductions.
H remains saved for backward, G stays CTA-local, and no full U/V or W/dW is
allocated. Atomic scatter counts and complete-chart norm scanning remain.
FP32 IEEE, disabled TF32 and disabled floating-point fusion are unchanged.

## Frozen comparison before GPU execution

N1024/N2048, batch32, floor(.05*N²) atoms, seed41, sigma3, MSE, fused AdamW
lr1e-4/weight-decay.01, existing research Graph update, and exactly24 updates
match the preceding cohort. All CST routes require independent full-site,
all-atom FP64 Y/dX/all-dP at initial and after24 checkpoints with both maxabs
and relative-L2<=4e-4. Twenty-one timed Graph replays follow the same2 warmup
and1 capture/replay updates. Peak allocated/reserved includes capture/replay.
Separate phase diagnostics are never added to or subtracted from step timing.

The new fixed cohort has22 workers: compact W, old direct, four stored direct
controls, four matched recompute variants and dense at each size. No route is
filtered before measurement. Recompute-vs-matching-stored comparisons isolate
storage/recomputation; adoption still compares against compact W. Qualification
requires >3% speed improvement, or >=5% allocated reduction with <=3% time
regression, at both sizes and independent reverse confirmation. The complete
controls remain in any inverse cohort. No public default is changed by a
benchmark result alone.

Compilation-heavy regression checks use two separate prerequisite jobs:
the original159 tests and the new73 runtime plus6 benchmark tests (238 unique
regression cases total). A third job checks all22 full correctness workers;
timing starts only after all prerequisites pass on the same frozen source.
Each job keeps the900s outer/875s driver/350s child limits. All collections,
partitions, JUnit mappings, source hashes and failures are preserved. This is
a separately declared implementation study, not an expanded retry budget for
the preceding failed gate. No tolerance relaxation or partial winner selection
is permitted.

The implementation and CPU/declaration/build checks are ready. GPU correctness,
speed, measured peak memory and adoption are pending.
