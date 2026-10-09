# Original Torus support preparation with lossless compact site IDs

Research-only candidate, public dispatcher unchanged. The preceding conservative
Grid-window route is rejected: it passed all exact support/physical-gradient
checks but lost36–67% complete-step latency. Its local measured/negative anchors
`kernel/torus-grid-window-measured-bdea` and
`kernel/torus-grid-window-negative-final` retain the source, full results and
negative history. Root archived and restore/fsck verified those anchors before
this separate study. No rejected window runtime is included here.

This candidate imports the original `sparse_weight.kernels.pack` without any
modification: the baseline conservative circle interval, original full-section
traversal, raw FP32 profile/stat reductions and exact full-axis capacity overflow
remain. Only packed site-ID buffers use signed int16 instead of int32. IDs are
nonnegative and must lie in0..32767; metadata explicitly guards that lossless
storage range while original execution support remains limited to axes<=2048.
Counts are reduced in int32 and stored exactly in the original FP32 Stats array
alongside norms and derivative moments. No count, raw-factor or norm quantization.

Every stored ID is widened to int32 immediately when loaded, before _circle,
_section's3*site offset, or W/dW row-stride multiplication. Padding is guarded
and maps tozero. Exact overflow reconstructs the complete-axis int32 arange.
The local floating `patches` and `contract` ASTs equal the baseline, retaining
one combined global normalization floor, the coupled q0 circle-radius/section
derivative, all five atom derivatives and dX. Source/sigma/radii/site snapshots,
retained backward, shared W/dW and research intrinsic updates are unchanged.

Only recipe C64/S256/T16/index_bits16 is permitted. No32bit alternative or
capacity sweep is added to the new route. Neither sparse_weight runtime files
nor shared helper files are modified. The new executor and contractions are
self-contained and do not import Grid-window code.

## Fixed comparison and gates

Same common MSE protocol: B32,N1024/2048,sigma3/8,~5%atoms,seed41,FP32 IEEE,
AdamWlr1e-4/decay.01,Polar intrinsic update. Predeclare all four cases in the
original order1024sigma3,2048sigma3,1024sigma8,2048sigma8. Per case compare
baseline sparse-W32,onecompact16 candidate andDense,12workers. The unchanged
central driver SHA256 is
`8ad59c07a7852963dc31000246a5d1ed6801ebc0512ede70f8a183d276647d05`.
No public-control, window route, retuning or threshold change.

Correctness budget900seconds: common30tests plus newcompact33tests
(1CPU+32GPU), followed by the fixed12verify-only workers. New tests cover
physical FP64 Y/dX/allP with floor1e-6/.5/100 and B1/3/32/64; independent
and empty gradients; retained forward;20live-geometry/width Graph updates with
public parameter/moment agreement; all four square seam/gap/overflow cases;
period discrepancy±.9/±1.1spacing; exact original packed IDs/raw/stats for both
storage widths; live grid/minor snapshots; synthetic ID32767 with3*site and2*W
strides plus padding; candidate copied-CST phase counters24/30; count-asserted asymmetric circle-only,
section-only and both-overflow fixtures using fixedsigma100 and full FP64
physical Y/dX/allP checks. These broad fixtures are correctness-only, separate
from the sigma3/8 comparison cohort.

Primary budget1500seconds: common30tests, then the fixed12workers. Initial
and24updated full-A/all-sites independent physical FP64 Y/dX/allP gates retain
4e-4maximum absolute/relativeL2 and existing elementwise checks. Complete-step
21samples and capture/replay allocated/reserved peaks are measured, with
separate five-sample phase diagnostics and counter30. Dense is measured on the
same actual GPU/runtime. Oracle scratch is excluded from the measurement pool.

An independent inverse is qualified only by>3%complete-step improvement, or
>=5%allocated reduction with<=3%time regression, after every primary worker
passes. No selection from partial or interrupted results, no silent rebudgeting.
Original window16memory savings suggest a hypothesis, not a performance claim
for this different full-scan implementation. All GPU work and submissions are
owned by root; raw logs/snapshots remain ignored. This candidate has no GPU
result at this checkpoint.

Local runtime checkpoint: full CPU suite1380passed,2616skipped,18warnings,
26.43seconds; initial new target1passed/29CUDA-skipped. After adding four Case
metadata declarations and three count-asserted correctness-only overflow cases,
targeted contributor/declaration/new tests21passed/32CUDA-skipped,1.33seconds.
Ruff/AST checks pass; offline `uv build --offline` built wheel and sdist. A
no-isolation build first failed because the main test venv lacks hatchling;
that setup log is preserved and the isolated offline build passed without
system package changes. GPU-only skips do not count as GPU validation.

Independent read-only reviewer confirmed exact baseline `patches`/`contract`
ASTs, the sole block ID-widening difference, imported unchanged pack, all ID
address consumers, fixed recipe, source snapshots and forced overflow branch
witnesses. Fifteen original Torus source/test files match the earlier validated
core snapshot byte-for-byte, preserving81onchip+45sparse-W tests without a
redundant rerun. The rejected window's42tests remain on its preserved branch
and are not part of this new source or reused route validation. The candidate
33tests and common30tests form63tests for the fresh correctness stage, followed
by all12 fixed full-shape verify workers. Evidence and source proof are under
ignored `output/torus-compact-ids/`.
