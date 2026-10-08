# Spatial support ordering for profile-product matrices

The completed split-K study identifies Product8192 assembly and canonical atom
VJP as large warm-event costs: about17.2 and9.7ms. Product2048 costs are
.129/.282ms for the same G8/P8 kernels. The8192 split1/split8 learning-Graph
diagnostic medians are43.21/43.42ms, so split scheduling alone has no observed
gain there. These are diagnostic evolving-state measurements, not isolated
primary time or peak-memory comparisons. Cache locality is a hypothesis, not a
counter-proven explanation. Evidence is in the completed split-K note/summary.

## Candidate and invariant

Keep canonical Parameter/optimizer storage unchanged. After exact preparation,
sort physical atom views lexicographically by output/input support tiles and
canonical ID. Tile16/32 are explicit alternatives. Unique signed-int64 keys
encode the owner above bit32 and original ID below it; casts happen before
multiplication. Copy all13 packed fields without arithmetic. Save the original
ID map with each forward and store every parameter cotangent at its canonical
ID. Each live forward/Graph replay rebuilds ordering from current preparation.

Whole-chart normalization/floors, every support site and atom, current width,
source/pitch/amplitude snapshots, input/dW matrix contractions, Polar pullback
and optimizer remain the same. No pruning, sampled norm, fixed width, reduced
precision, birth/death or Parameter remapping is introduced. Ordering changes
FP32 atomic accumulation order and must pass the existing FP64 tolerances.
Sort, temporary packed copy, canonical gradient scatter and allocator/Graph
costs belong in the complete-step measurements; no memory gain is assumed.

Strict small matrix revision v5 and large revision v3 use SpatialMatrixProduct/
StripRecipe. Earlier schemas and public dispatch remain unchanged. The shared
grouped VJP's optional ORDERED=False path keeps its old canonical addressing;
regression gates must cover previous revisions. This is a research candidate,
not an adopted optimization or literal Torus definition.

## Predeclared validation and comparison

CPU1234 passed/2231 skipped/18 warnings; the new suite has36 metadata passes
and162 CUDA tests pending. Four catalogs/twelve1024/2048/8192 rho3/8 cases and
two contributor prepare/check workflows pass, as do Ruff, wheel/sdist and
installed-wheel declaration imports with Triton/benchmarks blocked. An initial
declaration invocation used the primary checkout's editable installation;
its ModuleNotFoundError is retained. Setting PYTHONPATH=src:. validates the
actual research source. No mathematical/numerical gate was changed.

GPU gates: new spatial suite198, split223, matrix596, large48, with a1200s
outer budget. New coverage includes exact13-field payload and canonical-ID
bijection, integer keys above int32, atom tails/empty, full/support preparation,
full-site independent FP64 Y/dX/all parameter VJPs, normalization floors,
wide/singleton/empty support, strided inputs, partial Strip tiles/gaps, retained
old-forward snapshots, live pitch and20 captured AdamW/Polar updates including
moments/clocks. Product uses split1, Strip split8 in these operator scenarios.

Every catalog retains prepared, native split1/split8, spatial16/spatial32 and
Torch controls plus dense. Spatial Product keeps split1, Strip split8; the
respective baseline is the identical unsorted reduction schedule. Cases keep
B32,5% atoms,seed41, IEEE FP32/no TF32, initial rho3/8 with live widths1..16,
production fused capturable AdamW/Polar and21 isolated-route Graph samples.
Initial selected measurements after a successful gate: Product/Strip1024 and
2048 rho3/8 (600s child/1300s outer per two-rho family), and Product/Strip8192
rho3 only (Product1800s child/2000s outer; Strip600s/700s). Product8192 retains
all3,355,443 atoms and full default-chunk FP64 oracles, without sampling or
raising limits after a failed result. All six CST routes/dense remain present.

Measure actual capture/replay allocated/reserved peaks and retain unsuccessful
or slower alternatives. GPU process usage is unmeasured unless separately
instrumented. Independently reverse route order for any proposed adoption;
small gains require confirmation. Public dispatch/Torus mathematics are separate
decisions. At this source checkpoint CUDA validation and all performance are
pending; no ordering speedup or memory claim is made.

Raw drivers/logs/source snapshots/receipts live in ignored
benchmarks/cuda/linear/evidence/profile-product-spatial-order-20261008/.

## Strengthen moving-order verification

Add four targeted CUDA cases (Product/Strip, tile16/32) that capture a complete
Y/dX/all-atom VJP probe, reverse the four atoms' physical centres across support
owners, replay and compare with a new full-site FP64 oracle. Assert the live
order actually changes and remains a bijection. Also retain an earlier ordinary
forward across that replay and compare its Y/dX/dp with the original oracle.
Canonical Parameter/amplitude/width rows are never reordered. This covers a
changed permutation, beyond small centre motions in the20 optimizer updates.

Runtime, catalogs and cases remain byte-identical to initial checkpoint
c2a53adb528d30b72b5f5def6e6a90bec856df9d; only tests/notes change. The new
full CPU run gives1234 passed/2235 skipped/18 warnings. The original GPU gate
remains tied to its immutable198-case suite; a separate4-case motion gate must
pass on this new source before measurements. No gate or tolerance is removed.

## Interrupted full gate: no numerical outcome

Job l4job-967a02798112455fb964a51f542ecf93 atc2a53adb528d30b72b5f5def6e6a90bec856df9d
was interrupted when the Colab CLI exec command was not acknowledged within
its1440s transport budget (1200s driver request,1380s CLI execution request).
There is no downloaded result or receipt, so neither a CUDA pass/fail nor a
driver timeout can be inferred. All1,168 source/benchmark/test/kernel-dev file
sets and bytes match its Git archive, and the source/transport/spec records
are preserved in ignored jobs/ and interrupted-gate-proof.json. The supervisor
exited1 and the pool confirmed the owned endpoint stopped; all slots are stopped.
Documented recover completed successfully and marked the interrupted job
failed after verifying owned-session shutdown. A new single-L4 supervisor
now dispatches the separately queued motion gate. The original full gate is not retried or re-budgeted on this evidence.

The separate4-case centre-motion job l4job-f4ab40baa4384cdaa2346bd1f732bb55 at
215051da14dd9d2448b6f94b2c99f8b920ff2c6b has a300s driver budget. It can
diagnose small moving-order execution, but cannot replace the interrupted full
gate. No performance jobs are submitted until the complete validation scope
is established. PR#70 remains draft. Its head215051da passed exact-head CPU
Actions37703941653, including prepare/check, declarations, CPU and wheel/sdist.

## Motion fixture stream correction

The separate motion job returned a verified result (exit1, timed_out=false):
four failures,198 deselected. The first case fails during captured autograd:
a retained default-stream AccumulateGrad node forces an illegal dependency on
the capturing stream; later cases inherit its invalidated capture. This is a
fixture execution failure, not a measured spatial speedup or numerical pass.
Keep its original source, logs and receipt under ignored jobs/.

Create retained forwards and capture on the same explicit side stream. Also
unwrap the existing AtomState _ReadParameter lifetime wrapper before reading
the matrix's saved permutation; the wrapper's saved tensor is a CPU marker,
not the CUDA order. Assert matrix node, order shape and int32 dtype. All four
FP64, changed-order, bijection and retained-gradient checks stay unchanged.
Runtime/catalog/case bytes remain unchanged. CPU metadata36 passes/166 CUDA
skips. The corrected motion fixture is a new source/job with the same300s
budget; the original failure remains part of the disposition.

The corrected job l4job-34c65ba899ef45c18d14470263bb288e at
dd0fba892bb7ccf722d34ddf4ab35805d17d5af5 passed all4 targeted CUDA cases
(198 deselected,1 warning,15.47s pytest; driver exit0/no timeout). Source
archive SHA256 ed1719f61538fa46a24c1bcd6e1271e9050841e418159081e1168dcc9a702f44
and downloaded archive/receipt are verified and preserved in motion-proof.json.
Hardware/runtime: NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0.

With the concrete fixture correction independently passing, submit the entire
updated scope as a new job l4job-6a5ade859fe64a0ba410cfd7f21ad24f on the same
dd0fba89 source: spatial202 + split223 + matrix596 + large48. Retain the
original1200s driver budget and every original test/tolerance, plus the four
corrected cases. Preserve the earlier transport interruption and numerical
fixture failure; neither is discarded or reported as a pass. This is not a
performance rerun or budget increase. Complete GPU regression and primary
matched times/peaks remain pending.

## Completed full CUDA gate

Job l4job-6a5ade859fe64a0ba410cfd7f21ad24f passed all1,069 tests:
spatial202 (68.50s), split223 (29.91s), matrix596 (110.54s), large48 (26.50s).
Driver exit0/no timeout,248.19s total within the unchanged1200s budget.
Source archive SHA256 c8db0a302ee2399831253d9d0a7b40262381f6be63cd52fc70b1221c4cdc4f0e;
result archive SHA256141265e7d96b550b5309f9a5e054a7c3911ea2c5b8de2e6606e2973441df9bf2.
Archive/source/result hashes are verified and raw source/logs/receipt are
preserved in ignored jobs/ and full-gate-proof.json. Runtime/catalog/test bytes
match current head9b1d328e; its exact-head CPU Actions37705631786 also passed.

Proceed to the predeclared six primary comparison jobs using the same runtime:
Product/Euclidean Strip1024/2048 rho3+8 and8192 rho3. Keep all six CST routes
and dense, every oracle/optimizer gate and complete-step allocated/reserved
peaks. Literal Torus remains a separate distance/composition decision.
