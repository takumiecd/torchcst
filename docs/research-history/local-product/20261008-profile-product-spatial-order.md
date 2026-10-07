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
