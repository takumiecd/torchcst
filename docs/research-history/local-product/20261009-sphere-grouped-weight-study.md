# Exact grouped Sphere W and atom VJP study

This research candidate groups four or eight independent atoms in a CUDA CTA.
It preserves the existing two intrinsic-S² charts, Polar coordinates, separable
Triweight profiles, complete-chart L2 normalization with floor1e-6, and all six
parameter cotangents. It does not alter the public dispatcher or optimizer.

## Fixed candidate scope before GPU execution

The frozen recipes are G4/patch16/int32, G8/patch16/int32,
G4/patch32/int32 and G4/patch16/int16. Capacity is64 for initial sigma3 and256
for the broad sigma8 control. Fused W/patch32 is the unchanged control; bounded
H4096 and dense remain comparison references. The existing declared shape
support remains1..2048 sites per chart, batch1..64, IEEE FP32.

The fused preparation arithmetic and complete support packing are unchanged.
Its optional index dtype changes only allocation/storage. int16 IDs represent
site0..32767 without loss; direct preparation refuses larger charts for int16.
Every packed ID is explicitly widened to int32 before pointer arithmetic,
including the shared scalar overflow path. Counts, norms and raw profiles stay
int32/FP32. At A209715/CAP64 the two int16 index buffers save51.2MiB in tensor
storage; complete-step allocated/reserved savings must be measured separately.

Each non-overflow atom keeps its own lane and reductions consume site axes,
never the atom axis. A group with overflow executes the existing scalar `_weight`
and `_vjp` paths for those atoms inside CTA-uniform branches. Other atoms keep
the packed path. Empty support, partial atom groups and all overflow combinations
are masked exactly. There is no second all-atom fallback launch or extra scratch.

Grouping changes CTA scheduling and W atomic accumulation order. It does not
reduce the number of W atomic additions or complete support scans. Numerical
accuracy and actual speed remain unproven before the GPU gates. G8/patch16
versus G4/patch32 can change register use; a smaller launch grid alone is not
performance evidence.

## Predeclared validation and adoption

Independent FP64 Y/dX/all-dP must pass max-absolute and relative-L2<=4e-4;
existing sub-floor boundary tests retain absolute FP64<=4e-4 and FP32 reference
atol4e-5/rtol4e-4. No tolerance is relaxed. Tests cover mixed overflow on either
or both charts, group tails, zero atoms, requested gradient subsets, norm floors,
singleton/empty support, retained snapshots after live mutations, Graph live
widths and20 public optimizer Parameter/moment/step comparisons. A direct GPU
helper case accesses site32767 to check pointer arithmetic, without extending
public large-shape support.

The root orchestrator owns the shared L4 queue. Correctness budget is900 driver
seconds and the primary complete-step budget1500 seconds. Primary B32 square
N1024/2048,5% atoms, initial sigma3/8, seed41, IEEE FP32 uses the same evolving
model/input/target/optimizer protocol across control and candidates. Main gates
also require independent full-A FP64 correctness before and after24 updates.
Capture/replay allocated and reserved peaks are reported separately; physical
cache residency and GPU process usage are unmeasured.

A condition qualifies for an independent inverse run only with >3% complete-step
speed gain, or >=5% allocated reduction and <=3% time regression. All four fixed
recipes and broad controls must be retained in the primary evidence. Failed
conditions cannot be omitted or replaced with a post-hoc recipe. No runtime
adoption or measured speed claim is included in this initial checkpoint.
