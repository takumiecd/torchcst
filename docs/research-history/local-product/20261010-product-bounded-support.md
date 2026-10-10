# Regular Product: bounded support metadata

Research candidate only. No public dispatcher adoption or performance result
is declared by this initial checkpoint.

Keep the existing single Euclidean Product, two positive equally spaced Line
axes, normalized Triweight profile product and Polar[A,4]. The whole-product
denominator is `max(norm(u)*norm(v), floor)`. Live activity widths, all positive
supports, full norm derivatives, canonical gradients and the public Polar
optimizer contract remain unchanged. Existing matrix controls retain their code.

The candidate uses the same conservative coordinate-to-index span, grouped norm
preparation, grouped W assembly, IEEE GEMM and atom VJP kernels. Only executor
storage changes: prepare at most C atoms, assemble their W contributions, then
reuse the support scratch. Backward reconstructs the same metadata from the
forward's immutable parameters, all nine decoder scalars, grid origin/spacing,
normalization floor and recipe. Scratch is not an autograd saved tensor.

Metadata scratch is at most `13*4*min(K,C)` bytes per preparation, rather than
retaining `13*4*K` bytes until backward. C262144 reserves at most13 MiB; at
K3355443 the earlier support table alone is about166.4 MiB. These are tensor
budgets, not measured CUDA peaks. W/dW remain full-size; removing them is a
separate execution candidate. Recomputing norms may make this route slower.

The first frozen comparison uses B32, seed41, 5% atoms, initial rho3, live bounds
1..16 and the existing capturable AdamW/production Polar update. Compare native
G8P8 and Torch G8P8 with same-engine bounded counterparts (C262144), plus dense,
at N2048 and then8192 after the small/2048 correctness gates. Include complete
Graph step and capture/replay allocated/reserved peaks. Phase diagnostics are
separate instrumented graphs and do not replace the primary step measurements.

Predeclare no tolerance changes or retries of an unchanged failed source. Run
the independent full-site FP64 oracle for every atom and Y/dX/all four atom
gradients. Supplemental validation after24 captured learning updates requires
both max-absolute and relative-L2<=4e-4. Boundary tests cover chunk tails,
empty/singleton/tiny/floor/wide/precision supports, strides, scalar/lattice
mutation after old forward, zero atoms, required gradients and Graph moments.

Assess each engine against its own control. A memory candidate qualifies at
>=5% allocated-peak reduction with <=3% complete-step time regression; a speed
candidate requires >=3% step improvement with no allocated-peak increase. Small
improvements need independent confirmation. An initial losing measurement is
retained, not replaced by a more favorable run. The 2048 study can be negative
while the independently declared8192 memory objective remains meaningful.

Raw drivers, logs, source and pool receipts are retained under ignored output;
record measured source/result hashes and disposition here after retrieval.

Initial gate job `l4job-bdfab467e1ab460d890423ec572dbbcf` at source2dc2d1a
stopped after124 passed/1 failed: the retained-forward test changed only input
bandwidth scalars and the existing KernelSpec correctly rejected the second
forward's unequal input/output bounds. Correct the test to update both sides
equally, retaining changes to every decoder scalar. No executor/kernel or
tolerance changes; preserve this failed gate, and run the corrected source with
the same1000s job budget before any performance claim.

Corrected gate `l4job-7926548caf5a41539762fafc998b9150` passed125 tests and
the complete2048 comparison. Initial full-site oracle plus both candidates'
all209715-atom oracle after24 captured updates passed. The recompute variants
lost: native1.118382→1.377347ms and Torch0.941928→1.200411ms; allocated peaks
66493952→67336704 and99735040→100577792 bytes. C262144 exceeds this K, so this
case has no chunk-bound storage reduction. The already declared8192 study is
`l4job-dfd73c99fb604a46b4487c8a858b77fb`, same executor source and1600s budget.
Its outcome remains independent of the2048 loss.

The corrected2048 driver accidentally recorded the symbolic commit `HEAD`.
Keep that raw artifact unchanged; certify its complete non-driver source hash
manifest against the explicit measured source b469c48 in a companion proof.
Later submissions use explicit commits. This metadata issue is not a new run.

## Second candidate: lossless compact support cache

Revision v2 keeps five FP32 fields (precision, both norms and both norm-derivative
coefficients), four int16 interval ends and one uint8 singleton flag:29 bytes
per atom instead of52. The8192-site support bound guarantees every interval end,
including empty-support sentinels, fits int16. Centers are read from the saved
canonical Parameter; amplitude uses the exact original guarded Polar formula
and saved amplitude_max. No numerical factor is quantized or approximated.

The forward prepares a chunk using the unchanged norm kernel, assembles W and
encodes the immutable compact cache. Backward expands a chunk, then calls the
unchanged VJP. It does not repeat full norm/width preparation. With C262144,
logical cache+scratch is about105.8 MiB at K3355443, versus166.4 MiB for the
original table. Actual capture/replay peak and conversion/launch cost remain
unmeasured until the new comparison. W/dW still remain full-size.

Run the same boundary/Graph/snapshot suite and an additional bitwise layout
roundtrip, then the same-engine native/Torch controls plus dense at2048 and8192.
Use the same acceptance thresholds and full initial/24-update FP64 oracles,
1000s for the new2048 job and1600s for the new8192 job. This is a new execution
candidate after the observed2048 recomputation loss; do not replace that loss.
