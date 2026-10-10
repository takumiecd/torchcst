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


## Checkpoint and provenance

The compact-cache source is commit
`08b72c84732cfb428bee6b357d41be8b6e726093`; local CPU validation passed1412
with2699 skips, and wheel/sdist build passed. Its2048 GPU gate/comparison is
`l4job-98aa5d354344453494c60c70be55bccf` with the declared1000s budget.
No GPU result or adoption is implied by this checkpoint.

The corrected v1/2048 provenance proof matched all1591 non-driver source-file
SHA256 values to commit `b469c4853ce70ccfd9f74aefa837029b5a360c9d`.
The frozen driver SHA256 is
`09b71fbe44eb8fd90f7ba6fda794e73d865521272eb0c459946c9eb1a3ad1c24`.
The measured source archive SHA256 is
`005bd9b68f4acc1b920918592f749d3a164866f462ee01e5cdd9319a911a94cc`;
retrieved result archive SHA256 is
`09a8c61ec305f80e42f1537d5f713b1e73c4d45c2786b1436778d6b82846b630`.
The companion proof is ignored `output/product-bounded-2048-source-proof.json`.
Raw artifacts remain in the default pool job directory; the symbolic `HEAD`
field is not rewritten.

Runtime for this batch: NVIDIA L4, PyTorch2.11.0+cu130, CUDA13.0,
Triton3.6.0. Reproduce a candidate with the committed catalog/case via
`python -m benchmarks.cuda.linear.run --plans <catalog> --case <case>
--polar-update fused --phase-diagnostics --source-commit <explicit-commit>
--output <ignored-output-path>`. The frozen pool driver in each source archive
also preserves the independent gate and supplemental24-update oracle.


## Arithmetic support and retained mathematical state

For one positive regular axis, site coordinates are \(t_i=o+i\Delta\),
\(\Delta>0\). In exact arithmetic the positive Triweight support is
\[
\left\{i\in[0,N):\frac{c-\sigma-o}{\Delta}<i<
\frac{c+\sigma-o}{\Delta}\right\}.
\]
The runtime does not rely on those exact real-number inequalities at a rounded
boundary: `_candidate_span` expands the interval conservatively, applies the
original FP32 positive-gap predicate, and falls back to a full scan when the
spacing/coordinate/radius precision bound cannot certify the candidate. The
saved endpoints are the first positive site and one past the last positive site,
with `(N,0)` for an empty axis. Both input/output ranges fit signed16-bit integers
for the declared N<=8192; `precision` fallback is also representable.

Let \(v_i=(1-(t_i-c_{in})^2/\sigma^2)_+^3\),
\(u_j=(1-(s_j-c_{out})^2/\sigma^2)_+^3\). The matrix contribution remains
\[
W_{ji}=\sum_a\frac{A_a u_{a,j}v_{a,i}}
{\max(\|u_a\|_2\|v_a\|_2,F)}.
\]
The packed normalization factors and their center-derivative coefficients are
copied at FP32 precision, including the whole-product floor convention. The
stored ranges avoid backward's support search; they do not replace the complete
norm, omit positive sites, or freeze subsequent forwards' live widths.


## v1 large result: reject recomputation

Job `l4job-dfd73c99fb604a46b4487c8a858b77fb` succeeded: all four initial
complete FP64 oracles plus both updated candidate oracles passed. Both updated
candidates changed all3355443 widths in24 updates; worst updated Y maxabs was
2.180e-4, dX8.162e-10 and dP3.087e-11, all below the predeclared4e-4 gates.
Submission validation also passed. Native/Torch controls and candidates use the
same source b469c48 and inputs; this is one independent run with21 timed replays.

| N8192 route | Complete step ms | Allocated peak bytes | Reserved peak bytes |
| --- | ---: | ---: | ---: |
| Native control | 42.318327 | 1041040896 | 2501902336 |
| Native recompute | 47.926771 | 881241600 | 2164260864 |
| Torch control | 39.708207 | 1075119616 | 2522873856 |
| Torch recompute | 45.602354 | 915320320 | 2185232384 |
| Dense | 11.683064 | 1112017408 | 1642070016 |

The native candidate reduces allocated peak15.35% but slows13.25%; Torch reduces
14.86% but slows14.84%. Reject both under the stated <=3% regression requirement.
Phase diagnostics separately increase backward12.09→17.28ms (native) and
11.84→17.17ms (Torch), consistent with norm recomputation cost. The candidate
remains slower than dense; memory reduction does not establish training quality.
The source archive SHA256 is
`d0a651be339a321709a47e94c721587efebdade75a108720f1d96c86a6d87c93`;
retrieved result archive SHA256 is
`c324359ed38c2a330f08101177117e8a490db60a18312a89bc9d561b18ef2d7c`.
Preserve this validated negative route on the research branch; it is not eligible
for public dispatch adoption. The cache candidate is evaluated separately.


## v2/2048 gate and small-size rejection

Job `l4job-98aa5d354344453494c60c70be55bccf` passed156 tests, including
exact-numeric compact-layout roundtrip. All four initial FP64 oracle checks and both
24-update candidate full-atom oracles passed; each candidate changed all209715
widths, and the final Parameter hashes match the v1 updated oracle. This checks
live updates without interpreting21 timed replays as independent runs.

| N2048 route | Complete step ms | Allocated peak bytes | Reserved peak bytes |
| --- | ---: | ---: | ---: |
| Native control | 1.128098 | 66493952 | 163577856 |
| Native cached | 1.189505 | 71738368 | 188743680 |
| Torch control | 0.949566 | 99735040 | 205520896 |
| Torch cached | 1.033315 | 105817088 | 230686720 |
| Dense | 0.590983 | 102238720 | 148897792 |

Reject this size: native is5.44% slower with7.89% more allocated peak; Torch
is8.82% slower with6.10% more peak. With K<C the expanded scratch still equals
the original support table size, and the compact cache adds storage. Keep the
same C262144 for the already declared8192 study, rather than tuning away this
negative case. The large job is `l4job-5621868602464459b95173474dd48dea`,
source91968cf (only research-note changes after the v2 source08b72c8),1600s budget.

The2048 source archive SHA256 is
`0438d7e3240fef951f8452ab3195d1c43454e5d7050fd36da929b44c79c521b7`;
retrieved result archive SHA256 is
`7fede24bbaa6029a04eaf1639707fe011ad7c12720bed8fbc3eaec27ca411b9b`.
GPU process usage was not measured.


## Independent audit and offset correction

The independent read-only source audit found no additional contract/snapshot
violation or reason to invalidate the frozen C262144 comparisons. It identified
`START: constexpr` in layout encode/decode: C256 at K3355443 would compile26216
unique-offset variants (both directions), defeating scalable small-chunk use.
Change START to a runtime scalar in a separate commit; preserve the measured
source91968cf and its result. This changes layout addressing specialization,
not the mathematical preparation or VJP.

The correction also makes the inherited recipe-boundary test read its selected
algorithm revision, so cached v2 itself is checked. Extend the bitwise layout
test to N8192 endpoints, empty/zero-amplitude atoms and distinct nonzero offsets;
validate untouched prefix/suffix sentinels and exact field copies, in addition to
encode/decode equality. Targeted CPU validation is6 passed /56 skipped; the
GPU gate must run on the separately frozen correction before any new claim.


The follow-up audit confirms Triton3.6 integer specialization groups normal
START offsets into one i32/16-divisible class, so runtime START eliminates
O(chunks) value-specific variants. It also distinguishes exact numeric equality
from bit identity: the final layout test compares FP32 int32 views, including
negative zero. The queued gate `l4job-e60a5704381b4e9bbabe0853d85a101b` was
cancelled before execution and replaced with this stronger test/source freeze;
its snapshot is retained and it supplies no GPU measurement.
