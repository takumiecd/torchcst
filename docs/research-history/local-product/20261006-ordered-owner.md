# Small Linear: physical support order and owner splitting

Research branch `kernel/ordered-owner-split`, based on PR33's
`f2b612085cca4388b04c3bbd07bc340168e8ff7b`. PR33 remains a separate dependency;
this experiment does not merge it or change public dispatch.

## Scope and contract

The previous exact owner list reduces exploration but retains indirect loads
of physically distant atom values. Seven new explicit research routes compare:

| Alias | Placement and consumption |
| --- | --- |
| `local-ordered-band-atom32` | Compact full rebuild, rho bands and canonical ties; exact physical-ID lists |
| `local-ordered-index-atom32` | Same rebuild/lists, additionally order each band by relevant support start |
| `local-ordered-atom32` | Position-ordered values; consume one contiguous candidate range per owner/band |
| `local-ordered-cached-atom32` | Range route with wide forward H in matching physical atom order |
| `local-ordered-split2-atom32` | Local-H ranges divided into two atom-chunk groups, followed by output/dX reduction |
| `local-ordered-split4-atom32` | Same, four atom-chunk groups |
| `local-ordered-split4-i32-atom32` | Four groups, with int32 sort keys when their full bound fits |

Two controls from PR33 are retained: compact-ID local-H (the low-memory baseline)
and compact-ID cached-H with forward list release. Position-index versus
band-index isolates the ordering change within the compact rebuild design.
Position-ranges separately removes indirect IDs and list storage/construction;
its speedup cannot be attributed solely to caches. Compact rebuild also removes
persistent reserve/topology storage, which is distinct from physical ordering.

Each direction stores one compact 13-field value copy per atom. Forward orders
by output support start; dX by input support start. General atoms are grouped
by current rho band, singleton atoms by their site's 16-position owner. A
canonical-ID tie break makes ordering deterministic. Canonical Parameters and
Adam moments keep their identity. Per-call views/order/ranges protect outstanding
backwards from later center/width changes.

GPU preparation rebuilds all views from current parameters on every forward.
A compact-key candidate retains the same bucket/start/canonical ordering. It uses
int32 only when `stride * (max(K,N)+1) * atoms < 2**31-1`; larger cases retain
int64, including the padded sentinel.

Integer intervals select contributions; normalized Triweight coefficients and
all center/amplitude derivatives are evaluated with the existing arithmetic.
Full-domain normalization, floor, empty and exact singleton handling remain.
Variable-width interval ends need not be sorted: ranges bound the first/last
actual intersection, and the consumer retains exact overlap masks. No derivative
through a sort key replaces the existing parameter VJP.

The cached-H producer uses the reordered forward values. Consumers index H with
the same physical atom position; parameter VJP recomputes H and scatters gradients
through the per-call canonical map. Forward H is not retained for backward.

The existing 16-site output owner and 16 **batch rows** per block remain separate
axes. N128/B32 gives16 forward blocks and16 dX blocks. Split2/4 add an atom-group
axis for 32/64 blocks per consumer launch. Each group accumulates disjoint
32-atom chunks, including singleton buckets; a separate reduction combines the
partial outputs. A four-way N128/B32 FP32 partial buffer takes 64KiB per direction
while live. This is a tensor budget, not a measured complete-step peak.

## Protocol and results

The initial screen at `f3f5f6809729ecac96faaea0b5a9933b526d2b6d` passed58 GPU-host tests
(two declarations,56 CUDA tests), plus32 full-size independent scalar FP64
Y/dX/dP comparisons. Captured20-step tests include moving widths, Parameters,
Adam moments and step counts; slice/repeated/old-after-new backwards are covered.

Initial L4 complete-step medians in microseconds (one execution per cell):

| Size/rho | Baseline local-H | Band/index | Position/index | Position/range | Cached | Split2 | Split4 | Dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|64/3|65.85|67.05|66.33|67.01|67.74|64.24|59.83|38.88|
|64/8|65.75|66.83|67.18|66.99|64.84|61.03|57.42|38.96|
|128/3|82.24|93.79|91.53|94.73|95.72|85.33|84.29|45.44|
|128/8|97.79|108.49|108.46|110.67|102.03|97.07|87.46|45.53|

Peak allocated bytes, including capture/replay (same for both screened widths):

| Size | Baseline | Ordered IDs | Range | Cached range | Split2 | Split4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|64|130560|87040|84992|109056|99328|115712|
|128|341504|264192|237568|342528|270336|303104|

Position ordering/ranges reduce memory but do not independently establish a time
win. N128 layout diagnostics rise from about12.29us (baseline) to27.65us
(range), while four-way output/dX diagnostics fall from19.46/19.46us to
14.34/14.34us at rho3 and25.60/26.62us to13.31/14.34us at rho8.
Component event quantization and graph interference mean these values are
bottleneck diagnostics, not an additive reconstruction of primary timings.
The added int32-key candidate targets this measured preparation overhead.

### Repeated L4 sweep

Measured source `a482556a02ffa03ceb4e15fbbe5e2cba30b38a69`, two independent
executions per case, second execution reversing case and plan order. Each cell
is the median of the two21-sample execution medians, in microseconds.
All64 full-shape scalar-oracle comparisons passed and actual initial rho>1 and
updated widths were checked. Peak allocated remains the screen table's values
for each corresponding route. The compact-key candidate has the same peak as
Split4 (115712/303104 bytes),11.4%/11.2% below the baseline.

| Size/rho | Baseline | Range | Split4/int64 | Split4/int32 | Dense | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|64/1_25|58.74|59.70|56.18|55.67|39.09|5.2%|
|64/3|65.99|67.34|60.35|59.51|39.02|9.8%|
|64/8|65.89|67.25|57.64|57.04|38.98|13.4%|
|64/mixed|64.83|65.69|63.89|63.24|39.22|2.5%|
|128/1_25|74.71|87.42|80.03|75.78|45.85|-1.4%|
|128/3|82.60|95.35|84.37|79.62|45.83|3.6%|
|128/8|99.16|111.22|88.74|83.43|46.09|15.9%|
|128/mixed|90.04|105.13|90.44|85.84|46.11|4.7%|

Seven of eight observed medians improve with Split4/int32; N128/rho1.25 is1.4%
slower. Position/range consistently trades slower time for about30–35% lower
allocated peak. Splitting with int64 remains slower at N128/rho1.25, rho3 and
mixed, so splitting alone is not sufficient there.

At N128/rho3, compact keys lower the layout diagnostic from about27.65us to
22.56us and shared-memory scratch from8192 to4096 bytes; register counts change
96→94 with no spills. Output/dX retain the same four-way ownership and reduction.
The remaining large layout cost motivates reducing rebuild work in a later trial.
No public dispatcher adoption is inferred from these explicit recipes.

### Repeated Blackwell comparison

Actual NVIDIA RTX PRO6000 Blackwell Server Edition,188SM, CC12.0, same measured
source and byte-identical initial inputs/targets as L4. Initial Parameters are
byte-identical across plans and repeats within each device, but their hashes
differ between L4/G4. These are separate paired device comparisons; no
byte-identical cross-device Parameter or direct generation timing ratio is claimed.
Two independently ordered executions per case;24 full-shape scalar-oracle comparisons passed.
Times below are medians of the two execution medians, in microseconds.

| Size/rho | Baseline | Range | Split4/int32 | Dense | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
|64/3|61.60|61.28|54.22|32.22|12.0%|
|64/mixed|60.18|59.64|57.39|32.06|4.6%|
|128/3|78.28|87.26|70.52|42.50|9.9%|
|128/mixed|84.49|95.65|75.75|42.48|10.3%|

Allocated peaks are unchanged from L4 for these routes; reserved peaks and all
individual execution medians are in the compact companion JSON and raw artifacts.
G4 supports the split/key candidate independently of L4. The range route saves
memory and is slightly faster at N64, but remains slower at N128. L4 and G4 both
used PyTorch2.11.0+cu130/CUDA13.0/Triton3.6.0, FP32 IEEE/TF32 off. Two executions
establish observed repeatability, not a broad device or workload guarantee.

### Validation and disposition

- CPU:860 passed,493 CUDA/DB-dependent skipped;18 existing deprecation warnings.
- Wheel/sdist build and all eight `prepare`/`check` comparisons passed.
- GPU-host test stages:58 passed at the first screen;9 compact-key tests passed
  before the preserved driver declaration failure. Runtime hashes match the
  subsequent corrected source. These cover64 distinct CUDA checks and two
  declarations across the staged test sets.
- Independent full-size scalar FP64:120 plan/case comparisons across the screen,
  repeated L4 and repeated G4 stages; Y, dX and all atom gradients checked,
  with nonzero expected position gradients. Every successful measured stage
  asserts actual initial rho>1 and changing widths during fused AdamW/Polar.
- CUDA Graph timing includes output/dX partial reductions and preparation; peak
  allocated/reserved includes capture/replay. H is computed within consumer
  blocks for the selected split/key route; partial outputs remain global tensors.
- Successful raw runner artifacts are retained in the existing PostgreSQL store
  with verified byte-identical exports and idempotent reimports. Raw traces,
  logs and source snapshots remain in ignored evidence and central pool storage.

The preferred explicit time/memory candidate is Split4/int32 for the tested
middle/wide cases. N128/rho1.25 retains a small time regression, and the range
route offers the larger memory reduction at a time cost. Public dispatch remains
an independent decision. Further preparation reduction is the next investigation;
no persistent-sort cache or proven warm L2 residency is claimed here.

### Cold access diagnostics

Four valid Nsight probes each include exactly one forward and one dX launch at
N128/B32, initial rho3. Position-index versus band-index reduces global load
sectors per request from12.796 to10.790 (forward) and12.625 to10.422 (dX).
DRAM read bytes instead change145152→150656 and148864→151168. The range route
reads142976/149376 bytes; baseline reads147712/151424. L2 hit percentages are
valid (85.25–92.23%) but are cold instrumented probes, not warm residency
measurements. Zero measured DRAM writes during these probes does not establish
zero global-store or writeback traffic. These results support improved load
coalescing and do not establish a large DRAM-traffic reduction.

### Preserved non-result jobs

Two jobs were cancelled before execution: `l4job-c142311cbbfc4c2882933ac23a26cc27`
(pre-review cached-H stride) and `l4job-3c795c7645bf404b8a6b67613be0941f`
(wrong driver source label). No measurement is attributed to either.

`l4job-39204ce2169948cdaf235d3e6ea399f9` and
`l4job-92731e4691b8444d8b95c4439c8d4eed` failed declaration loading before
any measurements: filtering plans removed the case's cached-H baseline.
`l4job-b63878db44754a84a90d57db643fa217` passed9 compact-key tests but then
encountered the same declaration error. Cases now explicitly select the retained
local-H control; the fixed driver also asserts its presence. These are driver
configuration failures, not numerical failures or negative timings. The compact-key
test subprocess completed successfully before declaration loading; its196 runtime
files match the subsequent measured commit. A redundant retry
`l4job-4a71f1e912fe407f822dc2bc71ec70d2` was cancelled before execution once the
corrected full sweep passed all32 full-shape comparisons.

## Evidence and reproduction

Use `plans-local-ordered.json` with
`cases/local-ordered-{64,128}-rho{1_25,3,8,mixed}.json`. Initial actual rho>1 is
asserted; minimum/birth remain0.25, with production fused AdamW/Polar and evolving
widths. N64/N128, B32, A204/A819, FP32 IEEE/TF32 off, seed41,21 samples per
execution. CUDA Graph complete-step timing includes rebuilding and partial
reductions; compiler/phase diagnostics use separate graphs.

Prepare/check a comparison with the existing `tools.kernel_dev`; then run
`benchmarks.cuda.linear.run` with `--polar-update fused --phase-diagnostics
--kernel-diagnostics`. Use a fresh output directory and the measured source
checkout when specifying a historical source commit.

Ignored evidence directory: `benchmarks/cuda/linear/evidence/ordered-owner-20261006/`.
[The compact companion JSON](20261006-ordered-owner.json) records successful job
IDs, full source/result hashes, individual medians, peak allocated/reserved, width
updates, oracle maxima, cold metrics and28 verified DB receipts. All owned L4/G4
runtimes were stopped and the primary main checkout remained unchanged.

Central pool job directories preserve exact submitted drivers, source archives,
raw results, samples and SHA256 receipts. Completed runner artifacts use the
existing DB import with byte-identical export and idempotent reimport.

Cold Nsight probes, if available and valid, examine prepared fixed N128/rho3
output/dX kernels using `--cache-control all`. They exclude preparation/updates
and are distinct from primary warm complete-step timing and warm cache residency.
No worktree cleanup or public dispatcher adoption is performed.
