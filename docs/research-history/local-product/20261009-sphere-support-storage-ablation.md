# Sphere support storage ablation: compact full scan versus cell CSR

The matched primary comparison attributes the allocator reduction to lossless
int16 site-ID storage: the existing full-scan G8 route reaches exactly the same
capture/replay peaks as the CSR candidate. It does not establish a reliable two-run speed
win from CSR. The independent inverse and final disposition are recorded below.

## Mathematical and execution contract

Two independent intrinsic S² Explicit charts, Polar parameters `[A,6]`, separable
Triweight, complete-site per-chart discrete L2 floors `1e-6`, IEEE FP32, live
centres/widths/scalars/sites/radii and all six atom gradients remain unchanged.
This is the existing Sphere operator, not a newly defined single-chart Sphere
profile product or Sphere(4). Explicit/Product/Strip describe chart layouts;
Sphere/Torus describe geometry, Polar describes parameterization, and
separable/profile_product describes kernel composition. No API change is made.

`research_cuda_sphere_polar_grouped_compact/v1` accepts only four strict integer
fields CAP64/T16/G8/index16, including rejection of booleans. It calls the
existing `grouped_weight_linear`, preparer and kernels without copying or
changing their arithmetic. All sites remain in normalization and all overflow
support traverses the complete original domain. int16 changes storage only;
IDs are widened to int32 before address arithmetic. The algorithm retains the
existing measured Sphere domain up to2048 sites, with an additional32768-site
lossless ID guard; the isolated ID32767 test is not permission to expand that
metadata domain. `workspace_bound` remains unknown (`None`). Forward snapshots
and live CUDA Graph inputs retain their original ownership.

The old grouped recipe still restricts its int16 option to G4/T16. A distinct
fixed typed recipe adds the validated G8/int16 combination without changing that
published research recipe. Registration is benchmark-local; public dispatch is
unchanged. Broad sigma8 retains bounded H4096.

## Preregistered matched protocol

N1024/N2048 square, B32,5% atoms, seed41, initial sigma3, FP32 IEEE, MSE, fused
AdamW lr1e-4/decay0.01. Accepted G8/T16/CAP64/int32 baseline, full-scan compact,
cell CSR and dense each start independently from the same task/model/geometry
hashes. The fixed primary contains eight workers. Initial and24-update FP64
oracles evaluate every site and every atom for Y/dX/all-dP with both maxabs and
relativeL2 <=4e-4. Public20-step parameter/moment/clock gates, retained snapshots,
live geometry/scalars and support overflow are also tested.

Actual Graph clock is two eager warmups, one initial replay and21 timed replays:
24 updates. Separate fresh phase diagnostics run from24 to30. Their medians are
not added/subtracted from complete-step timings. Peak allocated/reserved bytes
include capture/replay, not initial/updated oracle or later phase scratch. GPU
process usage and physical cache residency are unmeasured.

Qualification is strictly >3% complete-step speed gain, or >=5% allocated
reduction with <=3% time regression. All eight primary workers must pass before
selection. Every qualified condition is rechecked once in reversed case/pair
order on an independent physical L4. The predeclared integration policy prefers
full-scan compact where both runs qualify; CSR must beat qualified full-scan
compact by the same speed/memory gate in both runs to justify its extra code.
No tolerance, condition, budget or recipe is retuned from results.

The gate outer/driver/pytest caps are900/875/500seconds; performance outer/driver
caps1500/1475seconds and each child400seconds. The repaired read-only assessment
utility enforces those bounds, complete gate63/zero skips/eight verify-only
workers, source archive/member hashes, exact args, full oracles, finite losses
and phase samples, schema1, fixtures and independent physical UUID. Initial
utility acceptance gaps were caught by peer negative probes and repaired before
qualification. Original utility and negative evidence remain preserved; runtime
and measurement source did not change.

## Gate and primary

Source `84188009d9a530562f1cd289e07f1e0baf0a9231`, fixed driver
`ea2f3aa87ae6e84e673e51d073d9050ff6a35afde7768be46a5ba86feb192584`.
Gate `l4job-9517953e13bf42d98c1d23dfe2d10714`:63 passed/zero skips,
25 new actual CUDA tests plus2 runtime CPU/6 benchmark CPU/common30, and all
eight full-shape verify-only workers passed. Driver231.719seconds. Maximum
initial/updated errors Y6.86e-6,dX9.11e-6,all-dP5.50e-5,relativeL2<=1.21e-6.
All1582 frozen files were hashed;1581 match Git84188009 and the extra file is the
fixed pool driver. All36 copied gate files were independently verified.

Primary `l4job-8a0da5eb19c1466d96d6f85b170bdb63`:all eight workers passed,
common30/zero skips, driver183.007seconds. Each timing uses21 positive finite
synchronized samples. Actual physical GPU is
`GPU-1bb31ecf-0b39-b799-a078-4d2f046c041a`, NVIDIA L4, Torch2.11.0+cu130,
CUDA13.0,Triton3.6.0,driver580.82.07. Full source-file map matches the gate.

| N | route | median ms | allocated MiB | reserved MiB |
|---:|---|---:|---:|---:|
| 1024 | baseline | 4.330065 | 105.662 | 166.000 |
| 1024 | grouped-compact | 4.293789 | 92.862 | 158.000 |
| 1024 | cell-support | 5.246658 | 92.862 | 158.000 |
| 1024 | dense | 0.086600 | 32.877 | 86.000 |
| 2048 | baseline | 23.393492 | 323.585 | 390.000 |
| 2048 | grouped-compact | 23.475462 | 271.585 | 338.000 |
| 2048 | cell-support | 23.418053 | 271.585 | 338.000 |
| 2048 | dense | 0.592572 | 81.502 | 106.000 |

Full-scan compact reduces allocated peak12.114%/16.070% at1024/2048. Its primary
time changes are0.838% faster and0.350% slower: a memory win, not a demonstrated
speed win. CSR has exactly the same peaks. At1024 CSR is21.168% slower than
baseline and22.192% slower than compact, so it fails qualification and receives
no inverse. At2048 CSR is0.105% slower than baseline and0.245% faster than
compact. It qualifies only by memory versus baseline; it fails the directed
CSR-versus-compact gate. All baseline-qualified records, including2048 CSR, are
kept in the seven-worker inverse; this is not selective omission.

## Independent inverse and disposition

Inverse `l4job-2e70ee1c8b9e408db3e1103b3643b4a1`.

All seven inverse workers passed with common30/zero skips in171.280driver
seconds on independent physical L4 `GPU-7caf1c9d-bef3-6645-1f08-054a3b2716c1`.
Its source dictionary and actual initial fixtures/runtime match primary exactly;
all33 copied files were verified. Original selected cases/pairs were reversed
once without retuning. Initial/24-update full FP64 oracles remain within gates.

| N | inverse baseline ms | compact ms | compact time change | allocated reduction |
|---:|---:|---:|---:|---:|
| 1024 | 4.238935 | 4.283246 | +1.045% | 12.114% |
| 2048 | 22.810373 | 22.758674 | -0.227% | 16.070% |

Compact qualifies by memory in both runs at both measured shapes. Time changes
across runs range from0.838% faster to1.045% slower at1024, and0.227% faster to
0.350% slower at2048. No >3% complete-step speed win is claimed.

2048 CSR inverse is21.856354ms versus compact22.758674ms,3.965% faster with
identical allocated/reserved peaks. Primary directed gain was only0.245%, so the
predeclared two-run additional CSR gate fails. This inverse improvement is
preserved, not omitted or retuned into a stable claim. 1024 CSR remains rejected.
Root disposition: integrate the compact full-scan research algorithm for
memory-conscious sigma3 comparisons; preserve CSR on its research branch and
exclude it from main. Further reductions in dominant Sphere work remain needed.


## Earlier CSR exploration and limits

The preceding exact B8 CSR study built512 cell histograms,513 prefix offsets,
original int32 site IDs and fresh actual-coordinate descriptors on every
forward. Its bounded query allowed16 xy rows/256 candidates and returned to the
original full scan for unsafe/invalid values, query/support overflow and
near-floor uncertainty. It never dropped a site or approximated distance.
Its four runtime files retain numeric checkpointda13153; measured source is
270b6c02. CSR order can change reduction/contraction access order, so identical
G8 source alone does not establish identical locality or kernel cost.

Initial gates be19ca7/a7a2e1 ended with80/82 tests passing respectively
(82/83 total,2/1 failures): a
stale-spec live-scalar oracle, an accidental singular-origin zeroamp fixture,
and PTX/TTIR parsing assumptions. The oracle was corrected to independent live
detached FP64 scalar values; the regular zeroamp witness usesp0=0,p1!=0. The
singular Polar origin's dtype-tiny contract was not redefined or claimed as a
new FP64 validated result. Actual saved IR was replayed on CPU before the final
parser-only repair. All failed artifacts and pre-submit repairs are retained.
The repaired83-test gate f7c7c32 passed all tests and six verify-only workers;
compiled fast-pack diagnostics used72/95registers,0spills and1024shared bytes at
1024/2048. These are compilation diagnostics, not complete-step measurements.

Earlier primary7d1d642 measured1024 baseline4.207257/CSR5.030339ms and2048
22.709985/22.025477ms. Independent2048 inverseb51746a measured22.438035/
21.822120ms,2.745% speed gain and16.070% allocated reduction. That preliminary
memory qualification motivated this matched compact control. The new ablation
shows why the memory benefit must not be attributed to CSR. Earlier3.014% speed
in one primary was not robustly reproduced. Exact search remains a research
prototype; broader speed claims and causal claims about individual kernels are
unsupported. The accepted baseline profile is preserved in the companion note.

## Integration and recovery

Compact-only integration applies nine files to main688: exact measured compact
metadata and numeric tests, additive benchmark registration, compact-only
catalog/two cases/six CPU checks, exact live-scalar oracle repair and its
independent CPU witness. All11 shared runtime files remain byte-identical to
both main688 and measured841. Compact runtime/test/oracle bytes also match841.
No cell runtime, test, registration or catalog is included. Final local CPU:
1389 passed/2644 GPU-DB skips/18 warnings in27.66seconds. Ruff, wheel/sdist,
two-plan/two-case declaration checks and actual prepare/check for both cases
passed. GPU source was not rerun solely for declaration/prose differences:
all numerical implementation and fixture bytes match the tested/measured source.
Owned GPU slots were confirmed stopped after the selected batch. The measured full research
source remains on `kernel/sphere-support-ablation` at84188009 and cell source on
`kernel/sphere-cell-support` at270b6c02. Raw jobs, failed proofs and verification
manifests remain in owner pool directories and ignored root
`output/sphere-{kernel-profile,cell-support,support-ablation}-20261009/`.
Recovery bundle/evidence archive and restore/fsck proof are saved under
`output/sphere-support-recovery-20261009/`; the ignored manifest verifies restored
branch heads and every evidence byte before remote branch cleanup.

For the integrated compact declaration use
`plans-sphere-grouped-compact.json` and its1024/2048-sigma3 cases with
`python -m tools.kernel_dev prepare`/`check`. The complete matched study,
including CSR controls, is reproduced from84188009 with
`benchmarks.cuda.linear.scaling_comparison --geometry sphere --size <N>
--sigma 3 --mode research_graph --linear-plan <plan JSON> --phases --output
<new file>`. Execute every declared primary worker before the reversed qualified
cohort; a single worker or phase diagnostic is not an equivalent comparison.

See the companion results JSON for full job IDs and source/result/cohort hashes.
