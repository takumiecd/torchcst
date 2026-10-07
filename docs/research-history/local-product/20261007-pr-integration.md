# Finish compact-view and batch-loop PR integration

The user requested a merge-or-close disposition for PR55 and PR46 instead of
keeping unresolved drafts. PR55 now incorporates current main `138aa083`,
PR46's complete history and both explicit low-memory alternatives. Neither
original source nor raw evidence was removed. Integration uses PR55; PR46 is
finished once its changes are present in main through that integration.

## Mathematical and execution contract

Retain normalized shared-width local PolarAmpWidth Triweight, dX, all source
and position gradients, exact singleton and normalization-floor behavior,
immutable per-forward snapshots and evolving production AdamW/Polar updates.
Full forward/parameter metadata remains13 fields; the explicit compact route
uses11 fields only for value-only dX. Batch-loop recipes explicitly use16 batch
rows and one parameter CTA partition; owner2 is a memory/time tradeoff.

Existing fusedback16 requires the full dX metadata to compute source VJP.
Compact dX and fused VJP are separate recipes, with declaration rejection and
a compiler assertion for their combination. Compact dX and batch-loop recipes
are also kept separate in this integration. Existing recipes retain defaults.
No global H/G buffer, public default dispatch change, new geometry or outer
GEMM schedule is introduced. Physical cache/DRAM traffic is unmeasured.

## Verification plan and checkpoints

Integration runtime source: `68b17947c91d2848c02e00bb0a9f709c5f1b48a6`.
CPU: **896 passed /1124 skipped**, Ruff and wheel/sdist build PASS.
Six prepare/check pairs compare seven recipes at N64 rho1.25/3/8/mixed and
N128 rho3/8. All performance fixtures have initial decoded rho greater than one.

One shared-pool L4 job `l4job-461b1de9762f4b8f96f6350b1b523567` runs the compact,
batch-loop, fused-backward and counter-free-cache regression suites first.
It then checks every performance Plan against the independent full-shape FP64
Y/dX/all-source-gradient oracle, including nonzero centers, before using the
existing complete-step runner. Current controls, compact uncached/repair4,
fused repair4 and batch-loop owner2 share input/target/Parameter bytes.
Timing includes production updates; allocated/reserved peaks include Graph
capture/replay. One independent execution/condition,21 samples/execution.
No G4 allocation is made for this integration.

## Verified integration result

Combined L4 suite **143 tests passed**; all42 independent full-shape FP64 Y/dX/source
comparisons passed with nonzero position cotangents. Maximum absolute
Y/dX/source errors: 2.102e-06 / 1.616e-06 / 2.964e-06. All six complete-step
artifacts passed consistency checks. Initial decoded rho>1, evolving widths,
and matched input/target/Parameter bytes were verified. All committed,
submitted and worker runtime files match the frozen source; source/result
archive hashes are preserved in the integration result record.

UC is uncached parameter-atom16/split2; R4 is counter-free repair4. Compact
variants reduce only value-only dX metadata. Fused16 is R4 atom16 backward
fusion already merged in PR57. Loop2 is PR46 batch16/owner2.

|N/rho|Copy8 us|UC us|R4 us|UC compact us|R4 compact us|Fused16 us|Loop2 us|dense us|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|64/1.25|54.87|49.69|49.93|49.54|49.48|46.39|52.37|39.02|
|64/3|58.98|51.08|52.30|50.73|51.86|52.17|57.03|38.85|
|64/8|56.48|50.69|51.47|50.60|51.28|51.18|55.40|38.87|
|64/mixed|61.95|55.84|56.41|55.70|56.77|54.63|57.49|38.74|
|128/3|69.66|67.28|68.58|67.99|68.89|70.11|70.34|45.62|
|128/8|72.95|69.97|65.67|70.28|66.05|73.01|79.68|45.24|

Complete-step Graph medians include production AdamW/Polar. One independent
execution per condition,21 samples; subpercent changes are small and are not
claimed as robust speed improvements. The original PR55 paired two-execution
record and PR46 screening retain their original source identities.

|Recipe|N64 allocated peak B|N128 allocated peak B|
|---|---:|---:|
|Copy8 / UC|115,712|303,104|
|R4|116,736|306,688|
|UC compact|114,176|296,448|
|R4 compact|115,200|300,032|
|Fused16|121,856|307,712|
|Loop2|99,328|270,336|
|dense|34,179,584|34,408,960|

Capture/replay allocated peaks above are identical across measured rho cases
for each shape/recipe. CST reserved peak is6,291,456 B; dense reserved is
48,234,496 B. Dense includes its workspace and has a different parameter
initialization/update trajectory; input/target/precision and time boundaries
match. Physical L2/DRAM traffic was not measured.

Compact UC saves1,536 B at N64 and6,656 B at N128 versus matched UC. N64 UC
compact medians are slightly better across the four conditions. N128 compact
medians are slower than matched controls; retain existing recipes for speed
priority. Fused16 remains useful at N64 rho1.25/mixed but does not win at N128.
Loop2 has the lowest allocated peak in all six conditions, at higher latency;
it saves16,384 B versus UC at N64 and32,768 B at N128.

## Database and dispatcher

All six raw artifacts export byte-identically from the benchmark DB and
reimport idempotently. The18 requests under
`benchmarks/dispatch/requests/local-pr-integration-l4-20261007/` use exact
source/case/environment/protocol filters and one independent run per condition.
Speed selects the fastest complete step; control-peak selects speed under the
fastest original control peak; memory selects minimum allocated peak.
All generated artifacts/leaderboards replay identically from frozen datasets.
The raw run IDs, hashes, provenance, all recipes and selected evidence IDs are
preserved in [the result record](20261007-pr-integration-results.json).
Dispatcher/adapter tests: **113 passed**.

These are caller-selected case-scoped exact tables. Current/initial rho is not
part of the runtime key, so do not combine different case policies as automatic
live-rho selection. No public default change or extra measured timing of the
selector wrapper is claimed.

Raw source/results, rejected earlier alternatives and checkpoints remain in
`benchmarks/cuda/linear/evidence/pr-integration-20261007/` and the shared pool job
directory. Supervisor exited0; all owned slots are stopped and no queued or
running pool job remains. No G4 was allocated. Final GitHub head CI must pass
before the authorized merge and local main fast-forward.
