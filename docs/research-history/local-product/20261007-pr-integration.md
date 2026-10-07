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

GPU results, source/result hashes, database export verification and generated
speed/memory policies will be recorded after retrieval. Original PR55's paired
two-execution record and PR46's four screened conditions remain in their
existing research reports; those measurements are not relabeled as integration
source results.

Raw drivers, frozen source, CPU/build/declaration logs and checkpoints:
`benchmarks/cuda/linear/evidence/pr-integration-20261007/` and the shared pool job
directory. All owned runtimes must be stopped before completion. GitHub CI and
the selected GPU gates must pass before merging PR55, followed by local main
fast-forward. No direct main push.
