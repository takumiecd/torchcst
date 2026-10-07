# Larger profile-product chart checkpoint

Following PR #67, this branch introduces explicit large Product/input-Strip
matrix and prepared algorithms. Product axes and Strip input accept up to8192
sites, with at most4,194,304 atoms (5% of8192 squared fits this bound). Batch1..64,
FP32 IEEE, regular Euclidean Line axes, Polar[A,4] and two Triweight profiles
remain required. Strip output stays2..128 with existing tile choices. Earlier
algorithms retain their1024-site and Product65536-atom bounds.

The executors and recipe fields are unchanged. Whole-chart normalization and
its single floor, complete support, all canonical gradients, retained-forward
snapshots and live width/pitch updates carry over. This is a scaling experiment;
no GEMM tuning or approximate support is mixed into it. Explicit large IDs use
revision v1 and reuse the strict grouped matrix/preparation recipe types.

CPU1114 passed/1898 skipped, fourteen new CPU declarations/boundaries, eight
case snapshots, prepare/check for both families, Ruff, wheel/sdist build and
isolated installed-wheel metadata imports without Triton/benchmarks passed.
The new48-test suite includes34 actual CUDA cases: full-site FP64 Y/dX/all-atom
gradients at2049x2051 and8192x8192 Product /8191-input partial Strip, full/support
preparation, both matrix engines and the factor control, saved-forward changes,
twenty captured optimizer updates, dynamic pitch, and sort/view/inverse keys
above the int32 limit. Existing regression suites accompany the GPU gate.

The GPU gate at source fb9c79dbb9824c37b82259708231815eb48f8efa subsequently
passed861 tests: large48, matrix596, preparation75, grouped56, global30 and
Strip56 (726 actual CUDA,135 CPU/metadata). Job:
l4job-d347d87e317e4f47ac47b3c63cfe34ed. Its driver argument/proof uses the
unambiguous short commit fb9c79db; the companion JSON resolves the full commit
and verifies exact implementation file sets/bytes against the frozen archive.
Torch2.11.0+cu130, CUDA13.0, Triton3.6.0, NVIDIA L4; no tolerance changes.

## Measured 2048 complete steps

All four primary artifacts and two independent reverse-order rho3 artifacts
passed the full runner, submission and adapter checks. Same seed41,5% atoms,
B32,FP32 IEEE, live widths, fused capturable AdamW and production Polar update;
21 synchronized Graph samples per isolated route. Times below are median ms.

| Family / width | Prepared | Native G4P8 | Native G8P8 | Torch G8P8 | Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product2048 rho3 primary |2.269217|1.229411|1.104699|0.922988|0.592839|
| Product2048 rho3 reverse |2.323882|1.274074|1.121938|0.945002|0.592709|
| Product2048 rho8 primary |2.912834|1.911339|1.787189|1.527401|0.587895|
| Strip64x2048 rho3 primary |0.186983|0.151120|0.148370|0.080766|0.057226|
| Strip64x2048 rho3 reverse |0.185208|0.151014|0.148319|0.081214|0.057191|
| Strip64x2048 rho8 primary |0.210176|0.171288|0.169688|0.100334|0.056834|

Capture/replay peaks in bytes (all runs of each family agree):

| Family / route | Allocated | Reserved |
| --- | ---: | ---: |
| Product / prepared |101938176|295698432|
| Product / native G4/G8 |66493952|163577856|
| Product / Torch |99735040|205520896|
| Product / dense |102238720|148897792|
| Strip / prepared |4676096|12582912|
| Strip / native G4/G8 |2575360|6291456|
| Strip / Torch |36654080|48234496|
| Strip / dense |36719104|50331648|

Native reduces both time and allocated memory against the prepared control,
including on the single Product chart. Torch contraction remains faster than
native; dense remains faster than both. This does not justify default dispatcher
adoption. Rho3 has two independent family jobs; rho8 has one. Maximum runner
absolute Y/dX/all-atom gradient errors are5.381e-5/4.231e-5/3.848e-6;
Polar update error is zero. Tolerances are unchanged.

Primary Product/Strip jobs:
l4job-f8f23b04647d44e5aa44aedcf87a30bd /
l4job-bb56666e248047169c46bbc03923223f.
Independent reverse Product/Strip jobs:
l4job-8d7bb2fa57d24e6f9dd7943f4139215e /
l4job-d8d221c3453845b699dcee85070a09b8.

## Full 8192 oracle and pending timing

Diagnostic job l4job-3e9ac4544164476887606ddc5491360e passed the independent
FP64 default-chunk512 oracle for every8192x8192 site and all3,355,443 atoms
atB32/rho3, without sampling. Oracle wall97.979956s; native maximum absolute
Y/dX/dp errors0.000180482/0.000179605/0.00000481145 and relative L2 errors
1.477e-6/1.485e-6/2.232e-7. First native call3.431354s includes compilation.
Its validation memory includes retained oracle tensors and is not complete-step
capture/replay memory. No timing improvement at8192 is claimed from this check.

Two first8192 rho3 complete-step jobs preserve the same four plans plus dense:
l4job-1db8c943dbc04f89b01e052025af36fc (Product) and
l4job-6d5d7464679b4221845a57001ebf93b1 (Strip). Their predeclared family/child
budgets are2000/1800s, based on the measured98s full oracle plus four independent
oracle workers and full support scans. These are new-case budgets; no failed
measurement was retried or re-budgeted. Actual Graph time and peaks are pending.

The supervisor stopped the preceding selected batch; all slots were confirmed
stopped before restarting one worker for these queued jobs. Current pending
jobs are not a runtime shutdown claim. Runtime source is frozen at
fb9c79dbb9824c37b82259708231815eb48f8efa for every measurement.

The companion summary retains source/result/driver hashes, independent job IDs,
all samples, correctness and width/support records. Raw CPU/build logs, frozen
archives, GPU artifacts and drivers remain in ignored
benchmarks/cuda/linear/evidence/profile-product-large-chart-20261008/.

## Strip width envelope correction before execution

A new synthetic8192 Strip artifact reproduces the5 MiB submission-limit failure
from retaining both full width arrays for every route in the consolidated JSON.
The global Product exporter already keeps the unchanged arrays in worker JSONs
and puts count/ranges/exact-array SHA256 into the consolidated artifact. Extend
that same exporter to profile-product Strip, with no metric or adapter semantics
change. Ten Product/Strip database/export tests passed, including unchanged
metadata, every other result field, both original series and exact hashes. The
5 MiB policy and all numerical/performance gates stay fixed.

Queued Strip job l4job-6d5d7464679b4221845a57001ebf93b1 was cancelled before
execution to avoid the proven envelope failure. It has no kernel/timing outcome,
is retained in the job ledger, and is not a dropped measured case. A new source
checkpoint will run the same four plans/dense/case and2000/1800s budgets after
this export fix. The already running Product job is unaffected. Runtime source
and mathematical contracts remain fb9c79dbb9824c37b82259708231815eb48f8efa;
new benchmark exporter source is recorded separately in the next job snapshot.

Export-fix source a5e0df7168fd7b8d3c9d65739e62b96ec12ef940 passed the complete
CPU gate1115/1898 skipped,18 warnings. The replacement first Strip8192 job is
l4job-9d0685beb773489d90b6087ae518164c, with exactly the same case/plans,
seed and2000/1800s family/child budgets. GPU correctness/performance remain
pending for this job. Exact runtime file bytes are unchanged from fb9c79db;
this source snapshot includes the writer and additional CPU envelope test.

## Original8192 Product experiment incomplete

Original job l4job-1db8c943dbc04f89b01e052025af36fc returned1 after its runner
child hit1800s. The outer2000s pool driver did not time out. Its consolidated
artifact remains RUNNING with exactly five records: four complete full8192
FP64 correctness workers (prepared/nativeG4/nativeG8/Torch) and only the prepared
measurement. Every correctness worker checked Y,dX,all3,355,443 canonical atom
gradients and the production Polar step; max errors across them are
1.80482e-4/1.79605e-4/4.81145e-6 and Polar0. These completed gates are reported
separately. No paired performance or peak-memory conclusion is taken from this
incomplete experiment; all partial artifacts and raw worker arrays are preserved.

The complete-axis CPU support reports before/after every measurement are an
identified scaling cost in code; timeout causality remains an inference pending
a new completed experiment. Bounded complete-support reports are independently
validated against dense reports in31 CPU scenarios, with original dense fallback
near the floor and for wide/numerically uncertain sites. This changes diagnostic
work only, preserving every atom/site in support and the full FP64 oracle.
See20261008-profile-product-support-report.md for the exact bounds/fallback law.

New job l4job-47ae54c49f5141db829be2e76ed4c923 uses source
1d0cbf16b5dd78113e69123f62e1b7e0c5c44c9f, the unchanged four plans/dense and
2000/1800s budgets. It first gates the31 CPU scenarios on Colab's Torch version,
checks every actual initial atom's support counts and records CPU diagnostic time,
then invokes the unchanged complete-step runner. It does not retry unchanged
source or increase the timeout. The earlier failed experiment remains visible.
