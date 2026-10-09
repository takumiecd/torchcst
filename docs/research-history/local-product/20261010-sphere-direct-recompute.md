# Sphere direct profile recomputation study

The [preceding CUDA Core direct cohort](20261009-sphere-direct-cuda-core.md)
was correct but failed the compact-W
time/memory adoption criterion. This separate study tests whether deleting
profile storage at allocation time improves that tradeoff. It preserves the
existing pair of intrinsic S² charts, Polar/Triweight chord profiles, complete
per-chart discrete L2 floors, live widths and all six atom gradients.

## Execution change and constraints

The new research Algorithm owns the storage choice; its Recipe has the same
CAP64/T16/int16/group1-or4/separate-or-merged fields as stored direct.
The existing decoder and site normalization are reused. A dedicated packer
stores complete Norm and Count plus ordered support IDs, never allocating Phi.
The contraction recomputes gap³/max(savedNorm,floor) from saved sites, centres
and precision. Backward does not reread mutable live kernel/chart buffers.
The original stored Algorithm retains its arithmetic and execution contract.

This removes `2*A*64*4` bytes of tensor capacity: 26,843,136 bytes at 1024 and
107,374,080 bytes at 2048. These estimates are not measured peak reductions.
H remains saved for backward, G stays CTA-local, and no full U/V or W/dW is
allocated. Atomic scatter counts and complete-chart norm scanning remain.
FP32 IEEE, disabled TF32 and disabled floating-point fusion are unchanged.

## Frozen comparison before GPU execution

N1024/N2048, batch32, floor(.05*N²) atoms, seed41, sigma3, MSE, fused AdamW
lr1e-4/weight-decay.01, existing research Graph update, and exactly24 updates
match the preceding cohort. All CST routes require independent full-site,
all-atom FP64 Y/dX/all-dP at initial and after24 checkpoints with both maxabs
and relative-L2<=4e-4. Twenty-one timed Graph replays follow the same2 warmup
and1 capture/replay updates. Peak allocated/reserved includes capture/replay.
Separate phase diagnostics are never added to or subtracted from step timing.

The new fixed cohort has22 workers: compact W, old direct, four stored direct
controls, four matched recompute variants and dense at each size. No route is
filtered before measurement. Recompute-vs-matching-stored comparisons isolate
storage/recomputation; adoption still compares against compact W. Qualification
requires >3% speed improvement, or >=5% allocated reduction with <=3% time
regression, at both sizes and independent reverse confirmation. The complete
controls remain in any inverse cohort. No public default is changed by a
benchmark result alone.

Compilation-heavy regression checks use two separate prerequisite jobs:
the original159 tests and the new73 runtime plus6 benchmark tests (238 unique
regression cases total). A third job checks all22 full correctness workers;
timing starts only after all prerequisites pass on the same frozen source.
Each job keeps the900s outer/875s driver/350s child limits. All collections,
partitions, JUnit mappings, source hashes and failures are preserved. This is
a separately declared implementation study, not an expanded retry budget for
the preceding failed gate. No tolerance relaxation or partial winner selection
is permitted.

## Completed GPU validation and matched results

The frozen implementation is commit
`3134fd2e32a110b5c8afe56e0cada39caf5d6575` on
`kernel/sphere-direct-recompute`. Original regressions passed159/159 and
recompute regressions passed79/79 with zero skips. The full correctness gate
passed all22 workers, followed by all22 primary workers and30 common tests.
Each CST worker passed the independent initial/after24 FP64 Y, dX and all-dP
checks without changing the4e-4 maxabs or relative-L2 limits. Primary driver
time was597.665s, within the frozen875s limit. Hardware was NVIDIA L4
(`GPU-d176e854-fb7f-9cc1-bf2d-7e9fd3badf50`, driver580.82.07), Torch2.11.0+cu130,
CUDA13.0 and Triton3.6.0.

These are matched primary complete-step medians and capture/replay memory
peaks. Memory columns are allocated / reserved MiB, not GPU process usage.
Each cell is one independent worker with21 timed replays, not21 independent
runs. Stored/recomputed pairs differ in Phi storage and recomputation only.

| Route | N1024 ms | N1024 MiB | N2048 ms | N2048 MiB |
| --- | ---: | ---: | ---: | ---: |
| compact W | 4.236103 | 92.862 / 158 | 23.708991 | 271.585 / 338 |
| old direct | 9.341014 | 87.662 / 164 | 46.502208 | 302.259 / 376 |
| stored G1 separate | 8.285995 | 74.862 / 132 | 42.212094 | 250.259 / 316 |
| recompute G1 separate | 8.567550 | 48.463 / 130 | 41.791453 | 146.259 / 212 |
| stored G4 separate | 6.809540 | 74.862 / 132 | 35.017720 | 250.259 / 316 |
| recompute G4 separate | 6.940207 | 48.463 / 130 | 34.833665 | 146.259 / 212 |
| stored G1 merged | 7.713684 | 74.862 / 132 | 38.834433 | 250.259 / 316 |
| recompute G1 merged | 7.813361 | 48.463 / 130 | 37.780611 | 146.259 / 212 |
| stored G4 merged | 6.392073 | 74.862 / 132 | 33.376390 | 250.259 / 316 |
| recompute G4 merged | 6.457436 | 48.463 / 130 | 32.783079 | 146.259 / 212 |
| dense | 0.085190 | 32.877 / 86 | 0.592783 | 81.502 / 106 |

Against matching stored routes, recomputation lowers allocated peak35.263%
at1024 and41.557% at2048. Time changes (positive means slower) are respectively
G1 separate +3.398% / -0.996%, G4 separate +1.919% / -0.526%, G1 merged
+1.292% / -2.714%, and G4 merged +1.023% / -1.778%. The measured peak reductions
are26.399MiB /104MiB; these include allocator/Graph lifetimes and are distinct
from the25.6MiB /102.4MiB eliminated Phi tensor capacity.

The fastest recompute route is G4 merged at both sizes. Relative to compact W,
its allocated peak falls47.812% /46.146%, but complete-step time rises52.438%
/38.273%. All four variants fail the predeclared compact-W adoption gate.
The assessor returned `PASS`, `primary_both_shapes_qualified=[]`,
`inverse_selection=[]`, and `automatic_adoption=false`. No inverse job was
eligible or run. The route remains a negative research result with a measured
memory benefit; it is not adopted and no public default changes.

## Compiler evidence and remaining hypotheses

The separate actual-N2048/B32/A209715 G4-merged resource diagnostic records
forward127 registers, zero spills and512 shared bytes; backward168 registers,
eight spills and1024 shared bytes. PTX contains FP32 multiply/add instructions
and no MMA or TF32. These compilation facts do not prove the cause of the
complete-step timing result: cache traffic, occupancy and atomic contention
were not attributed by this study.

The saved G4-merged PTX also confirms scalar X/DY loads with batch stride8192
bytes for the current row-major `[B,N]` layout. Four adjacent lanes select
different atoms before the batch index advances, while support IDs supply
random site offsets. A contiguous `[N,B]` layout could improve batch access
locality, but would add transpose buffers/launches and retain Y/dX atomic
counts and contention. This is an unmeasured future hypothesis, not an
implementation or explanation of the measured slowdown.

## Provenance and recovery

The candidate implementation, tests and catalogs remain on the retained
`kernel/sphere-direct-recompute` research branch at the frozen commit above.
They are not in `main`; this integration contains research notes only.
Local recovery artifacts are being collected under
`output/sphere-cuda-core-recovery-20261010/` in the primary checkout.
Its `checkpoint-manifest.json` records an intermediate bundle verification,
restore and fsck PASS. Final recovery-bundle and restore verification is pending,
including the final Site study heads and raw evidence; no final recovery or
cleanup completion is claimed here. Preservation follows the
[research operating rules](../../research-operations.ja.md).

All four stages used the same frozen tracked implementation and comparison
tools. Archive SHA values are listed separately below; the assessor verified
tracked-source identity across stages.

| Stage | Job ID | Source snapshot SHA256 |
| --- | --- | --- |
| original159 regressions | `l4job-6a9738646ee34c7f985337f0d55c5b4b` | `fbfa46ffd575373a03e3080e9471803d32dbfbfcf446c78e704d12da9087d3b8` |
| recompute79 regressions | `l4job-2f2d788389cd49cd8eff3dac3660b360` | `cb2c3e24a329e3b23ea16f1cc64a9bb3c9f0a13aad44901450452f61e12c2c2e` |
| full22-worker correctness | `l4job-24377266f68a46d28e08c04c690d246e` | `a316cab98552f0dc6bbccc7a4cd53e7146c93363c2122903b7e23d4f6c4857c8` |
| primary22 workers | `l4job-fe5b81a495cf473085b3ff874f80c187` | `b15191e6cb9b55f6cddf6a0c5676ea1010e32f9cf4c9b71379b5b5c685aaaf8c` |

Frozen tools and assessed results are retained in ignored
`output/sphere-direct-recompute/`; raw results/source manifests remain under
`~/.local/state/colab-l4-pool/jobs/<job-id>/results/`. All22 primary worker JSON
SHA values were independently checked against the primary cohort manifest.

- Protocol SHA256: `5cc4485386d4414062ee2aba5da900307e8f062fc1883bec4cc86e3fc002f375`.
- Driver SHA256: `e1df3987ffe061de67e28619cbf3726aa1cca5b801e48105c65ce0adb28935e2`.
- Assessor SHA256: `8fe45c0b4bfe41ba4cec524c812e7729436a8a93e9b477fc195f527d9ccccc50`.
- Assessment SHA256: `e31be1c34c197b5a659f1040168351e095331f67eba84bb35f7ab90898bb6d69`.
- Primary `cohort.json` SHA256: `76656bc5b27a8194a8c3f4b2d59f9808197dcd00314f8f4b7d7606d8929cca67`.
- Empty inverse-selection SHA256: `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.

Reproduction requires the retained research checkout at commit
`3134fd2e32a110b5c8afe56e0cada39caf5d6575` and its ignored frozen driver;
the following command is not available from the notes-only `main` tree:

```sh
python output/sphere-direct-recompute/cohort_driver.py \
  --stage primary --expected-tests 30 \
  --catalog benchmarks/cuda/linear/plans-sphere-direct-recompute.json
```

The three prerequisite stages and fixed budgets must remain in force rather
than being replaced with selective timing. GPU submission and resource
ownership remain with the orchestrator under the research operating rules.
