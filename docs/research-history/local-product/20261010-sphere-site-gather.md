# Sphere output ownership study

Stored-profile direct contractions reduced memory but were slower than compact
W. Profile recomputation has a separate completed study. This study changes
only output consumption to test the small-Linear H/owner idea on Sphere.
The existing pair of intrinsic S² charts, Polar/Triweight chord profiles,
complete per-chart L2 floors, live widths and all six atom gradients remain.

## Exact output support transpose

The unchanged ID-only preparer produces complete Norm/Count and packed IDs.
A CUDA Core producer writes unscaled H for every atom; input overflow visits
all input sites. Device count/prefix/scatter constructs output-site-to-atom
CSR for atoms whose complete output support fits CAP64. Each output owner
loops over its entire dynamic degree, recomputes profile values, reads H and
amplitude, and stores Y without atomic addition. Degree is not bounded by64.
Output-overflow atoms contribute exactly once in a separate full-site atomic
kernel launched after owner stores. This also handles both-sided overflow.

Backward is inherited unchanged from Phi-free direct, using saved H and
ID/norm/geometry snapshots with CTA-local G. CSR is forward scratch and is
not retained for backward. Empty atoms, signed amplitudes, norm floors,
strides, atom/site tails and retained forwards keep the same contract.
No full U/V, Phi or W/dW is allocated. FP32 IEEE, disabled TF32 and disabled
floating-point fusion remain. No reduction through a CSR key replaces a
parameter derivative.

The int32 CSR edge buffer reserves A*64 entries, 51.2MiB at N2048; metadata
and execution reject A*64 above signed-int32 capacity. Extra H reads and two
integer atomic construction passes may erase the output-scatter savings.
This is a tradeoff to measure, not an established speed or memory gain.

## Predeclared comparison

The fixed cohort contains compact W, Phi-free G4/merged direct, output
site-group1, output site-group4 and dense at both N1024/N2048 (10 workers).
Atom group4, CAP64, support tile16 and int16 packed site IDs stay fixed.
Both output-owner variants reuse merged direct backward.

Use batch32, floor(.05*N²) atoms, seed41, initial sigma3, MSE and fused AdamW
lr1e-4/weight-decay.01 with the existing research Graph update. Independent
full-site/all-atom FP64 Y/dX/all-dP checks run initially and after exactly24
updates, with maxabs AND relative-L2<=4e-4. The complete training-step median
comes from21 timed replays after2 warmups and1 capture/replay update.
Capture/replay peak allocated and reserved are recorded separately. Independent
30-update phase diagnostics are never summed into primary time.

The 43 new runtime tests and6 benchmark tests must pass with zero skips,
followed by all10 full correctness workers on the same frozen source, before
primary measurement. All preexisting runtime files are byte-identical to
parent3134fd2, whose238 original/recompute regression cases passed separately;
the new suite verifies the inherited backward, all gradients and actual CSR
contracts. Compiler/PTX diagnostics are isolated from primary timing.

Each job keeps900s outer/875s driver/350s child limits. Every collection,
partition, raw artifact and source hash is preserved; no omission, retry,
tolerance relaxation or partial-cohort selection is permitted. Qualification
is per size against compact W: >3% speed improvement, or >=5% allocated
reduction with <=3% time regression. A conditionally useful variant need not
win at every size. If any case qualifies, reverse the complete10-worker
cohort in an independent job before proposing any per-size policy. All
controls and both candidates remain present. Public defaults are a separate
decision.

## Completed validation and matched primary results

Frozen source is `a28c848896f6ca60ea6d03629901d158757f30a8` on
`kernel/sphere-site-gather`, retaining the parent3134fd2 lineage. All49 new
runtime/benchmark tests passed with zero skips. The correctness gate passed
30 common tests and all10 full workers; primary again passed30 common tests
and all10 workers. Every CST worker passed independent initial/after24 full
FP64 Y/dX/all-dP at the unchanged4e-4 maxabs AND relative-L2 gates. Primary
clocks are24 updates, and separate phase copies advance24→30. Primary driver
time was244.228s within the875s limit.

The primary ran on NVIDIA L4
(`GPU-05cdf79b-95cf-63d4-ef03-68266fcb15a6`, driver580.82.07), Torch2.11.0+cu130,
CUDA13.0 and Triton3.6.0. The table reports uninstrumented complete-step
medians and capture/replay allocated / reserved MiB. Process memory is
unmeasured. Each route has one independent worker and21 timing samples;
samples are not independent runs. Controls below were remeasured in this
cohort, rather than imported from the preceding recompute cohort.

| Route | N1024 ms | N1024 MiB | N2048 ms | N2048 MiB |
| --- | ---: | ---: | ---: | ---: |
| compact W | 4.020058 | 92.862 / 158 | 22.294632 | 271.585 / 338 |
| recompute G4 merged | 6.073122 | 48.463 / 130 | 30.972794 | 146.259 / 212 |
| site gather1 | 5.333492 | 60.625 / 130 | 28.373254 | 192.982 / 264 |
| site gather4 | 4.941968 | 60.625 / 130 | 28.517573 | 192.982 / 264 |
| dense | 0.084978 | 32.877 / 86 | 0.594941 | 81.502 / 106 |

Against matching recompute G4/merged, site gather1 improves time12.179%
/8.393% at1024/2048; site gather4 improves18.626% /7.927%. Both increase
allocated peak25.093% /31.945% (12.161MiB /46.723MiB). These measured peaks
reflect Graph/allocator lifetimes, not just the CSR capacity estimate.

Against compact W, site gather1 is32.672% /27.265% slower and site gather4
is22.933% /27.912% slower. Both lower allocated peak34.716% /28.942%, but
exceed the allowed3% time regression. Neither candidate qualifies at either
size. The assessor returned `PASS`, `inverse_selection=[]` and
`automatic_adoption=false`; no inverse was eligible or run. This is a negative
adoption result with a measured direct-route speed improvement. Runtime remains
on the research branch and no public default changes.

## Separate diagnostics and costs

These forward-loss / backward medians come from independent five-sample
instrumented phase copies ending at30 updates. They are not summed into,
subtracted from, or substituted for the complete-step medians.

| Route | N1024 forward-loss / backward ms | N2048 forward-loss / backward ms |
| --- | ---: | ---: |
| compact W | 2.519040 / 1.445888 | 14.556160 / 7.362560 |
| recompute G4 merged | 2.800640 / 2.397184 | 18.151424 / 12.938240 |
| site gather1 | 2.171904 / 2.394112 | 15.414272 / 13.014016 |
| site gather4 | 2.203648 / 2.393088 | 15.427584 / 13.159424 |

The independent actual-N2048/B32/A209715 compiler diagnostics report H
producer80 registers/zero spills/512 shared bytes and output owner40
registers/zero spills, with512 shared bytes for site-group1 and2048 for
site-group4. The unchanged backward uses168 registers/eight spills/1024
shared bytes. Count/prefix/scatter kernels use16/40/22 registers, zero spills
and512/64/2048 shared bytes respectively; overflow output uses38 registers,
zero spills and no shared memory. Saved PTX contains no MMA or TF32.
These resource facts and phase observations do not establish a timing cause.

The mechanism removes packed-output float atomics while adding fresh CSR
count/prefix/scatter and per-edge H gathers; overflow retains exact atomic
fallback and backward dX atomics are unchanged. CSR reserves51.2MiB int32
edge capacity at2048, although it is forward scratch. At about28 output
edges per atom, reading32 H values per edge implies roughly717MiB of logical
H read requests. This is a workload estimate, not measured DRAM traffic:
cache reuse, atomic contention and bandwidth were not attributed here.

## Provenance, reproduction and recovery

All stages share frozen tracked source and tools; stage archive digests are
listed separately. Raw files and full manifests remain under
`~/.local/state/colab-l4-pool/jobs/<job-id>/results/`, and ignored
`output/sphere-site-gather/` retains the protocol, drivers, assessor, assessment,
empty inverse selection and parent runtime hash proof.

| Stage | Job ID | Source snapshot SHA256 |
| --- | --- | --- |
|49 regressions | `l4job-cbc0c644be464bf9a6b47c7227abb85c` | `09556e6a9dd911605c9d27bd9a50d295d49f60d39b32caffac0cc36af9eaa0ab` |
|10-worker correctness | `l4job-62c196e5f134475daa44ce109899fe0e` | `86c1b70c6eb8a18f9c9e67260d5626dc814ac24ea0f1a16950ccad706fe8fc3f` |
|10-worker primary | `l4job-3cf8e1ab7521422f927a4ec0cbb4ff0b` | `3231d85d55c728ebd29433eba2ab01172161d160c6c5f12496bb7c82e24fbdcc` |

All10 primary worker JSON hashes were independently checked against the cohort
manifest, together with24/30 clocks and full numerical gates.

- Protocol SHA256: `656042ae3ad7d04064a5824cc1165783728460a8e4fe42b3def72a3b0af06960`.
- Driver SHA256: `ccd980986c10251d27a69d797a650203d3e0aa7e4d50c52d4f7042911c51973e`.
- Assessor SHA256: `358a8735834c6dfbaadcd69e040dba28fd71f9573b9469728b56e407a6ef4e14`.
- Assessment SHA256: `d6b3c0eab20c56d5ddde730e4a4635ca2fbd5f735acba3da167605cd9991f8c2`.
- Primary `cohort.json` SHA256: `7770dfc6005d0274ee0e5495ad9cc3bd104ae4ef5f4c1f33a26e68f3c5ed19a9`.
- Empty inverse-selection SHA256: `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.

Reproduction uses the retained driver after the two prerequisite stages:

```sh
python output/sphere-site-gather/cohort_driver.py \
  --stage primary --expected-tests 30 \
  --catalog benchmarks/cuda/linear/plans-sphere-site-gather.json
```

The fixed full cohort and900/875/350s budgets remain in force; selective timing
does not reproduce the adoption assessment. GPU ownership and submission stay
with the orchestrator under the research operating rules.
