# Sphere output ownership study

Stored-profile direct contractions reduced memory but were slower than compact
W. Profile recomputation is being measured separately. This study changes
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

CPU, declarations, wheel/sdist and static source checks are ready. GPU
correctness, complete-step time, measured peak memory and adoption are pending.
