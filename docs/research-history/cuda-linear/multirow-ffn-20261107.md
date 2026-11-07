# Multirow FFN reuse pilot

**English** | [日本語](multirow-ffn-20261107.ja.md)

## Contract and candidate

Optimize the existing 256-row, 256→1024 / 1024→256 FFN, A4096 per map,
Polar modes8/base1/merge2/spacing1. The existing active-native FP64-moment path
is the baseline. A separately registered research Algorithm,
`cuda_radius_cell_mass_row8/v1`, shares expansion and parameter-gradient geometry
across eight rows instead of four. The explicit `ROW8_NATIVE` Plan uses the
unchanged active-native recipe. It is not in the production catalog/default.

The CPU declarations and existing native checked entry retain their validation.
The candidate's source hash and build suffix distinguish its binary; default
builds still use four rows. It uses generic dispatch rather than the baseline's
specialized execution program. Tensor/function transforms fall back to the
existing Python entrance. This is an arithmetic-reuse pilot: FP64 Z/H/G, saved
Tensor ownership/version checks, support, detached radius task derivatives,
normalization, Atom count/shape, initialization and update law are unchanged.
Eight-row tail masks retain every input row and the original increasing batch
order in the final Atom-gradient reduction. Shared-memory layout is computed
for the selected group width; higher resource pressure can make it slower.

## Gates fixed before GPU timing

1. CPU metadata/support/serialization/selection and existing native regressions.
2. A100 MIG3g.40gb, record actual Torch/CUDA. Full 2×128 token inputs with
   noncontiguous layout: Y, dX and every Atom gradient in both FFN directions
   against the chunked CPU FP64 Fourier/autograd reference. Preserve the existing
   atol3e-5/rtol3e-4 admission. The independent polynomial quadrature check
   supplements the shared Fourier mathematics. Cover periodic edges, small and
   large support, zero amplitude, tail rows, retained backward/version checks,
   and reject Graph/capture rather than assuming support.
3. Compare candidate with current CUDA for outputs/gradients and optimizer
   Parameters/moments/clocks; test resume on the actual four-block FFN task.
   Record exactness separately from tolerance admission. No tolerance relaxation.
4. Only after correctness passes, run the existing cst-research FFN learner in
   fresh processes for baseline, row8 and Dense. Four pre-LN blocks, width256,
   FFN expansion4, head count4, c8, q0, minimum-001 envelope, native finite_chord
   update, Atom LR16×, baseLR3e-4, AdamW(.9,.95), decay0, clip1, TF32off, seed17.
   Batch2/context128/accumulation4 =256 rows per invocation and1024 tokens/update.
   Fixed20M token data/LR horizon,100 real updates (102400tokens), development
   only. No probes in timing; data transfer and all microsteps/updates are timed.
   Order A: baseline,Dense,row8; order B: row8,Dense,baseline. These are two
   fresh-process observations in one allocation, not six independent GPUs.
5. Report synchronized complete-update samples, setup separately, peak allocated
   and reserved including resident model/optimizer and eager temporaries. No
   Graph/capture or total process-memory claim. Compare full-model initial
   Parameters and calibration reports across the CST arms. Dense has its normal
   variance-compatible initialization; it is not an identical CST matrix.

Stop the frozen attempt on qualification failure/OOM/nonfinite/time budget.
Preserve negative results and actual source/result hashes. The candidate is not
adopted merely for passing correctness: judge complete-step time together with
memory; report remaining Dense gaps. This short performance pilot does not
establish long-run learning quality, minimum Atom counts or other GPU behavior.
RTX4090 is unavailable for this task. Raw evidence stays in ignored output.

This document records preparation, not completed GPU validation or a speedup.
