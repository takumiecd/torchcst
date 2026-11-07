# FP32 multirow intermediates and row-parallel Atom gradients

**English** | [日本語](multirow-packed-20261107.ja.md)

The preceding row8 pilot reduced update time by22.7% but retained FP64 Z/H/G.
This continuation preserves its mathematical operator, initialization, support,
normalization, detached-radius task VJP and native Polar update. No production
default or unrelated GPU is changed.

Three explicit research Algorithms use FP32 moment/response storage while
retaining FP64 geometry, profile coefficients, phase arithmetic and accumulation:

| Algorithm suffix | Retained through backward | Atom-gradient execution |
| --- | --- | --- |
| packed | FP32 Z/H | Original serial row reduction |
| packed_recompute | X/P/keys only; regenerate FP32 Z/H | Original serial row reduction |
| packed_parallel | X/P/keys only; regenerate FP32 Z/H | Eight-row groups plus ordered FP64 partial reduction |

All use eight-row response projection in addition to the existing eight-row
expansion. The parallel variant evaluates per-Atom geometry once into a small
transient FP64 table, then distributes row groups across CTAs. At R256/A4096,
its FP64 geometry/partial scratch is0.78125/3 MiB; these are temporary, not saved
activation arrays. No dense effective W, atom omission or per-tile renormalization
is introduced. Recompute reads forward-time SavedVariables and preserves version
checks, saved-Tensor hooks and first-order differentiation limits.

Scope: 256↔1024 FFN, A4096, FP32, R1..256, Polar modes8/base1/merge2/spacing1,
radius[.5,50]. Four-row and FP64 row8 comparators remain separately selectable.
Generic Tensor/function transforms explicitly fall back to the original path;
no memory/performance claim is made for those fallback modes. Graph is rejected.

Before timing, run metadata/serialization/retention tests, GPU edge/tail tests
and the existing global-native regressions. Test dX-only and Atom-only backward,
repeated backward and mutation rejection. Prove that saved bulk arrays are FP32,
and that recompute variants save no Z/H. Each new Algorithm must pass both full
R256/A4096 FFN VJPs against the CPU FP64 Fourier/autograd reference with unchanged
atol3e-5/rtol3e-4 and the independent polynomial-quadrature supplement. Actual
four-block task resume and3 matched updates versus row8 check gradients,
Parameters and Adam moments/clocks at the existing tolerances; exactness is
reported separately, not required after the precision change.

Use the unchanged cst-research FFN learner and previous TinyStories token/data
order. A10080GBPCIe MIG3g.40gb only, TF32off, width256/4blocks/4heads/FFN4N,
batch2×context128 (R256), accumulation4, seed17, c8/A4096, q0, minimum-001,
AtomLR16×, baseLR3e-4, AdamW(.9,.95)/decay0/clip1/native finite_chord update,
fixed20M-token LR/data horizon. Compare baseline4, row8, Dense and all three new
variants in fresh20-update screens. Screen timings use existing updates11..20;
all20 updates enter allocated/reserved peaks. These are candidate selection,
not adoption measurements or convergence runs.

After numerical gates, choose the fastest screened variant with allocated AND
reserved peak at/below Dense. If none meets both, choose the lowest-allocated
variant that is faster than baseline4 and label the remaining tradeoff. If none
qualifies, stop and preserve the negative screen. Confirm the selected variant
against baseline4, row8 and Dense in fresh100-update runs in two reversed orders
(A: baseline4,row8,Dense,candidate; B: candidate,Dense,row8,baseline4). Existing
learner medians use updates11..100; all100 updates enter memory. Setup, external
qualification/cold compilation, evaluation, checkpoint/probes remain separate.
No process-memory or Graph claim. Use identical initial CST Parameters and
calibration reports, record loss sanity checks, exact source/result hashes and
all independent process observations. One allocation is not hardware replication.
Stop a frozen attempt on numerical failure/nonfinite/OOM/time budget; preserve
all failures and do not relax tolerances. Raw evidence remains ignored output.

This is the fixed premeasurement protocol, not completed GPU evidence.
