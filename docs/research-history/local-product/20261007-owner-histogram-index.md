# 2026-10-07: direct owner index from position histogram

Branch kernel/owner-histogram-index based on verified-negative PR45. Old prefix routes retained. This distinct prototype avoids dependent bisections and repeated per-owner interval scans without materializing H or prefix scratch globally.

For each current direction/band, let l_a,h_a be integer clipped support bounds relative to current sliced domain, L=max_a(h_a-l_a), F(t)=number of active nonsingleton band atoms with l_a<t, and s the current band start in sorted order. Owner [j0,j1) uses begin=s+F(clamp(j0-L+1)), end=s+F(clamp(j1)), with bounds clipped into0..domain_count. If h_a>j0, then l_a>=j0-L+1; if l_a<j1, then a<end. Thus every contributor is retained even for nonmonotone ends/arbitrary current widths. Extra early noncontributors can remain; exact site factors and derivatives remain unchanged. No epsilon pruning or straight-through gradient. Actual singleton flags/buckets remain separate; rho alone never establishes one-hot.

The order CTA computes3 current support-span maxima,3 position histograms/prefix counts and direct owner-vector lookups. Position bins are next_power_of_2(max(input_count,output_count)+1); nonmembers/padding use a sentinel beyond every valid active start. All state is CTA-local. Following copy CTAs copy disjoint segments only. Canonical IDs, normalized coefficients and independent per-forward snapshots retain original contract.

Four variants: preparation4/8 warps x parameter atom32/batch split2 or atom16/warps4/batch split2. Output/input owner16 dimensions,atom chunk32,owner split4,batch16 rows. Controls copy8,param2 atom32,atom16/warps4/batch split2; dense same operator. N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off, actual initialrho>1 (1.25/3/8/mixed), production fused AdamW/Polar evolves widths.21samples/execution;rho3/8 screen first.

Host887 passed/868 skipped(22.21s),changed-file ruff/diff,8 prepare/check,isolated wheel/sdist build passed. GPU suite36 checks+declaration covers full support/snapshot coverage,independent FP64 Y/dX/all atom gradients/nonzero centers,slices/B1/32/64,N64/N12820 captured reference optimizer updates/moments/steps,old backward. Correctness and speed unverified. Extra safe-envelope work may erase preparation saving, especially mixed widths. Compiler and complete-step time/peak decide adoption. No physical DRAM/cache residency claim. Ignored raw evidence owner-histogram-index-20261007 preserved.

Frozen source779564e4; PR48(base PR45). GPU check `l4job-ce9f94ea70584f91894694e9f85e0ea2`,64screen `l4job-28daef7a16704eea932965f2cf9fa8eb`,128screen `l4job-e4f53041fe5147c4bd78ded66d2ac3f4` queued after cached-atom16 jobs on supervisor65855.

## Verified GPU correctness and N64 screening

37 tests passed (36 GPU +1 declaration),29.71s; source779564e4,197 committed/submitted/worker runtime hashes and source/result archive SHA256 agree. N64 screening14 full-shape FP64 comparisons pass including nonzero center gradients; all-plan initial Parameter/input bytes including dense match, actual initialrho>1 and changing production widths. N128 screening and inline-copy variant remain pending. One execution per condition,21samples; medianus.

|N64 rho|copy8|param2|atom16split2|hist8 param2|hist8 atom16split2|hist4 param2|hist4 atom16split2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|59.82|52.69|51.02|53.95|52.47|53.75|52.51|39.09|
|8|56.64|52.93|51.14|54.20|53.04|53.93|52.54|39.30|

All histogram routes lose to their matching exact owner-range scan controls in this N64 screen by about1–2us. The safe max-span envelope may include extra zero-support atom visits; exact site factors and position derivatives remain intact. Algorithmic exploration reduction is not a complete-step performance win. Peak115,712 bytes and reserved6,291,456 unchanged. No promotion from this screen; preserve the negative result. Inline metadata copy will test whether removing a launch offsets this preparation overhead.

N64 two artifacts verified by byte-identical database export/idempotent reimport. Separate layout diagnostic8.192us control vs9.216–10.240us histogram, ordered_layout48/54 registers vs27, all0 compiler spill slots,shared1024 bytes. Copy-only CTA24registers/shared0 vs32/shared16; logical scan savings do not compensate order-CTA work. Diagnostics are separate from complete-step timing. CPU CI676f36c5 passed.

## Completed N128 screening

N128 job also succeeded. Total28 full-shape FP64 comparisons, all-plan input and Parameter hashes including dense and197 runtime/archive proofs passed. Same one-execution screening scope.

|N128 rho|copy8|param2|atom16split2|hist8 param2|hist8 atom16split2|hist4 param2|hist4 atom16split2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|3|70.52|67.88|67.86|70.32|70.36|72.92|73.30|45.58|
|8|73.71|69.84|69.92|72.41|72.49|74.63|75.25|45.72|

All histogram routes lose to matching non-histogram parameter controls;hist8 adds about2.5us,hist4 about5us. Preparation4warps worsens N128 vs8warps. Peak303,104/reserved6,291,456 bytes unchanged. No adoption or full-repeat promotion. Inline copy and one joint histogram are distinct follow-up experiments in PR49/50 with separate sources/tests. N128 DB verification is in progress.

Completed N128 DB verification: all4 screening artifacts now have byte-identical export/idempotent reimport. Negative results and raw archives preserved.
