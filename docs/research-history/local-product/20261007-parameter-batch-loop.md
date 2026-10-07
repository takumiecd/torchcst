# 2026-10-07: parameter VJP batch16 loop without partial reduction

Branch kernel/parameter-batch-loop based on PR44. Two new research recipes explicitly use16 batch rows in the existing parameter VJP loop, atom records16,4 warps,1 parameter CTA partition. B32 is processed as2 tiles by one CTA per atom block; fixed-source Polar pullback follows accumulated task cotangents. No parameter partial buffer/reduction launch. Owner output/input16 dimensions, atom chunk32, split4 or split2. H not globally allocated. Position, width, normalization, dX and all atom gradients retain original contract.

N64 atom16 single control defaults BB32; this variant reduces register live tiles while giving up split2 CTA parallelism. N128 default already uses BB16, so owner-split4 variant is identical compiler configuration to atom16/warps4 control; alias timing differences there are noise, not a new algorithm. Owner-split2 is independently different. Measure complete time and allocated peak rather than claiming cache residency/zero DRAM.

N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off, initial decodedrho>1,production fused AdamW/Polar evolving sigma,21 samples/execution. Controls copy8,param2 atom32,atom16 single/4warps,atom16 split2/4warps; same dense operator. Independent FP64 Y/dX/all atom gradients/nonzero centers, singleton/norm-floor/slices/B1/32/64,20 captured reference updates/moments/steps N64/N128, old backward.

Host886 passed/832 skipped (22.37s), changed-file ruff,diff checks,8 prepare/check,isolated wheel/sdist build passed. Raw evidence parameter-batch-loop-20261007 ignored and preserved.

Frozen source889d9ef1; PR46 (base PR44). Check `l4job-824b070a35da4bfc9e60d8a0058c4fae`,64 screen `l4job-a1e8ab45c1d8464ba358ac562fcd5724`,128 screen `l4job-ccd359242e894bd39702625053fee898` completed; the source-specific results are below.

## Verified L4 screening,1 execution per condition

19 tests passed (18GPU+declaration),16.06s.197 runtime files match committed/submitted/worker,source/result archives verified. All24 full-shape FP64 Y/dX/all-atom-gradient comparisons and actual initialrho>1/nonzero positions/within-job input+Parameter hashes passed.

|N/rho|copy8|param2 atom32|atom16 single|atom16split2/4warps|batch16 loop|batch16 loop/owner2|dense|
|---|---:|---:|---:|---:|---:|---:|---:|
|64/3|59.02|52.58|52.60|50.63|52.64|56.86|39.33|
|64/8|56.49|52.53|51.62|51.21|51.78|55.85|39.27|
|128/3|70.25|67.88|69.11|67.56|69.09|70.41|45.69|
|128/8|73.68|69.45|70.19|69.35|70.28|80.42|45.92|

Complete-step median us. Owner4 peaks115712/303104 bytes;owner2 99328/270336;reserved6291456. Batch16 loop does not beat same-job parameter split2 in any screened case. N128 owner4 loop and atom16 single are identical configs; tiny alias timing differences are noise. Owner2 memory/time tradeoff retained; no route promoted. Raw negatives/proof saved,4 measurement artifacts DB byte-identical export/idempotent reimport checks passed in matching branch.

## Integration disposition

PR46 history is included in PR55 alongside compact dX snapshots and current
main. The combined-source gate and case-scoped speed/memory policies are
recorded in [the integration report](20261007-pr-integration.md). Original
screening measurements keep their original source identity.
