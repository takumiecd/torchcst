# 2026-10-07: parameter VJP batch16 loop without partial reduction

Branch kernel/parameter-batch-loop based on PR44. Two new research recipes explicitly use16 batch rows in the existing parameter VJP loop, atom records16,4 warps,1 parameter CTA partition. B32 is processed as2 tiles by one CTA per atom block; fixed-source Polar pullback follows accumulated task cotangents. No parameter partial buffer/reduction launch. Owner output/input16 dimensions, atom chunk32, split4 or split2. H not globally allocated. Position, width, normalization, dX and all atom gradients retain original contract.

N64 atom16 single control defaults BB32; this variant reduces register live tiles while giving up split2 CTA parallelism. N128 default already uses BB16, so owner-split4 variant is identical compiler configuration to atom16/warps4 control; alias timing differences there are noise, not a new algorithm. Owner-split2 is independently different. Measure complete time and allocated peak rather than claiming cache residency/zero DRAM.

N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off, initial decodedrho>1,production fused AdamW/Polar evolving sigma,21 samples/execution. Controls copy8,param2 atom32,atom16 single/4warps,atom16 split2/4warps; same dense operator. Independent FP64 Y/dX/all atom gradients/nonzero centers, singleton/norm-floor/slices/B1/32/64,20 captured reference updates/moments/steps N64/N128, old backward.

Host886 passed/832 skipped (22.37s), changed-file ruff,diff checks,8 prepare/check,isolated wheel/sdist build passed. GPU correctness/speed pending. Raw evidence parameter-batch-loop-20261007 ignored and preserved.
