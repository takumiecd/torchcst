# 2026-10-07: cache repair + parameter VJP, lower memory owner split

Branch kernel/small-linear-combined combines PR43 and PR42 on an isolated research branch. Existing recipes are retained. No main integration.

Three new cache routes combine current exact key-gather validation / bounded repair4 / repair8 with parameter atom32/batch split2. Every call still refreshes current coefficients, normalized support and independent autograd snapshots. Current canonical IDs retain exact gradient ownership; failed bounded repair performs full sort. New combined effects remain unmeasured.

One route keeps output/input owner16 dimensions, atom chunk32, parameter atom16/warps4/batch split2, and reduces owner split4 to split2. This halves the output/dX partial buffer capacity at the same tensor axes, but may lose GPU parallelism. Report actual complete-step peak and time rather than predicting peak from tensor budget. H is not allocated globally; physical DRAM/cache residency is not measured.

N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off; initial decoded rho>1 (1.25/3/8/mixed), production fused AdamW/Polar with evolving sigma. Controls copy8, param2 atom32 and atom16/warps4/batch2; dense same operator. 21 samples/execution, reverse case/plan order on repeat2. Independent FP64 Y/dX/all atom gradients and nonzero centers, singleton/norm-floor/empty support, slices/B1/32/64, old backward, N64/N128 20 captured reference optimizer updates/moments/steps.

Host:885 passed/814 GPU-or-resource skips (22.66s), changed-file ruff, isolated wheel/sdist build, 8 prepare/check passed. GPU checks and rho3/8 screening submitted only from committed snapshot. Ignored raw evidence under small-linear-combined-20261007.
