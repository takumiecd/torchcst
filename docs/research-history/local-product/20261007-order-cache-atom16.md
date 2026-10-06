# 2026-10-07: parameter atom16 with exact ordered cache

Branch kernel/ordered-cache-atom16 based on PR44/7c18dbd6. Only benchmark recipes change; runtime kernels unchanged. Three new gather/repair4/repair8 cache routes use parameter atom16,batch16 rows/split2,forced4 warps. Controls are copy8, atom16/warps4/batch split2, and matching three cached atom32/batch split2 routes. Output/input owner16 dimensions, atom chunk32,owner split4 and current per-forward normalized metadata/snapshots/canonical gradient IDs preserved. H not globally allocated.

N64 cached atom32 is slower than uncached atom16 in prior screen; this separates cache effect from parameter tile choice. N128 cached atom32 was best in rho3/8; atom16 is not assumed better there. Peak includes persistent ID-cache and parameter partials/reduction; no cache-residency/DRAM claim.

N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off, initial decodedrho>1,production fused AdamW/Polar evolving sigma,21 samples/execution. Independent FP64 Y/dX/all atom gradients/nonzero positions, support/normalization/slices/B1/32/64,20 captured reference updates/moments/steps N64/N128,old backward.

Host886 passed/841 skipped (22.24s), changed-file ruff,diff checks,8 prepare/check,isolated wheel/sdist build passed. GPU correctness/time pending; raw evidence order-cache-atom16-20261007 ignored and preserved.
