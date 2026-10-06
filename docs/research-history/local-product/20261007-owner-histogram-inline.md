# 2026-10-07: current histogram index with inline physical copy

Branch kernel/owner-histogram-inline based on PR48/676f36c5. The conservative current-support histogram-index formula and gradients are unchanged. Two recipes keep8 preparation warps and parameter atom32/batch split2 or atom16/warps4/batch split2. Physical normalized13-field values are copied by the same two direction CTAs that sort/index ranges. The following copy-only launch is omitted. Output/input owner16 dimensions,atom chunk32,owner split4,batch16 rows. No new global prefix/H/state.

Current canonical metadata and sorted IDs are available in these CTAs; each direction writes disjoint physical view, order, offsets and range slices. Gradient IDs remain canonical; current coefficients/norms/support and independent autograd snapshots are unchanged. This trades the previous parallel copy's throughput for one fewer launch/Order reread. Register/indirect-load cost may make inline copy slower; no performance or cache residency claim before GPU evidence.

Controls copy8,param2,atom16split2/4warps,and matching two parallel-copy histogram recipes. N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off,initial decodedrho>1,production fused AdamW/Polar with live sigma,21samples/execution. Independent FP64 Y/dX/all atom gradients/nonzero positions,full support/snapshot coverage,slices/B1/32/64,N64/N12820 captured reference updates/moments/steps,old backward.

Host888 passed/886 skipped(22.48s),changed-file ruff/diff,8 prepare/check,isolated wheel/sdist build passed. GPU suite18 checks+declaration and rho3/8 screen pending. Raw evidence owner-histogram-inline-20261007 ignored and preserved. Original histogram4/8-warp and negative bisection recipes remain intact.

Frozen sourcede62e6b4; PR49(base PR48). GPU check `l4job-9d94af39d688452fb94aae3088988666`,64screen `l4job-ee834a73bbc348188a3c597185c6b357`,128screen `l4job-ef19681399a242aaa682c993af56f706` queued on supervisor65855 after original histogram jobs.
