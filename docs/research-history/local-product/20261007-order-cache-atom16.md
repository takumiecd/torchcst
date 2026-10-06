# 2026-10-07: parameter atom16 with exact ordered cache

Branch kernel/ordered-cache-atom16 based on PR44/7c18dbd6. Only benchmark recipes change; runtime kernels unchanged. Three new gather/repair4/repair8 cache routes use parameter atom16,batch16 rows/split2,forced4 warps. Controls are copy8, atom16/warps4/batch split2, and matching three cached atom32/batch split2 routes. Output/input owner16 dimensions, atom chunk32,owner split4 and current per-forward normalized metadata/snapshots/canonical gradient IDs preserved. H not globally allocated.

N64 cached atom32 is slower than uncached atom16 in prior screen; this separates cache effect from parameter tile choice. N128 cached atom32 was best in rho3/8; atom16 is not assumed better there. Peak includes persistent ID-cache and parameter partials/reduction; no cache-residency/DRAM claim.

N64/N128 B32 A204/A819 seed41 FP32 IEEE/TF32off, initial decodedrho>1,production fused AdamW/Polar evolving sigma,21 samples/execution. Independent FP64 Y/dX/all atom gradients/nonzero positions, support/normalization/slices/B1/32/64,20 captured reference updates/moments/steps N64/N128,old backward.

Host886 passed/841 skipped (22.24s), changed-file ruff,diff checks,8 prepare/check,isolated wheel/sdist build passed. GPU correctness/time pending; raw evidence order-cache-atom16-20261007 ignored and preserved.

Frozen source4aa6823f; PR47. Check `l4job-6b7bad148a734e33a093493f964a64f0`,64screen `l4job-337c739d866946f8a6e954b1e04da9a7`,128screen `l4job-48f55393ade9438ca2b07140ec062f39` queued after combined full repeats on supervisor65855.

## Verified GPU screening

28 tests passed (27 GPU +1 declaration),8.11s. Source4aa6823f; committed/submitted/worker197 runtime hashes and both archive SHA256 proofs agree. All32 full-shape FP64 comparisons passed, nonzero position gradients, actual initialrho>1 and evolving production widths. Same initial canonical Parameter bytes across CST plans and input/target bytes across all plans including dense. One execution per condition,21 samples; medians us, independent full reruns remain pending.

|N/rho|copy8|uncached atom16split2|gather atom32|repair4 atom32|repair8 atom32|gather atom16|repair4 atom16|repair8 atom16|dense|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|64/3|59.54|51.18|53.16|52.82|53.01|51.63|51.19|51.48|39.06|
|64/8|56.66|51.20|52.32|52.26|51.76|50.39|50.75|50.67|39.26|
|128/3|70.50|67.88|68.58|69.69|66.09|68.94|70.03|66.29|45.73|
|128/8|74.28|69.85|66.31|66.40|66.37|66.52|66.48|66.58|45.81|

N64 wide cached atom16 improves the matching cached atom32 and uncached atom16 candidates in this screen. Middle atom16 repair4 is essentially tied with uncached atom16. N128 atom32 remains faster than each matching cached atom16 route in both screened conditions; no N128 full promotion. Peaks unchanged relative to matching routes: uncached115,712/303,104 bytes,cache117,248/307,200; CST reserved6,291,456. No physical DRAM/cache-residency inference.

N64 full rho1.25/3/8/mixed independent repeats queued: `l4job-cebf9a04d820407d9cdb8af8819ca8e1` and `l4job-9510bc7d2b6a4ca092935cc85cd5d69e`,source4aa6823f. Five CST routes (copy8,uncached atom16split2,three cached atom16) +dense; repeat2 reverses case/plan order. No new GPU or parallel consumer allocated.

All4 screening artifacts have verified byte-identical DB export and idempotent reimport. CPU CI at a9a3cb9c passed. Full repeat jobs remain queued.

## Completed N64 full independent repeat1 + repeat2

Both full jobs succeeded,source4aa6823f.40 full-shape FP64 comparisons,197 runtime/hash/archive proofs,initialrho>1 and changing widths,CST Parameter/all-plan input bytes match per condition and both runs.21samples per execution,second reverses case/plan order. Each cell median of2 execution medians(us). All8 full measurement artifacts plus4 screening artifacts have verified byte-identical DB export/idempotent reimport.

|N64 rho|copy8|uncached atom16split2|gather atom16|repair4 atom16|repair8 atom16|dense reference|
|---|---:|---:|---:|---:|---:|---:|
|1.25|55.46|50.12|49.36|49.57|49.61|39.39|
|3|59.49|51.56|51.94|51.16|51.81|39.32|
|8|57.31|51.36|50.76|50.33|50.66|39.26|
|mixed|63.24|56.92|56.99|56.58|56.95|39.33|

Narrow gather improves matched uncached atom16 in both runs (about0.76us aggregate);wide repair4 improves in both runs (about1.03us). Middle/mixed repair4 aggregate medians are slightly lower, but first-run direction is slower and second-run faster; no stable advantage claimed. Caches keep peak117,248 versus115,712 bytes uncached (1536bytes extra). All CST reserved6,291,456;dense allocated34,179,584/reserved48,234,496 bytes includes workspace. Dense is a separately initialized nn.Linear performance reference with identical dimensions/input/target/loss/capture settings,not the same initial CST matrix or source optimizer trajectory.

Only the stable narrow/wide N64 cases are selected for short G4 follow-up from unchanged validated runtime/benchmark source4aa6823f. The selected GPU test filter collects18 tests for gather/repair4;timing comparescopy8,uncached atom16split2,gather atom16,repair4 atom16+dense,21samples/two independent runs. G4 allocation waits for L4 owned-stop/server confirmation.

Frozen G4 source`4aa6823f2c475eaa2689ae0befaeb9ce778b84c4`; queued check/repeat1/repeat2 `colabjob-5da92cd7004b475f96cc0240f90c98c8`, `colabjob-5f462a23e7a9453b8de35957262bd48b`, `colabjob-66fc0ef89caf4ab3a03bc3fca1401015`. Only N64rho1.25/8 selected after consistent paired L4 gain; no new G4 allocation until current L4 drains/stops.

## Completed G4 narrow/wide independent repeats

Two selected executions on RTX PRO 6000 Blackwell Server Edition completed after an18-GPU-test correctness suite (40.97s,10 unrelated tests deselected). Source4aa6823f,197 runtime hashes and source/result archives match; within-G4 initial CST Parameter/input bytes agree across routes and repeats, decoded initial rho>1 and widths evolve. Four retrieved measurement artifacts have verified byte-identical DB export/idempotent reimport; family total16 artifacts. Each cell below is median of two execution medians(us),21 samples each, repeat2 reverses rho/plan order.

|N64 rho|copy8|uncached atom16 split2|gather atom16|repair4 atom16|dense|
|---|---:|---:|---:|---:|---:|
|1.25|52.23|48.91|48.36|48.28|33.83|
|8|52.39|49.14|49.15|49.29|33.96|

Narrow gather and repair4 improve uncached in both executions, but their rank changes between runs. Wide cached routes change direction vsuncached and are effectively tied/slightly slower in aggregate; L4’s stable wide advantage is not reproduced on G4. No universal cache default. Cache allocated117,248 B vsuncached115,712 B, CST reserved6,291,456 B; dense allocated34,179,584 B/reserved48,234,496 B. All source-position/normalization derivatives retained; no new global H and no measured L2-residency/DRAM claim.

Supervisor25329 exited0, owned G4 session6202ee0da80b stopped with Session terminated/server No active sessions found and all slots stopped. Only after this confirmation, one final short L4-validated N64 tight-histogram comparison was submitted on G4.
