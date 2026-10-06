# 2026-10-07: CTA-local prefix-maximum owner search

Branch kernel/owner-prefix-search based on combined PR44. Exact normalized triweight/source Polar contract unchanged. This prototype reduces repeated range exploration, independently of key cache reuse.

For each current band, sorted support starts l[a] are monotone; support ends h[a] can be nonmonotone. Compute m[a]=max_{k<=a, same band} h[k] in the order-building CTA. For owner [j0,j1), begin=first a with m[a]>j0, end=first a with l[a]>=j1, clamp begin<=end. The first relevant atom is retained and every overlapping atom satisfies a<end. Trailing noncontributors can remain in the safe envelope; existing exact site-factor evaluation eliminates them. This does not assume width/geometry makes h monotone. Singleton buckets remain separate.

Prefix maxima and sorted interval arrays stay CTA-local; no global prefix/H buffer and no extra launch. Current sort CTA computes all owner bounds by bounded bisections into these arrays. The subsequent parallel-copy CTA only copies its disjoint physical segment; it no longer recomputes all13-field intervals over all C for every owner. API uses Triton associative_scan with JIT maximum reducer ([official documentation](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html)). Compiler registers/shared/spills and complete-step time/peak decide whether this theoretical search reduction helps. Cache+prefix combination is not implemented.

Two routes keep output16/input16, atom chunk32, owner split4, batch16 rows: parameter atom32/batch split2 control variant and atom16/warps4/batch split2. Controls copy8, param2, atom16/warps4/batch2, same dense operator. N64/N128 B32 A204/A819, seed41, initial decodedrho>1, FP32 IEEE/TF32off, production fused AdamW/Polar with live sigma. 21 samples/execution; rho3/8 screen first.

Host first CPU run caught wrong test expectation atom16 for atom32 route (885 passed/1 failed); fixture assertion corrected before submission. 8 prepare/check and isolated wheel/sdist build passed. GPU tests check full normalized support/snapshot coverage, slices/B1/32/64, independent FP64 Y/dX/all atom gradients/positions,20 reference optimizer updates/moments/steps N64/N128 and old backward. Runtime and speed remain unverified. Ignored evidence owner-prefix-search-20261007 preserved.

Declaration correction rerun1 passed/18 GPU skipped; changed-file ruff/diff checks pass. Frozen source633b109b; PR45. GPU check `l4job-46b3812d2f394c05a89ec9a938b32980`, N64 screen `l4job-9116aca8479a4ba1a23afe385b1c493c`, N128 screen `l4job-040d0532f0344732aae36530118b9b62`. These isolated source jobs follow combined checks/screens on supervisor65855.

## L4 verified negative result (1 execution)

19 tests passed (18 GPU+declaration),33.70s. Source/result archives and197 committed/submitted/worker runtime files match. All20 full-shape FP64 comparisons/initialrho>1/nonzero positions/within-job initial Parameter and input bytes incl dense passed. No prefix route promoted.

|N/rho|copy8|param2|atom16split2/4warps|prefix param2|prefix atom16split2/4warps|dense|
|---|---:|---:|---:|---:|---:|---:|
|64/3|58.99|52.73|51.20|55.90|54.50|39.29|
|64/8|56.63|52.94|51.09|55.53|53.90|39.24|
|128/3|70.14|67.76|67.78|71.56|71.69|45.78|
|128/8|73.77|70.11|70.08|73.27|73.98|45.65|

Complete-step median us. Peak allocated115712/303104 bytes and reserved6291456 unchanged. Separate layout event64rho3 8.19→11.26us,128rho3 13.31→17.41us. Layout compiler64prefix32regs/0spill/shared1024 vs27/0/1024,128prefix48/0/4096 vs46/0/4096. Copy-only CTA drops64regs32→24 and12856→40, shared16→0. No measured spill increase; serial prefix/bisection work in two order CTAs adds latency even though other CTAs avoid interval scan. Hardware DRAM traffic not measured.

Raw correct/negative evidence preserved.4 measurement artifacts are being independently imported/exported/idempotence-checked in DB. CPU CI f5ac36a4 SUCCESS. Next distinct prototype computes conservative support-span bounds with position histogram prefix counts and direct lookup; this result is not replaced.
