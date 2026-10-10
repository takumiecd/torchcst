# Regular Product matrix-free comparison and update-clock audit

2026-10-10. This experiment uses a regular two-axis Euclidean Product of positive
equally spaced Line axes (same spacing), normalized Triweight profile product and
canonical Polar `[z0,z1,co,ci]`. It preserves every atom/site contribution, detached
live activity-dependent task width and whole-product normalization
`D=max(||u|| ||v||,floor)`. It does not replace this with two independent floors.

The bounded CSR route removes W/dW and full-K H/G. Preparation saves13 exact FP32
fields per atom, including norms, centers and positive support intervals. H/G and
owner lists are bounded by atom chunk; normal atoms belong to at most4 owner lists,
and wider supports use a disjoint complete overflow list. No support is truncated.
Temporary owner order never changes Parameter/Adam moment order. Integer atomic
CSR insertion may change floating reduction order; correctness is tolerance-based.

Runtime is preserved on `kernel/product-support-direct` at
`198c835e924f385af0bc42a73fe3fb0b81588861`. The comparison fails the joint speed/memory
gate at both sizes and is not promoted to main or public dispatch. The fused source
VJP follow-up remains on `kernel/product-fused-backward`; see its result below.

## Supplemental clock correction

The old driver performed one eager step plus22 Graph replays, counted recording
as another update, and hard-coded24. Recording did not execute an update. The raw
JSON therefore certifies23 actual updates, not the planned24 checkpoint. Frozen
drivers of the prior bounded/cache jobs likewise have this defect. The historical
[bounded/cache note](20261010-product-bounded-support.md) is corrected to23; its
planned24 requirement, raw evidence, initial gates and negative dispositions remain.

The repaired driver performs23 replays plus one eager step, reads AdamW's actual
step counter and asserts24. It runs oracle-only for the old CSR source, without
repeating any timing or changing a runtime/fixture/tolerance. This supplemental
model is outside primary timing/peak measurements. Primary runner warmup5 plus
21 eager measurements plus5 side warmups plus21 replays means52 real updates;
Graph recording adds none. Separate phase graphs continue evolving widths.

## Validation and reproduction

Measured198c835 GPU gate:151 tests passed with no skips. Includes chunk tails,
empty/singleton/floor/precision fallback, wide overflow, retained snapshots,
requested gradients,20 replay optimizer moments/clock and live widths. Separate
B1/B64 x group4/8 x full/support audit job721506bcd58e4379859aadb4785fa56f passes
all8 full FP64 Y/dX/dP cases. Initial full-site/all-atom FP64 gates for all4 plans
require maxabs AND relative-L2<=4e-4. Host suite1419 passed/2687 skipped; wheel/sdist
build and runtime byte checks passed. Host skips do not count as GPU validation.

On the preserved source, catalog `benchmarks/cuda/linear/plans-regular-product-matrix-free.json`
and cases `regular-product-matrix-free-{2048,8192}-rho3.json` reproduce with
`python -m benchmarks.cuda.linear.run --plans <catalog> --case <case> --polar-update fused
--phase-diagnostics --source-commit 198c835e924f385af0bc42a73fe3fb0b81588861 --output <ignored-path>`.
The exact pool driver is stored as `__pool_driver__.py` in each source archive.
The files named above are research-branch files; no negative runtime is added here.
All pool evidence remains under `~/.local/state/colab-l4-pool/jobs/<job-id>/`;
ignored `output/` companions retain source proof, driver and build artifacts.

## CSR V1 small and large raw complete-step results

Measured commit198c835e924f385af0bc42a73fe3fb0b81588861, B32, seed41, K=floor(.05N²), initialrho3, FP32 IEEE/TF32off, live widths and fused AdamW/Polar updates. Hardware NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0. Each size is one independent timing run;21 timed samples are not independent runs. All raw archive/manifest hashes checked.

| N | Route | Complete Graph step ms | Peak allocated bytes | Peak reserved bytes |
| --- | --- | ---: | ---: | ---: |
| 2048 | native-g8-p8 | 1.105304 | 66493952 | 163577856 |
| 2048 | torch-g8-p8 | 0.928656 | 99735040 | 205520896 |
| 2048 | csr-c64k | 2.307620 | 52078592 | 140509184 |
| 2048 | csr-c256k | 2.609106 | 92711424 | 249561088 |
| 2048 | dense | 0.588896 | 102238720 | 148897792 |
| 8192 | native-g8-p8 | 42.411351 | 1041040896 | 2501902336 |
| 8192 | torch-g8-p8 | 39.509333 | 1075119616 | 2522873856 |
| 8192 | csr-c64k | 54.069071 | 526459392 | 1361051648 |
| 8192 | csr-c256k | 51.648795 | 580723200 | 1495269376 |
| 8192 | dense | 11.682435 | 1112017408 | 1642070016 |

Qualification remains >3% faster with no allocated increase, or >=5% lower allocated with <=3% slower step. Each CSR candidate must be reported against both valid controls. Dense is a different parameterization comparison.

| N / candidate | Time change vs native / Torch | Allocated reduction vs native / Torch | Joint result |
| --- | ---: | ---: | --- |
| 2048 / csr-c64k | +108.78% / +148.49% | +21.68% / +47.78% | REJECT |
| 2048 / csr-c256k | +136.05% / +180.96% | -39.43% / +7.04% | REJECT |
| 8192 / csr-c64k | +27.49% / +36.85% | +49.43% / +51.03% | REJECT |
| 8192 / csr-c256k | +21.78% / +30.73% | +44.22% / +45.99% | REJECT |

N2048: C64K cuts peak but more than doubles time; C256K is slower and increases peak against native. N8192: both chunks materially reduce peak, but slow against both controls by well over3%. Neither size qualifies; do not submit an independent reverse-order performance confirmation for this losing V1 freeze. Preserve the unfavorable small and large measurements jointly. No process peak or DRAM-traffic claim follows from allocator/time data.

## CSR frozen source / retrieval proof

Both V1 archives contain identical1594 file hashes, including290 `src/` files. The corrected clock-only archives contain exactly the same1593 non-driver file hashes; only `__pool_driver__.py` differs. Frozen V1 driver SHA256 fa6be9a6433cf83244cb2ce8d687d9bf4489e731648091ae958d820707897a7a; corrected driver34a7603b2f037bfaefc3b5e44935d17654d7a97cb4e5583866d034d349dc6e17. No runtime/fixture/threshold difference was introduced by the clock repair.

| Job / scope | Source archive SHA256 | Receipt result archive SHA256 | State observed |
| --- | --- | --- | --- |
| l4job-b6d848f6362d4833b7f78f3c43b3b415 | 5034a1c02d4a80b72ec264e415734295ad31befb99dfb5cd820a86f2631ac959 | 490131545df73d2ab91f29af9a7cf6db409f650ed5059a1f69bb13f983f0969e | completed returncode0, timed_outfalse |
| l4job-adfcb15929574cc3877db8dbd8742f09 | d737af1d50c6e75f7c2e699992603a38473e05cbe71537b0896ccf0689fd77fa | d4d69a5f26f77956da46603dec32f27b2eb0283787bce99fee45dc64e271b876 | completed returncode0, timed_outfalse |
| l4job-2cddd6e7dedc492e9e746f39741e683b | ae9439a565fc2cdc15111f96bbf332636749fc938a5af78b60ad58cb11dd69c3 | 936e67a104d5cb7efe6febbdc94b9e8fd5642432cc6320838abab589c3bb5981 | completed returncode0, timed_outfalse |
| l4job-6f83d08dcb974a9387ddb7a2b314910d | 698c31af4900089c50938eb8da7bbfb44dcd7719eba0d07cd369a97435e8f3c1 | not yet available | no retrieved result yet; no PASS claim |

All available result archive hashes match receipts; every extracted result manifest file hash matches its on-disk bytes. Archive source hashes match spec and result metadata. Raw JSON retains its original hard-coded24 fields and original finalParameter hashes.

## Corrected clock-only oracle evidence

N2048 job2cddd6e7dedc492e9e746f39741e683b runs `--oracle-only`: no timing or memory comparison was rerun. All4 plans pass full-site Y/dX/all209715 atom cotangents with observed AdamW clock24; all209715 widths changed. Maximum Ymaxabs across plans4.4800361038710435e-05, dX7.586370937375832e-10, dP9.625540528480567e-11, and max relativeL2 7.574498944926368e-07, all below the unchanged maxabs AND relativeL2 gate4e-4.

N8192 job6f83d08dcb974a9387ddb7a2b314910d has the same corrected driver and1593 unchanged non-driver hashes. No retrieved result existed at this audit;24-update large correctness remains pending. Its eventual result must be appended independently, without replacing either V1 timing run.

