# Exact grouped Sphere W/VJP: conditional speed and memory gains

This checkpoint preserves the exact existing Sphere contract: two intrinsic S² charts, Polar parameterization, separable Triweight profiles, full per-chart L2 normalization with floor 1e-6, live widths and all six atom gradients. Runtime source remains byte-identical to the 280-test numeric gate (`95beaac`). The benchmark-only composition is `aa9a91f3b68feff96d6ae6948cd4adc7de2ff733`.

## Frozen protocol and evidence

Primary job: `l4job-dfce72e6185e4814b6b590b15a23b3b4`, L4 GPU `e6e685ce-4e06-ea58-d7ec-25eaec7a7fbd`, Torch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0. All 24 isolated workers passed in 635.87 driver seconds; shared GPU regressions were 30 passed with zero skips. Pool return code is zero, with no timeout. All 70 downloaded file hashes and the copied manifest were independently verified.

- Primary cohort SHA256: `f172241bb5b00fdc53b19c440cf6596b1a9c225ee6d2a7d7ef57590faa194b97`.
- Pool manifest SHA256: `f776e393cff99ed0bb0e8fc592c4985f3b24e1881e602c58b0ca13bfcc2840cc`.
- Driver SHA256: `8ad59c07a7852963dc31000246a5d1ed6801ebc0512ede70f8a183d276647d05`.
- Strict inverse-selection SHA256: `26c101f770dc493a85acac661e11be39c8f8b6081d5d8687372d0697b7049edd`.

B32, square N=1024/2048, five-percent atoms, seed41, initial sigma3/8, IEEE FP32, identical input/target/fixed cotangent, MSE loss, fused AdamW lr1e-4 and decay0.01. Full initial and updated FP64 oracles use all atoms and all sites. Graph execution actually performs two eager warmups, one initial replay, and 21 timed replays: exactly24 updates. Separate fresh diagnostics execute six updates from step24 to30; their phase timings are descriptive and are not added to/subtracted from complete-step results. Public optimizer guards remain outside this explicit research Graph protocol.

CAP64 applies to every narrow candidate; CAP256 applies to every broad candidate. Narrow control is fused W64/patch32. Broad control is bounded H4096. All four fixed recipes and both broad conditions are retained, without retuning or filtering. Capture/replay allocated and reserved peaks include the model, optimizer, task tensors, fixed dy and Graph allocations; initial/updated oracle scratch and later phase diagnostics are outside the primary peak. GPU process memory and physical cache residency are unmeasured.

## Primary measurements

| N | sigma | recipe | complete step ms | gain vs control | allocated MiB | allocated reduction | reserved MiB | strict inverse |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1024 | 3 | control | 4.77552 | — | 105.662 | — | 166.000 | — |
| 1024 | 3 | 64-g4-t16 | 4.59869 | 3.70% | 105.662 | 0.00% | 166.000 | yes |
| 1024 | 3 | 64-g8-t16 | 4.18609 | 12.34% | 105.662 | 0.00% | 166.000 | yes |
| 1024 | 3 | 64-g4-t32 | 5.45146 | -14.15% | 105.662 | 0.00% | 166.000 | no |
| 1024 | 3 | 64-g4-t16-i16 | 4.70024 | 1.58% | 92.862 | 12.11% | 158.000 | yes |
| 1024 | 3 | dense Graph | 0.08537 | — | 32.877 | — | 86.000 | — |
| 2048 | 3 | control | 24.51412 | — | 323.585 | — | 390.000 | — |
| 2048 | 3 | 64-g4-t16 | 25.13544 | -2.53% | 323.585 | 0.00% | 390.000 | no |
| 2048 | 3 | 64-g8-t16 | 22.82009 | 6.91% | 323.585 | 0.00% | 390.000 | yes |
| 2048 | 3 | 64-g4-t32 | 27.32070 | -11.45% | 323.585 | 0.00% | 390.000 | no |
| 2048 | 3 | 64-g4-t16-i16 | 25.40118 | -3.62% | 271.585 | 16.07% | 338.000 | no |
| 2048 | 3 | dense Graph | 0.59390 | — | 81.502 | — | 106.000 | — |
| 1024 | 8 | control | 10.10943 | — | 148.833 | — | 206.000 | — |
| 1024 | 8 | 256-g4-t16 | 114.61956 | -1033.79% | 262.188 | -76.16% | 318.000 | no |
| 1024 | 8 | 256-g8-t16 | 102.23829 | -911.32% | 262.188 | -76.16% | 318.000 | no |
| 1024 | 8 | 256-g4-t32 | 107.82477 | -966.58% | 262.188 | -76.16% | 318.000 | no |
| 1024 | 8 | 256-g4-t16-i16 | 114.26653 | -1030.30% | 210.188 | -41.22% | 266.000 | no |
| 1024 | 8 | dense Graph | 0.08446 | — | 32.877 | — | 86.000 | — |
| 2048 | 8 | control | 95.00990 | — | 299.432 | — | 390.000 | — |
| 2048 | 8 | 256-g4-t16 | 510.91935 | -437.75% | 934.784 | -212.19% | 1006.000 | no |
| 2048 | 8 | 256-g8-t16 | 458.78674 | -382.88% | 934.784 | -212.19% | 1006.000 | no |
| 2048 | 8 | 256-g4-t32 | 473.97183 | -398.87% | 934.784 | -212.19% | 1006.000 | no |
| 2048 | 8 | 256-g4-t16-i16 | 510.26753 | -437.07% | 729.984 | -143.79% | 802.000 | no |
| 2048 | 8 | dense Graph | 0.59363 | — | 81.502 | — | 106.000 | — |

The strict gate was predeclared: more than3% complete-step speed gain, or at least5% allocated reduction with at most3% time regression. Only N1024/sigma3 G4T16/i32, G8T16/i32, G4T16/i16 and N2048/sigma3 G8T16/i32 qualified. N2048/i16 saves16.07% allocated but is3.62% slower, so it does not qualify. G4T32 fails both narrow conditions. All broad candidates fail both gates, including i16.

## Numerical evidence

Across initial and updated independent full-A FP64 checks in every CST worker, maximum absolute errors are Y7.1783e-6, dX9.1478e-6 and all-dP5.5012e-5. Maximum relative-L2 errors are Y1.2113e-6, dX1.1193e-6 and all-dP1.0749e-6. All are below the unchanged4e-4 gates. All 296 source hashes reported by workers match the local frozen source. Initial parameter, geometry declaration, buffer/parameter dtype-shape/content, input/target and fixed-dy hashes match within each condition; live widths evolve. Every actual candidate plan equals its frozen catalog declaration.

## Mechanism and negative dispositions

Grouped execution lowers the launch grid and keeps independent atoms in the leading vector dimension while reducing only site axes. It leaves full-site support scans and the number of W atomic additions unchanged. The int16 option is lossless storage for site IDs with explicit int32 address widening; raw profiles and arithmetic remain FP32. The narrow primary G8 gains are consistent with improved grouped W/VJP execution, but are not an explanation of every component cost.

The broad slowdown occurs with zero independently observed support overflow: initial maximum chart supports are250/237 for N1024 and240/231 for N2048, within CAP256, and remain within capacity after24 updates. Thus the observed broad failure is not explained by overflow fallback. Broad grouped G8 separate diagnostics show forward/backward70.30/30.78ms atN1024 and304.02/144.03ms atN2048, versus H4096 control2.71/6.53ms and27.03/67.15ms. The likely mechanism is the Cartesian product of broad supports and W atomic assembly/VJP; no lower-level profiler proves atomic contention or register spills here. Bounded H/G should remain the broad comparison route.

Narrow G8 separate diagnostics show forward/backward2.51/1.47ms atN1024 and14.60/7.29ms atN2048. Research optimizer diagnostics are roughly0.06/0.13ms, so optimizer work is not the dominant measured phase. Full support scans remain an unoptimized structural cost. No claim of approaching the historical Euclidean one-millisecond result follows from these Sphere measurements.

Broad recipes and G4T32 are negative performance outcomes and must not receive a recommendation or automatic selection. The exact runtime branch and all raw data remain recoverable; the research catalog can preserve the negative controls. Public dispatch is unchanged.

## Independent inverse and disposition

Job `l4job-a43439f3314c48409de0ce0a1ebe02f0` passed all eight isolated workers in158.72 driver seconds on independent L4 GPU `d51ffe33-358c-65c9-3d1b-81d2dfbe7560`, with30 shared GPU regressions and zero skips. It used identical runtime/benchmark file hashes, actual model/task/geometry initialization and declared recipes. The central driver reversed the original qualified case/recipe order. All26 downloaded file hashes and the copied manifest were independently verified. No recipe, tolerance, capacity or timing scope changed.

- Inverse cohort SHA256: `f550515a5da046a133f957fc62fc35311e4cccd8e3e9955a1b804684b0fd63a6`.
- Inverse manifest SHA256: `6055079c86029b98376d8a67d7d05af6b7ce5a68f311507e9c17e059f30eff79`.

| N | sigma | recipe | inverse control ms | inverse candidate ms | inverse speed gain | inverse allocated reduction | both runs pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 1024 | 3 | 64-g4-t16 | 4.85024 | 4.87455 | -0.50% | 0.00% | no |
| 1024 | 3 | 64-g8-t16 | 4.85024 | 4.32457 | 10.84% | 0.00% | yes |
| 1024 | 3 | 64-g4-t16-i16 | 4.85024 | 4.75183 | 2.03% | 12.11% | yes |
| 2048 | 3 | 64-g8-t16 | 24.74828 | 22.74274 | 8.10% | 0.00% | yes |

The accepted research conditions are G8/patch16/int32 at N1024 and N2048 with initial sigma3, and G4/patch16/int16 at N1024 with initial sigma3 when allocated memory is the objective. G8 saves10.84–12.34% complete-step time atN1024 and6.91–8.10% atN2048 across the two runs. The N1024 int16 route reduces allocated peak12.11% in both runs, with1.58–2.03% faster complete steps. Narrow G4/int32 fails independent confirmation: the inverse is0.50% slower, despite the primary3.70% gain. It receives no performance recommendation.

This is conditional research integration. The registry/catalog preserve explicit negative controls, but there is no automatic policy that selects broad grouped W or the rejected narrow G4/int32/G4T32 recipes. Public dispatch is unchanged; arbitrary shape/width extrapolation is unvalidated. Dense Graph remains substantially faster in the reported conditions. The Sphere contract is preserved, and the exact grouped source can serve subsequent support-search research without treating grouping alone as a complete solution.

The pre-GPU declaration is retained in numeric checkpoint `95beaac122e8a413dba70daf0c00c5be81f05d46`; the measured source is `aa9a91f3b68feff96d6ae6948cd4adc7de2ff733`. Raw primary/inverse output, copied manifests, strict selection and independent analyses are archived in ignored `output/sphere-grouped-weight/`. The local recovery branch `kernel/sphere-grouped-measured-aa9` retains the measured source before integration rebasing. Owner pool evidence remains keyed by the job IDs above.

## Reproduction

The research driver uses `benchmarks.cuda.linear.scaling_comparison`, rather than the public eager optimizer path. Example qualifying worker:

```sh
python -m benchmarks.cuda.linear.scaling_comparison \
  --geometry sphere --size 1024 --sigma 3 --mode research_graph \
  --linear-plan <G8/patch16/CAP64/int32 plan JSON> --phases \
  --output output/sphere-grouped-worker.json
```

Use the frozen catalog `benchmarks/cuda/linear/plans-sphere-polar-grouped.json` and all four declared Cases through `tools.kernel_dev check` for public runner declarations. To reproduce the comparison, run every fixed primary case/recipe/control/dense worker in isolation, then reverse the strict qualified condition order with unchanged source and protocol. Individual phase timings or a single worker cannot replace this paired complete-step gate.
