# Sphere kernel attribution before the cell-support experiment

The user prioritizes the existing Sphere operator over Strip+Torus. Its contract remains two independent intrinsic S² Explicit charts and separable Polar/Triweight, with separate complete-chart L2 floors. Chart kinds are Explicit, Product and Strip; Sphere/Torus describe Geometry, and profile_product describes Kernel composition. ProductChart is a site/coordinate layout within one geometry, not a declaration of S²×S². Grouping the existing pair behind a single-chart API would require preserving both radii, distances, normalization domains and blockwise centre updates; Sphere(4) instead describes S⁴. No chart/API change is part of this speed experiment.

## Fixed diagnostic scope

Four isolated workers: B32, N1024/N2048, five-percent atoms, seed41, initial sigma3/8, IEEE FP32, MSE/AdamW. Sigma3 uses the previously validated G8/T16/CAP64/int32 W route; sigma8 uses bounded H4096. Each unchanged scaling_comparison.worker validates independent FP64 Y, dX and every atom gradient before and after24 actual updates, then reports complete-step time and capture/replay allocated/reserved peaks.

The driver retains Python references to that primary instance without allocating extra GPU tensors in its measurement window. After the primary, it copies the actual24-step model, task and optimizer state. Parameters, moments, clock, task and geometry match exactly. Diagnostic Graph recording executes no update; one initial replay plus five profiler replays advances the copy to30. The primary remains at24 with unchanged parameters/moments. CUDA kernel self times below come from that separate instrumented Graph: they are descriptive and must not be summed into or subtracted from complete-step timings. This cohort selects an optimization target, not a performance winner.

| N | sigma | complete step ms | allocated MiB | principal diagnostic kernels, ms/replay |
|---:|---:|---:|---:|---|
| 1024 | 3 | 4.175070 | 105.662 | pack 0.912, assemble 1.545, vjp 1.453 |
| 2048 | 3 | 22.756552 | 323.585 | pack 7.356, assemble 6.540, vjp 6.936 |
| 1024 | 8 | 9.961442 | 148.833 | center_vjp 3.001, profiles 1.911 |
| 2048 | 8 | 93.612000 | 299.432 | center_vjp 32.936, profiles 18.780 |

At narrow N2048, full-site packing is the largest named diagnostic kernel family; W assembly and VJP remain substantial. The next fixed experiment therefore replaces support search with a fresh3D index of the actual normalized Points, while retaining the W/VJP mathematics. At broad widths, centre VJP and repeated full profiles are distinct targets; the narrow candidate is not a broad recommendation. Optimizer work is not the leading cost in these traces. Neither profiler values nor CPU candidate-count estimates establish a speed improvement.

## Provenance and failure history

Source commit: `688cdab5fc51498321471d1b956a3292504ae621`. Fixed diagnostic driver SHA256: `56578f605e6b1c83f7ba9521ace46e158d1475ee635000576e4c9468a4c4bafa`. Completed pool job: `l4job-26b904e77517448281480c5560b59b88`; all4 workers PASS, no pending workers, elapsed driver142.611s within the unchanged900s outer/875s driver budget. Actual GPU: `NVIDIA L4, GPU-4b0c6cc5-2627-6a2a-1752-0b1912b66a64, 580.82.07, 23034 MiB`. Runtime: Python3.13.15, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0. Source archive SHA256: `8df4bd8f41074151f39d04e305a8da89ad9c8a9590b657ebdb6990f2ee0c40a6`; result archive SHA256: `3a98430339b2349c74c5b598cdb0243864692c78e596559856294cacd395bbc0`.

Preceding job l4job-5244aa8181424e9fb48011cf313d468e failed at the independent repeated-training p24 check for N2048/sigma3: maxabs3.814697265625e-6 exceeded its predeclared2e-6 threshold (relL2 1.2155120057897763e-9). It did not finish the four-worker cohort; no partial performance selection was accepted. The correction copies the actual validated primary24 state, as established phase_diagnostics does, instead of repeating learning independently. It changes no oracle/optimizer tolerance, condition, deadline or kernel source. V1 source, its18 job files and all logs are retained. V2's36 job files, full Chrome traces, per-event/kernel averages and complete-worker JSONs were copied and SHA256-verified. All owned GPU slots stopped after each batch.

Frozen scripts/specs/source/results are retained in the host-wide pool job archives and root ignored output/sphere-kernel-profile-20261009/jobs/. The ignored driver copy is output/sphere-next-profile/diagnostic_driver_v2.py in the sphere-baseline worktree. Reproduction uses the exact main source and that hash-verified driver through the shared pool with args[] and timeout900; a direct owner-independent GPU run can invoke it with CST_JOB_OUTPUT set on an actual L4. Its four complete workers call the existing scaling_comparison module. Raw traces and snapshots are not committed.
