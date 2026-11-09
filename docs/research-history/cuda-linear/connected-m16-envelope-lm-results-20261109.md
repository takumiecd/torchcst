# Ordered M16 envelope qualification and cost

**English** | [日本語](connected-m16-envelope-lm-results-20261109.ja.md)

The separate ordered physical/envelope schedules pass83 GPU tests, including
rectangular independent quadrature with the raw map chain, three bitwise repeats
at mixed radii and the legacy regressions. Actual width512/2048,A8192 full-map
oracles pass initially and after976 task updates. All four task arms pass the
original exact serialized model/optimizer next-update and hook-invariance gates.
No tolerance relaxation, public dispatcher/default or deterministic/Graph support.
The preceding unordered schedule remains separately identified and preserved.

The [short LM task study](https://github.com/takumiecd/cst-research/pull/36) finds
radius prediction fidelity improves substantially while the one-seed1M-token
quality gap remains. Its read-only signal analysis and interpretation belong in
cst-research; this operator record is not a quality-adoption claim.

## Matched complete isolated Linear steps

RTX4090 CC8.9, Torch2.9.1+cu128/CUDA12.8/Triton3.5.1, IEEE FP32, B256,N8192,
A65536, five warmups/fifteen samples, one independent run. Existing Linear
runner/physical fixture includes forward,MSE,backward including dX,fused AdamW
and radius/center restoration; transposes/geometry/recomputation are inside timing,
compilation and proof are outside. Dense uses its standard initialization as a
compute/memory control. Eager only, no Graph.

| Plan | Median ms | Peak allocated / reserved MiB |
|---|---:|---:|
| preceding coalesced | 12.810783 | 525.502930 / 560 |
| ordered direct levels | 12.922911 | 557.502930 / 592 |
| Dense | 4.908139 | 1072.251953 / 1342 |

The ordered schedule is about0.9% slower in this run with32MiB extra scratch,
matching4*B*N FP32 cells. It is2.63x slower than Dense with52.0% of its allocated
peak. These are allocator peaks including model/gradients/Adam state/temporaries,
not total GPU process usage. This is not a repeated speedup claim or another
GPU/shape qualification.

Full A65536/B256 VJP versus coalesced: Y/dX max absolute1.193e-7 and relative
L2<=5.377e-8; physical dP/all four columns are bitwise equal on this fixture.
Independent20-Atom boundary/wide quadrature and four-step AdamW state gates also
pass in the existing runner. A separate task checkpoint audit verifies all eight
milestone file hashes, four final model Parameter digests, finite FP32 moments,
and clocks976 (DST973). Checkpoints remain on NFS; raw copies/logs stay ignored.

## Source and reproduction

Task execution used operator00aae6afaf3615da6bccd986c75c14cd7b2ca64b.
Benchmark registration used752240cd94ab8f9cc8b95ff3d4c9cebc389cd8c8; both have
identical runtime tree0fe7b7214529217e257df8c0369f31a2905dbe77.
Job: connected-m16-ordered-4090-20261109-01. Source archive
`da4490ecd41b2526241314c141d3198dabdfab7ae8ab4cf1e2191fa43873a5fd`;
results `6ec0c5e958cb6c1b9163cf8b49ae064ea1a9010320d807bb2d1f08bd827d97c1`.
All1370 recorded source hash comparisons match the frozen archive. Both local
retrieval and NFS evidence are retained. GPU idle0%/40MiB after completion; Hub
left running, no unrelated processes or data changed.

```bash
python -m benchmarks.cuda.linear.run \
 --plans benchmarks/cuda/linear/plans-connected-radius-ordered.json \
 --case benchmarks/cuda/linear/cases/connected-radius-8192-r256-c8-ordered.json \
 --polar-update radius --source-commit 752240cd94ab8f9cc8b95ff3d4c9cebc389cd8c8 \
 --output output/ordered-step.json
```

[Curated result](connected-m16-envelope-lm-results-20261109.json).
Local CPU runtime suite2255passed2654skipped,19warnings; subsequent catalog
roundtrip and declaration check pass separately. CUDA skips are not GPU evidence.
Graph,higher-order gradients,other GPUs and public default adoption remain untested.
