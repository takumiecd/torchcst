# Radius fraction API and bounded FFN speed probe

English (canonical) | [日本語](radius-fraction-speed-20261109.ja.md)

Current API names are `radius_fraction_init`, initialization keyword/report
`radius_fraction`, and `field_lrs={"radius_fraction": ...}`. The public q keyword,
report field and aliases were removed before merge at the owner's request.
The formula may still use q as a mathematical symbol. Historical frozen sources
and evidence retain their original names; they are not checkpoint migrations.

## Fixed-state measurements

RTX6000 Ada Generation48GB, Torch2.9.1+cu128, CUDA12.8, FP32, TF32 off;
8192 Atoms,256 rows, width512/expansion4, tau0.03, initial fraction0,
radius[0.5,50]. All arms start from byte-matched physical rows/weights and inputs.
The existing `benchmarks.cuda.linear.radius_cell_mass.train` performs zero_grad,
x.grad clear, MSE forward/backward and AdamW. Timings synchronize before/after
complete steps; setup is excluded. Each arm runs in a fresh process,5 warmups
and21 timed steps. Two reversed orders are reported separately, not averaged.

| Direction / order | Dense ms | Physical core ms | Envelope ms | Envelope + public CST policy ms |
| --- | ---: | ---: | ---: | ---: |
| 512->2048 / forward | 0.218 | 3.274 | 3.966 | 3.983 |
| 512->2048 / reverse | 0.310 | 3.277 | 3.980 | 3.975 |
| 2048->512 / forward | 0.210 | 3.259 | 3.975 | 3.959 |
| 2048->512 / reverse | 0.202 | 3.272 | 3.972 | 3.998 |

With the same plain AdamW, the envelope adds21.1–22.0% over the physical core.
This includes the Python map, its gradient chain and nested dispatch; it does
not identify a single bottleneck or compare directly with original Polar's
detached-radius gradient/update semantics. Dense is faster in this small cohort.

| Direction | Dense allocated MiB | Physical allocated MiB | Envelope allocated MiB |
| --- | ---: | ---: | ---: |
| 512->2048 | 45.25 | 66.94 | 69.50 |
| 2048->512 | 45.25 | 65.44 | 68.00 |

Peaks include warmup/Adam state and exclude setup. Reserved peaks are62–66MiB
Dense,78–80MiB physical and80–82MiB envelope. Public CST policy down-projection
peak is67.88MiB; the other is69.50MiB. Natural allocated spread across21 envelope
steps was1.88MiB. This is not a long-run leak test or total GPU process usage.

## Scope and evidence

The three core arms use LR0: gradients/moments/counters are computed while
holding the workload fixed. The public CST arm uses positive sub-ULP LR1e-20
to exercise projection/transport; decoded physical rows are verified byte
unchanged afterward. This measures a fixed-state eager Linear workload, not LM
convergence, typical nonzero-LR trajectories, Graph capture or large-width speed.
No runtime optimization was adopted from this probe.

Measurement source `2e41c3e8a50012cd23e53821bd4a4fbc956a9d29`, archive
`c618b50e7d67f0510c8c270d8e63612eb34587f677dd5f7bc58a32831c8d491e`.
That source contains the initial rename plus temporary initializer aliases;
the subsequent alias removal affects initialization API/report/call sites,
not the timed train loop or envelope/CUDA execution. The initializer math is
unchanged. Renamed GPU qualification passed41 checks before timing; after alias
removal90 targeted CPU checks passed and the full local suite passed2233 with
2613 skipped (native opt-ins disabled). Old q keywords are explicitly rejected.

Job `envelope-speed-20261109-01` produced16 fresh timing workers plus one
qualification process. Retrieval checked37 files and1310 source files against
the local Git archive, matched every physical/input hash, and confirmed all18
recorded processes inactive. GPU postflight was0%/39MiB. Evidence archive SHA256:
`5345513619a6c3a2e8904712cf20dbfd11ffe222c96aeeae8df996b3caf9226a`.
[Per-order compact results](radius-fraction-speed-20261109.json).
Raw logs/source/builds remain in ignored owner storage under the job ID.

Reproduce on the pinned source and qualified environment:

```bash
python -m benchmarks.training.radius_envelope_probe \
  --arm envelope --ni 512 --no 2048 --output NEW_RESULT.json
```

Other arms are dense,physical,envelope_cst. The driver records the full order
and per-command runtime; benchmark setup includes the shared initializer and
an initial independent physical-weight output check. The source is experimental
and remains in the draft PR; there is no main/default integration.
