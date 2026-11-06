# Scoped 2048-wide FFN validation

English (canonical) | [日本語](ffn-rectangular-2048-20261107.ja.md)

The explicit `GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048` route enables the
previously unsupported fixed-coefficient FFN cohorts. It reuses existing native
arithmetic and keeps the original Plan guards, recipe fields and dispatcher
defaults. The supported shapes/counts remain exactly512↔2048/A8192 and
128↔2048/A4096, FP32, at most256 token rows, Polar M8/base1/merge2/spacing1,
radius bounds[.5,50]. This is eager-only scope.

## Executed checks

The expanded R3 qualification passed all12 native cases on RTX4090, including
all four2048-site directions, noncontiguous B2T128, Y/dX/all Atom-coordinate
VJPs against chunked CPU FP64 references, independent polynomial quadrature,
and small controlled optimizer/model serialized-resume checks.

The R8 recipe qualification separately checked512→2048 at target std.02 and
2048→512 at std.02/sqrt(8), with8192 Atoms each and the actual q0/bound1/w_c.3/
birth4 recipe. It passed both native directions, quadrature, mixed ordinary/
Atom optimizer reference checks and small model resume. The actual four-block,
width512 causal model then completed20 fixed-batch updates in Dense/CST arms:
all eight FFN projections are CST in the latter; attention/embedding/output
stay Dense. Microbatch2×context128, accumulation4, FP32, TF32off, seed17,
AdamW baseLR3e-4, betas(.9,.95), decay0 and global clip1. Planned LR/data horizon
is20M tokens. This repeats eight training windows; it is not heldout feasibility.

A separate terminal audit verified checkpoint/curve/summary hashes, finite
optimizer state/counters, exact model-load/cache-generation semantics, all eight
trained CST projection VJPs at B1T8 and independently recomputed CE. Two task
updates in disposable copies verified serialization and the next loss, parameters,
moments and counters; maximum parameter difference was0 in both arms. The
original saved runs were not resumed or overwritten.

## Short complete-step observations

| Width512 FFN | Median update ms | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: |
| Dense | 18.40 | 952.66 | 1168 |
| CST reference | 726.25 | 1187.04 | 1492 |

These are10 synchronized complete-update samples after excluding the first10
updates from timing, in one Dense-then-CST job, without motion sampling. All20
smoke updates are still inside the planned200-update LR warmup. Data transfer and
four microbatches are timed; separate probes/evaluation are excluded. They are
short functional observations, not independent repetitions or primary scaling
measurements. CST has272384 FFN parameters versus8398848 for Dense, yet its
measured GPU peak is larger. No practical speed or memory advantage is shown.
Selected-recipe heldout pilots and retained seed29 observations are recorded
below and in the [research study](https://github.com/takumiecd/cst-research/pull/29).

## Selected-recipe heldout pilots (R10)

Both supported 2048 cohorts also completed the separate selected Atom-LR16
recipe at seed17, with fresh Dense/no-FFN/CST arms. Each arm completed976
updates/999,424 tokens with a common planned20M LR/data horizon, not fixed-batch
repetition. All eight FFN projections in four blocks are CST. Ordinary-parameter
LR stays unchanged; the16 multiplier affects the whole Atom group, not only
amplitude. This uses the same frozen R8 consumer and tested operator above.

| Model width / expansion | Dense DEV CE | CST DEV CE | No-FFN DEV CE | Dense / CST update ms | Dense / CST allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 4 | 3.28466 | 3.40110 | 3.71174 | 18.48 / 792.80 | 952.66 / 1187.04 |
| 128 / 16 | 3.91922 | 3.96799 | 4.35479 | 9.18 / 501.67 | 373.07 / 697.90 |

Each cohort passed fresh exact-shape/count/recipe qualification at both actual
target standard deviations, then full-model preflight and the independent
terminal trained-checkpoint audit described above: all eight CST Y/dX/Atom VJPs,
finite optimizer/counters, recomputed CE and serialized next-task-step checks.
The four2048 directions are therefore covered in actual task learning, beyond
the earlier20-update reference smoke. Both retrieved archives and all22 raw
files matched their manifests.

These single development-seed results improve on no-FFN while trailing Dense.
They do not establish three-seed confirmation, convergence or asymptotic quality.
The966 synchronized timing samples per arm exclude the first10 updates;
reserved peaks are1168/1492MiB (Dense/CST at N512) and460/788MiB (N128 expansion16).
These sequential eager FP32 observations still show no speed or GPU-memory win.
The owner stopped the unstarted seed43/71 cases in favor of short field-LR
experiments. Seed29 at all three main widths finished and was audited/retrieved;
the planned three-seed confirmation is incomplete. Attention replacement has not started.

R10 evidence SHA256, under the same research evidence locator:

| Cohort | Native qualification | Trained checkpoint audit | Small-evidence archive |
| --- | --- | --- | --- |
| N512 expansion4 | `c45dc18eaa0dd074d3f38e7c8488037ac48ae4d60b265d4f04c8e12903b58cf0` | `74ceefd58caa2dac38c5eaa77cf270d4c1176ad85d5d41b8c6cd3996ae22348a` | `e59e990cd947d5c3ac538e0cac28cf20b3b9ad0c9df1811fabcab18f4559c5c9` |
| N128 expansion16 | `78d37a661f5e01811c84f92fd2b76177f655d62f2595affa4502850906f7cc08` | `fbc172e1f38eef8d57efcee819391f5b4554d3f33cfda54165fb41feb1fa835f` | `0edb48531af793f9079d9911b9e9f3b06bbd24f321238f80b4f84c13a8bf732f` |

Archives are `width512-e4-lr16-r8-01-evidence.tar.gz` and
`width128-e16-lr16-r8-01-evidence.tar.gz`. Frozen bootstrap exact argv and source
manifests remain in the evidence locator; use fresh output paths to reproduce.
These task pilots do not widen this operator's shape/count/precision admission.

## Retained width512 seed29 observation (R11, owner-stopped family)

Each fresh Dense/no-FFN/CST arm completed4882 updates/4,999,168 tokens with the
same selected recipe and planned20M horizon. Final DEV CE was2.57642/2.83565/
2.68898, and confirmation CE was2.48887/2.74543/2.60475 respectively. All eight
trained CST Y/dX/all-Atom VJPs, independent CE, finite optimizer/counters and
serialized next-task-step audits passed; resume maximum parameter error was0
for all three arms. Eleven raw files matched the retrieved manifest.

Dense/CST median update times were18.57/822.90ms (4872 synchronized samples,
first10 excluded), with allocated952.66/1187.04MiB and reserved1168/1492MiB.
This longer one-seed task observation supports functionality of the bounded
route; it establishes neither three-seed repeatability nor a speed/memory win.
The original R11 supervisor was deliberately stopped and its old running journal
is stale. The final owner-stop receipt is authoritative; cancelled cases must not
be treated as completed or numerical failures.

- Native qualification SHA256: `3b3a80bf3c7dabf6fe1a4088dddf3f7e4c027eabee39fdb0746f79dc0a3fc309`.
- Trained audit SHA256: `9764902e1b0acdee490f2a63f728a5369d6aeee2c17bf0a3604d08e324f78ec7`.
- Archive `width512-e4-lr16-seed29-r8-confirm-01-evidence.tar.gz` SHA256:
  `a1973aac3b176ba1e63a66373cd67a7305f9c9813335b7fbd09afb4709dbca74`.
- Final owner-stop receipt SHA256:
  `59d7a4e28c60d84b8a92bdd1b442abbe74c81daeb8267fcaf8b51ff8674b7f27`.

Consumer70e02d8/operator2b587 are unchanged. Full curves and the separate
owner-stopped observation report are in the research repository at
`docs/ffn-feasibility-r11-seed29-20261107.md` and its Japanese counterpart.

## Provenance and reproduction

- Operator: `2b587e0f04ddf64e2bfb0c2e3db591e87a6f6440`.
- Expanded R3 consumer: `036d0c5beb4a6f675ce1dbeafd4224e6c270f342`.
- R3 native receipt SHA256: `419ab7750182bca2302d7152c6fc01c1a961efe69f19bf5bb22355f3aa299ce6`.
- R8 consumer: `70e02d864e661d3a87cd8aea322b49ae3f4a1f8f`.
- R8 native receipt SHA256: `b40ac9ffa47814408eb22d61f65c2a0313f0420632566609f168c269131785d6`.
- R8 terminal receipt SHA256: `13e60f4b96786fb300c074891bbc02a1e6051c7fc2e934d4279c3f058b4348b0`.
- R8 trained-state audit SHA256: `08080c751cf5c2c80d57663daa4031bcb7fd231a401187ace7d16161511a9593`.
- Retrieved small-evidence archive SHA256: `932013048726fdc91c94f9c158ad39f96feee3884cb20511e0fad24327b21c37`.

Actual runtime: NVIDIA GeForce RTX4090,24GB class, CC8.9, Torch2.11.0+cu126.
Resolve the research evidence locator
`$CST_RESEARCH_EVIDENCE_ROOT/ffn-feasibility-20261106` for
`width512-reference-r8-01-evidence.tar.gz`, raw scripts and source manifests.
All seven bundled raw files matched the retrieved manifest; bulk checkpoints
remain on the GPU host with hashes. The earlier CPU suite at the tested operator
reported2086 passed/2483 skipped; skipped GPU cases are not counted as validation.

Use those frozen sources and fresh output paths. Run the research consumer's
`ffn_qualification --cohort recipe --width 512 --expansion 4 --coefficient 8`,
then `ffn_run --kind dense` and `--kind cst`, both with `--width 512 --tokens 20480
--schedule-tokens 20000000 --repeat-fixed-batches`, passing the actual new
qualification/data manifests and their SHA256 arguments. The recorded bootstrap
and audit scripts retain environment setup, exact argv, PIDs and source checks.
Reproduction does not qualify another consumer, scalar recipe, GPU or Atom count.
