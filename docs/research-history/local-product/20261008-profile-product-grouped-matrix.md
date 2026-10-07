# Grouped profile-product aggregate matrices

Grouping atoms within a CTA removes much of the aggregate matrix assembly and
parameter VJP cost. On the paired L4 Product1024 cases, native group8/patch8
reduces complete learning step time from0.561313 to0.309897ms at rho3 and
from0.655156 to0.457504ms at rho8. Allocated peak falls from25,604,096 to
16,688,640 bytes. These routes and their v1/v2 controls are integrated as explicit
research algorithms; the public registry and default dispatcher are unchanged.
Only profile-product execution is optimized. Both measured families are
Euclidean: single Product chart and input Strip. Literal Torus is unimplemented.

## Contract and implementation

The [preceding matrix checkpoint](20261008-profile-product-matrix.md) records
the unchanged equations. For complete chart profiles u_a(i), v_a(j),

\[
D_a=\max(\|u_a\|_2\|v_a\|_2,\varepsilon),\quad
W_{ij}=\sum_a\alpha_a u_a(i)v_a(j)/D_a,\quad
Y=XW^T,\quad \bar X=\bar YW,\quad \bar W=\bar Y^TX.
\]

Normalization uses one whole-chart floor, including its derivative; it does
not independently floor each axis or each Strip tile. Preparation and every
support patch use current widths/centers/pitch on each forward and Graph replay.
Y/dX and all canonical atom gradients retain FP32 IEEE arithmetic. The existing
Polar pullback and optimizer operation are unchanged. Retained forwards save
Source, amplitude limit, packed statistics, W and Strip pitch. Singleton center
derivatives remain exactly zero; complete support has no capacity truncation.

v1/v2 retain their original recipe meanings. Strict v3 adds atom_group1/4/8
and patch_sites8/16/32, with preparation/support group/sites and torch/triton GEMM
choice. Defaults are group8/patch8/native. The grouped kernel uses independent
atom lanes, loops over the maximum current span in its group, and masks each
atom's own bounds before forming products. Empty atoms sharing a CTA with live
neighbors contribute zero. Assembly uses FP32 weight atomics; parameter VJP
reduces each atom's cotangents without gradient atomics. Workspace tensor sizes
are unchanged from v2. No per-atom matrix is stored.

Implementation commit1bf450644f807147fd5bf513a10ee55af2a25f64; GPU source
a3c74c2946a10dfe0002fad371c5c09a7d361959 adds only prior checkpoint notes.
All eight job snapshots have exact implementation file sets/bytes checked
against their declared commit. Subsequent common-main merge and this record do
not change those runtime, benchmark or test bytes.

## Paired complete learning steps

NVIDIA L4, Torch2.11.0+cu130, CUDA13.0, Triton3.6.0, Python3.13.15. Batch32,
seed41, about5% atoms: Product1024x1024 A52429; Strip64x1024 A3276, tile64/pitch68.
Initial rho3/8, width bounds1..16, fused capturable AdamW lr1e-4/wd.01 plus
production Polar update. Every plan runs in an isolated process; all use the
same initial atom/input/target hashes, update widths, and report21 synchronized
samples of an uninstrumented complete CUDA Graph learning step. No samples or
unfavorable plans are discarded. Dense has its own weight parameters under the
same batch/input/target, precision and optimizer protocol.

Primary times in ms (G=atom group, P=patch sites):

| plan | Product rho3 | Product rho8 | Strip rho3 | Strip rho8 |
| --- | ---: | ---: | ---: | ---: |
| prepared factor v3 | 0.561313 | 0.655156 | 0.113058 | 0.122965 |
| v2 native P16 | 0.644230 | 0.630258 | 0.114725 | 0.112580 |
| v3 native G1P8 | 0.558362 | 0.715928 | 0.107793 | 0.123079 |
| v3 native G4P8 | 0.337871 | 0.486603 | 0.093656 | 0.104634 |
| v3 native G8P8 | 0.309897 | 0.457504 | 0.093195 | 0.105389 |
| v3 native G4P16 | 0.579758 | 0.531572 | 0.111245 | 0.108445 |
| v3 native G8P16 | 0.680836 | 0.551998 | 0.118389 | 0.113875 |
| v3 Torch G8P8 | 0.243491 | 0.387239 | 0.062823 | 0.074681 |
| dense | 0.083319 | 0.082748 | 0.054569 | 0.054414 |

Primary jobs: Product l4job-77c56e3025064f43a507b020a6634233;
Strip l4job-6e0d0edbedae4256a4e3ab5fcda69b14. Each contains rho3 and rho8.
Two additional independent rho3 jobs per family confirm the direction:

| family/run | prepared ms | native G8P8 ms | Torch G8P8 ms | dense ms |
| --- | ---: | ---: | ---: | ---: |
| Product pilot | 0.563585 | 0.315670 | 0.246158 | 0.083307 |
| Product reverse order | 0.561694 | 0.311056 | 0.243543 | 0.083110 |
| Strip pilot | 0.115255 | 0.096254 | 0.063976 | 0.054855 |
| Strip reverse order | 0.113365 | 0.093208 | 0.062582 | 0.054466 |

Pilot jobs:l4job-94dd9586d17149d280c723ca7e51b068 /
l4job-f4e50d90b45f4ca3be9a15bd08693f13. Reverse catalog order jobs:
l4job-aa5db39aa6244595bdaf08b79b3004c8 /
l4job-a2c9a89026104a20a1252402ee035399. All plans and samples are in the companion
JSON. Rho3 has three independent jobs per family; rho8 has one. No fastest run
is substituted for the primary result. 256/512 declarations are provided but
not measured here; sizes beyond1024 and other devices remain unverified.

## Memory and disposition

Actual warmed capture/replay peaks include optimizer and framework/library
allocations. Every measured case has these peaks; native grouping introduces
no additional tensor workspace. GPU process usage is unmeasured.

| family/route | allocated bytes | reserved bytes |
| --- | ---: | ---: |
| Product prepared | 25,604,096 | 102,760,448 |
| Product native (all listed variants) | 16,688,640 | 56,623,104 |
| Product Torch G8P8 | 50,767,360 | 115,343,360 |
| Product dense | 51,382,784 | 111,149,056 |
| Strip prepared | 2,355,200 | 8,388,608 |
| Strip native (all listed variants) | 1,305,088 | 6,291,456 |
| Strip Torch G8P8 | 35,383,808 | 48,234,496 |
| Strip dense | 35,408,384 | 48,234,496 |

Native G8P8 gains44.79%/30.17% time on Product rho3/rho8 and17.57%/14.29% on
Strip, while allocated peaks fall34.82%/44.59%. Strip rho8 slightly favors G4P8.
G8P16 regresses on Product/Strip rho3, so grouping is not a universal gain
independent of patch width. Torch G8P8 is faster but substantially increases
memory. Integrate explicit recipes and their validated controls for further
research; retain the existing public selection policy. Native remains slower
than dense. No dense parity or general size/width winner is claimed.

## Validation and diagnostics

CPU1100 passed/1864 skipped; all12 declarations and Product/Strip prepare/check,
Ruff, wheel/sdist build and isolated installed-wheel metadata imports without
Triton/benchmarks passed. Gate l4job-cefa3d62abf74e7eb2ab71e68e40477e passed813:
matrix596, preparation75, grouped56, global30, Strip56 (692 actual CUDA and121
CPU/metadata). Coverage includes independent full-site FP64 Y/dX/all atom VJPs,
strided inputs/matrix tails, atom tails, full/support preparation, empty/sharp/
floor/wide profiles, partial tiles/valid pitch gaps, saved-forward mutations,
gradient branches, and20 captured optimizer updates comparing parameters,
moments and clocks with live widths/pitch. Tolerances are unchanged.
All eight completed performance artifacts pass adapters6/5 and the unchanged
public submission policy. Max abs Y/dX/dp:2.45942124e-5 /2.10344630e-5 /
4.25495323e-6. Same-cotangent production Polar update agrees exactly.

Separate warm job l4job-87971247a563444b95ee9c684b0c00da passes seven plan gates.
It uses sequential plans and external events after a learning Graph with widths
continuing to update; its timings are diagnostics, not primary step/memory
evidence. Nested wrappers must not be summed or subtracted from primary time.
Product native P16 assembly/VJP events are110.59/313.34us; G8P8 reduces them to
27.65/63.49us. G8P16 has105.47/354.30us. Compiled VJP registers rise from72
(G8P8) to181(G8P16); both report zero spills. Greater register demand and masked
patch work are plausible contributors to the regression, not an isolated
causal proof. Native Product Y remains about76.8us, preparation64.5us and VJP
63.5us. Native GEMM scheduling and larger sizes are separate next experiments.
Full event samples and compiler reports are preserved in the companion JSON.

All current jobs succeeded; the supervisor returned exit0 and all pool slots
are stopped. A prior interrupted matrix job required documented pool recovery
before this batch. An initial identity check aborted before allocation; the
existing authorized account was subsequently verified, without a new login,
and the selected batch proceeded. Transport/auth logs, all archives/receipts,
drivers and raw artifacts remain in ignored
benchmarks/cuda/linear/evidence/profile-product-grouped-matrix-20261008/ and the
host-wide pool job directories. No source/numerical gate was relaxed.

Reproduce from the frozen GPU commit with the contributor workflow:

```bash
python -m tools.kernel_dev prepare \
  --plans benchmarks/cuda/linear/plans-profile-product-grouped-matrix-global.json \
  --case benchmarks/cuda/linear/cases/profile-product-grouped-matrix-global-1024-rho3.json \
  --candidate native-g8-p8 --candidate torch-g8-p8 --output output/grouped-comparison
python -m tools.kernel_dev check \
  --plans output/grouped-comparison/plans.json --case output/grouped-comparison/case.json
python -m benchmarks.cuda.linear.run \
  --plans output/grouped-comparison/plans.json --case output/grouped-comparison/case.json \
  --polar-update fused --source-commit a3c74c2946a10dfe0002fad371c5c09a7d361959 \
  --output output/grouped-step.json
```

The prepared comparison retains native16 as its same-engine baseline. Use the
original complete Case/catalog and preserved measurement driver to reproduce
the eight-plan paired comparison, and the Strip equivalents for that family.
