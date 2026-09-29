# CUDA Linear research notes

These are dated measurements and rejected or unpromoted experiments, not a list
of runtime backends. Start with the [candidate decisions](../cuda-kernel-shortlist.ja.md)
and the adjacent implementation before interpreting an old result.

| Subject | Starting points |
| --- | --- |
| Exact mapped CST | [Ada bounded candidates](five-percent-bounded-lists-ada.ja.md), [Blackwell repeat](blackwell-bounded-cst-20260929.ja.md), [small shapes](small-shape-ada-ampere-20260929.ja.md) |
| L4 bottlenecks and alternatives | [Current bottleneck](l4-current-bottleneck-20260929.ja.md), [small tiles](l4-small-tiles-20260929.ja.md), [on-chip fusion](flash-cst-l4-20260929.ja.md) |
| Approximate anchor CST | [1024² training](l4-anchor-atom-training-20260929.ja.md), [optimization](l4-anchor-optimization-20260929.ja.md), [8192² result](l4-anchor-8192-dispatch-handoff-20260929.ja.md) |
| Other exact candidates | [Atom prefix](atom-prefix-linear.ja.md), [local atom](local-atom-linear.ja.md), [low-density GEMM](low-density-gemm.ja.md) |

All other files here remain searchable as historical experiment records. Small
saved measurements live in [benchmark evidence](../../../../benchmarks/cuda/linear/evidence/README.md).
