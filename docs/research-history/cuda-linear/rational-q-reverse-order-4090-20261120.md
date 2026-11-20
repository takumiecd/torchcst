# Reusing output-tile order in rational-radius Linear backward on RTX 4090

**English** | [日本語](rational-q-reverse-order-4090-20261120.ja.md)

The [per-call Atom ordering](rational-q-atom-order-4090-20261120.md) groups
Atoms by radius bank and input tile. The experimental explicit Plan
`cuda_rational_radius_wide_reverse_ordered` also saves the existing
output-tile order from forward and uses it to assign CUDA lanes in the
reverse response. The kernel writes each response to its original index in
the gathered layout, preserving the inputs and accumulation order for the
gradient consumers.

## Timing and validation

RTX 4090, PyTorch 2.9.1+cu128, eager IEEE FP32, M8 cell mass, 128 rows,
32N Atoms at width N. Both Plans use the same native binary and Atom replay.
Times are the means of the medians of two isolated processes per Plan, with
eight warmups and 24 measured iterations per process. The update includes
forward, backward and CSTAdamW. Peaks are PyTorch allocated memory during
the update.

| Width | Forward, ordered → reverse (ms) | Backward, ordered → reverse (ms) | F+B, ordered → reverse (ms) | Update, ordered → reverse (ms) | Peak, ordered → reverse (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 3.23 → 3.24 | 5.98 → 4.48 | 9.21 → 7.71 | 9.33 → 7.71 | 241.74 → 242.74 |
| 8192 | 4.52 → 4.57 | 11.50 → 8.04 | 16.03 → 12.60 | 16.86 → 12.88 | 484.49 → 486.49 |
| 16384 | 7.56 → 7.57 | 23.63 → 15.70 | 31.19 → 23.25 | 32.87 → 24.16 | 966.99 → 970.97 |

The width-16384 reverse-response kernel fell from 9.66 to 1.69 ms in separate
profiler runs. `batched_parameter_kernel` still took 9.23 ms, about 59% of
the new backward time. All three replay fixtures matched byte-for-byte for
the output, input gradient and four Atom gradient fields. Input-only and
Atom-only backward also matched at seven rows. The targeted CUDA suite had
14 passes. The CPU suite had 2,206 passes and 2,585 skips.

The seven-layer width-16384 capacity fixture matched all three update losses.
The mean of updates 2–3 changed from 0.2200 to 0.1636 s, with peak allocated
memory of 4.06 and 4.08 GiB, respectively. The prior FP32 Dense run exhausted
memory at its first optimizer step. This fixture uses synthetic inputs and
tests eager training capacity.

## Physical Atom order and versioning

In the width-16384 fixture, `AtomState.relayout()` reordered the Parameter,
existing gradient and both AdamW moments consistently, with zero relative
output difference. CSTAdamW changed the Parameter's PyTorch `_version` from
1 to 3 and left `layout_version` at 0. The subsequent relayout advanced
`_version` to 4 and `layout_version` to 1. The Polar CUDA update also
explicitly calls `torch.autograd.graph.increment_version` after its direct
write.

In a separate 32-update probe with the earlier Plans, rebuilding the order
per call took a median of 32.89 ms over the 24 timed updates. Sorting
physically once and reusing the placement took 37.00 ms per update, with an
initial 109.73 ms relayout excluded from that timing. An isolated warmed
relayout check took 51.91 ms. By update 32 the current keys differed from
their initial values for 50.58% of Atoms. Adjacent key inversions in the
physical order rose from zero to 24.21%.

The reverse Plan rebuilds the order from current values in each forward and
only shares data within the corresponding forward/backward pair. An eager
cache across forward calls would key the permutation by the Parameter
identity and storage, its PyTorch mutation version, the Atom layout version,
and the sorting configuration. That permits reuse across microbatches while
the Atoms are unchanged. Native writes through raw pointers must report the
mutation explicitly. A fixed physical order remains valid for Atom ownership
after optimizer updates, with potential loss of locality as the keys move.

## Artifacts

Measured source `570c740f25a1d97dca9c2fb44d7bef2941bd16d5`.
The local experiment directory is `output/rational-q-reverse-order-20261120-01`.
It contains `summary.json`, per-process timings, `reverse-profile.json`,
`physical-{verify,online,physical}.json`, and the replay, profile, capacity
and physical relayout drivers. The native SHA-256 is
`4394778ee16ad722f768ea7ae6cf16813d991aabd88ad199410bc6b269259758`.
The source snapshot SHA-256 is
`9ee7763583f4484621f10fae01b6b88438b000e0d1f1dae62c0ae7c1f5637e56`.
The core evidence archive SHA-256 is
`aa3e2ad6b4005347285de262a727272b5e79d9926fa6570bb5a7ae808568c225`.
