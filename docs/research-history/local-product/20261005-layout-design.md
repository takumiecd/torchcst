# Small Linear: support-ordered values and owner ranges

Design checkpoint on `kernel/local-product-onchip-h`, based on runtime source
`c3fe23157ad2771ec6234024b80c351075c33171`. The user asks to make memory layout
the foundation of the next kernel. No CUDA implementation, GPU allocation or
performance adoption is made at this checkpoint.

## First candidate

Use one support-ordered SoA execution view for each direction, with a small
begin/end index for each spatial owner and rho band. Forward orders by output
support start; dX orders by input support start. This keeps the values needed
by neighboring atom lanes adjacent, rather than only sorting an ID list whose
payload remains in arbitrary canonical order.

The canonical Parameter and optimizer moments keep their current identity/order.
Only derived execution views are reordered. Each direction stores an atom once;
owners can read overlapping ranges without duplicating the atom's13 metadata
values. Copies across forward/dX views already exist in the current layout.

Proposed layout, with atom as the contiguous dimension:

```text
canonical Parameter / Adam moments: current atom identity and shape
canonical decoded metadata:         [13 fields, A canonical atoms]

forward execution values:           [13 fields, padded A], band then output lo
dX execution values:                [13 fields, padded A], band then input lo
physical -> canonical IDs:          [2 directions, padded A]
owner ranges:                      [2 directions, spatial owner, 3 bands, begin/end]
wide saved H, if retained:          [batch, forward physical atom position]
```

13 fields retain the current normalized arithmetic and flags; reducing fields
is a separate experiment. Band starts are32-atom aligned in the offline budget.
Owner ranges can start inside a band, with exact validity/support masks. Padding
is never treated as a live canonical atom.

`H`'s producer and consumer must share the physical permutation; sorting only
the metadata would leave H as a canonical-ID gather. Parameter VJP continues
to return canonical gradients. The recompute-VJP recipe is the initial control,
so its parameter contraction need not use a scattered saved-H permutation.
Both permutations/ranges are per-forward snapshots, preserving outstanding
backwards even when later calls repair or rebuild the layout.

Consecutive values are a design objective, not a proven transaction reduction:
the compiler's lane mapping and generated loads still require inspection.
[NVIDIA's coalescing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
motivates contiguous fields across neighboring atom lanes.

## Candidate ranges and exactness

Within each band, sort live atoms by the relevant support start `lo`. For a
spatial owner `[s,t)`, a valid contribution satisfies `lo < t` and `hi > s`.
Take the smallest contiguous sorted range containing every such contribution.
The kernel still masks `lo < t && hi > s` inside this range: variable widths
can leave unrelated atoms between the endpoints.

One construction uses a prefix maximum of `hi` in each sorted band: begin is
the first prefix maximum greater than `s`; end is the first `lo >= t`, clamped
to an empty range if begin >= end. A fused preparation can instead reduce the
minimum/maximum matching physical index for each owner. Their preparation cost
is not assumed to be zero. GPU construction remains to implement and verify.

Do not assume ordering by `lo` also orders `hi` when widths differ. Do not assume
rho<1 implies one-hot. Exact singleton flags, inactive support, normalization
floors and full-domain norms retain the existing contracts. Boundary-spanning
atoms can occur in multiple reading ranges but contribute once per output site.

## Offline coverage and geometry

Eight actual initial CUDA snapshots from the
[scan investigation](20261005-scan-investigation.md) are analyzed. Two additional
CPU fixtures use production decoding, initial rho1.25/3/8 at fractions0.5/0.3/0.2,
with the same bounds, centers and seed. All initial decoded widths exceed1.
The mixed fixtures are CPU geometry diagnostics, not new GPU benchmark Cases.
Twenty direction-level coverage checks recover exactly the intended atom IDs.

Counts cover spatial owners once, excluding the batch-block multiplier.

| N / initial rho | Current forward scans | Useful visits | Support-ordered range visits |
| --- | ---: | ---: | ---: |
| 64 / 1.25 | 816 | 225 | 225 |
| 64 / 3 | 816 | 246 | 246 |
| 64 / 4 | 816 | 267 | 267 |
| 64 / 8 | 816 | 348 | 348 |
| 128 / 1.25 | 6,552 | 914 | 914 |
| 128 / 3 | 6,552 | 1,057 | 1,057 |
| 128 / 4 | 6,552 | 1,139 | 1,139 |
| 128 / 8 | 6,552 | 1,487 | 1,487 |
| 64 / mixed1.25/3/8 | 816 | 256 | 263 |
| 128 / mixed1.25/3/8 | 6,552 | 1,072 | 1,130 |

Homogeneous widths happen to give exact ranges in these snapshots; this is not
a general guarantee. Mixed N128 dX has1,067 useful visits in1,142 candidate
visits. Bounds must be refreshed from current support even if the ordering
does not change. No speedup follows directly from these counts.

## Value replication versus contiguous ranges

An alternative packs every owner's complete metadata separately. It eliminates
false positives, but repeats boundary-spanning payloads. An ID-only owner list
avoids payload replication, but leaves indirect metadata loads unless the
physical values are also arranged for those lists.

FP32 payload budgets below count only forward plus dX13-field views, each padded
to32 atoms per band or per owner/band. They exclude canonical preparation,
permutations, H, persistent slack, temporary sorting storage, optimizer and graph
capture. These are tensor budgets, not measured peaks or traffic.

| N / rho | One ordered copy per direction | Separate complete values per owner |
| --- | ---: | ---: |
| 64 / 1.25 | 23,296B | 28,288B |
| 64 / 8 | 23,296B | 44,928B |
| 64 / mixed | 26,624B | 39,936B |
| 128 / 1.25 | 86,528B | 108,160B |
| 128 / 8 | 86,528B | 168,064B |
| 128 / mixed | 89,856B | 138,112B |

Current persistent execution views alone are48,256B at N64 and124,800B at N128;
their reserve slots are included. Smaller snapshot payloads do not establish
a smaller complete-step peak: a new layout may require more sorting workspace.
Forward+dX owner-range indices take192B/384B for N64/N128. Canonical ID maps
and any persistent repair state are additional buffers.

## Input and H reuse within a block

For the existing16 input vectors per block, a cooperative input stage would
contain16×64 FP32 values (4KiB) or16×128 (8KiB). These are whole input vectors,
not a selection of16 coordinates. Local H for32 atoms contains16×32 values
(2KiB); the16-site output accumulator contains1KiB. Registers/shared placement,
bank behavior and occupancy remain compiler/runtime questions.

Try the ordered range schedule first. Then separately compare shared input
staging against current masked gathers, and register/shared local H choices.
Keep wide H reuse as a separate baseline. Do not widen output ownership as a
substitute for spatially useful atom groups; the prior32/64-owner experiment
was slower.

Global buffers for metadata or saved H remain global allocations. Small working
sets and a locality-conscious order may favor L2 reuse; a persisting access
policy preferentially retains selected data rather than guaranteeing no DRAM
access ([NVIDIA programming guide](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-c-programming-guide/index.html#l2-access-properties)).
Physical DRAM traffic and warm H residency must be measured on the candidate.

## Implementation and validation sequence

1. Add an explicit research route with fused GPU preparation of ordered views
   and ranges; keep current recipes intact. A full rebuild provides a correctness
   baseline before assuming incremental repair will be cheaper.
2. Connect the range consumer and matching H permutation. Retain canonical
   parameter gradients and per-forward snapshot lifetimes.
3. Check support boundaries, mixed widths, empty/singleton/floor paths, partial
   domains, retained backward, width/center movement and CUDA Graph replay with
   independent FP64 scalar Y/dX/all-atom-gradient and production update checks.
   Initial benchmark widths stay >1; smaller supports remain valid during training.
4. Compare baseline/candidate/dense using the existing runner. Include all layout
   rebuild/repair costs in the uninstrumented complete step and capture/replay
   allocated/reserved peaks. Measure metadata, input and H traffic separately.
5. If one long owner range still limits parallelism, test division along atoms
   with bounded partial outputs; count reduction and scratch costs explicitly.

Reproduction: `PYTHONPATH=src:. python
benchmarks/cuda/linear/evidence/layout-design-20261005/analyze.py` from the active
worktree, using the preserved CUDA snapshots. Raw interval tables/script are
ignored evidence. [Machine summary](20261005-layout-design.json) records geometry,
payload estimates and snapshot hashes. This checkpoint changes no runtime kernel,
optimizer, public dispatcher or benchmark registration.
