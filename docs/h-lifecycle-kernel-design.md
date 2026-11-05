# Kernel design around H generation, reuse and lifetime

**English** | [日本語](h-lifecycle-kernel-design.ja.md)

This is the design policy agreed on 2026-10-10 for profile-product Linear on
RegularGrid and future supporting kernels. **Design around how X produces H,
how H contributes to outputs, and where and for how long H is retained.** Group
overlapping input reads and output aggregation, then generate and reuse H within
those groups. H is a logical intermediate; it does not require allocating a
Tensor covering all atoms and all batches.

See the [RegularGrid specification](regular-grid.md) for the public Chart
contract, the [kernel development guide](kernel-development.md) for
implementation and registration, and the [PeriodicGrid comparison](periodic-comparison.ja.md)
for existing measurements. This guide defines execution-design policy; it does
not change KernelSpec mathematics or the public dispatcher. RegularGrid
declarations and the Torch reference are implemented. GPU performance of the new
fusion and aggregation strategies below has not been validated.

## Fixed policy and choices to measure

| Fixed policy | Implementation choices to measure |
| --- | --- |
| Optimize both X→H and H→Y | Whether to prioritize grouping on the input or output side |
| Partition atoms and batches to control H workspace | Atoms per group, microbatch size and input/output interval sizes |
| Keep uses of H close together and reuse workspace after its last use | Placement in registers, shared memory or small global buffers |
| Aggregate overlapping output contributions locally to reduce atomics and writes | Output ownership, routing to intervals and atom order |
| Distinguish reuse within forward from retention through backward | Recompute/capacity tradeoffs for no, partial or full retention |
| Compare complete-step time and total peak memory with the same mathematics and update rules | Candidate admission per condition and future selection rules |

Start by comparing a few explicit execution strategies. Do not require a
general-purpose compiler or automatic search framework. Generating a long-lived
value earlier simply extends its lifetime. The default approach is to generate
it just before its first use, group its uses and release it after its last use.
Choose precomputation or overlap with transfers only after measuring the cost.

## Mathematics and the meaning of H

Here batches occupy columns: X is `[Ni,B]` and Y is `[No,B]`. The public Linear
uses X `[...,Ni]` and Y `[...,No]`; only the orientation differs, not the
operation.

Let qᵛ and qᵘ be the raw profile products for the input and output coordinates,
and A the amplitude. The profiles' shared live sigma follows the existing Polar
rule and is stop-gradient for the task VJP.

\[
D_a=\max\bigl(\|q^u_a\|_2\|q^v_a\|_2,\epsilon\bigr),\qquad
W_{ji}=\sum_a A_a\frac{q^u_{ja}q^v_{ai}}{D_a}.
\]

For example, setting `U_ja=qᵘ_ja/√D_a` and `V_ai=qᵛ_ai/√D_a` gives:

\[
W=U\,\operatorname{diag}(A)\,V,\qquad
H=VX,\qquad Y=U\,\operatorname{diag}(A)\,H.
\]

H is the result of each atom reading the input X. Its meaning is fixed as the
value before multiplying by amplitude. The denominator need not be split
uniquely between U/V; existing CUDA preparation uses a different split. Raw H
values need not match between candidates that change this split. Validate
equivalence of W/Y and the correct pullback. Normalization covers all observed
sites, with the floor applied once after the full product. Do not renormalize
per output interval or partition, or apply a separate floor to each axis.

For atom groups g and batch partitions t, compute:

\[
Y_t=\sum_g U_g\,\operatorname{diag}(A_g)\,(V_gX_t)
\]

Batch partitions are independent; atom partitions contribute sums to Y. Do not
truncate support or omit atoms when partitioning. Validate changes in
floating-point summation order against a fixed error criterion.

## X→H: share reads

Group atoms with overlapping input support to reuse contiguous X reads and
profile evaluations. Consider both reuse when several atoms access the same
input and coalesced reads when neighboring GPU lanes access contiguous
addresses. Do not assume that atom reordering alone improves cache efficiency.

With `grid_shape=((16,16),(128,))`, the output side has two coordinates and the
input side has one. Each side's profile is the product of its coordinate
profiles. A multidimensional support box need not become one contiguous
interval after flattening; use contiguous intervals along the innermost grid
axis.

Derive candidate support ranges directly from centers, spacing and sigma, and
also check the actual profile support condition. Effective bandwidth is
`rho_d=sigma/spacing_d` for each coordinate axis; one rho may not describe all
axes. With narrow support, generating H reduces to reading one or a few sites.
For freely moving centers, `rho<1` alone does not always imply one-hot support.
Truncating an infinite-support profile such as Gaussian to a finite range is
outside this policy.

Euclidean support is clipped at the finite grid boundary. FlatTorus uses
periodic displacement and handles support crossing a periodic boundary.
**Wrapping coordinates and applying `% n` to site indices have different
requirements.** RegularGrid can also declare partially periodic grids by taking
a concrete geometry. Kernels that simply wrap site indices must check layout
conditions such as `period=n*spacing`. Outside those conditions, use a supported
exact path or reject the configuration in Algorithm support checks. Wide
support or exhausted search capacity must still count each observed site
correctly, without truncating support.

## H→Y: group and aggregate contributions

Group atoms with overlapping output support and accumulate Y locally for each
interval. Initial candidates fuse normalized U evaluation and amplitude
multiplication instead of constructing a full `AU` Tensor. Reuse the same
`A_a U_ja` within a microbatch. Fusing amplitude into H is another candidate
with the same mathematics; compare it using support-site counts, microbatch
size and fusion cost.

| Unit of ownership | Intended reuse | Required decision |
| --- | --- | --- |
| Atom group × microbatch | Share X and reuse generated H across outputs | Atomics and local aggregation when different owners write the same Y |
| Output interval × microbatch | Aggregate all contributions to an interval and write Y together | Recompute or temporarily share H for atoms spanning multiple intervals |

Grouping atoms that share an output does not eliminate atomics by itself. To
avoid atomics, one owner must aggregate every contribution to an output element,
or a separate reduction must combine partial sums. Include the latter's buffer
and extra processing in the cost. An output-interval owner must include all
atoms whose support overlaps the interval, even when their centers lie outside.

Input grouping and output grouping need not use the same order. Evaluate how
much retained storage can reduce duplicated H computation and repeated X reads
introduced by output aggregation. Consider interval candidates and range
metadata rather than assuming a full per-site support-index table or CSR.
Routing and sorting also incur construction and storage costs. After centers or
live bandwidths change, do not reuse old routing without validation.

## H/G lifetime and backward

Reusing H for several outputs within forward and retaining H through backward
are separate choices. If another output interval still needs H after one
forward partition finishes, plan to complete that use or recompute H before
overwriting its workspace.

Let E=dY and G=UᵀE. For a fixed factorization:

\[
dX=V^T\operatorname{diag}(A)G,\qquad
dA_a=\sum_b H_{ab}G_{ab}.
\]

Center gradients also require derivatives of profile evaluation and
normalization. Depending on the denominator split, both U and V may depend on
the same center; include both pullbacks. Full dU/dV Tensors are unnecessary:
return the required local contributions to the atom's Parameters. Even when
amplitude is fused into U, retain enough information to recover unscaled U/G
for the amplitude gradient. Recovery by dividing by amplitude fails at zero
amplitude. A zero-amplitude atom can still have a nonzero amplitude gradient;
do not exclude atoms based only on their value.

| Data | Use and retention policy |
| --- | --- |
| Chart layout, Parameters and optimizer state | Preserve existing ownership contracts; include their capacity throughout training |
| Prepared support ranges, norms, amplitudes and related metadata | Reuse across partitions of the same forward state; evaluate retention versus recomputation |
| H | Retain from generation through its last output use; decide separately how much survives through backward |
| X and forward-time state | Save the information needed to recompute H and profiles correctly in backward |
| G and H regenerated in backward | Generate in partitions and reuse workspace after processing dX/atom-gradient contributions |
| Y/dX partial sums and atom-gradient partial sums | Aggregate all contributions within the owned range, including gradients across batch partitions |

Recomputation must use forward-time Parameters, bandwidths, layout and fixed
scalars. Do not change retained-backward semantics by recomputing H from updated
Parameters. Align snapshots and saved-Tensor version checks with the existing
autograd contract, and include their storage in the budget.

For the initial processing-order comparison, use the same policy of recomputing
H rather than retaining it through backward. Then, with the same processing
order, compare no retention, partial retention with a capacity bound, and full
retention as a reference. For partial retention, use recomputation time saved
per retained byte as a guide, and measure the selection cost itself. Full
retention is a comparison control, not an exemption from adoption memory
requirements.

## Memory budget and the role of W

If the dtype uses s bytes, a group contains C atoms and the microbatch size is b,
H values alone require `s C b` bytes; holding H/G together requires `2 s C b`
bytes. With FP32, C=4096 and b=8, H is 128KiB and H/G together are 256KiB. This is
an example of value workspace, not a claim that it fits in a particular GPU's
shared memory.

The actual budget also includes preparation metadata, routing, snapshots,
partial sums, norm derivatives, Parameter gradients, optimizer state and
allocator/Graph pools. Check register spilling and reduced parallelism from
shared-memory capacity. Overly fine partitions increase rereads and launch
counts; do not select a candidate solely for its small workspace.

Constructing W is not required. Aggregating local contributions into W first and
reusing them across the batch can be compared using the same reuse, lifetime
and storage criteria. A full FP32 W takes `4 No Ni` bytes regardless of batch
size; full H takes `4 K B` bytes, proportional to batch size. Evaluate local W
including assembly, regeneration and duplicated work, as a comparison candidate
for the H path. Do not fix a boundary from rho alone. Consider support size,
atom overlap, batch reuse and workspace bounds.

## Comparison and records for the next implementation

First reproduce the existing 2-coordinate configuration with RegularGrid and
check mathematical equivalence. Compare atom ownership and output-interval
ownership with the same inputs, atoms, geometry, profiles, update rules and
retention policy. Then extend the scope to grids with 2 output coordinates,
Euclidean geometry and different batches. Distinguish dimensions that a Chart
can declare from the validated support of each CUDA Algorithm.

Record the following in research notes and results for every candidate:

1. Ownership unit, atom grouping, H generation/use order and amplitude-fusion location.
2. Where X/H are reused, duplicated computation, Y/dX aggregation and remaining atomics.
3. Saved Tensors, forward-state handling, backward recomputation and workspace bounds.
4. Complete-step time including routing/preparation, and peak allocated/reserved memory including capture/replay.
5. Independent-oracle Y/dX/all-atom gradients, normalization and coverage of empty/single/wide support, periodic boundaries, variable bandwidth, updated state, retained backward and Graph replay.
6. Source/result hashes, hardware, runtime, conditions, independent run count and reasons for adoption, deferral or rejection.

The adoption objective for this study is for total peak allocated and reserved
memory each to stay at or below a complete dense training step under the same
conditions. Choose by complete-step time among candidates meeting that
requirement. Measure controls as well in independent processes, release oracle
scratch and use matching measurement boundaries. Fix the scope and preserve
records of research candidates that miss the memory requirement. Do not sum
diagnostic stage timings as a substitute for complete-step time. Allocator
peaks do not measure physical DRAM traffic or L2 hit rates. Establish physical
cache/atomic explanations with the necessary traces/counters; do not present
unmeasured mechanisms as established causes.
