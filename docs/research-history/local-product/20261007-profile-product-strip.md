# Polar profile product: whole-chart input Strip

## Contract and initial scope

The single Strip Chart spans all operator sites. Input tiles schedule computation;
they do not define independent atoms or independent normalization domains.
For atom a, whole input sites j and output sites i,

\[
W_a(i,j)=\mathrm{amp}_a\frac{u_a(i)v_a(j)}{D_a},\qquad
D_a=\max(\|u_a\|_2\|v_a\|_2,\epsilon).
\]

Input coordinates include the live pitch:
\(s_j=o_i+(j\bmod T)s+\lfloor j/T\rfloor P\).
The global norm, floor state, singleton certificates and center corrections
are computed once. Each small contraction receives this metadata restricted to
its input tile. Forward contributions and canonical atom VJPs sum across tiles.
Width task gradients remain detached; the existing Polar update evolves width.

Initial support: Euclidean Line axes with equal spacing, input Strip axis 1,
2..128 output sites, 16..1024 input sites, tile sizes 16/32/64/128, CUDA FP32,
batch 1..64, Polar triweight product. Chart parameter gradients are unsupported;
nontrainable live pitch is read on replay. This does not tile both axes.

The `tiled` recipe uses the small local contraction; `reuse` uses the tuned
ordered layout and launch settings from PR #59. The initial Python scheduler
visits each tile and retains its backward metadata. Correctness and complete
step time/allocated/reserved peaks determine the next optimization; no speedup
is assumed from tiling alone. These research plans are not public dispatch policy.

## Validation and evidence

Pending actual GPU checks and measurements. The independent oracle constructs
full raw matrices per atom (chunked in atom count), normalizes over every site
pair, and checks Y, dX and all canonical gradients. Boundary/tail/floor cases,
source/scalar/pitch snapshots and captured Polar updates are separate checks.
Benchmark fixtures use output 64, input 256/512/1024, tile 64, pitch 68,
batch 32, approximately 5% atoms and initial width 3 or 8.
