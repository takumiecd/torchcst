# Connected M16 contiguous-row experiment (2026-11-09)

This benchmark-local algorithm preserves the revision-9 connected physical kernel,
its FP32 positive coefficient generator, complete cross terms, and first-order
VJP. It is not registered in the production dispatcher.

The candidate transposes input and output cotangents to `[site, row]`, stores
shared moments as `[node, feature, row]`, and assigns one atom to a 256-thread
CTA with adjacent lanes handling adjacent rows. Atom responses use `[atom, row]`.
The parameter VJP reduces rows with warp shuffles and reuses the response already
computed for dX. Owner aggregation and the hierarchy keep the preceding schedule.
Input transposes and moment reconstruction remain inside the timed eager step;
forward responses and moments are not saved for backward.

Qualification uses the existing Linear benchmark runner, a separate Plan catalog,
and Cases at B=256, A=8N, N=2048/8192/16384. The controls are the preceding owner
schedule and Dense, without changing their identities. All are measured as complete
forward, loss, backward (including dX), fused AdamW, and physical-domain restoration
steps, with allocated and reserved memory peaks. Cold compilation is excluded.

Before interpreting speed, run independent FP64 oracle Y/dX/all physical gradients,
optimizer-state gates from the fixture, retained/multiple forwards and version-guard
tests, then full A/B comparisons against the preceding qualified schedule.
[Results are recorded separately](connected-m16-coalesced-results-20261109.md).
No training quality, CUDA Graph, or default adoption claim.
