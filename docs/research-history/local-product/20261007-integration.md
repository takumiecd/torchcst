# Small Linear validated integration

The integration branch `kernel/small-linear-integration` brings the measured
N64/N128 support ordering, local H, parameter partitions, exact order repair,
counter-free caches and tight histogram recipes together. It includes the final
records from PRs #33–#45 and #47–#53, with their source and result identities.
The serial parameter batch-loop experiment (#46) remains on its research branch;
its negative outcome is summarized in the morning report.

The integrated recipes are explicit research selections in the existing Linear
runner. Public selection is unchanged. Negative comparison recipes inherited by
the successful routes remain reproducible, without a speed recommendation.

## Contract and measured evidence

See the [morning report](20261007-overnight-morning.md) and
[mathematical/comparison scope](20261007-comparison-methodology.md). Full-domain
discrete L2 normalization, actual-support singleton checks, multi-site position
gradients, Y/dX/all source gradients and current production AdamW/Polar updates
are preserved. Forward widths evolve; the existing fixed-decoded-width VJP
contract is unchanged. No new global H allocation or cache-residency claim is
introduced. Performance fixtures start with decoded rho > 1.

The GPU measurements and independent gradient/optimizer checks were performed
on the recorded constituent sources, not this integration head. The integration
adds no new numerical algorithm. It combines the histogram and cache options in
their separate constexpr paths. Merge conflicts retained the complete cache
recipe registrations and current counter-free diagnostic implementation.

## Integration validation

- Full CPU suite: 893 passed, 1,012 CUDA/DB-dependent skipped, 22.78 seconds.
- Changed Python files pass Ruff; `git diff --check` passes.
- Wheel and source distribution build successfully.
- 24 local plan catalogs / 178 Cases pass declaration and worker-snapshot checks;
  24 fresh baseline/candidate/dense prepare/check pairs pass.
- All 38 recipe occurrences in the combined, cache atom16, counter-free cache
  and tight histogram catalogs retain all 38 effective recipe properties from
  their respective measured branch heads.
- GPU correctness and complete-step timing are inherited from the independently
  verified constituent records. No new GPU experiment or timing was run for the
  integration head; CUDA skips are not counted as GPU validation.

Host logs, package builds, prepared comparisons and recipe-equivalence results
remain in ignored `benchmarks/cuda/linear/evidence/integration-*` paths. Original
raw evidence and research branches are retained. GitHub CI for the integration
PR must pass before merging; local main is fast-forwarded only after that merge.

The next investigation is N128 middle/mixed update cost and work division, with
compact physical metadata views and G4 partitioning left as unmeasured proposals.
