# Backward fusion: condition-scoped dispatcher adoption

The user requested selecting kernels where they win instead of requiring one
kernel to beat every existing route. Keep the measured fusion implementations
and use the existing dispatcher generator to select the observed winner for
each workload. This checkpoint changes generation requests and research records;
it adds no runtime code or GPU experiments.

## Registered policies

Read each fixed PostgreSQL snapshot filtered by adapter revision 3, exact source,
GPU, Case, environment and protocol. Retain all five measured Plans, including
the slower fusion alternatives. Generate two policies per scope:

- `speed`: lowest complete-step CUDA Graph time.
- `control-peak`: lowest time under the fastest non-fused control's measured
  capture/replay allocated peak. This ceiling is a comparison policy, not a
  user-specified numerical memory limit.

Both policies fall back to the fastest measured non-fused control for that
scope. The winning Plan includes its recipe and original observation evidence
IDs. The [registration receipt](20261007-backward-fusion-dispatch.json) records
requests, source/Case filters, dataset and dispatch hashes, winners, timing,
allocated/reserved peaks, run counts and replay checks.

| Workload | Speed choice | Control-peak choice |
| --- | --- | --- |
| L4 N64 rho1.25 | repair4 fusedback16 | uncached control |
| L4 N64 mixed | repair4 fusedback16 | uncached control |
| L4 N64 rho3 / rho8 | uncached control | uncached control |
| L4 N128 rho1.25 / mixed / rho8 | repair4 control | repair4 control |
| L4 N128 rho3 | uncached control | uncached control |
| G4 N64 rho1.25 | repair4 fusedback16 | uncached control |
| G4 N64 mixed | uncached control | uncached control |

The independent L4 repeat also selects repair4 fusedback16 for rho1.25 and mixed,
and non-fused controls under the peak ceiling. Timing samples are not independent
runs. Initial actual rho is greater than one and widths evolve during each
measured production AdamW/Polar step.

## Scope and live widths

The existing exact-table runtime key has GPU, shape, Operator declarations,
precision, required gradients and execution mode. It does **not** contain initial
or current atom rho values. A caller chooses the artifact for a measured workload;
these artifacts cannot be combined into a live rho-switching dispatcher. A check
with two actual rho-profile datasets confirms that the generator rejects their
ambiguous shared runtime key. No new support threshold is inferred from the
rho1.25 result, and arbitrary width evolution is not certified to stay faster.

These are usable research-selector artifacts with the benchmark-local Registry
and the existing model-owned persistent execution state. Public CSTLinear's
default Registry/selector is unchanged. The selector wrapper was structurally
validated but not newly timed on a GPU. The kernel complete-step measurements
remain those in the [fusion record](20261007-backward-fusion.md).

## Source identity and verification

The L4 full batch records commit `b0bbc45d`; its independent repeat and G4 batch
record full commit `b0bbc45d4cefc68bb87cc25053422fcb908e3dbf`. Their runtime source
hashes match the committed/submitted/worker proof, but the stored source IDs are
different. Preserve both identities without rewriting raw observations or
pooling them into a synthetic two-run dataset. Each generated scope uses one
independent execution and explicitly sets `min_runs=1`.

There are 24 registrations: 16 L4 full-batch policies, four L4-repeat policies
and four G4 policies. All preserve five input Plan observations, reproduce the
exact artifact and leaderboard from the frozen dataset offline, and retain the
target GPU's recorded Torch/Triton versions. Runtime loading must verify those
versions normally; CPU structural validation does not bypass that requirement
for execution. Related dispatcher/generation/local-product database tests:
**77 passed**.

Requests are tracked under
`benchmarks/dispatch/requests/local-backward-fusion-20261007/{l4,l4-repeat,g4}/`.
Full dispatch/leaderboard/dataset JSONs and the generation script remain ignored
under `benchmarks/cuda/linear/evidence/backward-fusion-20261007/dispatch/`.
Original source/result archives and all 16 verified database imports remain
preserved. No GPU was allocated for this checkpoint.

With a private configured `DATABASE_URL`, an example generation is:

```bash
PYTHONPATH=src:. python -m benchmarks.dispatch \
  --request benchmarks/dispatch/requests/local-backward-fusion-20261007/l4/64-rho1_25-speed.json \
  --output output/backward-fusion-l4-narrow
```

For offline reproduction, add `--dataset` pointing to the saved scope's
`dataset.json` and choose another new output directory. Load on the matching GPU
runtime using `load_selector` with
`benchmarks.cuda.linear.manifest.REGISTRY`; execution uses the existing research
model/layout binding. A selector replacement requires Graph recapture.
