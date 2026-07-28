# Absorb & GramService design

Status: **approved design, staged implementation** (2026-07-28).
Theory source: `cst/theory/drafts/null_set_and_init.tex` v0.4 (P1 null-set
classification, P2 gauge soundness / locality theorem, P3 logdet init) plus the
discussion recorded in the cst arctx run. This document is the contract for the
implementation stages below; the tests listed per stage are the acceptance
criteria.

## 1. What absorb is

One structural primitive that subsumes merge and prune-with-refit.

For a live atom `k` with amplitude `c_k` and atom matrix
`psi_k = u(s_k) v(t_k)^T`, solve the ridge-regularized local least squares

```
alpha* = argmin_alpha || psi_k - sum_{j in N(k)} alpha_j psi_j ||_D^2 + ridge * ||alpha||^2
res2   = || psi_k - sum_j alpha*_j psi_j ||_D^2
```

over the neighborhood `N(k)` = live atoms within radius `R` of `z_k = (s_k, t_k)`
(locality theorem: Gram inverse entries decay exponentially under tau-separation,
so the neighborhood solve approximates the full leave-one-out solve with
certified error).

The **absorb operation** = kill atom `k` and redistribute its mass:

```
w_j += c_k * alpha*_j   for j in N(k)      # then death(k)
```

Case behavior falls out of the same formula — this is the point:

| situation                    | res2      | alpha*            | effect            |
|------------------------------|-----------|-------------------|-------------------|
| near-duplicate (rho -> 1)    | ~0        | concentrated ~1   | merge             |
| group dependency (oversample)| ~0        | spread over group | group absorb      |
| no influence (c~0 or D-null) | ~0        | ~0                | free prune        |
| isolated, effective          | ~||psi||^2| ~0                | paid prune        |

**Acceptance is a single number**: cost `S = 0.5 * c_k^2 * res2` (the exact
leave-one-out training-loss increase for quadratic loss). alpha* is only the
delivery manifest, never an acceptance input.

Two guards (not acceptance criteria — safety rails):

1. **Double audit**: res2 is D-weighted (data second moment). An atom invisible
   to the data (empirical null) absorbs for free on train but moves the
   function. Always record `||Delta W||_F = |c_k| * res_F` (Frobenius residual
   of the same alpha*) alongside the D cost.
2. **Trust region on alpha***: near eps-null combinations the unridged solve is
   ill-conditioned and alpha* explodes. The ridge term bounds ||alpha*||; the
   res2 reported must be the ridged residual (bias is paid honestly in the
   cost).

Sequential semantics: absorbing one member of an over-dense cluster changes the
residuals of the rest (support shrinks). Chains are planned by simulation at
decision time (Gram depends only on positions, so removing a row/col and
updating running amplitudes is exact — no re-forward needed) and applied **in
list order** inside one atomic ticket.

## 2. Layering (decided)

- **Storage layer**: new op `SynapseAbsorb` in the op vocabulary. Neutral
  mechanism, any policy may emit it.
- **Policy layer**: composable `ActionSpec.synapse_absorb(AbsorbCourt(...))`
  plus a `GramService` the rule consults at plan time. Existing courts
  (magnitude prune, similarity merge, LC_merge) remain untouched as contrasts;
  cSET/cRigL stay faithful baselines with no absorb.
- **Catalog layer**: a named RENT preset is added **only after** the cst-side
  verification gates (V1/V5/V7) pass. No catalog entry before that.

## 3. Stage 1 — `SynapseAbsorb` (storage)

File: `src/torchcst/storage/synapse.py` (additive), export in
`storage/__init__.py`.

```python
@dataclass(frozen=True)
class SynapseAbsorb:
    site: str
    dying: int                 # one logical id
    receivers: Tensor          # int64 [r], logical ids
    delta_w: Tensor            # float [r], = c_dying * alpha*
```

Contract:

- One op absorbs one atom. A structural event may carry an ordered list of
  absorbs; `prepare()` validates them **in order** against the running state:
  at each step `dying` must be live, all `receivers` live, `dying not in
  receivers`. A receiver of an earlier op may die in a later op (chains are
  legal); a dead atom may never receive.
- Apply = `w[rows(receivers)] += delta_w` then the existing death path for
  `dying` (slot free, id retired, lineage recorded). All inside the existing
  two-phase prepare/commit; a validation failure anywhere rolls back the whole
  ticket (nothing partially applied).
- `s`/`t` are never touched: Gram is position-only, which is what makes chain
  simulation exact.
- Optimizer follower: the dying row is zeroed exactly as death does today;
  **receiver rows keep their optimizer moments** (the amplitude shift is a
  structural transfer, not a gradient event). Document this in the docstring.
- Version bump, op log, deterministic replay: same obligations as
  birth/death/merge.

Tests (`tests/torchcst/test_absorb_op.py`):

- exact-duplicate pair, `delta_w = c_dying` on the twin: represented matrix `W`
  is **bit-preserved** (gauge move).
- ordered chain: absorb A into B, then B into C, in one ticket; final state and
  op log deterministic; replay reproduces it.
- invalid ops (dead dying id, receiver == dying, dead receiver mid-chain in
  wrong order) fail `prepare()` and leave state untouched.
- optimizer follower: dying row zeroed, receiver moments preserved.
- quota-free storage level: `store.apply([...])` works without any policy.

## 4. Stage 2 — `GramService` (representation)

New file: `src/torchcst/representation/gram.py`, export in
`representation/__init__.py`. Read-only; imports representation + torch only
(no policy/, no engine).

Inputs at construction: the factor matrices for live atoms
`U [n_out, K]`, `V [n_in, K]` (obtained by the caller via `KernelPort.columns`
on the atom coordinates), amplitudes `w [K]`, positions `z = (s, t)`, options
`radius`, `ridge`, and optional `data` batch `X [n, n_in]` for the D metric
(`Sigma_x = X^T X / n`); identity metric when absent.

Gram entries are assembled separably: `Gamma_ij = (U^T U)_ij * (V^T Sigma_x V)_ij`
(Hadamard). v1 may build neighborhoods from a chunked `cdist` on `z`; the
**contract** is radius-based neighborhoods (cell lists are a later
optimization, not a semantic change).

API (all indices are positions into the live view, ids mapped by the caller):

```python
class GramService:
    def neighbors(self, k) -> Tensor                 # indices within radius
    def residual(self, k) -> AbsorbAssessment        # res2_D, res2_F, alpha, receiver idx
    def cost(self, k) -> float                       # 0.5 * w[k]^2 * res2_D
    def local_lambda_min(self, k) -> float           # lambda_min of normalized
                                                     # neighborhood Gram incl. k
    def plan_chain(self, order_hint=None, budget=..., cost_cap=...) -> list[...]
                                                     # sequential simulation:
                                                     # pick cheapest, remove,
                                                     # update running w, repeat
```

`local_lambda_min` exists because pairwise coherence is structurally blind to
(a) collective dependency (many moderate overlaps) and (b) fiber alignment
(distinct z, dependent factor vectors). It is the detector; `residual` is the
assessor; `plan_chain` is the planner emitting `(dying, receivers, delta_w)`
tuples ready to wrap into `SynapseAbsorb` ops.

Tests (`tests/torchcst/test_gram_service.py`):

- Gram matches the dense direct computation `<psi_i, psi_j>_D` on random
  configs (identity and data metric).
- planted exact duplicate: `res2 ~ 0`, alpha concentrated (~rho) on the twin.
- planted oversampled cluster (spacing < sigma): `res2 ~ 0` with alpha spread
  over several receivers, while **all pairwise coherences stay below** a
  threshold that would catch a duplicate — the case similarity merge cannot
  see.
- isolated atom: `res2 ~ ||psi_k||_D^2`, alpha ~ 0.
- ridge bounds `||alpha||` on a near-singular cluster; unridged lstsq blows up
  (demonstrated, not shipped).
- **cost identity**: on a quadratic toy objective, executing the absorb
  (Stage 1 op) raises the training loss by `0.5 c^2 res2_D` to numerical
  precision. This is the tex Schur identity and the V1-shaped acceptance test.
- `local_lambda_min` flags a planted fiber-aligned set (same t, spread s,
  z-distances large) that `neighbors`-radius screening alone would pass.
- `plan_chain` on a planted m-copy cluster (m=8): absorbs m-1 copies at ~zero
  cost each, leaves exactly one survivor carrying the summed mass; total W
  displacement ~ 0. (Sequential-prune soundness, tex prop 2.6(b).)

## 5. Stage 3 — policy wiring (split into 3a / 3b)

**Stage 3a (first): whole `StructuralPolicy`.** Per the framework's own
extension guidance (authoring path 2), an `AbsorbPolicy` object implementing
`capture(clock)`, `bind_instruments(...)` (to receive `KernelPort`), and
`plan(context)`: build a `GramService` from the live view + kernel columns at
event time, run `plan_chain(budget=..., cost_cap=rent)`, map view positions to
entity ids, emit the ordered `SynapseAbsorb` ops as one plan. Acceptance is
`cost <= rent` only; audit payload records per-step `cost`, `res2_D`,
`||Delta W||_F`, `||alpha||`, receiver count. Requires at most additive
admission of `SynapseAbsorb` to plan/bundle op unions — no `ActionKind`, no
quota field, no engine plan-assembly changes.

**Stage 3b (after 3a proves semantics in cst experiments): composed route.**
`ActionKind.SYNAPSE_ABSORB`, `StructuralQuota.synapse_absorb`,
`ActionSpec.synapse_absorb(AbsorbCourt(...))`, engine plan-order rule (absorb
runs **before** prune courts: canonicalization first). Deferred deliberately:
it touches contract/engine surface and should follow, not precede, evidence.

## 6. Stage 4 (later) — `torchcst/init/`

Pure functions, engine-free, emitting `list[SynapseBirth]`:
`coverage_lattice(...)` and `logdet_greedy(...)` (greedy D-optimal / DPP-MAP
using the same marginal-gain quantity `log res2` from GramService; rent-based
stopping = the K dial). Amplitudes by the variance-preservation rule
(`c^T Gamma c` matched to target second moment, random signs). Initialization
is birth on the empty state — no new score exists by design.

## 7. Out of scope / explicitly rejected

- No global dense `K x K` Gram anywhere on the hot path; neighborhoods only.
- No acceptance branching on alpha* shape (merge vs prune is an outcome, not a
  mode).
- No modification of cSET/cRigL/LC_merge semantics.
- Gaussian-family `spec.merge_atoms` (pairwise positional merge, "replace pair
  by a new atom at a fitted position") is a possible later complement; absorb
  does not need it and does not move positions.
