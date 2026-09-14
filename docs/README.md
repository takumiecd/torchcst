# Design documents

`README.md` is the public API contract. The documents retained here contain the
mathematical derivations and implementation design of the continuous-only
rewrite. Experiment runners and reports live in the companion
[cst-experiments repository](https://github.com/takumiecd/cst-experiments):

- [local-adam-math.ja.md](local-adam-math.ja.md) — CSTLocalAdam equations: atom-local α/C transport, whitening, regularized direct updates, state and approximation boundaries;
- [local-visible-adam-math.ja.md](local-visible-adam-math.ja.md) — projected visible diagonal-Adam operator equations, Γ transport, geometric-mean metric, and approximation boundaries;
- [dense-visible-adam-math.ja.md](dense-visible-adam-math.ja.md) — dense visible m/v baseline, exact atom-local metric, state cost, and approximation boundary;
- [first-order-rebuild.ja.md](first-order-rebuild.ja.md) — current optimizer API, equations, experimental metrics and memory boundaries;
- [normalized-optimizer.ja.md](normalized-optimizer.ja.md) — composable normalized optimizers: LinearJGHAtomGrad, independent numerator/denominator moments, replaceable solvers, and elementwise box updates;
- [tangent-operator-design.ja.md](tangent-operator-design.ja.md) — proposed parameter-driven, kernel-aware Gram actions and matrix-free recompression;
- `implicit-projected-adam-decisions.ja.md` — historical second-order implementation decisions;
- `implicit-projected-moment-transport.md` — moment-transport derivation;
- `implicit-projected-moment-transport.ja.md` — Japanese version;
- `second-order-pullback-moment-spaces.md` — second-order moment-space analysis.

Historical structural-policy and mutation designs remain available in Git
history but are not part of the current architecture.

- [CSTParameterAdam: パラメータ数に比例する状態](parameter-adam.ja.md)
