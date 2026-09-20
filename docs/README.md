# Design documents

`README.md` is the public API contract. The documents retained here describe
the supported continuous-only optimizer surface. Experiment runners and reports live in the companion
[cst-experiments repository](https://github.com/takumiecd/cst-experiments):

- [normalized-optimizer.ja.md](normalized-optimizer.ja.md) — composable normalized optimizers: LinearJGHAtomGrad, independent numerator/denominator moments, replaceable solvers, and elementwise box updates;
- [optimizer-selection.ja.md](optimizer-selection.ja.md) — current Triweight + Polar + Quadratic recommendation, fixed-LR best practices, and retained optimizer API;
- [parameter-adam.ja.md](parameter-adam.ja.md) — parameter-coordinate Adam baseline and its bounded state.

Removed optimizer experiments and their derivations remain available in Git
history but are not part of the package or current architecture.
