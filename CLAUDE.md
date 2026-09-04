# CLAUDE.md

## Commands

```bash
.venv/bin/pytest
.venv/bin/ruff check .
python -m pip install -e ".[dev]"
```

## Contract

`README.md` is the public API contract. The implementation is a ground-up,
fixed-shape, continuous-only rewrite. Do not restore compatibility surfaces for
stores, structural policies, birth, death, merge, absorb, slot remapping, or the
old Pullback Adam implementation.

## Package boundaries

| Package | Responsibility |
| --- | --- |
| `geometry/` | fixed-cardinality charts and future coordinate geometry |
| `atoms/` | fixed-shape atom amplitudes and opaque kernel-coordinate rows |
| `kernels/` | stateless interpretation, initialization, and evaluation of opaque atom coordinates |
| `nn/` | user-facing CST modules such as `CSTLinear` and future convolution modules |
| `_derivatives/` | internal displacement, JVP, VJP, HVP, and contraction machinery shared across module families |
| `optim/` | model-level parameter ownership, compact moments, quartic solve, and exact-loss acceptance |

Public objects are re-exported from `torchcst`; internal placement must not leak
into the README API. Keep module ownership separate from derivative evaluation,
and keep optimizer logic independent of the concrete Linear or Conv family.

The canonical operator is `sum(kernel(p[a], charts))`. A factored matrix
expression is an optional kernel capability and must not become the model
definition. Amplitude and every other trainable kernel property belong in each
atom's opaque `p` row; kernel objects own only fixed configuration or buffers.

## Development order

Every landed milestone must remain testable. Establish dense autograd oracles
before optimizing a contraction, and establish the complete quartic correctness
path before introducing numerical shortcuts.
