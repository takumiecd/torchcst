# CUDA backend notes

These notes describe the implementation in the adjacent `_backends` package.
They are not additional public backend names.

| Topic | State | Note |
| --- | --- | --- |
| Strip + Torus Linear | Implemented backend and earlier measurements | [Execution design](strip-torus-gemm-prototype.ja.md) |
| Support layout | Implemented preparation and boundary routing | [Layout](support-layout.ja.md), [batched preparation](batched-support-preparation.ja.md), [station routing](local-strip-routing.ja.md) |
| CUDA plan selection | Proposal; no tree or YAML loader in the runtime yet | [Dispatch design](cuda-dispatch-design.ja.md) |

The public `backend="auto"` still selects the existing exact CST implementations.
Candidate algorithms remain in [CUDA Linear experiments](../../../../../experiments/cuda/linear/README.md).
