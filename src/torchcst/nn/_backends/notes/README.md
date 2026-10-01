# CUDA backend notes

These notes describe the implementation in the adjacent `_backends` package.
They are not additional public backend names.

| Topic | State | Note |
| --- | --- | --- |
| Strip + Torus Linear | Implemented backend and earlier measurements | [Execution design](strip-torus-gemm-prototype.ja.md) |
| Support layout | Implemented preparation and boundary routing | [Layout](support-layout.ja.md), [batched preparation](batched-support-preparation.ja.md), [station routing](local-strip-routing.ja.md) |
| CUDA plan selection | General performance tree is a proposal; normalized full/window now have a registry and compatibility selector on the research branch | [Dispatch design](cuda-dispatch-design.ja.md), [registry v1](cuda-registry-v1.ja.md) |

The public `backend="auto"` still selects the existing exact CST implementations.
Candidate algorithms remain in [CUDA Linear experiments](../../../../../experiments/cuda/linear/README.md).
