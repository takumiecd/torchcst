# Geometry declarations and state

**English** | [日本語](README.ja.md)

GeometrySpec is an immutable mathematical declaration of Euclidean / Sphere /
FlatTorus geometry. GeometryState owns its metadata, with no fixed Tensor
buffers. Point-sequence Patterns live in [patterns/](../patterns/README.md),
and observation-layout Charts in [charts/](../charts/README.md). Backend
functions perform computation.

| Location | Responsibility |
| --- | --- |
| `spec.py` | Concrete GeometrySpec types; validation of dimensions, metrics and fixed values |
| `presets.py` | Pure euclidean / sphere / flat_torus construction |
| `state.py` | Fixed metadata, checkpoints and initialization placement hints |
| `../_backends/torch/geometry/` | Distances, embeddings, projection, retraction and vector transport |

| Declaration | Meaning |
| --- | --- |
| EuclideanGeometrySpec | intrinsic_dim and the ordinary Euclidean metric |
| FlatTorusGeometrySpec | intrinsic_dim and periods; shortest distances over independent periodic coordinates |
| SphereGeometrySpec | intrinsic_dim, radius, representation and chart_margin |

embedding_dim / center_parameter_dim / metric follow from the concrete type.
Sphere uses ambient chord distance and retains ambient / intrinsic center
representations. `radius` / `chart_margin` are Python scalars; `periods` is a
tuple. Module.to() does not round these values. Tensor arguments determine the
actual device/dtype for Torch computation; atom Parameters determine them for
CUDA execution. Required device constants live in AlgorithmState temporary
caches, not Geometry training state or checkpoint buffers.

```python
import torch
from torchcst import GeometryState, geometry_presets as geometry
from torchcst._backends.torch.geometry import execution

spec = geometry.sphere(2, radius=2.0, representation="intrinsic")
state = GeometryState(spec, dtype=torch.float64)
assert not tuple(state.buffers())
centers = torch.tensor([[0.1, 0.2]], dtype=torch.float64, requires_grad=True)
sites = execution.decode_centers(state, centers)
assert sites.shape == (1, 3)
execution.squared_distance(state, sites.detach(), centers).sum().backward()
assert torch.isfinite(centers.grad).all()
```

`declaration()` validates metadata at configuration boundaries, without reading
host scalars back during forward/backward. Replace fixed values with a newly
validated Spec when changing them. When changing RegularGrid periods, also keep
Chart spacing and the geometry declaration consistent.

Checkpoint extra state stores the full declaration, including fixed values, and
validates it against the destination. There is no loader reinterpreting it as a
model with different fixed values or the old Tensor-buffer format.
ExplicitChartState owns observation-point Parameters; Atoms owns atom centers.
