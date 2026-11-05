# Pattern declarations and state

**English** | [日本語](README.ja.md)

A Pattern declares a sequence of points along one axis or subspace. Geometry,
which defines the space's metric, and Chart, which combines point sequences,
live in separate directories at the same level.

```text
patterns/
  spec.py       PatternSpec / LinePatternSpec / GridPatternSpec / PointsPatternSpec
  presets.py    pure line / grid / points declaration construction
  state.py      PatternState's small Tensors and checkpoints
_backends/torch/patterns/
  execution.py  implementation selection by declaration type and revision
  grid.py       generation of needed grid points
  points.py     selection of explicit point sequences
```

Line / Grid hold shape, start and spacing; Points holds a coordinate table.
PatternState owns only axis scalars and coordinate tables, not the full
Cartesian product. Backend functions compute positions / bounds. In the layout
above, presets.py constructs pure declarations; state.py owns small Tensors and
checkpoints. execution.py selects implementations by declaration type and
revision; grid.py generates needed grid points, and points.py selects explicit
points.

```python
import torch
from torchcst import PatternState, pattern_presets as patterns
from torchcst._backends.torch.patterns import execution

spec = patterns.grid((2, 3), spacing=(0.2, 0.3))
state = PatternState(spec)
selected = execution.positions(state, torch.tensor([0, 5]))
assert selected.shape == (2, 2)
```

spacing and low/high are mutually exclusive, with no implicit range. A singleton
axis specified through low/high can have spacing=0. Declarations are snapshots
taken only at configuration boundaries; computation uses live Tensors.
