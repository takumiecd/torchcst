# Pattern の宣言と State

Pattern は一つの軸・部分空間の点列を宣言する。空間の metric を定める Geometry と、
点列を組み合わせる Chart は同じ階層の別ディレクトリに置く。

```text
patterns/
  spec.py       PatternSpec / LinePatternSpec / GridPatternSpec / PointsPatternSpec
  presets.py    line / grid / points の純粋な宣言構築
  state.py      PatternState の小さな Tensor と checkpoint
_backends/torch/patterns/
  execution.py  宣言型と版から実装を選ぶ
  grid.py       必要な格子点の生成
  points.py     明示した点列の選択
```

Line / Grid は shape、start、spacing を持つ。Points は座標表を持つ。
PatternState は axis の scalar・座標表だけを所有し、全 Cartesian product を保持しない。
positions / bounds の計算は backend の関数に置く。

```python
import torch
from torchcst import PatternState, pattern_presets as patterns
from torchcst._backends.torch.patterns import execution

spec = patterns.grid((2, 3), spacing=(0.2, 0.3))
state = PatternState(spec)
selected = execution.positions(state, torch.tensor([0, 5]))
assert selected.shape == (2, 2)
```

spacing と low/high は排他的。暗黙の範囲はない。low/high の singleton axis は
spacing=0 を持てる。declaration は設定時だけ snapshot し、演算は live Tensor を使う。
