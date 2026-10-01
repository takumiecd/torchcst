# Operator の宣言と実状態

単一 Chart を基本の形とし、現行の Kernel に必要な入出力 Chart の組も扱う。
二つの大きな実行系を作らず、観測点の配置だけを区別する。

| 型 / field | 意味 |
| --- | --- |
| `SingleChartSpec.chart` | logical shape が `[out, in]` の一枚の Chart |
| `ChartPairSpec.input_chart / output_chart` | 独立した入力・出力の観測点。matrix shape は `[output.features, input.features]` |
| `OperatorSpec.layout` | 上記どちらか一つ。入力・出力 Tensor と Chart の枚数は別の概念 |
| `OperatorSpec.kernel` | atom の Profile・振幅・幅・正規化・更新の宣言 |
| `OperatorSpec.operation_id / revision` | この段階では `linear` / `1` |
| `Operator.charts / kernel / atoms` | 既存 Module が持つ実状態への参照 |
| `Operator.p` | `atoms.p` の現在の Tensor。コピーや独自 Parameter は作らない |

単一 Chart の Kernel は radial、Chart の組は separable とする。amplitude の
wrapper は inner Kernel に同じ判定を適用する。片方の配置をもう片方として
解釈したり、Chart を複製して枚数を合わせたりしない。

```python
from torchcst import CSTLinear

# chart.shape == (out_features, in_features)
single = CSTLinear(chart=chart, atoms=8, kernel=direct_kernel)
single_spec = single.declaration()  # 設定時の不変な snapshot
operator = single.operator  # 計算用の live view
y = operator.apply(x)  # Torch の materialized 参照実装

pair = CSTLinear(input_chart, output_chart, atoms=8, kernel=separable_kernel)
y = pair.operator.apply(x, algorithm="factored")

# Spec と既存の状態の整合を、設定時に確認して結びつける。
bound = single_spec.bind(
    charts=single.cst_charts(), kernel=single.kernel, atoms=single.atoms
)
```

`CSTLinear` の Torch materialized / factored 実装、dense weight、atom の評価も
この入口を使う。計算本体は `_backends/torch/operators/linear.py` に置く。
Torch の autograd は入力、全 atom、学習可能な Chart の Tensor を追跡する。
`p` を明示すれば、derivative engine の候補点や atom 部分集合も評価できる。

`Operator` は nn.Module ではない。所有権、checkpoint の key、optimizer の
Parameter 参照は既存 Module のまま。`atoms.p` の更新や置き換え、dtype / device
変換後も live view は現在の値を使う。Module 自体を差し替えたときは CSTLinear
が view を再構成する。custom Kernel の既存実行は宣言を要求しないが、宣言を
使う際には従来通り `declaration()` の明示実装が必要である。

`declaration()` と `spec.bind()` は設定境界の処理。scalar の読み戻しや明示点の
snapshot は forward / backward / CUDA Graph capture 中に行わない。trainable な
座標の snapshot は固定の現在値であり、実行時の Tensor や勾配を置き換えない。
view は live な設定を使うので、設定変更後に宣言を利用する場合は改めて取得する。

Spec だけから新しい Module を生成する factory はまだない。Triton / tiled の
既存実行と normalized Strip の CUDA registry は、既存の最適化入口を維持する。
汎用宣言を受け入れる Algorithm は、その数学的・数値的契約に適合するものだけを
登録する。入出力 Chart の組を将来なくす場合も、正規化領域と勾配を照合して移行する。

## normalized Strip の CUDA 接続

`NormalizedStripLinear.declaration()` も共通の `OperatorSpec` を返す。これは一枚の
3D Euclidean Chart 上で、Triweight を operator 全体の点で L2 正規化する radial
演算。座標は signed amplitude / clamped log width / center で、DirectAmpWidth の
activity 座標や Chart ごとの正規化とは別の意味である。

CUDA 内の `NormalizedStripSpec.from_declaration()` はこの固定の数学契約だけを
受け入れる。Kernel、Profile、正規化領域・床、幅、Geometry、配置の異なる宣言を
拒否し、対応 device / precision / Recipe の判定は既存 registry に任せる。
以前の CUDA 内 `OperatorSpec` 名は互換 alias として残す。

contiguous Strip は同じ点を表す Product Chart に canonicalize する。非連続な
tile pitch は受け入れない。公開 NormalizedStripLinear は従来通り metadata を
snapshot して所有し、CSTLinear の live Chart binding と区別する。
