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
`Operator.apply()` uses ordinary autograd. CSTOptimizer consumes the resulting
Parameter gradients without installing an optimizer-specific backward route.

`Operator` は nn.Module ではない。所有権、checkpoint の key、optimizer の
Parameter 参照は既存 Module のまま。`atoms.p` の更新や置き換え、dtype / device
変換後も live view は現在の値を使う。Module 自体を差し替えたときは CSTLinear
が view を再構成する。Chart と Kernel は具体的な ChartState / 共通 KernelState を参照する。
利用者は Layer に ChartSpec / KernelSpec を渡し、新しい意味には対応する backend 実装を用意する。

`declaration()` と `spec.bind()` は設定境界の処理。scalar の読み戻しや明示点の
snapshot は forward / backward / CUDA Graph capture 中に行わない。trainable な
座標の snapshot は固定の現在値であり、実行時の Tensor や勾配を置き換えない。
view は live な設定を使うので、設定変更後に宣言を利用する場合は改めて取得する。

CSTLinear / CSTConv2d は宣言から State を構築する。CSTLinear はbackend共通の
Registry / SelectorからTorch・CUDAのAlgorithmを選択する。
汎用宣言を受け入れる Algorithm は、その数学的・数値的契約に適合するものだけを
登録する。入出力 Chart の組を将来なくす場合も、正規化領域と勾配を照合して移行する。

## normalized radial の共通 dispatch

`CSTLinear` の `chart` と `kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT` が、
一枚の regular 3D Euclidean Chart 上で Triweight を演算子全体の点で L2 正規化する
radial 演算を宣言する。signed amplitude / clamped log width / center の座標であり、
DirectAmpWidth の activity 座標や Chart ごとの正規化とは別の意味である。

`atoms=` は整数、Tensor、nn.Parameter、Atoms を受け取る。Tensor は detach して
複製、Parameter は同一オブジェクトを登録、Atoms は owner ごと再利用する。
正規化用にも同じ live Operator / KernelState / ChartState / Atoms を使う。
公開 Linear は CSTLinear に統一し、旧専用クラスと checkpoint adapter は削除した。

共通RegistryとSelectorがPlanを選び、各Algorithmが対応条件・入力・実行を扱う。
Operatorは共通Chart/Geometry/Kernelへの入口を保ち、normalized専用のOperator型や
adapterは追加しない。regular siteの配置情報はnormalized Algorithm内で既存宣言から得る。
regular Product と連続 Strip は同じ点配置で実行できる。
固定 kernel・正規化・幅・Euclidean site の意味が一致しない場合は、この専用
algorithm に流さず一般の Torch 参照経路を使う。

`CSTLinear(selector=...)` の選択器は実行方針で、OperatorSpec と checkpoint には入らない。
full / window も通常の登録済み Algorithm として Plan に記述する。
launch 設定は各 Algorithm の Recipe に置く。対応する固定 metadata は設定境界で
snapshot し、buffer の置換・version・dtype/device が変わったら再構成する。
atom の値は毎回現在の Tensor を使う。通常 forward / backward / Graph capture では
snapshot しない。metadata を変えた後は capture の外で一度 forward する。

ChartSpec / ChartState は ABC。具体型は charts/ に置き、Layer は compile_chart で対応する
State を構築する。Geometry と Pattern はそれぞれ geometry/ と patterns/ に置く。
