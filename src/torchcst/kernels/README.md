# Kernel と Profile の宣言

第一段階として、数学的な宣言と組み込みクラスの Torch 計算本体を分離した。
公開 `Kernel` / `Profile` は、既存 constructor、buffer、checkpoint、custom subclass
との互換入口として残す。新しい宣言は frozen dataclass で、Tensor 評価や device
選択を持たない。Geometry、Chart、CUDA registry への接続は段階的に進める。

## 配置と責務

| 配置 | 責務 |
| --- | --- |
| `profiles/` | `TriweightSpec`、`GaussianSpec` などの無次元距離に対する関数の形。幅や正規化は含まない。 |
| `parameterizations/` | 固定幅、振幅依存幅、Direct / Polar の座標解釈と数値制約。 |
| `normalization.py` | 非正規化 / 離散 L2、対象領域、ノルム床。 |
| `spec.py` | `ProfileBinding`、`KernelSpec`、初期化・座標更新の宣言。 |
| `_declarations.py` | 既存クラスから宣言を作る設定用 adapter。 |
| 既存の公開クラス | 設定と buffer の所有、checkpoint 検証、計算への遅延接続。 |
| `../_backends/torch/profiles/` | Profile の評価と解析微分。 |
| `../_backends/torch/parameterizations/` | 振幅・activity・幅の解釈。 |
| `../_backends/torch/kernels/` | atom の合成、重み生成、初期化、factor 微分。 |
| `../_backends/torch/updates/` | Kernel 座標の勾配射影、更新、optimizer 状態の輸送。 |

`KernelSpec.parameterization` が幅を定義する場合、各 `ProfileBinding` の固定幅は
省く。幅を二重に宣言せず、入力・出力の幅の違いは parameterization の bounds に
記録する。`AmpWidthSpec.couple_bandwidth` は微分の契約なので Recipe に移さない。
`site_chunk`、`atom_chunk`、`checkpoint_blocks` は数学的な宣言に含めない。

## 設定の参照

```python
from torchcst import AmpWidth, Triweight

kernel = AmpWidth(sigma_min=0.2, sigma_max=1.0, profile=Triweight(0.2))
spec = kernel.declaration()
assert spec.profiles[0].profile.id == "triweight"
assert spec.profiles[0].normalization.domain == "chart_sites"
assert spec.parameterization.id == "amplitude_dependent_width"
```

`DirectAmpWidth.declaration(chart_count=1)` は単一 chart の radial 演算、既定の
`chart_count=2` は入力・出力 profile の積を宣言する。単一 chart の経路は従来通り
非正規化 profile を要求する。custom subclass の演算を組み込み ID と誤認しない
よう、宣言を使う custom class は `declaration()` を明示実装する。既存の計算
メソッドによる拡張は引き続き使用できる。

`declaration()` は設定時点の独立した snapshot である。`.to()`、buffer の変更、
checkpoint load 後は必要に応じて作り直す。scalar buffer の参照は GPU と同期し得る
ため、forward / backward / CUDA Graph capture 内で呼ばない。従来の計算経路は
この snapshot を読み戻さず、現在の buffer を使う。

## 段階的な移行

この段階では Spec 単体を実行する factory や、KernelSpec を受け取る CUDA dispatch
はまだ導入していない。公開クラスの削除や checkpoint 形式の変更も行っていない。
[Operator](../operators/README.md) が単一 Chart / Chart の組と Kernel を束ね、
既存 Module の実状態を参照する。Torch linear はその入口を使う。
次は CUDA の契約への接続を整え、その後に互換メソッドを減らす。

Geometry / Chart の宣言は `../geometry/spec.py`、座標の計算は
`../_backends/torch/geometry/`、`charts/`、`patterns/` に分ける。
Torch 実装は CPU 限定ではなく、既存の Torch Tensor device 上で実行する。
CUDA 専用の最適化は `nn/_backends/cuda/algorithms/` に引き続き置く。
