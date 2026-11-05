# Kernel の宣言と状態

[English](README.md) | **日本語**

`KernelSpec` は atom 関数の数学的な意味を宣言する。Tensor や backend の選択は持たない。
公開 API の正本は [README](../../../README.ja.md)。旧 `presets` の互換 alias は置かない。

## 三つの部品と結合

1. **Profile** は関数の形。`GaussianSpec()`、`TriweightSpec()`、`ChebyshevSpec(modes=8)` など。
2. **Factor** は評価と正規化。`point` は点評価、`l2` は点評価＋離散 L2、
   `cell_mass` はセル積分＋全質量 1 の正規化。
3. **CoordinateSpec** は atom の座標・初期化・更新の宣言。`polar`、`direct`、`physical` で構築する。

```python
from torchcst import BandwidthBounds, RadiusBounds, TriweightSpec, kernels

bounds = BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1)
coordinates = kernels.polar(amplitude_max=1.0, w_c=0.1, bounds=bounds)
product = kernels.product(kernels.l2(TriweightSpec(), floor=1e-6), coordinates=coordinates)
radial = kernels.radial(kernels.l2(TriweightSpec()), coordinates=coordinates)

cells = kernels.product(
    kernels.cell_mass(base_tile_sites=1, merge_factor=2),
    coordinates=kernels.polar(
        amplitude_max=1.0, w_c=0.3,
        bounds=RadiusBounds(minimum=0.5, birth=4.0, maximum=50.0, upper_floor=1.0),
    ),
)
physical = kernels.product(kernels.cell_mass(), coordinates=kernels.physical(radius_init=(0.5, 1.0)))
```

`product(output_factor, input_factor, coordinates=...)` は Chart ごとに距離・評価・正規化し、
その結果の積をとる。Factor を一つだけ渡すと両 Chart で共用する。多次元 Chart 内では
全次元の距離から一つの Factor を作り、全サイトで正規化する。座標軸ごとの積ではない。

`radial(factor, coordinates=...)` は各 Chart の無次元距離の二乗を加算してから一度だけ
Profile を評価し、L2 は Weight 全サイトで正規化する。同じ `(output_chart, input_chart)`
を使える。現状は Torch materialized 参照経路。単一 `chart=` の radial も維持する。
`cell_mass` は二つの 1D FlatTorus RegularGrid と半径座標の product に限定し、radial は拒否する。

`bounds=` は共有境界、`input_bounds` / `output_bounds` は Chart ごとの境界。点評価の
`BandwidthBounds` は sigma、セル積分の `RadiusBounds` は物理支持半径で、意味を混ぜない。
動的幅は CoordinateSpec が所有し、Factor の `sigma=` との二重宣言を拒否する。
固定幅は `product(l2(..., sigma=...))` と宣言し、必要なら `amplitude(...)` で振幅を加える。

コンパクト Profile の `l2(..., floor=...)` は正のノルム床を持つ。Gaussian の L2 は
log-softmax による安定した正規化で床を持たない。`cell_mass` の質量正規化は評価定義に
含まれ、`NormalizationSpec` に独立した continuous-mass 規則を宣言しない。

`polar` は角度方向の task motion、上限付き radial activity、activity に依存する幅と
radial regularization を使う。`direct` は上限付きの符号付き振幅と、採用された振幅の
移動量で進む独立した activity を使う。どちらも導出幅を通る task 勾配は detach する。
Sphere / FlatTorus の atom 中心には Geometry の勾配射影・retraction・vector transport を
適用する。学習する非 Euclidean 観測 Chart は別の更新 policy を必要とするため、
現状の CST optimizer では拒否する。セル質量の半径・階層・CUDA Plan は
[半径セル質量ガイド](../../../docs/radius-cell-mass.ja.md) を参照する。

## 配置と所有

| 配置 | 責務 |
| --- | --- |
| `profiles/` | 関数の形の不変な宣言 |
| `parameterizations/` | 振幅・幅・activity の座標解釈 |
| `normalization.py` | 点評価の正規化規則・領域・床 |
| `spec.py` | ProfileBinding、CoordinateSpec、KernelSpec、初期化・更新方針 |
| `factories.py` | 利用者向けの純粋な宣言構築 |
| `options.py` | site / atom chunk、checkpoint など実行設定 |
| `state.py` | 共通 KernelState / ProfileState、固定 Tensor と checkpoint |
| `../_backends/torch/` | 評価・微分・初期化・更新・geometry の参照計算 |
| `../_backends/cuda/algorithms/` | 対応する意味を実行する CUDA Algorithm |

`KernelState` に演算メソッドを置かず、学習する `atoms.p` は Module が所有する。
`declaration()` は設定境界の snapshot であり、forward / backward / CUDA Graph capture では
呼ばない。更新時は既存の座標法則と geometry の射影・retraction・vector transport を維持する。
`KernelOptions`、実行 Plan、CUDA launch の選択は数学的な宣言から分離する。

model / optimizer の checkpoint は数学的な宣言を検証し、`torch.load(weights_only=True)`
で復元できる。旧形式・異なる Profile / 正規化 / 座標法則を読み替えない。
未対応の形・revision・状態方針を拒否し、組み込み ID を継承しただけの subclass を実行しない。

`NORMALIZED_RADIAL_TRIWEIGHT` は単一 operator Chart 上の全体 L2 と log-width の別契約。
`provided_atoms` のため `atoms=Tensor | nn.Parameter | Atoms` で既存座標を渡す。
LogWidthSpec の端点は FP64 buffer で保持する。任意の宣言が CUDA Registry に自動登録される
わけではなく、対応・検証・承認・既定 dispatch への採用は別の段階である。
