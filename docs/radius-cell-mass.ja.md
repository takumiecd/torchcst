# 半径によるセル質量カーネル

[English](radius-cell-mass.md) | **日本語**

現在の公開 API は [README](../README.ja.md#kernel-を組み立てる) を正本とする。
このガイドは `cell_mass` の利用、階層の数式、明示的な CUDA Plan 選択を扱う。
測定・実装の経緯は [研究記録](cell-mass-kernel-flow.ja.md) に残す。
現在の最適化された公開 eager 構成は
[推奨 CUDA 構成](../README.ja.md#公開-api-で推奨する-cuda-構成)を参照する。

## CPU で動かす

`RadiusBounds` は物理半径を宣言する。点評価の `BandwidthBounds` が表す sigma とは
単位が異なる。`ChebyshevSpec(modes=M)` は有限 Fourier 密度のモード数を選び、
整数 M=1..512 に対応する。指定半径の外にある確率は Profile の結果であり、
一定の許容誤差まで M を自動的に増やす契約ではない。

```python
import torch
from torchcst import CSTLinear, ChebyshevSpec, RadiusBounds, chart_presets, kernels

n = 128
spacing = 128 / n
grid = chart_presets.regular_grid(n, spacing=spacing, geometry="flat_torus")
layer = CSTLinear(
    charts=(grid, grid), atoms=8,
    device="cpu", dtype=torch.float64, backend="factored",
    kernel=kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8), base_tile_sites=1, merge_factor=2),
        coordinates=kernels.polar(
            amplitude_max=1.0, w_c=0.3,
            bounds=RadiusBounds(
                minimum=spacing / 2, birth=4 * spacing,
                maximum=50 * spacing, upper_floor=spacing,
            ),
        ),
    ),
)
y = layer(torch.randn(4, n, dtype=torch.float64))
input_radius, output_radius = layer.support_radii()
assert y.shape == (4, n)
```

必要なのは固定の 1D FlatTorus regular grid 二つと `product` 結合である。
`radial(cell_mass(...))` は未定義なので拒否する。`support_radii()` の戻り値は
`(input_radius, output_radius)`。Chart と中心列の `(output, input)` 順とは区別する。

## 階層と座標の契約

Chart のサイト数と `base_tile_sites` の比は `merge_factor` の冪でなければならない。
各 atom は半径 R 以上の最小のタイル幅 T を選び、局所周期を次で定める。

```math
P = \left(\left\lceil \frac{2R}{T}\right\rceil + 1\right)T = 2L.
```

最も近いタイル境界に整列した一周期分の窓を保持する。セルを Chart 周期で折り返して
加算し、P が Chart 周長を超える場合の重複像も含め、質量の総和を 1 とする。
階層と窓の離散的な選択は、各 backward 分岐内で固定する。

| 座標則 | atom の列 | 半径の更新 |
| --- | --- | --- |
| `kernels.polar(...)` | `[s, t, output_center, input_center]` | Polar activity から導出し、半径を通る task 勾配は detach |
| `kernels.physical(...)` | `[amplitude, support_radius, output_center, input_center]` | 物理半径を直接微分し、`radius_init` / `radius_floor` で初期化・下限を宣言 |

Polar は既存の更新則と moment transport を使う。半径の境界・初期値は各 Chart の
半周期に収める。Gaussian は sigma の契約を保ち、このセル積分には入らない。
旧 sigma-cell-integral API や preset の checkpoint を半径として読み替えない。

## CUDA Plan を明示する

CPU 参照は一時的な Profile factor を使う。CUDA の標準的な小規模構成は
N<=128、A<=4096、M=1..16（奇数も含む）の一次 forward/backward を対象とする。
以下は CPU 例に続けて、CUDA 環境で実行する。

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import BLOCK_LOCAL

layer = layer.to("cuda")
layer.selector = FixedSelector(BLOCK_LOCAL, registry=make_algorithm_registry())
y = layer(torch.randn(4, n, device="cuda", dtype=torch.float64))
y.square().sum().backward()
```

選択した Plan は shape・dtype・設定を別途検査する。shared memory の上限などにより、
上記 N/A/M の範囲内でも使えない組合せがある。自動 dispatcher はこれらの研究用 Plan
を選ばない。Plan 定数はすべて
`torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans` にある。

明示的な候補`GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048`は、512→2048/2048→512で
A8192、または128→2048/2048→128でA4096、入力最大256行、FP32、
Polar M8/base1/merge2/spacing1・radius bounds[.5,50]だけを
対象に、既存のgeneral native演算を再利用します。元の`GLOBAL_ACTIVE_GRAD4_NATIVE`の
1024-site制約は維持します。独立検証を行うFFN実験に向けた準備であり、公開dispatcherの
既定採用、汎用的な長方形サイズ拡張、実測の高速化を示すものではありません。

| Plan | 対象・計算方法 |
| --- | --- |
| `REFERENCE` | 階層間で一致する周波数の moment を再利用 |
| `DIRECT` | moment を個別に構築。moment 自体を迂回する意味ではない |
| `HYBRID_SERIAL` / `HYBRID_OVERLAP` | 入力・出力で独立に指定した低階層を直接射影・展開。後者は補助 stream と合流を使う |
| `BLOCK_LOCAL` | batch 行ごとに moment と完全 response を一度構築し、出力タイルで再利用 |
| `LOCAL_TILE_MOMENTS` | N128 Polar、B<=32、A<=4096、M8、base1/merge2。小タイル moment を shared に置き、global moment は level5 以降を保持 |
| `GLOBAL_OWNER_COMPACT_PARTIALS` | 入出力が同じ N1024 Polar、B<=4、A<=65536、M8、base1/merge2。完全 response・中心 Jacobian と小さい係数 partial を再利用 |
| `GLOBAL_OWNER_COMPACT_KEYED` | 同じ N1024 条件。forward owner key から必要な moment bank を選ぶ |

研究用の`GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32`と
`GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C`は、同一軸のN1024/2048/4096/8192/16384、
FP32・B<=4・A<=2^24、Polar M8/base1/merge2/spacing1・radius[.5,50]に対応する。
他の旧owner Planやshared経路の対応範囲は拡張していない。公開入口は引き続きeager-onlyで、
研究用whole-step Graphは固定5%atom/B4 cohortに限定する。
大規模の正しさと時間・capture込みメモリの検証範囲は
[大規模研究記録](research-history/cuda-linear/radius-large-scale-20261105.ja.md)を参照する。

N1024 Plan は上の N128 例には指定できない。対応する Chart・atom 構成を作ってから
`FixedSelector` へ渡す。各 Plan の性能とメモリは対象 workload ごとの検証であり、
モデル全体で得た比率を単独 layer に転用しない。例えば `LOCAL_TILE_MOMENTS` は保存量を
減らせても、単独 B32 layer では `BLOCK_LOCAL_BATCHED_CACHED_PARALLEL` より遅い場合がある。
測定値・条件は [研究記録](cell-mass-kernel-flow.ja.md) を参照する。

これらの pipeline は eager 専用。autocast、決定的 reduction の資格付け、CUDA Graph、
高階微分には対応しない。CUDA 上の atom 値検査は非同期で、不正値の device assertion は
後続の CUDA API または同期で報告される。shape・設定の不正は host で同期的に検査する。
FP64 の Linear 微分が動くことは、FP64 GPU optimizer 更新の資格付けを意味しない。

## Linear と optimizer の選択を分ける

FP32 radius Polar の eager SGD / AdamW 学習では、更新用の
`RADIUS_FLAT_TORUS_FUSED` を `CSTOptimizer(update_selector=...)` へ明示的に渡せる。
以下は、FP32 の CUDA model と base optimizer を用意した後の設定例である。

```python
from torchcst import CSTOptimizer, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.polar_update.plans import RADIUS_FLAT_TORUS_FUSED

optimizer = CSTOptimizer(
    base_optimizer,
    model=model,
    update_selector=FixedSelector(
        RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry(),
    ),
)
```

これは既存 Polar 更新則の実装選択であり、Linear の Plan とは独立している。
モデルの device / dtype を決めてから optimizer を作る。上の FP64 Linear 例に
そのままこの更新 Plan を適用しない。共通の group・moment・finite 検査の意味は
[optimizer ガイド](cst-optimizer.ja.md) を参照する。

## 保存値と実装の境界

`BLOCK_LOCAL` は global に保存した moment と完全 response を全出力タイルで共有し、
係数構築とサイト展開を次の compute kernel の shared memory で融合する。
動的 shared memory は 48 KiB を上限とする。

内部 forward は `(Y, input_moments, atom_response)`、reverse は
`(dX, grad_output_moments, grad_response)` を返し、atom-gradient kernel は両側の
保存射影を再利用する。公開 forward の戻り値は Y のままである。atom-only backward は
input 勾配を省き、一つの projection kernel で必要な moment / response を作る。
未使用 bank は省略し、必要な merge 依存は保持する。固定容量 moment buffer の省略 slot
はゼロと定義する。参照経路は FP64 moment と完全 response を global に保存し、
hybrid は直接計算する低階層の buffer を省く。

N1024 compact Plan は完全 response と中心 Jacobian を一度構築し、射影後に一時 moment
を解放する。backward は dX を完成させてから、同じ保存値を atom 勾配で再利用する。
これらの保存方針は数学的な Profile や Polar 更新則を変更しない。
