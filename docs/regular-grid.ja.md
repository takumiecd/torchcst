# 一つの階層を宣言する等間隔格子

`chart_presets.regular_grid` は、一つの Weight 軸に配置するサイトを宣言する。
全サイトの座標表・index 表・固定 Tensor buffer は持たず、必要な座標だけを生成する。
Geometry は Euclidean または FlatTorus、点の配置は grid_shape / origin / spacing で定める。

```python
import torch
from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, kernels
from torchcst import chart_presets as charts

out_chart = charts.regular_grid((16, 16), spacing=0.1, geometry="flat_torus")
in_chart = charts.regular_grid(128, spacing=0.1, geometry="flat_torus")
assert out_chart.ncs == 256
assert out_chart.geometry.periods == (1.6, 1.6)

kernel = kernels.product(
    kernels.l2(TriweightSpec()),
    coordinates=kernels.polar(
        amplitude_max=1.0, w_c=0.1,
        bounds=BandwidthBounds(minimum=0.02, birth=0.4, maximum=0.8, upper_floor=0.2),
    ),
)
model = CSTLinear(charts=(out_chart, in_chart), atoms=8, kernel=kernel, backend="factored")
y = model(torch.randn(4, 128))
assert y.shape == (4, 256)
assert not tuple(model.output_chart.buffers())
```

## サイトと Weight 軸の対応

D 次元の一つの Chart は `grid_shape=(n_0,...,n_{D-1})` とサイト数

\[
N=\prod_{d=0}^{D-1}n_d,\qquad x_d(i_d)=o_d+i_d h_d,\quad 0\le i_d<n_d
\]

を持つ。最後の座標軸を最速とする row-major 順に平坦化する。
`charts=(out_chart,in_chart)` なら Weight は `[out_chart.ncs,in_chart.ncs]`。
多次元 Chart も一つの階層であり、旧二重タプルは受け付けない。

`kernels.product` は各 Chart の全次元の距離から一つの Factor を作り、その積をとる。
L2 は Chart 内の全サイトで正規化する。座標軸ごとの Profile 積ではない。
`kernels.radial` は同じ Chart ペアの無次元距離の二乗を加算し、一度 Profile を評価する。
その L2 は Weight 全サイトを対象とする。詳細は [Kernel API](../README.md#compose-a-kernel)。

## 間隔と幾何

`spacing` は有限の正値で、scalar は D 座標へ broadcast する。
`origin` も scalar または D 個の値で、最初の観測点を表す。atom の中心とは別の値。
`"flat_torus"` は各周期を `L_d=n_d h_d` として導出し、終端点を重複せず一周する。
具体的な Euclidean / FlatTorus GeometrySpec も受け付け、次元数の一致を検証する。
具体的に渡した周期は上書きしない。

FlatTorus は最短周期差、Euclidean は通常の距離を使う。Sphere には別の座標写像が
必要なので RegularGrid は受け付けない。Sphere のサイトは Explicit / Product Chart で宣言する。
半径セル積分 `cell_mass` は一周のサイトと周期が整合する二つの 1D FlatTorus Grid に限定する。

## 保存・placement・復元

origin / spacing / periods は Python の tuple、半径などは Python scalar。
保存量は O(D) で、Tensor buffer はゼロ。これは固定配置だけの保存量であり、
学習 step の GPU ピークメモリを表すものではない。

固定値を `.half()` / `.float()` / `.double()` で量子化しない。初期化前は Tensor を保持しない
placement hints を使い、実行時の距離計算は atom の中心 Tensor の device / dtype に従う。
座標を低精度に出力した場合の隣接点の数値的な識別は保証しない。指定 dtype で座標が
非有限になる配置は、座標生成時に拒否する。

CUDA 更新に必要な周期 Tensor は AlgorithmState の一時キャッシュに置く。
周期 metadata、atom placement、定数 version が変われば再構築し、checkpoint には含めない。
固定値を変える場合は新しい検証済み Spec に置き換え、Grid の間隔と Geometry の周期を整合させる。

checkpoint の extra state は固定値と配置を保存し、復元先の宣言との一致を検証する。
shape が同じでも異なる grid_shape / origin / spacing / geometry は拒否する。
旧 Tensor-buffer 形式の loader や読み替えは置かない。

## 検証

`test_regular_grid.py` は配置・境界・低精度・checkpoint を検証する。
`test_static_coordinate_metadata.py` は Tensor を保存しないこと、dtype 往復での固定値保存、
実際の atom placement による距離と勾配、周期キャッシュ更新を独立した値と比較する。
`test_kernel_composition.py` は非対称の多次元 Chart に対する Y / dX / 全 atom 勾配を検証する。
CUDA の数値・optimizer・prepared guard は実機で別途確認し、CPU skip を実機資格とは扱わない。
