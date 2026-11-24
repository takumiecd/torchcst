# 半径セル質量 Conv2d のチャネル Chart

[English](radius-cell-mass-channel-charts-20261124.md) | **日本語**

## 最初の候補での重みの定義

入力・出力チャネルに、それぞれ spacing 1 の一次元周期 Chart を置く。3×3では9個の
空間オフセットそれぞれに固定の Atom 部分集合を割り当てる。各部分集合がチャネル行列
$W_s[o,i]=\sum_{j\in A_s} a_j m^{\mathrm{out}}_j(o)m^{\mathrm{in}}_j(i)$ を作る。
各行列を Conv2d の重みに並べ、stride・padding・dilation は一回の通常の Conv2d で扱う。
1×1はオフセットが一つの同じ構成になる。これでチャネルを一つ移る意味がどのオフセットでも
揃う。周期的なチャネル順序はモデル化の仮定なので、同じ Dense アーキテクチャで検証する。

全オフセットで Atom の総数を固定する。rational 座標の学習可能スカラーは1 Atom あたり4個。
空間因子を別に持ち、チャネル因子をオフセット間で共有する構成も候補になる。この構成では
表現力、パラメータ数、半径則に使う振幅の意味が変わるため、別の比較が必要になる。
3×3パッチをチャネル優先で平坦化すると、1チャネル移動が9サイトになり、行やチャネルの
境界で離れた空間オフセットが隣接する。既存の864サイトの構成は学習比較で別の重み定義として扱う。

## 半径上限までの二進階層

両チャネル軸で base tile 1、merge factor 2、spacing 1を使う。96→128の最初の条件では、
前回の層計測と同じ $R_{\max}=32$ にする。実際に到達するタイル幅
$T\in\{1,2,4,8,16,32\}$ はすべて両方の Chart 周長を割り切る。一般に、到達するタイルの
サイト数で Chart サイズが割り切れ、宣言した最大タイルの物理幅が $R_{\max}$ 以上であればよい。
局所周期と窓の選択は既存のセル質量の定義に従う。この条件では
$P=(\lceil 2R/T\rceil+1)T\le96$ となり、96サイトの周期と折り返しを保てる。

現行の公開宣言、Python の AxisPlan、native parser は、一定の merge factor で Chart サイズまで
到達する完全な階層を要求する。半径上限までの階層を明示する場合は、この3箇所で bank の範囲と
半径上限の検証を揃える。この条件で $R>32$ を許すには別の階層が必要になる。採用前に、
境界値、その両側、周期をまたぐ窓を検証する。128→128の条件なら、96サイト対応を進める間も
既存の公開二進階層で参照実装を使える。

以前の864サイトを平坦化した入力で base tile 27の場合、$R=4$ の局所周期は54になる。
96サイトのチャネル Chart で base tile 1の場合は、同じ半径で12になる。以前の実測は
その宣言のもとでの結果である。128サイトに埋めて96チャネルに切り出す場合は、固有の境界と
質量の契約が生じるため、別の学習比較を要する。

## Linear からの移植

重みだけを作る演算を切り出す。rational 座標の復号には既存の `physical_parameters` を使い、
振幅から半径、および $q$ から半径へのヤコビアンを保つ。選択された局所窓ごとに、
$B_i,B_o$ が M8 の実数係数17個を周期 Chart のサイトに展開する。
$B_o C B_i^\mathsf{T}$ で各係数ブロックの和を重みに書き込む。Conv2d の backward から
タップごとの $G_s=\partial\ell/\partial W_s$ を受け取り、$B_o^\mathsf{T}G_s B_i$ に射影し、
forward で選んだ分岐の Atom 係数の微分と縮約する。空間方向の padding と重複加算を含む
$dX$・$dW$ は通常の畳み込みで計算する。

| Linear の実装 | Conv2d での適用範囲 |
| --- | --- |
| `packed_make_geometry`、セルの Fourier 係数と導関数 | 軸ごとの計算を流用し、Atom ごとに両軸の窓情報を保存する。この条件では同じ物理半径、spacing、階層なので両軸で level と bank が一致する。owner の折り返し先は Chart サイズが異なる。 |
| `packed_make_dense_weight`、`packed_dense_weight_grad` | forward と VJP の候補。VJP はすでに密な $dW$ を受け取る。重み専用 autograd を切り出し、level の範囲・owner 数を形状から求め、タップをグループ分けかバッチ縮約で扱う。浅い階層、周期境界、矩形レイアウトと Tensor のメモリ stride を検証する。 |
| `axis_parameter_gradient` と行を直接射影する Plan | 軸ごとの微分式と処理順の考え方を流用する。この launch は行ごとの信号と Atom 応答を受け取る。重みを作る Conv2d 経路では畳み込みが余接を返し、軸ペアの射影で微分する。 |

現行の native の重み合成と VJP は固定の level 範囲を走査し、対応する大幅の形状に依存する。
現行の wrapper は入力行列を受け取り $dW=dY^\mathsf{T}X$ を作る。新しい wrapper では
タップごとの重み余接を受け取り、そのレイアウト変換も計測に含める。短い窓の直接加算と
広い窓の Fourier owner ペアによる縮約を同じ重みで比較する。小さいチャネル行列で速い境界は
計測で決める。dispatcher での採用には optimizer 込みのステップ時間と allocated / reserved のピークを使う。

## FP64 での照合

CPU で96→128、stride 2、padding 0または1、1オフセットに rational M8 Atom 5個の
参照計算を実行した。Chart の両側の継ぎ目と半径上限・階層境界で、総質量1と
Chebyshev の独立した数値積分を照合した。重み合成と重み余接から Atom への逆伝播を
別々に計算した。1×1・3×3を通した最大絶対誤差は、数値積分に対して $W$ が
$5.8\times10^{-15}$、$Y$ が $2.1\times10^{-14}$ だった。セル質量の直接参照に対して
$dX$ が $5.6\times10^{-16}$、直接 autograd に対して Atom 全4フィールドが
$2.5\times10^{-14}$ だった。独立した数値積分の方向微分との誤差は $8.2\times10^{-9}$。
数学的な設計と CPU 参照の検証であり、CUDA の切り出し、launch の調整、学習比較は今後の対象とする。

参照計算は base tile 1の96サイト軸を内部メタデータとして構成し、選んだタイルが
Chart サイズを割り切ることを確認している。公開宣言の対応は今後行う。
Git 管理対象外の `output/conv-radius-chart-20261124/` に `check.py` と `result.json` がある。
worktree のルートから以下を実行する。

```sh
OMP_NUM_THREADS=1 PYTHONPATH=src:tests python output/conv-radius-chart-20261124/check.py
```

ソースは `680742aa8a73b8fe7c4f85be5ce1e5be52c79bdf`。検証スクリプトの SHA-256 は
`afd3dd2a0e07c7f460cb6eb74c4340f78d19eebdf2a27913e6cde9b138d0b14a`。
結果の SHA-256 は `6fdbb8453b960f0d0260a0d641916022910153ea7c6adc0032237c33edb5b8dd`。
