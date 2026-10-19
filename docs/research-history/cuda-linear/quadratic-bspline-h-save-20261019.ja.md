# Quadratic B-spline: backwardの分解とforward Hの保存

2026-10-19。前の[batch連続配置比較](quadratic-bspline-h-coalesced-20261019.ja.md)のB32混合σ・約1.99ms、そのうち広いσ側backward融合区間約.86msを基準にする。研究用revision v14。公開dispatcherや数学的KernelSpecを変更しない。

## 保持する数学と精度

生の入力・出力profileをそれぞれ `fi, fo` として、

\[
D_a=\max\bigl(\sqrt{Q_{i,a}Q_{o,a}},\epsilon\bigr),\quad
H_{ab}=\sum_i f_{i,a}(i)X_{bi},\quad
G_{ab}=\sum_j f_{o,a}(j)dY_{bj}.
\]

`Q`は全観測siteにおけるprofile二乗和。既存のdegree-4離散momentによる正確なsampled L2、積の後で一度だけのfloor、その中心微分を維持する。幅はPolar状態からatomごと・stepごとにdecodeし、task VJPの幅stop-gradientという既存契約を維持する。幅の直接微分を追加した実験ではない。

入力側中心微分を `Hμ`、出力側を `Gμ` とし、

\[
S_a=\sum_b H_{ab}G_{ab},\quad
T_{i,a}=\sum_b H_{\mu_i,ab}G_{ab},\quad
T_{o,a}=\sum_b H_{ab}G_{\mu_o,ab},
\]
\[
dA_a=S_a/D_a,\qquad
d\mu_{d,a}=(A_a/D_a)\left(T_{d,a}-S_a\partial_{\mu_d}\log D_a\right),
\]
\[
dX_{bi}=\sum_a (A_a/D_a) f_{i,a}(i)G_{ab}.
\]

振幅勾配は既存Polarの2成分へ戻す。floor枝では分母の中心微分は0。振幅0でも振幅勾配は残す。
Hだけを保存しても入力中心勾配のためHμは必要になる。G/Gμはbackwardで初めて得るdYに依存するため、forwardには保存できない。

準備・係数・累積moment・縮約は従来どおりFP64、H/Hμ/G/Gμは既存queryが行っていたFP32丸めを保つ。保存版はそのFP32値を保存し、backwardでFP64へ戻して使う。shape、支持、微分式、IEEE設定、誤差閾値を変更しない。

## 保存と実行の候補

- `save_h=none`: forward Hを保存せず、backwardで入力bankとH/Hμを再計算。融合版は測定済みv13と同じCUDA forward/backward kernelを呼ぶ対照。
- `save_h=h`: forwardの未スケールHだけを保存。backwardの入力bankは必要だが、Hμだけを2個のmomentと微分係数から再計算し、Hの係数積と3番目のmoment読み取りを省く。
- `save_h=hp`: forwardのHとHμを保存。backwardの入力bank再構築とH/Hμ queryを省く。forwardではHのmoment読み取りをHμにも共有するが、微分係数・FP64演算・保存writeは追加される。

保存はautogradの各forward専用Tensorで、同じforwardのmetadata/coefficients/count/orderと結び付ける。更新後のσ・中心や別のXへ使い回さない。次のforwardは別の保存領域を作り、retained backwardの旧状態を上書きしない。

保存配列は `[A,B]` FP32。Graph内でnative数が変わってもshapeは固定し、native slotは未使用。Hのみは`4AB` byte、H/Hμは`8AB` byte。A=52,428ならB32で6.4/12.8MiB、B64で12.8/25.6MiBの容量となる。これは保存Tensorの容量であり、capture/replay込みの測定peakとは別。
dPを要求しないforwardでは保存しない。Xはnative側のatom勾配にも必要なため、H/Hμ保存版でもforward状態として保持する。

`backward_mode=fused`はG/Gμ生成・H再計算またはload・勾配縮約・dX差分注入を融合する。
`backward_mode=staged`は診断対照として、H/Hμ再計算、G/Gμ再計算、broad atom勾配縮約、dX差分注入を別kernelに分ける。native側VJP、bank構築、dXのscan/復元も別に記録する。

stagedはH/Hμ/G/Gμの一時Tensorと追加launchを伴う。各stage時間の和を融合版.86msの厳密な内訳とは呼ばない。融合を解いた場合に各仕事が要する時間と、融合を維持した完全stepの両方を示す。融合モードの個別stage markerは空であり、0msの仕事を意味しない。

## 検証と測定

L4一台を共有poolから排他的に使う。1024²、A=52,428、spacing .1、初期σ .1/.4/1.6/6.4/12各20%、可変幅を含む既存fixture、B32を主対照としB64・切替付近・小σだけも別に比較する。
AdamWと既存fused Polar更新、構築・routing・sampled L2・全勾配を含む完全step、capture/replay込みのallocated/reserved peakを測る。診断は主測定後の別Graphで、未計装の完全stepを置き換えない。

Y/dX/全atom4成分の独立FP64 gateはmaxabsとrelative L2をともに4e-4未満とする。live幅・中心、毎stepの新しいX、Graph内の幅切替とX変更、retained backward、ゼロ振幅、勾配の部分要求、空atom、B64も検査する。

測定・検証結果は、回収archiveとmanifestを確認してから以下へ記録する。

## 初回B32混合σの比較

job `l4job-56eef9fd777e4c438ade972b2c316632`、source `304c924e`。新規対象61件（60実CUDA + 1宣言）が通り、全shapeの独立FP64 Y/dX/全atom gateも通過した。以下は1 processの探索結果。2 processの最終比較は別に記録する。

| 方法 | 完全step ms | allocated / reserved MiB | broad backward融合区間 ms |
|---|---:|---:|---:|
| v13再計算対照 | 1.9879 | 68.488 / 134 | .8582 |
| v14再計算 | 1.9823 | 68.488 / 134 | .8704 |
| Hのみ保存 | 1.8312 | 74.888 / 174 | .7096 |
| H/Hμ保存 | 1.7082 | 81.288 / 172 | .5929 |
| W range reuse | 3.2259 | 78.590 / 154 | — |
| Dense | .0823 | 49.002 / 106 | — |

H/Hμ保存時は入力bank再構築も省ける（再計算版は別区間で約.032ms）。forward融合区間は.453～.456msで、この条件ではHμ計算と保存writeによる大きな遅延は観測されなかった。backward register/threadは126→56（H保存）→48（H/Hμ保存）、すべてspill 0。これはcompiler報告で、occupancyやcache hit率の実測ではない。

### 融合を解いた診断対照

| 保存方法 | H/Hμ再計算 ms | G/Gμ再計算 ms | broad atom縮約 ms | dX差分注入 ms | 完全step ms |
|---|---:|---:|---:|---:|---:|
| なし | .2888 | .2888 | .1475 | .4413 | 2.3203 |
| Hのみ | .1208（Hμのみ） | .2888 | .1475 | .4413 | 2.1494 |
| H/Hμ | skip | .2908 | .1495 | .5192 | 2.0073 |

native側のatom VJPは別区間で約.110ms。dXのscan/多項式復元、native側GEMM、bank構築も完全stepに含まれるが、上の4つのkernel区間の外で記録する。stagedにはmaterializedなH/Hμ/G/Gμのwrite/readとlaunchが増え、保存なしで94.643/214MiBのpeakになった。この対照の時間を足して融合版.86msを再現したものとは扱わない。

保存により入力H/Hμの仕事を減らせたが、G/GμはdYから生成する必要があり、dX差分注入も残る。staged対照ではdX注入が単独で最も長い区間だった。
