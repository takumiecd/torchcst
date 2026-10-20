# 局所多項式 / Fourier共有の新kernel候補: CPU screen

2026-10-21。旧quadratic B-spline CUDA最適化は一時停止のまま。
この調査は新しい数学的kernel候補の形・連続性・計算量をCPU FP64で比較する。
runtimeのKernelSpec、既定dispatcher、optimizer、GPU実装は変更しない。

## 母形状と幅の意味

\[
P(u)=(1-u^2)^4_+,\quad R=\sqrt{11}\sigma,\quad
p(x;\mu,\sigma)=P((x-\mu)/R).
\]

無限連続領域でL1正規化したpの標準偏差はsigma。
L2正規化後のエネルギー分布p²の幅は別であり、母形状とのenergy RMS比で比較する。
固定格子の切り出し・Fourier混合後にこのsigmaが厳密な標準偏差になるとは仮定しない。
中心とsigmaはatomごとに連続。幅の微分も調査し、既存Polarの幅stop-gradientとは別範囲。

周期Lでの複素級数は正負のharmonic -M..M（M=4/8）、real feature数は1+2M。

\[
c_k(R,L)=\frac RL\int_{-1}^1(1-u^2)^4\cos(2\pi kRu/L)du,\qquad
f_L(x)=\sum_{k=-M}^{M}c_ke^{2\pi ik(x-\mu)/L}.
\]

係数は球Besselの閉形式 `I(z)=768 j4(z)/z^4` と
`z I'(z)=-768 j5(z)/z^3` から計算する。z<4は20項の安定級数、
z>=4はsin/cosの有理式。係数とsigma微分はO(M)、小zの巨大な差分は避ける。
独立Gauss-Legendre128点と適応積分で照合する。

## タイル、周期、切替

spacing=.1、1軸tile幅は32/64/128点。Tを物理tile幅とする。
`L_j=L0*2^j`。L0=4T/rho=.22、L0=8T/rho=.34を対照にする。
各周期の有効集合は `K=L/T-1` 個の隣接する丸ごとのtile。
centerの所属tileを中心とする奇数個の集合であり、native支持を囲む余白を持たせる。
Fourier側の境界サイトをnativeで直接計算する補正はしない。

native/FourierはR/L0=.16..22、周期間はR=rho*Ljの±8%で、
quintic smoothstep `s(t)=6t^5-15t^4+10t^3` により混ぜる。
その重みのsigma微分も含める。hard switchは負の対照として残す。
早期切替対照はR/L0=.05..075（L0=8Tなら支持直径が約.8T..1.2T）。
大きいtileでも「1tile程度で即Fourier」が成立するかを別に調べる。

中心の所属tileによる不連続を検出するため、hard tile集合と、隣接2集合を
`s(frac(mu/T))` で混ぜる `tile_blend=True` を分ける。
後者のmu微分は位相微分だけでなく混合比の微分も含む。
これはnew kernelのスカラー係数としての定義で、既存profileに追加窓を掛けた同義計算ではない。
最大2周期×2集合を参照する。重複tileの入力momentは共有する。
mu/sigmaが滑らかでも空間xのtile端が厳密に滑らかとは限らず、別のmetricで残す。

## sampled L2と共有

入力依存bankは `T[t,k,L,b]=sum_(i in tile t) X[b,i] exp(2pi ik xi/L)`。
入力Xが変われば作り直す。同じXでsigma/centerだけが変わる場合の継続利用と混同しない。
隣接周期の偶数harmonicは同一周波数として共有する。
原型は正負を両方保存したcomplex128。real packingした容量モデルとは区別する。

正規化用のgeometry moment `Z[t,k,L]=sum exp(2pi ik xi/L)` は入力にもatomにも依存しない。
等間隔の連続siteでは、n点の和を `n*sinc(n theta/2pi)/sinc(theta/2pi)` と
中央siteの位相で厳密に計算できる。最終CPU時間はこの経路を使い、geometry tableのpayloadは0。
入力bankの構築が不要になるわけではない。geometry tile和を保存する対照とも照合する。
Fourier同士のL2交差項は係数の線形convolutionと、集合の共通tileのgeometry momentで計算する。
単一周期では正の2Mまで、隣接2周期の混合では長周期単位で正の4Mまで必要。
直交性、積分による離散和の代用、tileごとの再正規化は使わない。

native混合区間では `||wp p+sum f||²` のnative/Fourier交差項も含む。
この原型はnative支持のKp点でKp*(2M+1)のmixed和を明示計算し、費用を隠さない。
Fourierのみではnative境界の直接計算はない。全Cartesian productのnormは軸normの積にし、
floorは全積の後に一度だけ適用する検査を行う。

## 検証と計測の範囲

- sigma=.1..10の連続sweepと切替の両側、複数のcenter phase、tile境界両側。
- shapeの母形状との差、normalized mu/sigma勾配の母形状との差、裾の振幅/energy、負の裾、幅。
- candidate自身の解析微分を、独立Torch quadrature/autogradと有限差分で照合。
- sampled L2とその微分、共有H/Hmu/Hsigmaを直接site和と照合。
- 任意の2次元入力に対する軸別productとmaterialized積、全積floorの一致。
- CPU時間は1軸projectionとそのmu/sigma微分、係数・構築・L2込み。Y/dXやoptimizerを含む
  CST完全step、GPU性能、allocator peak、全atom更新を測った値ではない。
- bank payloadは数値arrayのみ。Python辞書、index、temporaries、saved forward、optimizer等を含むpeakではない。

再現（numpy/scipy/matplotlib/torchを含む研究環境）:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_polynomial_fourier_cpu \
  --output output/poly-fourier-cpu-20261021-final
python -m pytest tests/test_polynomial_fourier_cpu_study.py -q
```

結果JSONにはsource commit/2 source hash、環境、全測定sampleを保存する。
raw結果と図はignored outputに保全し、この文書には検証後の要点のみ追記する。

## 検証済み結果

測定source `15dcbe88935f2826ef50eb53ccd1243db74540e7`、macOS arm64、numpy2.4.6、
BLAS/OMP/Accelerateは各1thread。最終原本は `output/poly-fourier-cpu-20261021-qualified/`。
results SHA256 `2356215c8af25c4a839a54a560944db0a2b53ebcacccf9b01d53bf0216001040`。
source2本とtest1本のGit byte一致を確認し、proof.jsonへ保存した。
先行 `*-final/` は旧sourceかつCPU testと並行したため時間の根拠から除外する。

shape10,368条件、切替786記録、独立直接和120状態。
係数/幅微分の独立quadrature誤差は3.06e-15/1.16e-15、
norm²/そのmu/sigma内積誤差は5.69e-14/1.02e-14/6.76e-14、
table geometryと有限等比級数geometryの誤差は5.69e-14以下。
共有H/Hmu/Hsigmaの直接評価との差は2.06e-15/3.56e-15/6.94e-15。
normalized profileの有限差分relL2誤差はmu1.67e-9、sigma3.47e-10。
独立Torch quadrature/autograd・動的X・中心のtile越え・Cartesian product/floorは21test通過。
CPU全suiteは2066 passed /5651 skipped /18既存warnings（42.12s）。
GPU・CST完全step・全atom optimizer照合を通したという意味ではない。

### 形・裾・勾配

主条件T=3.2（32点）、L0=8T、smooth sigma/period + smooth tile集合、sigma=.1..10。
以下は母形状pとの差やcandidate自身の特性の最大値である。
母形状との差はnew candidateの数学的特性で、candidate内部の勾配計算誤差とは異なる。

| 指標 | M4 | M8 |
| --- | ---: | ---: |
| sampled L2正規化後の母形状との相対L2差 | 12.063% | 0.6053% |
| 母形状との中心勾配relL2差 | 33.405% | 2.8883% |
| 母形状とのsigma勾配relL2差 | 202.75% | 10.941% |
| native支持外の最大振幅 / peak | 7.0492% | 0.3624% |
| native支持外のL2 energy / 全energy | 0.6532% | 0.001774% |
| 負の振れ / peak | 7.0492% | 0.4297% |
| energy RMS幅 / 母形状の幅 | .998139..1.264754 | .999828..1.000377 |
| 空間xのtile端に残る値のjump / peak | 3.9871% | 0.1815% |

M8の山の幅はよく保たれるが、非負性・xの連続性を厳密に満たさない。
mu/sigmaに関する滑らかさと、空間xに関する滑らかさを同一視しない。

| M8 / tile点数 | 母形状との最大相対L2差 | 最大裾振幅 / peak | native/Fourier重なりsigma |
| --- | ---: | ---: | --- |
| 32 | .6053% | .3624% | 約1.235..1.698 |
| 64 | .5532% | .3544% | 約2.470..3.396 |
| 128 | .1987% | .1421% | 約4.940..6.793 |

大きいtileほど本調査範囲内でnativeに残るsigmaが増えるため、この表だけで大tileの方が
Fourier精度・速度に優れるとは言えない。主条件のnative終端では支持直径は3.52T。
local側は「1tileしか読まない」制約を置かず、約3.5tile幅まで読む必要があった。
sigma=.1はnativeのままで、支持は約6～7site（1軸）。

「支持直径が.8T..1.2T」で早期切替する対照は、M8でも母形状との差27.52%、
裾振幅12.52%、energy幅比最大2.314となった。M4は形状差62.46%。
L0=4Tの狭い周期対照もM8で形状差4.67%、裾2.53%となった。
少数M、丸ごとのtileのみ、十分な余白、早期切替のすべてを同時には満たせていない。

### 切替と連続性

hardな周期/経路切替ではnormalized shape jumpはM4最大15.39%、M8最大.8097%。
quintic重なりではsigmaを相対±1e-7ずらした差はM4で4.31e-7、M8で1.58e-7以下。
これは有限差分幅での変化であり、数学的jumpの下限やゼロの測定ではない。
重み端で2階まで微分が消え、同じ期間profileへ接続する構成にしている。

hardなtile集合の入替ではmuをtile端±1e-9で比較したshape差がM4最大2.426%、M8最大.1128%。
隣接2集合のsmooth混合では両Mとも1.33e-8以下、中心勾配差2.21e-8以下だった。
代償は通常2集合、周期重なりで最大4集合を持つこと。空間xの端のjumpは上表のとおり残る。

### 正規化込みの費用・保存量・再利用

この実装では1軸native支持Kp、参照tile数Kt、batchBとすると、
nativeはO(B Kp)、Fourier共有H/微分はO(B M Kt)。
入力bank構築は要求されたtile/周波数のsiteを一度読む。
同じXなら係数・sigmaが変わってもmomentを共有できる。Xが変わればそのmomentは無効。
隣接周期の偶数周波数aliasも共有する。tile上のrange prefixを導入すればHのKt依存を
減らす余地はあるが、この原型の時間には導入していない。

L2の係数convolutionは直接O(M²)。geometryは有限等比級数でO(M)、保存表0。
原型は全component組についてE/Emu/Esigmaを個別計算し、通常2集合で12回、
2周期×2集合で48回のconvolutionを行う。対称性・同じ周波数の処理をまとめる余地が残る。
native/Fourier重なりにはO(Kp M)の交差項が加わる。追加の空間窓やFourier境界直接補正はない。

N1024、B8、A128、32点tile、3stepでmu/sigma/Xをすべて変更。
同じA/stepのX/mu/sigma hashはMとtile条件をまたいで一致した。
下表の時間は各step3時間sampleの中央値を3step平均したCPU Python原型で、
CST完全stepやGPUの速度・ボトルネックを示すものではない。
各部分は別測定であり、足し合わせて主測定の代わりにしない。

| 1軸projectionとmu/sigma微分 | M4 | M8 |
| --- | ---: | ---: |
| 共有版（係数・新Xの構築・L2込み） | 137.819ms | 246.046ms |
| 同じcandidateのsite直接評価 | 16.970ms | 22.455ms |
| 係数生成だけ | 4.187ms | 4.217ms |
| 入力bank構築・tile要求だけ | 6.755ms | 12.773ms |
| L2とその微分だけ | 121.420ms | 226.524ms |

Python原型の共有版にCPU速度の利益はなかった。特に個別のL2交差項・多数の小さい
geometry/convolution呼び出しが重い。低階数の数学的構造をそのまま高性能実装と扱わない。

step0の入力bankはM4が69KiB、M8が134KiB（complex128で正負を両方保存）。
比較用に全atomのcoeff/omega/両微分arrayを保持したpayloadは62.02/117.14KiB、
H/Hmu/Hsigmaの3出力は合計24KiB、元のXは64KiB。geometry表は0。
Python object/index、一時buffer、autograd保存、optimizer、allocator peakは含まない。
係数を全atom保持せずatom単位で処理する容量と混同しない。

同じstepで独立atomごとにbankを作る対照のtile-frequency数は8558/16218、
共有版は552/1072で構築重複を93.55%/93.39%減らした。
これはqueryや全演算を93%削減した意味ではない。A8では約65%、A32では約87%だった。
別の5step reuse検査では同じXならstep1以降に新規moment構築は0、
毎step異なるXならstepごとにM4約552/M8約1072の構築が必要だった。

2次元profile productのstraight tensor bank容量モデル（FP64 real packing、B32、1周期/level）:
32×32tileでM4は9²=81 feature=20.25KiB、M8は17²=289 feature=72.25KiB。
元の入力tileは256KiB。複数周期・入出力・勾配・metadataを加えた総peakではない。
実際の2D bankをGPU/CPUで構築してこの容量を実測した値でもない。
profileが軸別に分解できても任意入力Xが因数分解できるとは仮定せず、
任意の2D Xで軸順の収縮と明示productの一致を検査した。

`shape-and-tail.png`、`switch.png`、`tile-seam.png`を保存した。
曲線の視認性のためshape/switch/seamのprofile描画はspacing=.0125を使う。
上表の数値・shared contraction・timingはprotocolどおりspacing=.1である。

## 現時点の判断

M4はこの母形状/周期/余白では裾・形・幅の変化が大きく、そのまま採用しない。
M8は狭いsigmaをnativeで残す条件では形とenergy幅が保たれ、連続mu/sigmaの共有計算も成立した。
一方、1tile程度で即切替は失敗、空間tile端の微小jump、負の裾、幅方向の母形状との勾配差、
混合L2費用が残る。候補の数学的な制約を詰める段階であり、GPUへ採用・再開する根拠にはしない。
