# 小tileから親へ積み上げるFourier集約のCPU検証

2026-10-21。ユーザーの指示は「小tileのatomを処理し、親用の値を集約し、
親が揃ったらその尺度のatomへ適用する」というbottom-upのデータフロー。
親から子へ入力集約を作る方式とは区別する。GPU実装・公開dispatcherではない。

## 共通kernel

多項式との混合を使わず、全sigmaで有限Fourierを使う。
母多項式(1-(r/(sqrt(11)*sigma))²)^4_+への制約付き連続L2射影で係数を定義する。
M4は端の0/2階微分0、M8は0/2/4階微分0、中心値1。
[混合kernelの係数定義](tile-hybrid-profile-design-20261021.ja.md)と同じ固定射影行列。
今回そのPはFourier係数を決める母形状であり、実行時にPへ切り替えない。
負の小さい裾は許容した数学prototype。厳密な非負性を主張しない。

spacing=.1、L_j=4*spacing*2^j (j>=0)、支持[-L_j,L_j]、周期2L_j。
R=sqrt(11)*sigmaが.85L_jの±12%にある間は、quintic smoothstepでF_LとF_2Lを接続する。
区間外は単一F。各成分は支持外0、kernel全体は最大active Lの外で0。
各atom・軸の中心/sigmaは連続で自由。尺度割当だけを離散化する。

sampled L2を各軸で正規化する。2Dは直積格子のprofile productで、L2²=Ex*Ey。
全ての方式は同じ係数・支持・正規化・mu/sigma微分を使う。
小Lでは格子上の周波数折り返しが重要。
正規化のgeometry和でも整数周波数を周期内へ折り返し、aliasしたDCを正確なnsite和として扱う。
以前のsinc(integer)/sinc(integer)式を小Lにそのまま使うと精度が不足したため、
新prototype内だけでalias-awareな幾何和を実装した。旧測定source/helperは変更していない。

## 計算方式

入力は1D線分または2D正方格子。最小leafは4/8/16点（2Dは各軸の点数）。
これは論理集約単位であり、GPU launch tileが4点等であるという意味ではない。
支持区間/長方形を、完全に含むdyadic nodeと部分leafへ厳密に分解する。
部分leafは短い1D/2D prefixで処理し、境界サイトのkernel直接評価や裾の省略を使わない。

- direct: 同じFourier関数をatomごとの支持内で直接評価し、Hと各軸mu/sigma pullbackを求める。
- independent: 同一nodeを使うatomは共有するが、別node/尺度のmomentは入力から独立に構築する。
  完了したnodeはqueryへ適用後に破棄する、streaming baseline。
- bottom_up: 小leafで必要な周波数を一度集約し、そのleafのatomへ適用する。
  親へ必要なmomentを足し上げ、親のatomへ適用したら子の作業配列を破棄する。

atomの位置やsigmaからquery metadataを先に決める。親が要求する周波数集合を子へ伝えるのは
metadata処理だけであり、入力依存の値の計算はleafから親への順序である。
親用の奇数周波数もleafで計算する。親の偶数成分だけを再利用して残りを無料とは扱わない。
同一周波数、格子alias、実入力の共役対称性をまとめて重複計算を減らす。
2Dは片軸だけの共役折り返しではなく、(qx,qy)と(-qx,-qy)の対をまとめる。

各leafの入力を連続コピーし、周波数chunk32で積和する。
複数atomの部分leaf queryにも同じ短いprefixを使う。
親用だけの周波数にはprefixを作らない。Xが変わるstepでは入力依存momentを再構築する。
同じ相対leaf位置のsin/cos表は固定geometryとして共有する。

## 比較範囲とカウンタ

M4/8、1D N256/B4/A16または128、2D 32²/B2/A8または64。
leaf4/8/16、small/mixed sigma分布。各atom・軸のsigmaは異なり、stepごとにX/mu/sigmaを更新。
smallは約.1〜.15、mixedはlog(.1, side*spacing/8)。同じfixtureを全M/leaf/方式で使う。
3動的step、各状態3回のCPU時間中央値。係数・L2・group/metadata・準備・query込み。
固定geometryは先にwarmし、相対leaf basisのsetup時間と配列payloadを別記する。

独立Torch quadrature/autogradで、合成出力・振幅/中心/sigma勾配を照合する。
identity入力で実装の全入力Jacobianを取り出し、Torchの全入力勾配も確認する。
性能測定は正規化profile-bank Hとmu/sigma pullbackまで。
高速なbackward executor、CST全体のY/dX、optimizer、GPU完全stepではない。

カウンタは下記を別々に記録する。

- input_values: 各leafでXを連続ロードする論理要素数（batch込み）。
- weighted_products: 親用の追加成分も含む、入力*Fourier basisの複素積和項数。
- prefix_values: 部分leafだけのscan要素数。
- merge_adds: 子から親へ足す複素要素数。
- traffic_model_bytes: 上記の主要配列・moment・係数・query出力更新の論理read/writeモデル。
  hardware transaction、cache miss、GPU帯域を実測した値ではない。
- scratch/carry payload: 実装が数えた配列の容量モデル。プロセスpeakではない。
- cached_bank_payload: query nodeの全bankを保存した場合の仮想payload。実行時にはstreamして破棄する。

代表条件のみ、call内の追加tracemalloc peakも測る。既存fixture/geometryの保存は別。
入力読み取り削減だけで、全メモリアクセスや速度が同率に改善したとは主張しない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_bottom_up_fourier_cpu \
  --output output/bottom-up-fourier-cpu-20261021
```



最初の32²測定を保全後、`--scaling-only`で2D 64²/128²、A64、leaf8、mixedを追加する。
M4/8を比較し、既存の32²測定は繰り返さない。kernel/helperのbyteは同じまま使用する。

その後、profile productの軸分解を入力moment準備にも使う追加候補を実装する。
2D full momentは`Ex^T X Ey`の順で計算し、必要な(qx,qy)だけ取り出す。
leaf辺長n、必要pair数K、異なるx周波数Fxに対し、
従来のO(B*n²*K)からO(B*n²*Fx+B*n*K)へ変える。
部分leafは従来の短いprefixを使い、同じXの連続ロードを共有する。
これは近似やkernel変更ではない。`--separable-only`で1Dと2D 32²/64²/128²を同条件比較する。

最終候補の再測定はsource `f7a580f1`で、既存原本と別のoutputへ保存する。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src:. \
python -m benchmarks.research.run_bottom_up_fourier_cpu --separable-only \
  --output output/bottom-up-fourier-separable-reproduction
```

## 結果: 積み上げは準備を減らすが、速度・容量の十分条件ではない

sourceは初回`2a500eb9`、領域拡大`f8cf8cba`、軸分解追加`f7a580f1`。
原本はignored `output/bottom-up-fourier-cpu-20261021/`、
`output/bottom-up-fourier-scaling-cpu-20261021/`、
`output/bottom-up-fourier-separable-cpu-20261021/`。
各proof.jsonにGit source snapshot/hash、result/figure/diagnostic hashを保存した。
全M/leaf/方式・追加source間で、同一条件のX/mu/sigma hashを照合して一致。

Apple M4 Pro、NumPy、FP64/complex128、BLAS/OMP/Accelerate各1thread。
初回144記録、領域拡大12記録、軸分解24記録。同一fixtureの再測定を含み、180独立runではない。
それぞれ3動的step、各step3時間sample。固定geometry/mappingは測定前にwarm。
`geometry_setup_ms`は相対leaf basis（最終版では軸basisも）の生成時間で、
全cold起動や初めてのLを導入する費用を含む値ではない。
係数・L2・query metadata・入力依存moment構築・merge・H/queryは各callで作り直し、時間に含めた。

### 正しさ

新規9test pass。独立Torch quadrature/autogradとの合成出力、入力の全Jacobian、
全atomの振幅・各軸中心・sigma勾配を照合。
1D/2D・M4/8・leaf4/8・2動的stepを検査した。照合はabs/rel2e-9（Jacobian abs2e-10）。
benchmark上の全H/各軸pullbackの直接式との差は初回最大2.70e-12、軸分解版最大1.28e-12。
CPU suiteは最終sourceで**2124 passed / 5651 skipped / 18 warnings、44.97s**。
Ruff/diff check pass。skipをGPU確認と数えない。
高速なdX executorを測ったわけではなく、CST完全step/GPU gateではない。

### 計算量・メモリアクセスのモデル

相対値は3step合計のbottom_up_separable / independent。
leaf8、1DはA128/B4、2DはA64/B2、sigmaはmixed。

入力依存集約の実数FLOP概算を`4W+8C+2P+2A`とする。
Wは実入力*複素basisと縮約、Cは軸分解後の複素積和、Pはprefix scan、Aはmergeの要素数。
これは係数/L2、固定geometry、query係数との積和を含む全step FLOPではない。
配列アクセスモデルには主要なweighted中間値、basis、prefix、moment、query係数・出力を含める。
GPUのcache/transactionや全配列一時値を完全に数えるモデルではない。

| M | 入力格子 | 入力読取の削減 | 入力依存集約FLOPモデルの削減 | 主要配列アクセスモデルの削減 |
| --- | --- | ---: | ---: | ---: |
| 4 | 256 | 83.33% | 58.28% | 49.11% |
| 8 | 256 | 83.33% | 57.80% | 49.97% |
| 4 | 32² | 50.00% | 22.53% | 16.52% |
| 8 | 32² | 50.00% | 27.64% | 19.75% |
| 4 | 64² | 66.67% | 30.75% | 25.07% |
| 8 | 64² | 66.67% | 33.84% | 27.39% |
| 4 | 128² | 74.64% | 41.54% | 36.61% |
| 8 | 128² | 74.64% | 43.22% | 37.53% |

**親用の追加奇数周波数を計算しても、独立に作る重複を減らせる条件は確認できた。**
入力を1回読めることと、全周波数が加算だけで作れることは別。
親の新周波数はleafで実際に計算しており、上表のW/Cに含めている。
2Dの周波数pair数も実装・カウンタに含め、軸数の和だけで計算したものではない。

small sigmaだけのM8/leaf8では、通常bottom-upの入力読取削減は1D18.6%、2D29.4%、
配列アクセスモデルの削減は5.9%、9.9%に留まった。
共有atomが少ない、leafが大きく実質1段だけの条件などでは積み上げの利益も減る。
全条件で同じ削減率になるとは主張しない。

### CPU時間

同じ最終source内での比較。単位ms、各step3回中央値からさらに3step中央値。
全方式が同じFourier-only kernelを評価する。

| M | 入力格子 | 直接 | 尺度/node独立 | 通常bottom-up | 軸分解bottom-up |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 | 256 | 28.455 | 35.309 | 33.554 | 34.378 |
| 8 | 256 | 31.427 | 36.838 | 34.950 | 35.763 |
| 4 | 32² | 28.001 | 74.591 | 72.924 | 73.684 |
| 8 | 32² | 27.625 | 136.141 | 138.423 | 133.378 |
| 4 | 64² | 27.361 | 128.097 | 126.731 | 124.008 |
| 8 | 64² | 28.919 | 321.765 | 308.388 | 308.934 |
| 4 | 128² | 29.545 | 314.512 | 295.382 | 283.727 |
| 8 | 128² | 32.159 | 943.359 | 867.704 | 828.302 |

M8/128²では独立943.36ms→軸分解828.30ms、約12.2%短縮。
ただし同じFourier関数の直接式は32.16msで、今回のCPU共有参照はまだ大幅に遅い。
M8/1Dは通常bottom-upが34.95ms、軸分解35.76msで、軸分解を常に選ぶ根拠もない。
64²のM8などは両bottom-upが同程度の範囲であり、小差を普遍的な勝利とはしない。
直接式のprofile productは軸別に重みを作ってXと縮約できるため、小支持で有利。
CPUの時間差をGPUでの同率改善やDense等価性へ外挿しない。

### 保存容量と残る時間

代表例M8/128²・step0のcall内追加tracemalloc peakは、直接.521MiB、独立3.042MiB、
通常bottom-up3.197MiB、軸分解3.280MiB。固定geometry配列は別に約1.110MiB。
これは入力fixtureと既存geometryを除いたPython/NumPyの追加追跡peak。
RSS、GPU allocated/reserved、CST全体のpeakではない。
**今回、CPU追加peakの削減は確認できなかった。**
全node bankを常設しない実装にはしたが、独立方式もstreamingできる。
bottom-upだけが省メモリになると仮定しない。carry、query metadata、geometry保存も必要。

追加のcProfile診断では、M8/128²/軸分解のinstrumented全体約1.19sのうち、
部分leaf用のmoment/prefix/query helperが累積約.87s。
係数/L2を含むprofile準備は約.033s、query metadataは約.034s。
累積時間は入れ子で重なるため加算しない。非計装の828msとも直接比較しない。
原本`diagnostic.pstats`、`diagnostic.json`、再生成用`analysis.py`を保存した。

親への積み上げで重複は減ったが、部分leaf scanの要素数とquery自体はほぼ変わらない。
そこが残る大きな費用であり、入力読み取り75%削減がCPU時間75%削減になるわけではない。
今回の空間分解は正方quadtree。軸別rectangle treeや、同じFourier関数を小leafで直接評価する
条件別実行方式は、この比較で実装・採用したものではない。
GPU並列化時のH書戻しatomic/segmented reductionや、実DRAMアクセスも未検証。

## 判断

**bottom-upでFourier準備の計算と主要配列アクセスを減らせることは確認できた。**
M8の2D128²では、入力集約のFLOPモデル約43%、配列アクセスモデル約38%の削減。
一方、CPU速度は独立集約比約12%の改善に留まり、直接計算を超えず、追加peakも減らない。
GPU高速化の採用判断はまだできない。

次の実装判断では、上位nodeへの積み上げだけでなく、部分leafの準備/queryと書戻しまで含める。
小さい支持・共有atomが少ない条件を含め、同じFourier関数の局所直接評価も対照に残す。
中心/sigmaの自由度を減らして得た結果ではない。

![input work and traffic models](../../../output/bottom-up-fourier-separable-cpu-20261021/separable-work.png)
