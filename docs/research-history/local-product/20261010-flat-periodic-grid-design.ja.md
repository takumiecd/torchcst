# N次元の平坦な周期格子と profile product

2026-10-10。利用者は独立した周期軸・等間隔配置・profile productを採用し、
3次元以上への拡張と速度・メモリ両方の診断を希望した。この文書は数学と
実験の設計であり、公開API、実装済みAlgorithm、GPU性能結果ではない。
起点はmain `1f79f564`。既存Sphere/Euclidean/embedded Torusの測定値を
この新しい演算の結果へ読み替えない。

## 数学と配置

空間は矩形の平坦なTorus

\[
\mathbb T^D=\prod_{d=0}^{D-1}\mathbb R/L_d\mathbb Z,
\qquad L_d>0.
\]

各軸に点数 \(n_d\)、原点 \(o_d\)、周期 \(L_d\) を宣言する。
間隔は \(h_d=L_d/n_d\) から導出し、独立したspacingとperiodを重複して
保存しない。siteは \(t_d(k)=o_d+h_d k\)、\(0\le k<n_d\)。
端点 \(o_d+L_d\) は原点と同じ点なので重複して追加しない。
全site座標Tensorは不要で、配置の宣言/固定bufferはO(D)。任意点表、
siteの学習、3Dドーナツへの埋め込みは初版の対象外。

Chartは点配置を、Geometryは周期距離を、Kernelはprofile合成と正規化を
宣言する。仮称PeriodicGridChart/FlatTorusGeometryであり、公開名は未確定。
既存ProductChartのLine/Grid構造、metadata-only supports判定、状態所有を
再利用できる範囲を調べる。既存TorusGeometryはambient chordであり、同じ
ID/revisionを使ってこの距離へ変更しない。周期数、点数、原点、軸割当、
flatten順序でdispatchし、座標Tensorから毎回規則性を推定しない。

### Dと行列のshapeは別

単一chartを使い、先頭p軸を出力、残りを入力に割り当てる。
各側は最後の軸が最速のrow-major。論理shapeは常に

\[
N_{out}=\prod_{d<p}n_d,\qquad N_{in}=\prod_{d\ge p}n_d,
\qquad W\in\mathbb R^{N_{out}\times N_{in}}.
\]

D=3なら出力を2D、入力を1Dなどにできる。座標の次元を増やしても
Linearの入出力Tensorの意味は変えない。現行SingleChartSpecは論理shapeが
[out,in]なので、D次元の格子shapeと2次元のoperator shapeを明示的に分ける。
まずD=2/3/4を独立に検証し、数学的には任意Dへ拡張可能な宣言にする。

### 周期差とprofile

\[
\delta_L(z)=z-L\left\lfloor z/L+1/2\right\rfloor,
\qquad \delta_L\in[-L/2,L/2).
\]

初版のraw profileはTriweight、共有する現在の幅を \(\sigma_a\) とする。

\[
g_{a,d}(k)=\left[1-
  (\delta_{L_d}(t_d(k)-c_{a,d})/\sigma_a)^2\right]_+^3,
\quad P_a(\mathbf k)=\prod_d g_{a,d}(k_d).
\]

全格子はCartesian productなので、重複siteを作らなければ

\[
n_{a,d}=\sqrt{\sum_{k=0}^{n_d-1}g_{a,d}(k)^2},
\quad \|P_a\|_2=\prod_d n_{a,d},
\quad W_{ji}=\sum_a A_a\frac{P_a(j,i)}
 {\max(\prod_d n_{a,d},\epsilon)}.
\]

全operatorで一つのnorm floorを使う。軸ごとのfloorへ置き換えない。
振幅・幅は既存Polar/activity則を候補とし、task VJPの幅stop-gradientを
明示的に維持する。source parameterは[D+2]、元のcanonical atom順を保つ。
中心更新は周期座標のwrapとし、平坦な接空間のvector momentsはwrapによって
回転しない。既存Sphere/embedded Torusのretractionへは流さない。

周期差の反対側 \(|\delta|=L/2\) にはcut locusがある。Triweightの支持半径が
半周期未満ならその付近のprofileはゼロだが、広い支持では中心微分が非滑らかに
なり得る。Torch/CUDAで同じ枝・片側微分規約を定めるまでoptimizer対応は完了と
扱わない。半周期以上の幅を勝手に禁止して性能を良く見せず、支持集合/値/normは
全軸評価で照合し、cut locusのVJP規約と通常点の勾配検証を区別する。

## 支持範囲と保存量

\(\sigma_a<L_d/2\) なら正支持は円上の連続区間。軸ごとにstart/countを
保存し、番号は(start+offset) mod n_dで生成する。seamを跨いでも各siteを
一度だけ処理する。保守的な候補boundsを丸め誤差の外側へ広げ、最後に
raw profileの正支持判定を行う。算術が安全でない場合はその全軸を走査する。
\(\sigma_a\ge L_d/2\) は全軸候補であり、capacityによる切り捨てはしない。
Gaussianなど無限支持profileに同じ有限範囲を適用しない。

範囲情報の保存はO(KD)。全候補site IDを持つCSRは最初の方式にしない。
normと中心微分の統計も支持のある軸siteだけから正確に集計できる。
forwardごとの中心・幅・範囲・normのsnapshotをbackwardでも使う。
幅/中心は毎step更新し、Algorithmインスタンスに古い支持を保存しない。

ただし、配置O(D)と完全stepメモリは異なる。W/dWはO(Nout Nin)、global H/Gは
O(BK)、source/勾配/optimizer状態はO(K(D+2))、範囲はO(KD)。
norm/factor保存とGraph allocator poolsも必要量に含める。

## 次元・幅・アクセスを切り分ける

軸dの正支持site数をm[a,d]とすると、軸factorの準備はおおむね
sum_d m[a,d]、Wへの寄与生成はproduct_d m[a,d]。この二つを混同しない。
各軸で約6点なら2D/3D/4Dの支持直積は36/216/1296点であり、探索が軽くても
寄与生成とVJPが重くなる可能性がある。これは計算量の例で、実測時間ではない。

最後の座標軸を最速とし、最内側の支持区間を連続アクセスにする。
高次元ではstride付きの行/面、seamの分割、atom間の重なりが残る。
支持boxが近いatomの並び替えは候補だが、sort/pack/inverse mappingの費用を
完全stepへ含める。G8は既存の比較候補で、D=3/4でも最速とは仮定しない。

| 段階 | 確認する量 | 結果による変更候補 |
| --- | --- | --- |
| 周期範囲・profile/norm準備 | 全軸走査数、実支持数、準備時間、保存byte | 算術bounds、軸factor共有、保存/再計算 |
| W組み立て | 支持直積の総数、時間、atomic数 | 既存grouped patch、site/atom配置 |
| forward/dX/dW contraction | 時間、X/dY/W読み出し | 既存GEMM、bounded W窓、bounded H/G |
| atom VJP | 時間、factor再評価、norm微分 | shared dW、既存融合方式 |
| optimizer | 実更新clock、source/moment byte、時間 | 支配的な場合だけ変更 |

allocatorピークだけでアクセス問題と断定しない。疑わしいkernelについて
DRAM転送量・帯域、L2 hit、register/spill、occupancyを独立に調べる。
計測済みtimeとpeak、実装からの容量計算、profilerによる因果診断を分ける。

## 実装・検証・測定の順序

1. 純粋な宣言、Torchの全格子FP64参照、周期中心の更新規約を確立する。
   D=2/3/4、異なるperiod/origin、seam、単一点/空支持、広い支持、norm floor、
   反対点、retained backward、全中心勾配、optimizer moments/clockを検証する。
2. 既存のsupport範囲・grouped W+GEMMを出発点に、新契約の最小CUDA基準を
   作る。支持探索/正規化/assembly/GEMM/VJPを分けて診断できるようにする。
   新たな数値精度近似・候補capacity切り捨て・全site表は導入しない。
3. 同じ新契約のfull-axis Torch参照と算術support版を比較する。
   旧Sphere/embedded Torusは別モデルであり、同じ演算のbaselineにしない。
4. 最初はB32、D2/3/4、初期rho=1.25/3/8で幅を更新する。
   1024x1024の比較では格子shapeを(1024,1024)、(32,32,1024)、
   (32,32,32,32)、出力分割pを1/2/2とする。総site数・入出力数は同じ。
   各軸rhoが同じでも支持直積数は異なるので、総支持数を合わせた診断も追加する。
5. atom数K固定と、K(D+2)のparameter scalar予算を固定した診断を分ける。
   新契約内の全方式は同じ初期P/X/targetとoptimizerを使う。
6. 独立全量FP64 Y/dX/all-source VJPと更新clock gate後、既存runnerで
   未計装の完全stepとcapture/replay allocated/reservedを測る。
   別phase Graphを足し引きしない。物理アクセスの計測はその後の独立診断。
7. 支配的な段階を一箇所変更して同条件で再比較する。速度>3%改善かつ割当増なし、
   または割当>=5%削減かつ時間悪化<=3%を独立確認の候補条件とする。
   速度のみ有利な候補も記録するが、両目的を満たした採用と区別する。

新しいmath/chart/state/update/serializationとbenchmark fixtureの接続が必要。
元のSphere用catalogやcaseをそのまま使わず、runtimeと独立oracleを確立してから
固定case/sourceを登録する。この文書だけではGPU割当も計測も実施していない。

## 根拠

- [Flat tori in three-dimensional space and convex integration](https://abel.math.harvard.edu/~knill/teaching/summer2012/exhibits/flattorus/article.pdf):
  平坦な周期矩形と3Dの回転Torusでは距離が異なる。
- [既存profile product契約](../../profile-product.ja.md): 単一chart、全積norm、
  座標次元と論理shapeの区別。周期Geometryはこの版に未実装。
- [grouped Product実測](20261008-profile-product-grouped-matrix.md): 既存計算方式の
  再利用候補。新しい周期演算の性能保証ではない。
- [過去のmatrix-free結果](20261010-product-matrix-free-results.md): 保存量を減らしても
  完全stepが速くなるとは限らず、同条件の時間/peakを両方測る。

## CPUの小格子数学監査

seed41、D2/3/4の各100条件、合計300条件をCPUのPython scalarで照合した。
格子shapeは(7,9)/(3,5,7)/(3,4,5,6)。軸ごとに異なるperiod/origin、複数周期外の
中心、seam近傍、幅0.015/0.12/0.4/0.5/0.8倍の最小period、norm floor1e-6/10を
含む。独立参照は全格子を列挙し、min(正の周期差, period-周期差)で距離と積を
計算する。候補側はstart/countの周期範囲とsigned wrapから軸factorを計算する。

全条件でraw productの最大絶対差5.8842e-15、全量L2と軸L2の積の差5.8842e-15、
正規化後の最大差5.0515e-15。候補IDの重複なしも確認した。
これは有限の小格子における数学チェックで、FP32 boundsの安全性、勾配、
optimizer、実装済みChart/API、GPU精度・速度を検証した結果ではない。

監査sourceと全300記録を、この文書のworktreeのignored
`output/flat-periodic-grid-design/cpu_math_audit.py`と`cpu_math_audit.json`へ保存した。
source SHA256 `6fb907559a4e9070a7b30101c80ec7116f5c1a715b41ee984e4dfe8ff81826fb`。
再現は`python output/flat-periodic-grid-design/cpu_math_audit.py`。
GPU実測・公開API実装は未実施。
