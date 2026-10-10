# Hの生成・再利用・寿命を軸にしたkernel設計

2026-10-10に合意した、RegularGrid上のprofile-product Linearと今後の対応kernelの設計方針。
**XからHをどう作り、Hを出力への寄与にどう使い、いつまでどこに保持するかを設計の軸にする。**
入力の読み取りと出力の集約が重なる範囲を束ね、そのまとまりの中でHを生成・再利用する。
Hは論理的な中間値であり、全atom・全batchのTensorを確保することを要求しない。

Chartの公開契約は[RegularGridの仕様](regular-grid.ja.md)、実装・登録の手順は
[kernel開発ガイド](kernel-development.ja.md)、既存の測定は
[PeriodicGridの比較](periodic-comparison.ja.md)を参照する。
この文書は実行方式の設計方針であり、KernelSpecの数学や公開dispatcherを変更しない。
RegularGridの宣言・Torch参照は実装済み。D2 FlatTorusの初期kernelをL4で検証・比較し、
[研究ノート](research-history/cuda-linear/regular-grid-h-lifecycle.ja.md)へ記録した。
以下の設計を高次元・別geometry・別GPUへ広げた性能は未検証。

## 決めたことと計測で決めること

| 決めた方針 | 計測して決める実装 |
| --- | --- |
| X→HとH→Yの両方を最適化する | 入力側・出力側のどちらを優先して束ねるか |
| atomとbatchを分割し、Hの作業領域を制御する | 一度に扱うatom数・小batch・入力／出力区間の大きさ |
| Hの利用を近づけ、最後の利用後に作業領域を再利用する | register/shared memory/小さなglobal bufferへの配置 |
| 重なる出力寄与を局所集約し、atomicと書き込みを減らす | 出力の担当単位、区間への振り分け、atomの順序 |
| forward内の再利用とbackwardまでの保存を区別する | 非保存・一部保存・全保存の再計算／容量の交換 |
| 同じ数学・更新則で完全step時間と総peak memoryを比較する | 条件ごとの候補採否と、将来の選択規則 |

最初は少数の実行方法を明示して比較する。汎用コンパイラや自動探索基盤の構築を前提にしない。
長く使う値を単に早く作ると保持時間が延びる。基本は最初の利用の直前に作り、
利用をまとめ、最後の利用後に解放する。先行生成・転送との重ね合わせは費用を測って選ぶ。

## 数学とHの意味

この文書では列方向にbatchを置き、Xは`[Ni,B]`、Yは`[No,B]`と書く。
公開LinearのXは`[...,Ni]`、Yは`[...,No]`であり、向きの違いだけで演算は同じ。

入力／出力座標群の生profile積をそれぞれqᵛ、qᵘ、振幅をAとする。
profileの共有live sigmaは既存Polar則を使い、task VJPではstop-gradientとする。

\[
D_a=\max\bigl(\|q^u_a\|_2\|q^v_a\|_2,\epsilon\bigr),\qquad
W_{ji}=\sum_a A_a\frac{q^u_{ja}q^v_{ai}}{D_a}.
\]

例えば`U_ja=qᵘ_ja/√D_a`、`V_ai=qᵛ_ai/√D_a`と置けば、

\[
W=U\,\operatorname{diag}(A)\,V,\qquad
H=VX,\qquad Y=U\,\operatorname{diag}(A)\,H.
\]

Hは各atomが入力Xを読み取った結果である。振幅を掛ける前の値として意味を固定する。
U/Vへの分母の配分は一意ではなく、既存CUDA準備は別の配分を使う。
配分を変える候補では生のHが一致するとは限らない。W/Yと正しいpullbackの同値性を検証する。
正規化は全観測siteを対象にし、全積の後でfloorを一度だけ適用する。
出力区間や小分けごとに正規化し直したり、軸ごとにfloorを掛けたりしない。

atom群g、batch部分tごとに、

\[
Y_t=\sum_g U_g\,\operatorname{diag}(A_g)\,(V_gX_t)
\]

と処理できる。batchの分割は独立、atomの分割はYへの合計である。
分割で支持やatomを切り捨てない。浮動小数点の加算順序の差は固定した誤差基準で検証する。

## X→H：読み取りを共有する

入力の支持範囲が重なるatomをまとめ、連続するXの読み取りとprofile評価を再利用する。
同じ入力に複数atomがアクセスすることと、隣接GPU laneが連続addressを読むことの両方を見る。
atomの並び替えだけでcache効率が上がるとは仮定しない。

`grid_shape=((16,16),(128,))`では出力側に二つ、入力側に一つの座標がある。
各側のprofileはそれぞれの座標profileの積。複数座標の支持boxはflatten後に
常に一つの連続区間になるとは限らないので、最内側の格子軸に沿う連続区間を利用する。

支持は中心・spacing・sigmaから直接候補範囲を算出し、実際のprofileの支持条件も確認する。
有効幅は座標軸ごとの`rho_d=sigma/spacing_d`であり、一つのrhoで全軸を扱えるとは限らない。
狭い支持ではH生成が一つ／少数のsiteの読み取りに縮退する。
自由に動く中心について`rho<1`だけで常にone-hotと仮定しない。
Gaussianなど無限支持のprofileを有限範囲で打ち切ることは、この方針に含めない。

Euclideanは有限格子の端で範囲を切る。FlatTorusは周期差を使い、周期境界をまたぐ支持を扱う。
**座標の周期wrapと、site番号の`% n`は同じ条件ではない。**
RegularGridには具体geometryを渡して部分周期の格子も宣言できる。
単純なsite番号のwrapを使うkernelは`period=n*spacing`等の配置条件を確認する。
条件外は対応する正確な経路へ落とすか、Algorithmの対応判定で拒否する。
広い支持や探索容量超過でも、支持を切り捨てず各観測siteを正しく数える。

## H→Y：寄与を束ねて集約する

出力の支持範囲が重なるatomをまとめ、その区間のYを局所的に累積する。
初期候補では正規化したUの評価とampの乗算を融合し、`AU`の全体Tensorを作らない。
同じ`A_a U_ja`を小batch内で再利用する。
H側へampを掛ける融合も同じ数学の候補であり、支持site数と小batch数、融合費用で比較する。

| 担当する単位 | 再利用の狙い | 必要な判断 |
| --- | --- | --- |
| atom群×小batch | Xを共有し、作ったHを複数の出力へ利用 | 異なる担当が同じYへ書く場合のatomicと局所集約 |
| 出力区間×小batch | 区間への全寄与を集約し、Yへまとめて書く | 複数区間にまたがるatomのHの再計算／一時共有 |

同じ出力先のatomを集めるだけではatomicは消えない。
atomicを不要にするには、その出力要素への全寄与を一つの担当が集約するか、
部分和を別のreductionで合計する必要がある。後者のbufferと追加処理も費用に含める。
出力区間を所有する場合は、中心がその区間の外でも支持が重なるatomをすべて拾う。

入力側をまとめる順序と、出力側をまとめる順序は一致するとは限らない。
出力集約のために増えたHの重複計算・Xの再読込を、どれだけの保持量で減らせるかを評価する。
全siteの支持index表やCSRを前提にせず、区間の候補・範囲情報を検討する。
区間への振り分けや並び替えにも構築費用と保存量がある。
中心・live幅が更新されたstepでは、古い振り分けを無検証で使い回さない。

## H/Gの寿命とbackward

forward内で複数出力にHを再利用することと、backwardまでHを残すことは別の判断である。
forwardの部分計算が終わっても、他の出力区間に利用先が残っていれば、
その利用を終えるか再計算する計画を決めてから作業領域を上書きする。

E=dY、G=UᵀEと置くと、固定した因数分解について、

\[
dX=V^T\operatorname{diag}(A)G,\qquad
dA_a=\sum_b H_{ab}G_{ab}.
\]

中心勾配にはprofile評価と正規化の微分も必要になる。
分母の配分次第ではUもVも同じ中心に依存するため、両側のpullbackを含める。
dU/dVの全体Tensorを作る必要はなく、必要な局所寄与をatomのParameterへ戻す。
ampをUに融合しても、振幅勾配用の未スケールU/Gを復元できる情報を残す。
ampによる割り算を使う復元はゼロ振幅で成立しない。
振幅がゼロでも振幅方向の勾配は非ゼロになり得るので、値だけでatomを除外しない。

| データ | 利用と保持の方針 |
| --- | --- |
| chart配置・Parameter・optimizer状態 | 既存の所有契約を維持。学習全体の容量に含める |
| 支持範囲・norm・amp等の準備情報 | 同じforward状態の小分けで再利用。保存と再計算の費用を評価 |
| H | 生成から最後の出力利用まで保持。backwardまで残す量は別に決める |
| Xとforward時点の状態 | Hやprofileをbackwardで正しく再計算するための情報を保存 |
| Gとbackwardで再生成するH | 小分けに生成し、dX／atom勾配への寄与を処理したら再利用 |
| Y/dXの部分和、atom勾配の部分和 | 担当範囲の全寄与を集約。batch分割分の勾配も合計する |

再計算にはforward時点のParameter・幅・配置・固定scalarを使う。
更新後のParameterからHを再計算してretained backwardの意味を変えない。
必要なsnapshotや保存Tensorのversion検査を既存autograd契約と整合させ、容量にも含める。

最初の処理順序の比較では、backwardへHを保存せず再計算する方針を共通にする。
その後、同じ処理順序で非保存・容量上限付きの一部保存・全保存の参照を比較する。
一部保存では「保存byteあたり省ける再計算時間」を目安にし、選別自体の費用も測る。
全保存は比較の対照であり、採用時のメモリ条件を免除しない。

## メモリ予算とWの位置付け

dtypeのbyte数をs、一度に処理するatom数をC、小batchをbとすると、
Hの値だけなら`s C b` byte、H/Gを同時保持すれば`2 s C b` byte。
FP32のC=4096、b=8ではHが128KiB、H/Gの合計が256KiBである。
これは値の作業領域の例であり、特定GPUのshared memoryに収まるという主張ではない。

実際の予算には、準備情報・routing・snapshot・部分和・norm微分・Parameter勾配・
optimizer状態・allocator/Graph poolなどを含める。
register圧力による退避やshared memory容量による並列性の低下も確認する。
小分けを細かくしすぎると再読込や起動回数が増えるため、小さい作業領域だけで候補を選ばない。

Wの作成は必須ではない。局所的な寄与を先にWへ集約し、batchへ使い回す方法も
同じ再利用・寿命・保存量の評価軸で比較できる。
FP32の全Wは`4 No Ni` byteでbatchに依存せず、全Hは`4 K B` byteでbatchに比例する。
local WはW組立・再生成・重複処理を含めて評価し、H経路との比較候補として位置付ける。
rhoだけから境界を固定しない。支持範囲、atomの重なり、batch再利用量、作業領域の上限を見る。

## 比較と次の実装で残す記録

初めに既存の2座標条件をRegularGridで再現し、同じ数学であることを確認する。
atom担当と出力区間担当を、同じ入力・atom・geometry・profile・更新則・保存方針で比較する。
その後に出力側2座標などの格子、Euclidean、異なるbatchへ範囲を広げる。
chartの宣言可能な次元数と各CUDA Algorithmの検証済み対応範囲は区別する。

各候補について、次の情報を研究ノートと結果に残す。

1. 担当単位、atomの束ね方、Hの生成／利用順序、ampの融合位置。
2. 同じXとHを再利用する範囲、重複計算、Y/dXの集約方法と残るatomic。
3. 保存Tensor、forward状態の扱い、backward再計算、作業領域の上界。
4. 振り分け・準備を含む完全step時間と、capture/replay込みのallocated/reserved peak。
5. 独立oracleのY/dX/全atom勾配、正規化・空／単一／広い支持・周期境界・
   可変幅・更新状態・retained backward・Graph replayの検証範囲。
6. source/result hash、実機・runtime・条件・独立run数、採用／保留／不採用の理由。

今回の採用目標は、同条件のdense完全学習stepに対し、allocatedとreservedの総peakを
それぞれ上回らないこと。その条件を満たす候補から完全step時間で選ぶ。
対照を含め独立processで測り、oracle用scratchを解放して同じ測定境界を使う。
scopeを固定し、メモリ条件を満たさない研究候補の記録も保存する。
診断用の各stage時間を足して完全step時間の代用にしない。
allocatorのpeakは物理DRAM転送量やL2 hit率ではない。cache/atomicの物理的な原因は
必要なtrace・counterで確認し、未測定の部分を確定した理由として書かない。

## D2実装での正規化配分とforward scale

現在のRegularGrid/Triweight D2実装は、raw入力・出力profileをqv、quとして、

\[
D=\max(\|q_v\|_2\|q_u\|_2,\epsilon),\qquad
(S_v,S_u)=\begin{cases}
(\|q_v\|_2,\|q_u\|_2) & \|q_v\|_2\|q_u\|_2\ge\epsilon,\\
(\sqrt\epsilon,\sqrt\epsilon) & \text{otherwise}.
\end{cases}
\]

したがってSv Su=Dであり、各軸へ独立にfloorを適用する方式ではない。
この配分で `H=(qv/Sv)X`、`β=amp/Su`、`Y=sum_a qu*β*H` と計算する。
新しいprepared H経路のβはamp/D全体ではなく、出力側分母Suの先行計算である。
Hの生成方法・backwardの正規化微分は元の配分を維持する。

β・inverse width・output centreだけをHのsorted positionへ置く。値は12A bytesで、
forward中のすべての小batchに再利用する。βの非finite検出用flagは4 bytes。
flag、3項目、routing、Hはいずれもbackwardへ保存せず、次forwardで更新状態から再生成する。
backwardには元のforward snapshotとnorm/VJP情報を残す。ゼロ振幅からの割算復元は使わない。

FP32ではamp/Suだけがoverflowしても、(qu/Su)*ampが有限になる場合がある。
非finite βでは元の演算順序を使い、通常βではsorted positionから積和する。
この選択はGPU flagで行い、host同期を入れない。guard用の準備と空launchも完全step費用に含める。
候補数、H容量、3項目の保持量、backward再計算は別々に評価し、
[実測と範囲regression](research-history/cuda-linear/regular-grid-h-lifecycle.ja.md)を残す。

## 出力の束と対称なG寿命

H生成のatom groupと、出力集約のatom groupは同じにする必要はない。
Hは8 atomずつ生成し、H→Yは32 atomずつ束ねる研究Planを追加した。
Hのbatch容量、3項目の保持量、forward snapshotの保存量は増えない。
広い束のreduction順序とGPU register配置は変わるので、全勾配oracleと完全stepで検証する。

backwardの対称候補では `G[b,a]=sum_j dY[b,j]*qu[a,j]/Su[a]` を作り、
`dX[b,i]=sum_a G[b,a]*amp[a]*qv[a,i]/Sv[a]` を入力site担当で集める。
Gをinput中心で整列したpositionに置き、同じ小batch内で再利用する。
G・routing・inputの3項目はbackward内だけで寿命を終え、次chunkで上書きする。
forwardからH/Gを保存せず、必要なparameter partialは元のatom IDへ一度ずつ書く。

`damp=sum_b H*G`、`dci=amp*sum_b G*dH`、`dco=amp*sum_b H*dG` とし、
正規化微分・単一siteの微分flag・Polarのsource VJPは既存と同じ。
G/dGとH/dHはforward時点のpacked情報から計算し、live Parameterやchartを読み直さない。
dXが不要なら従来のparameter-only計算を使い、G用routing/bufferを作らない。

G担当方式では入力owner tile8、forward出力tile16を最初の比較条件にする。
両方向のsupportを連結するCSRは作らず、中心binのprefixと支持のexact評価を使う。
βinput=amp/Svの範囲guard、routing構築、G生成、dX集約、全parameter partial/reductionも
完全stepに含める。atomicを消すだけで速くなるとは仮定せず、allocated/reserved総peakが
dense以内かを含めて従来のatom-owned backwardと比較する。
