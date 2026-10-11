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

過去の実装と実測から移せる処理順序は
[歴史的kernelとHの寿命](research-history/cuda-linear/regular-grid-h-historical-kernels.ja.md)に整理した。
入力断片共有・owner内のG再利用・batch内勾配累積・選択保存を、現行で実装済みの
再利用と区別して比較する。

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
Triweightでは`rho<0.5`なら各軸で高々1site。ただし空支持があり、単一siteでも
正規化floorが活性な場合はampだけの重みに縮退しない。これらを区別して軽量経路を検証する。
両側の支持がそれぞれ1siteでnorm積がfloorより大きいTriweight atomなら、
局所的に `W_a[j,i] = A_a * 1[j=j_a] * 1[i=i_a]` となる。
HはXの単一site参照に、GはdYの単一site参照に置き換えられ、中心のtask勾配は0。
振幅のtask勾配は `sum_b dY[b,j_a] * X[b,i_a]`、Polarへのpullbackは維持する。
空支持とfloor活性のatomを同じamp-only経路へ入れない。これは今後の候補であり実装済みではない。
Gaussianなど無限支持のprofileを有限範囲で打ち切ることは、この方針に含めない。

速度・メモリの評価は狭い・中間・広い単一幅だけでなく、幅の混在率を変えた条件で行う。
初期幅の平均だけでなく、両側の支持site数と空支持、幅分位点、更新後の変化を記録する。
共通の幅境界を持つ[分布別比較](research-history/cuda-linear/regular-grid-width-distributions.ja.md)を用い、
単一幅で速い方式をmixでも速いと仮定しない。

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

GとHのbatch容量は独立に選べる。streaming研究Planではforward Hを16/32batch、
backward Gを8batchに固定し、parameter partialもchunkごとに畳んで上書きする。
BM=8、B=32の場合、Gは8A floats、parameter用はpartialの3Aとphysical累積の3A
だけである。全batch partialの12A floatsを保持しない。各chunkでphysicalな
(damp,dci,dco)を累積し、最後にPolar source VJPを一度適用する。
chunk終了の順序はG生成→dX集約→physical累積→buffer上書きで固定する。
追加の累積kernel、batch和のFP32順序変更、Graph poolの総peakも比較に含める。
この候補はparameterのみ必要な場合には従来経路を維持する。

## 中心site prefixでforward候補を絞る

出力側の中心site keyを `k=floor(phase/spacing)` としてsortし、`N+1`個のprefixを置く。
Hと3項目は同じOrderで生成する。Y owner開始siteをs、実在幅をw=min(BO,N-s)、
既存prepared span由来のguard付き最大距離をDとすると、中心候補は
`[s-D,s+w+D)` の周期窓に含まれる。窓長w+2D>=Nなら全候補を一度だけ処理し、
短ければ重複しない最大2区間をprefixから読む。CSRやatomごとの支持site ID列は作らない。
追加prefixの大きさはatom支持数ではなくgridのsite数で決まる。

この候補はforwardの探索だけを変更する。H容量、G容量、joint norm floor、packed VJP、
periodic_rawのFP32演算、input-owner backwardのcoarse routingは維持する。
一つのbroad atomでDが大きくなれば全探索に戻るため、速さは支持分布に依存する。
owner内の無効site積和は残る。粗いbinの検査比8倍/3倍がそのまま速度向上倍率ではない。
sort/searchとprefixの準備費用、atom加算順の変更、Graph両peakを含めて測る。

半周期の比較だけでperiodic_rawのdivision/floorを置換しない。例えばL=1024、
FP32 delta=nextafter(512,0)では、delta/L+0.5が1へ丸まり、現行wrapは-512になる。
単純なdelta>=512比較では正側に残り、広い支持の中心微分を変える場合がある。
今回はその近似を混ぜず、既存の距離式とnorm/VJPを保つ。

### H生成幅とY担当batch幅を分ける候補（2026-10-11）

centre-site prefixの後、H producerのBM8・tile-major配置を維持し、
Y ownerだけBM16へ広げる。独立した `OwnerBatchHRecipe` と
`OwnerBatchStreamingHRecipe` を用い、既存recipeの意味を変更しない。
後者はG8、入力側coarse routing、physical部分和の再利用を維持する。

owner内のlocal batch rowを r、sorted atom positionを pos とすると、
Hの読出しoffsetは `((r // 8) * A + pos) * 8 + r % 8`。
Hは引き続き `[Hcap/8, A, 8]`、Hcapは16または32。
actual batch rowでmaskし、最後のchunkやbatch33/64でも保存範囲を越えない。
H/Gをbackwardへ追加保存せず、raw profile・一つのjoint L2 floor・
Unsafe beta時の元の演算順序を変えない。

B32のY担当CTAは各output ownerにつき4から2になる。
profileとroutingのbatch間の重複を減らす一方、GROUP32×BO16×BM16の
中間値はBM8の倍であり、register圧力・spill・実測時間の悪化はあり得る。
追加global scratchは必要ないが、総allocator peakの同等は実測で確認する。
H16/H32×通常/G8の4候補を同じ条件で比較し、未計装完全stepと
allocated/reserved総peakの両方で判断する。

post24の実routing prefixから、guarded距離・候補atom訪問数・
実GROUP別の反復数・live grid site候補検査数をCPUで数える。
時刻計測後の診断内でのみコピーし、primary graph/peakに入れない。
候補数は正支持数や物理memory trafficではなく、batch多重度と
padded laneを除く値。Yのactive/launch CTA数も別に記録する。

raw UのPhi16表を持つ案は次段階に置く。正確な支持envelopeによる
span検査とbroad fallbackが必要で、H32をH16へ縮めても表の転送や
追加metadata・Graph poolのreservedが費用になる。まずscratchを
増やさないbatch共有を切り分ける。

### Producerとconsumerを両方考えた配置と診断

現在のHは `[batch slab, output-centre-sorted atom, 8]`。
同じatomの8batch値は連続し、Y ownerは出力座標の近いatomのHを連続して読む。
一方、Xは `[batch,input site]` なので、各atomの支持内は隣接siteを読めても、
同じproducer group内のatomの入力中心が近いとは限らない。
Gはinput-centre順でdX consumerを優先しており、G producerのdY読出しには対称な問題がある。
N2048/B32/FP32ではX自体は256KiB、A209715/H32のHは約25.6MiB。
論理参照量だけからcache missや物理trafficを推定しない。

未実装の配置候補は、output siteを第一キー、input siteを第二キーとする
`key=output_site*NI+input_site` の単一sort。出力siteごとの連続性を保ち、
同じoutput site内のproducerを入力中心順にする。Boundsはsorted keyに対し
`arange(NO+1)*NI` をsearchsortedする。同scopeの最大8192²ならint32に収まる。
別のHコピーやatom数に比例する追加permutationは不要だが、key生成とsortの費用、
atom加算順、group内最大支持長が変わる。cache改善は未検証で、採用候補ではない。

中心座標から格子番号は直接計算できる。現在のindex構築はその後のsort、
site/binの開始位置作成、guarded最大距離のreductionも含む。
routingはそのforward/backward内の一時状態であり、forwardのOrderをbackwardへ保存しない。
一時的であることと構築費用がゼロであることは異なる。今回の約0.08msの構築より、
実際の集約・profile評価・H/G読出しを先に調べる。

backwardの通常経路はG生成・H再計算・dX atomic scatter・parameter partialを融合する。
G8経路はGを一時保存し、dX owner reductionを使う。両者を同じ「atomicなし」と呼ばない。
G8のpost24固定snapshotについて、既存runnerの別診断Graphで次を測る。

1. dX初期化、入力routing、入力係数などの準備。
2. G/dG生成、H/dH再計算、3種類のparameter partial生成を融合したproducer。
3. GからdXへのowner集約（通常/unsafe fallbackの2launchを含む）。
4. chunkのphysical parameter部分和を累積。
5. source Parameterへの最終VJP。

同じsnapshotの実backwardとdX/all-dPを照合し、source/packedが変化しないことを検査する。
主測定の完全stepと両peakを保存した後だけ診断を行い、5 replayと実input routing censusを記録する。
producer内をさらに分けるprobeは既存contractを用いてG/dGとH/dHをそれぞれ評価できるが、
融合解除でregister配置・中間転送が変わるため、その時間を元kernelから差し引かない。

実測したG8/H16・owner BM16の内訳（N2048/B32、post24固定診断、ms）：

| 区間 | rho3 | rho8 |
| --- | ---: | ---: |
| 入力routing・係数・dX初期化など | 0.0881 | 0.0891 |
| G/dG＋H/dH＋parameter partial | 1.2964 | 1.7664 |
| G→dX owner集約 | 0.9677 | 1.5585 |
| physical部分和累積 | 0.0614 | 0.0625 |
| source VJP | 0.0154 | 0.0164 |

生成と集約の双方が重い。これだけでcache missやatomic stallを原因と断定しない。
5種類のG8 route、両rhoで診断と実backwardのdX/all-dPの差は0だった。
完全stepの独立FP64 gateと両peak、raw evidenceは研究履歴へ記録した。

次の二候補を独立したbenchmark-local recipeとして実装し、forwardを固定して単独で比較する。

- 入力site-prefix：8siteの区分を1siteにし、dX候補を絞る。N2048ではprefixが
  257→2049個（int64で＋14KiB）、検索用arangeは＋7KiB。global D10かつ均一分布なら
  候補範囲40→28siteで約30%減を予想するが、実boundsで数え直す。broad支持なら効果は消える。
- 入力bin内の並べ替え：既存区分を保持し、`key=input_bin*NO+output_site` でsortする。
  prefixと候補数を維持し、G producerのdY局所性を狙う。既存Keysへ書くため追加A-sized配列は
  不要。同じinput-binには平均約819atomあり、input-site第一キーの約102atomより大きな集合を
  第二キーで並べられる一方、bin内の入力site順は保証しない。メモリpeakの同等は実測事項。

比較planは `input-site-stream-g8-h32` と `input-order-stream-g8-h32`。
基準 `site-owner-bm16-stream-g8-h32` とH32/G8、forward BM16を一致させる。
いずれもatom加算順・producer groupが変わるため、全勾配・floor・境界・snapshot・Graphの
検証を通してから完全stepを測る。既存forwardの二座標sort案は別に比較する。


実装source `323ff210` はL4で976 contract tests（skipなし）を通過した。
forwardのrouting/Yのbitwise一致、入力site-prefixの支持包含、coarse副キーのbin人口・
候補集合の同一性を直接検査した。全勾配・unsafe fallback・partial batch・retained snapshot・
Graph更新も新routeで検査する。候補訪問数とallocatorピークは、比較ごとに実測する。


### 入力routing独立比較の結論

Colab L4、N2048/B32/A209715、rho3/8、seed41、source `323ff210`。
各rhoで独立2回、各方式24 AdamW更新・21 primary timing samplesを測った。
下表は各runのmedianを2回で平均した値で、単位はms。
生成はG/dG＋H/dH再計算＋parameter部分和を含み、Gだけの時間ではない。

| rho | 方式 | setup | 生成 | G→dX集約 | 完全step |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | 基準 | 0.0880 | 1.2759 | 0.9636 | 4.7098 |
| 3 | 入力site候補削減 | 0.0937 | 1.2836 | 0.7076 | 4.5038 |
| 3 | 入力bin内の副キーsort | 0.0968 | 1.2186 | 0.9610 | 4.7530 |
| 8 | 基準 | 0.0891 | 1.7306 | 1.5334 | 6.5957 |
| 8 | 入力site候補削減 | 0.0947 | 1.7413 | 1.0829 | 6.1267 |
| 8 | 入力bin内の副キーsort | 0.0972 | 1.6451 | 1.5252 | 6.6024 |

- 入力site候補削減はrho3で候補訪問25.0%、rho8で30.0%減。
  dX集約は全4runで26.0〜29.8%短縮、完全stepはrho3で3.4〜5.3%、rho8で6.2〜8.1%短縮。
  生成部分は0.4〜0.9%増で、改善は主にdX集約に現れた。
  **測定条件での次の実験基準は入力site候補削減版とする。**
- 副キーsortは候補集合を維持し、生成診断は4.3〜5.3%短縮したが、完全stepは
  rho3で0.5%短縮／2.4%増、rho8で1.8%増／1.5%短縮と符号が反転した。
  **完全stepの再現した利得がないため保留。** A+Bの組合せや公開dispatcher変更は行わない。
- 3方式とも全4runでallocated/reserved peakは**57.43/104 MiB**。
  denseの**81.75/106 MiB**を両方下回るが、今回の変更によるpeak削減はない。
  site prefixの論理容量増（14KiB＋arange7KiB）は主測定peakを変えなかった。
- denseの完全stepは約0.59msで依然大幅に速い。入力候補削減だけでは差は埋まらない。
  次は生成部分を主な調査対象にする。

setup・生成・集約は別Graph診断であり、その和を完全stepとみなさない。
forwardのコード・候補・更新後パラメータは同一だがforward診断時間にも変動があるため、
その差を入力routing変更の効果としない。cache hit率や物理traffic改善は未測定。
976 GPU contract testsと48 initial/post24全atom FP64 gatesが成功し、全12診断の
実backwardとの差はdX/all-dPとも0。全4source/result archive、各1629 source entries・
37 result manifest entries、28 workers、同一rhoの反復条件と更新後hashを独立監査した。
全owned GPU slotsの停止を確認済み。


### producer分解で分かったこと

入力site候補削減版、source `ecfbb914`、Colab L4/N2048/B32/rho3,8の各独立2回。
下表はG→H分割Graphの各run medianを2回で平均した値。単位はms。

| rho | G/dG生成 | H/dH再計算 | parameter partial | 値のみG | 値のみH |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.8970 | 0.8515 | 0.3343 | 0.6651 | 0.5837 |
| 8 | 1.1924 | 1.1310 | 0.3287 | 0.9313 | 0.7931 |

G/dGとH/dHの両方が大きく、Gだけが突出する結果ではない。rho3→8で両contractの時間が
増える一方、最後の3部分和は約0.33msでほぼ一定だった。H→Gの逆順でも同じ順位で、
G/dGはrho3 0.8909/rho8 1.1894ms、H/dHは0.8520/1.1336msだった。
値だけのprobeもproduction NEED_P=FalseのG値と一致し、G値の計測は0.6656/0.9329ms。
ただし微分付きとの差には追加store・別specializationの影響もあり、純粋な微分費用としない。

**次の調査対象はG/H共通の `_contract` 内部**とする。profile評価・正規化微分と、
X/dYのgather・積和reductionのどちらが大きいかを分けて調べる。今回の結果から
cache miss、実traffic、算術律速のいずれかに原因を確定することはできない。
最後のparameter partialだけを先に最適化する根拠は弱い。

測定文脈による差も大きい。元のbackward schedule内で融合producerはrho3平均1.3271ms、
rho8平均1.8079msだったが、producerだけを回す別Graphでは1.8560/2.4028ms。
分割Graph全体は2.0838/2.6527msだった。**分割各段やその差・割合を、本番融合kernelの
厳密な内訳へ換算しない。** 主測定や融合の性能改善としても扱わない。

本番の既存runtimeは変更していない。primary完全stepはrho3 4.7369〜4.7509ms、
rho8 6.5390〜6.5537ms、全4runでallocated/reserved peak **57.43/104 MiB**。
denseは約0.59ms、81.75/106 MiB。診断の4中間bufferは論理25.60MiBで、productionの
G scratchに比べ追加3field=19.20MiBを使う。照合用bufferは別途15.20MiBであり、
これらは主測定後だけの診断storageであってprimary peakや最適化案のmemoryではない。

全4cohortで同じsnapshotのGは差0、partialの最大差9.32e-10以下、value-onlyの差0。
実backwardのdX/all-dPとの差は0。18 GPU testsと32 initial/post24全atom FP64 gatesを
通過した。各rhoの候補削減版は更新後Parameterがbitwise一致し、同じ条件の独立反復で
診断の順位を確認した。公開dispatcherや新しい実行方式の採用は行わない。

### 微分と実行方式の費用を分ける比較

G/Hの値のみと微分付きの差を見るときは、batch単位・atom順・出力field数を揃える。
値のみでも第2bufferへ同じ値を保存するcontrolを置き、微分付きと同じ論理保存量・
allocator増分になったことを確認する。この差は必要な微分処理を加えた実装上の差であり、
追加packed読出し・依存関係・register配置を含むため純粋なFLOPs時間とは呼ばない。

容量8/32の比較でBM8を固定した場合、変わるのは4回のchunk launchと1回の4-slab launch、
およびscratch容量である。32batchへprofileを一度だけ評価する共有の実験ではない。
atom順を変えたときはgroup内最大支持長に由来するpadding loopも確認する。
診断の`valid_sites`はprepared envelope長の合計で、full-axis fallback時の内部ゼロを
除いたpositive支持数ではない。loop数が同じでも、時間差だけでcache missを原因と断定しない。

G/DG/H/DHの保存を加えた融合版と、同じbufferを使う分割版を分けて比較する。
前者と元producerの差は追加store・値の寿命・compiler配置、後者との差はscratch再読出し・
追加launch・Order/P読出し等を含む。これらの診断差を足して本番backwardの厳密な内訳としない。

Orderはproducerとconsumer双方の都合で決める。G生成が出力順で速くても、dX consumerが
入力順を必要とするなら並べ替え／別Order／index変換の費用を完全stepへ含めて評価する。
H生成とY consumerにも同じ制約がある。両Orderを置いたproducerだけのpeakは本番stepの
peakとは区別し、採用は完全stepとallocated/reserved両peakで判断する。

source `1abbd00c` のL4/N2048/B32/rho3,8、各独立2回では、入力順・容量8の微分追加は
G +0.2188/+0.2738ms、H +0.2356/+0.2940msで、同じ2bufferのallocated/reserved peakは一致した。
元の融合producerは1.7468/2.3441ms・22.90/56MiB、中間値3fieldの保存追加で
2.1091/2.7065ms・42.90/76MiBになった。容量32も元の融合より約6.2%/4.9%遅かった。
Gは出力順、Hは入力順に利点があったが、融合全体の順序差は小さくrho3では反復で順位が逆転した。

この範囲では融合・容量8とconsumerに合う順序を維持し、微分を含むcontract内部の共有を
次の対象にする。全保存・分割・容量拡大は採用しない。これらの診断差は本番stepの内訳ではない。
[全条件・検証・監査記録](research-history/cuda-linear/regular-grid-h-lifecycle.ja.md)を参照する。

### Producer内の正規化共有を比較する研究候補

`input-site-reciprocal-g8-h32` は既存の `input-site-stream-g8-h32` と同じ
H32/G8、BM8、BK8、atom group8、input-site順、融合したG/H/parameter partialsを使う。
変更はbackward producer内の各軸について、atomごとに正しく丸めた `1 / norm` を求め、
`raw / norm` と `raw_derivative / norm` を乗算へ置き換えることだけ。
正規化を縮約の外へ出さず、joint floor、profile評価、中心微分、source VJPを維持する。
個別normが `[1e-6,128]` の外、NaN、またはsingleton flagを含むatom groupでは
その軸の従来の除算へ戻す。joint floorは個別normの下限を保証しないため、このfallbackを残す。
これはFP32の評価順を変える研究候補であり、bitwise同値を前提にしない。
既存Recipe/Algorithmおよび公開defaultを維持し、候補には独立したIDを与える。

比較前に固定する条件: L4、N2048、B32、A209715、rho3/8、seed41、FP32/TF32 off、
24 AdamW更新、各21 primary samples。候補・input-site除算版・matrix・factor・onchip・denseを
各fresh processで測り、独立2 cohort（2回目は順番を反転）とする。
初期/更新後の全Y/dX/all-dPの独立FP64 gateを両方通す。
未計装の完全stepとcapture/replay peak allocated/reservedを判断に使い、
別Graphでのproducer/owner/VJPの時間を完全stepから差し引かない。
低norm、empty、singleton、stride、retained backward、Graph更新を先に検証する。
GPU correctnessが失敗した場合は同sourceで性能測定へ進まない。

この比較は完了し、生成段は9〜11%短縮、完全stepはrho8で5.5〜11.9%短縮を2回確認した。
rho3は差なし／5.9%短縮で、全体改善を一律には主張しない。peak allocated/reservedは
除算版と同じ57.431/104MiB。詳細は[研究記録](research-history/cuda-linear/regular-grid-h-lifecycle.ja.md)の
2026-10-11の逆数共有比較を参照。候補は研究routeに保持し、公開defaultは変更しない。

### Backward producerのbatch共有を分離する比較

RTX 6000 Adaで、基準BM8/G8に対しBM8/G16とBM16/G16を比較する。
forwardのH32・producer BM8・Y owner BM16、input-site順、dX owner BM8、
BK8/group8、profileと正規化微分、inverse normの共有は同じ。
BM8/G16は保存容量とchunk launchの変更を含む対照で、BM16/G16との比較で
producer内のbatch共有を広げる効果を調べる。

新しい `InputBatchProducerHRecipe` はcomputeの `producer_batch_tile` と
G storageの `batch_tile=8` を分ける。Gは両対照とも `[2,A,8]`。
local row rの書込みoffsetは `((r//8)*A+sorted_position)*8+r%8`。
dX consumerは同じslabを読む。G/Hの値と中心微分はproducer内で融合し、
H/dH/dGをglobalへ追加保存せず、Gとparameter partialはchunkごとに上書きする。

parameter partialは `[ceil(min(B,16)/producer_batch_tile),3,A]`。
BM16のbatch reductionとG16のchunk累積ではFP32の加算順が変わるため、
全atom FP64 gate、tail/stride/floor/retained snapshot/Graph更新を先に検証する。
G16 scratchはG8より4*A*8 bytes増え、BM8/G16はpartialも1slab増える。
この論理容量をGPU peakと混同せず、capture/replayのallocated/reservedを実測する。
compute幅を広げることでregister圧力やoccupancyが悪化する可能性もある。

N2048/B32/A209715、rho3/8、seed41、24 AdamW更新を固定し、
3方式とdenseをfresh processで測る。各rho独立2cohort、2回目は逆順。
診断は同じpost24 snapshotの別Graphでproducer/dX/physical/source VJPを測り、
完全stepの内訳として足したり差し引いたりしない。公開defaultは変更しない。

比較はsource `b3e74be6`、RTX 6000 Adaのrho3/8・各独立2cohortで完了した。
BM16/G16はBM8/G8に対し完全stepが12.0–14.9%、生成段が24.8–35.6%短縮した。
BM8/G16対照に対しても完全stepが6.0–10.0%短縮した。
3方式の総allocated/reserved peakはすべて57.431/104 MiBで、denseの両peak以下。
この測定条件の次の研究比較基準を `input-site-reciprocal-bm16-g16-h32` とする。
119件の資格検証と48の全shape・全atom FP64 gatesはPASS。
詳細・加算順の差・証拠・実行コマンドは[研究記録](research-history/cuda-linear/regular-grid-h-lifecycle.ja.md)に残す。

### G32容量とproducer BM32を分離する次の比較

BM16/G16を固定基準とし、BM8/G32、BM16/G32、BM32/G32を独立した
`InputWideBatchProducerHRecipe` / Algorithmで比較する。既存G16 recipeの意味は変えない。
forward H32 / producer BM8 / Y owner BM16、入力順、dX owner BM8、BK8/group8、
正規化・微分・source VJPは維持する。G保存は `[4,A,8]`、partialは
`[ceil(min(B,32)/producer_batch_tile),3,A]`。H/dH/dGは追加保存しない。
BM16/G32対BM16/G16で容量・chunk数の差、BM32/G32対BM16/G32で
producer内の共有幅の差を調べる。BM8/G32は保存容量を揃えたcompute幅対照とする。

RTX 6000 Ada、N2048/B32/A209715、rho3/8、seed41、FP32/TF32 off、
24 AdamW更新、各21 primary samplesを固定する。上記4 routeとmatrix/factor/onchip/denseを
fresh processで実行し、各rho独立2cohort、2回目は逆順にする。
初期・post24の全shape/all-atom FP64 gateを変えず、tail/stride/低norm/retained snapshot/
Graph更新を先に検証する。判断指標は未計装の完全stepとcapture/replayの実測
allocated/reserved peak。別Graphのstage診断とstatic compiler情報を補助に使い、
register増加だけから実機occupancy低下を断定しない。結果に従って研究基準を選び、公開defaultは維持する。

比較はsource `6813b75c`で完了し、211資格検証/56 FP64 gatesがPASSした。
BM32/G32は完全stepを5.4–7.1%、生成段を26.4–29.7%短縮したが、総allocated/reserved peakは
66.731/126 MiBへ増えた。denseの81.752/106 MiBに対してreservedが上回るため、
両peak条件を満たすBM16/G16（57.431/104 MiB）を研究基準に維持する。
BM16/G32の容量だけの拡大は完全stepが0.2–0.9%短縮に留まり、生成段はほぼ同じ。
次はG16容量のままproducer内部の共有・読取りを調べる。全G32候補は測定済みの研究対照として
残し、公開defaultは変更しない。詳細は[研究記録](research-history/cuda-linear/regular-grid-h-lifecycle.ja.md)を参照。

### G16の詳細診断

採用候補の変更前に、BM16/G16のpost24 snapshotを固定して、元の融合producer、Gのみ、
G/Hそれぞれのvalue-only・同じ2bufferへのvalue書込み・中心微分、Pの並べ替え、partial書込み順、
batch連続入力、profile事前計算を別の対照として測る。最後の2つは追加buffer／準備費用を含め、
個別probe時間を融合時間から引かない。constant-site入力と256MiBの外部eviction pressureは
アクセス負荷の対照であり、実workloadの性能候補ではない。すべての数学対照をproduction値と
照合し、axisの縮約はmaterialized per-site fieldsを使ったFP64縮約とも照合する。

実際のGraph replayのCUPTI traceを別のmodel/optimizer copyで採り、kernel数・各kernel時間・
GPU時間間隔を保存する。元のpost24 parameterを変えず、完全stepとcapture/replay peakは
既存runnerで別に測る。RTX 6000 Adaのhardware countersはRmProfilingAdminOnly=1で制限され、
Nsight Computeも未導入。CUDA driverのoccupancy APIはresource上限だけを示し、達成occupancyや
cache hit/stallを測ったとは表現しない。支持・padding・CTA内アドレス重複・32byte sector集計も
実cache trafficとは区別する。rho3/8を各独立2回、2回目はprobeとprimary worker順を逆転する。
既存全shape FP64 Y/dX/all-dP gate・完全step/両peakを維持し、診断結果の精査後に次の実装を選ぶ。

### G16詳細診断からの実装優先順位（RTX 6000 Ada）

[19対照と全体traceの測定](research-history/cuda-linear/regular-grid-fine-bottlenecks.ja.md)では、
packed fieldsの物理配置をinput-orderへ揃える対照がproducerを約10–12%短縮した。
ただし別copyの準備に約0.036msと10.4MiBのpayloadが必要であり、完全stepの改善は未認定。
次は、G16/H32 capacityを保ち、余分な全量copyを増やさずにpacked/physical layoutを揃える。
raw atom IDへの対応は最後のphysical→source VJPまで明示的に保持する。

input順の8atomはH側の支持を約7.9倍重複して参照するが、G側は約1.01–1.03倍。
HとGのアクセス非対称を測り、output順G生成を試す場合はmappingとbuffer寿命も含める。
global profile cacheは追加25.6/51.2MiBで、rho8の生成込みprobeを約11%悪化させた。
batch連続入力も遅くなったため、連続化方向や全量保存を一律に選ばない。

Y/dX owner集約は合計で最大級のkernel群であり、候補siteの余分なcheckとvalue再参照を
次の独立対象にする。AdamW約0.249ms、Polar/periodic更新約0.118msも別対象として記録する。
allocator確保・解放は測定した生成Graph内で0回だった。実cache hit/stallと達成occupancyは
未計測であり、compilerの94registers/0spillsや理論occupancyだけで原因を断定しない。


### Joint input/output cell内でH/Gを使い切る比較

入力・出力の中心binを一度だけsortし、forwardのindexをbackwardで再利用する研究候補を
比較した。全量H/Gを作らず局所Y/dXの部分和へ消費し、最後のowner reductionで集約する。
Register保持量とhaloを削った修正版も含め、RTX 6000 AdaのN2048/B32/rho3+8では現行
BM16/G16より遅く、reservedがdenseを上回ったため不採用。index構築は約0.054msで、重いのは
H/G生成と局所集約を融合したkernelに残った。
詳細・正確なfallback・全gradient gates・完全step/両peak・source/result hashesは
[joint cell比較](research-history/cuda-linear/regular-grid-joint-local.ja.md)を参照。
入力・出力の近さだけで高速化を保証せず、局所保持量と支持外paddingも費用として扱う。

### 支持タイルとIEEE FP32小行列積

中心cellではなく支持区間から最大2タイル/軸へ所属させ、同じ開始タイルのatomで
X/dY panelを共有する候補をBM8/BM16で比較した。indexはforwardからbackwardへ再利用。
全支持のH/Gを集約してからparameter勾配を作り、タイル間の交差項と正規化を維持する。
H/dH/G/dGは全atom×小batchのslabであり、寿命はchunk単位。atom束内だけでは完結しない。
狭支持producerと広支持fallbackを分離してspillを0にした最終版でも、Ada N2048/B32の
完全stepはBM16で現行の2.11–2.26倍、両peak93.717/150MiBへ増えた。採用せず現行を維持する。
重いstageはH/Gと中心微分の生成で、sort約0.111msと最後のreductionは相対的に小さい。
支持に対するpanel paddingと微分slabの寿命を次の仮説として扱い、実cacheの原因は未断定。
127資格検証/48 full FP64 gatesと独立2cohort/rhoの完全step・メモリ・source修正の経緯は
[支持タイル研究記録](research-history/cuda-linear/regular-grid-tile-dot.ja.md)を参照。

### 生成粒度と微分寿命を独立に変更した比較

32-site routing/ownerは固定し、生成だけをlive支持和集合の16-site panelにする版と、
固定32-site生成でdH/dGをphysical勾配へ直ちに消費する版を別々に比較した。
支持幅版はH/G生成を約54–59%、完全stepを固定tile対照から37.70–39.92%短縮した。
しかし現行より28.85–40.14%遅く、両peak93.717/150MiBのため採用しない。
微分寿命版は4 slabからH/Gの2 slabへ減らし、allocated/reservedを68.117/122MiBへ削減したが、
H valueを先に用意して入力profile/微分を再生成するpassが増え、固定tileより10.24–11.02%遅かった。
生成粒度と保持量の改善を混ぜず、追加passとownerアクセスを寿命の費用として扱う。
現行BM16/G16/H32を維持する。194資格検証/56全shape FP64 gates、独立2cohort/rhoの
完全step・両peak・6固定診断・復旧hashは
[粒度と微分寿命の研究記録](research-history/cuda-linear/regular-grid-tile-schedule.ja.md)を参照。

### 支持幅生成と即時消費の組み合わせ

二つの効果を独立したflagで組み合わせた研究planを比較した。
H/Gの2 slabだけを保持し、dH/dGをproducer内でphysical勾配へ消費する。
支持幅単独からallocatedを25.601MiB/reservedを28MiB減らしたが、完全stepは2.85–3.22%遅い。
組み合わせはrho3=2.051–2.055ms/rho8=2.421–2.426ms、68.117/122MiB。
現行より29.39–44.88%遅く、reservedはdense106MiBより大きいため採用しない。
追加のH value生成passは約0.235/0.292ms残る。全支持縮約後に勾配を作る契約は維持する。
224資格検証/64 full FP64 gatesと独立2cohort/rho、8固定診断を確認。
詳細は[組み合わせ比較](research-history/cuda-linear/regular-grid-tile-combined.ja.md)。
任意の支持tile数への一般化は次の設計議論とし、今回の測定へ混ぜない。

### 支持幅bucketと可変タイル数

各軸にA個のint32 IDを保存し、幅区分と開始タイルで候補範囲を絞る方式を実装した。
H/Gはatom束が支持全体をregisterで足し切り、即時消費のH/G 2 slabを維持する。
2タイル制限と広支持all-atom fallbackを外し、周期・短い末尾tileの重複を避ける。
幅16は組み合わせ対照より1.37–6.40%速くallocated62.523MiBへ減ったが、
現行より27.31–35.17%遅くreserved124MiBがdense106MiBを上回る。幅32も不採用。
H/G生成は約半分になった一方、Y/dX ownerが幅16で約2倍、幅32で約3.2–3.4倍へ悪化した。
幅区分・候補範囲・所属判定の反復とrangeの分割/paddingを次の切り分け対象とする。
今回の候補数は準備済みinterval所属数と一致しており、単なる候補過剰では説明できない。
251資格検証/72 full FP64 gates、独立2cohort/rhoと10固定診断の詳細は
[支持幅bucket比較](research-history/cuda-linear/regular-grid-tile-bucket.ja.md)へ保存した。

### タイル16 ownerの準備移動

幅区分・範囲・所属情報を準備へ移す単独版と組み合わせ版を同条件で比較した。
幅区分だけで完全stepを元bucket16から17.57–20.93%、全部で25.96–29.03%短縮。
全部版は現行より3.32–6.49%速く、Y/dX ownerは約63–67%短くなった。
追加準備は約0.0285ms、allocated65.975MiB/reserved124MiB。reservedがdense106MiBを
上回るため採用は保留し、現行の両peak適合baselineを維持する。
410資格検証、72 full FP64 gates、独立2cohort/rho、10固定診断をPASS。
複数周期の代表座標による共通勾配誤差も、全量コピーなしの準備内canonical reductionで修正。
詳細は[owner準備移動の記録](research-history/cuda-linear/regular-grid-tile-owner-prep.ja.md)。

### Graph poolとbuffer寿命の切り分け

owner組み合わせのreserved124 MiBを通常pool60/Graph pool64 MiBに分けた。
H/partial共用は130 MiB、dP遅延だけは124 MiB、arenaだけは122 MiBで不採用。
AdamW moments事前初期化・capture前warmup勾配解放・dP遅延を組み合わせると
完全stepのallocated/reserved=62.774/88 MiB、同条件dense=81.752/104 MiB。
速度差は通常ownerに対し−0.255〜+0.239%、rho3約1.381 ms/rho8約1.716–1.718 ms。
545資格検証/112 full FP64 gatesをPASSし、24更新後のParameter/全勾配/moments/clockはbitwise一致。
通常pool22/Graph pool66 MiBで、主効果は通常poolの断片化とwarmup所有を減らしたこと。
peakはwarmupを含めてリセットせず、初期化自体の確保も含める。
研究runnerの明示policyであり公開dispatcherの既定は変更しない。
[Graph pool寿命の記録](research-history/cuda-linear/regular-grid-graph-pool-lifetime.ja.md)に負の結果・policy別比較・復旧hashを保存した。


## 共通カーネル内の支持モード（2026-10-11）

単一の実行方式で幅の混在を扱う。rhoに基づく公開dispatcherを増やさず、
現在の支持とjoint norm floorの状態で処理を選ぶ。文字どおりGPU起動を一回に
限定せず、H/Gの再利用とatomicなしのowner集約に必要な段階境界は残す。

研究Plan `support-modes-bm16` は現行 `tile-owner-late-bm16` と同じexecutor、
producer、ownerを使う。既存13-field packedのsingle-site flagsと支持区間から
両軸単一・floor非適用を判定し、各軸のsort keyの先頭bandにまとめる。
一般支持は従来のdyadic bandを一つずらし、空支持はowner範囲の末尾へ置く。
全量packedコピー、CSR、新しいatom ID配列は追加しない。

- 空支持: producerの寄与を省き、ownerから除く。Parameter/optimizer stateは維持。
- 両軸単一・floor非適用: ownerがX/dYを直接参照し、amplitude勾配を
  `sum_b X[b,i_a] dY[b,j_a]` から求める。中心task勾配は0。
  H/G slabへの読み書きもprofile再評価も行わない。
- その他: floor適用単一サイトを含め、既存の正規化・微分・小batch H/G生成を使う。

初版は支持モードを学習中に毎forward再分類し、snapshotをbackwardで使う。
混在率と支持が更新されてもGraph replayできるよう、H/G容量は引き続き
`atom_count × batch_tile` を確保する。単一サイト部分の読み書きを省くことと、
allocator peakを削ることは別の改善であり、初版で容量削減を主張しない。
追加のowner band/rangesが小さなメモリ費用となる。
初版はD2 full-period RegularGrid/Triweight/POLARに限定する。
独立FP64のY/dX/全atom勾配、floor境界、空支持、retained backward、
Graphの可変支持を検証してから、準備込み完全stepとpeak allocated/reservedを比較する。
