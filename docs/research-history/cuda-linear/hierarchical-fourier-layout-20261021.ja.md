# 階層Fourier tileの分割・保持・転置を詰める

2026-10-21。`DyadicFourierSpec`を追加したsource `395cc27b`で、次の実行方式を整理する。
これは設計と支持分割の診断であり、新しいGPU executorや速度の測定ではない。
中心・sigma・形・支持・L2を変えず、[既存bottom-up研究](bottom-up-fourier-cpu-20261021.ja.md)
の部分leaf費用を減らすことを目的にする。

## 小tileから親への値の受け渡し

固定したグローバル座標の基底に対して、入力tile tのmomentを

\[
Z_t(\boldsymbol\omega)=\sum_{\mathbf i\in t}X_{\mathbf i}
 e^{i\boldsymbol\omega\cdot\mathbf x_i}
\]

と置く。同一周波数なら、親pの値は子の和Z_p(ω)=Σ_c Z_c(ω)。
局所座標の基底を使う実装には原点移動の位相が必要だが、この式には不要。
中心の位相・sigma依存係数はatomを評価するときに掛ける。

ただし親の周期は子の2倍であり、親の奇数harmonicは子の通常の整数harmonicと一致しない。
子のM成分だけから親の全M成分を厳密に作ることはできない。
現研究の解は、上位で必要な周波数をmetadataとして末端まで伝え、その追加成分も計算すること。
各nodeの必要集合はQ_t=Q_own(t)∪Q_carry(t)であり、実際の容量と費用は|Q_t|で数える。
単に各tileを常にM個の値として数えない。実入力の共役対称性・格子aliasも考慮する。

2次元の一般のXでは、profileが軸の積でもjoint momentが必要。
非aliasの単一尺度で(2M+1)^2個の実基底成分に相当するため、
M4は81、M8は289。軸ごとの個数の和だけで全momentを保持できるわけではない。
小さい周期ではaliasで減るが、複数尺度の追加周波数も別に必要になる。

## 正方quadtreeだけで支持長方形を切らない

各軸の支持site区間を独立にdyadic区間と末端の部分区間へ分割し、直積を取る。
長い境界stripを多数の小正方形に分けず、例えば64×8の矩形として扱える。
この方式の準備では、X→x方向moment→y方向momentなど軸ごとの縮約を使う。
全ての矩形・周波数を常設することは要求しない。集約DAGとその順序・保存量の比較が必要。

### 支持分割だけのCPU診断

既存runnerのfixtures(2,128,64,regime)、M8、3動的step、spacing=.1。
sigmaはatom・軸・stepごとに異なる。smallは約.1〜.15、mixedは約.1〜1.6。
尺度混合の各成分のboxも数えるため、box数はatom数とは異なる。
各分割を整数site上に再構成し、重複・欠落なしで元のboxと一致することをassertした。
最小Lとsigma→Lの規則はMに依存しないが、今回はM8のfixtureのみを実行した。

| 分布 | leaf辺長 | 正方query数 | 矩形query数 | 正方の部分query | 矩形の部分query |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 8 | 2200 | 2110 | 1938 | 1876 |
| small | 4 | 5032 | 3446 | 3335 | 2377 |
| small | 1 | 13458 | 4448 | 0 | 0 |
| mixed | 8 | 10235 | 3847 | 6405 | 2540 |
| mixed | 4 | 21103 | 5269 | 10211 | 2664 |
| mixed | 1 | 55762 | 6640 | 0 | 0 |

mixed/leaf8ではquery数62.4%、部分query数60.3%の削減。
ただしqueryはatom・尺度成分からの延べ参照数であり、unique保存node数ではない。
周波数数、moment構築、carry、prefixの各要素数、準備metadata、L2、Y/dXは含まない。
この表だけでメモリアクセス全体や速度が同率で減るとは判断しない。
leaf1では部分queryが消えるが、smallの矩形query数はleaf8の2倍以上に増える。
細分化だけで最適になるわけではない。
原本と再生成driverはignored `output/hierarchical-fourier-layout-20261021/`に保存する。

## 論理的な小tileとGPUの実行単位を分ける

4/8/16点のleafは支持とmomentを共有する単位。GPU blockをその小ささにする必要はない。
32/64/128点等のまとまりを実行単位の候補にし、その内部で小tile→親を処理する。
具体的なthread数、各軸辺長、batch chunkは未決定で、GPUで測る。

小tileの全momentをglobal memoryに保存すると、入力そのものより大きくなる場合がある。
実行単位内ではraw X、basis、momentをregister/shared memoryで再利用し、
小さいsigmaのatomを評価後、親で使う成分だけを残す候補を優先する。
globalへ渡すのは実行単位をまたぐcarryと必要なH等に限定する。
全祖先の周波数集合がshared memoryへ収まるとは仮定せず、周波数chunkと小batchで費用を数える。
chunk間にXを再読する場合も、sharedからの再読とglobalからの再読を区別する。

## H生成とY/dXを同じplanの転置として設計する

実数sin/cos基底で入力moment生成をA、atom係数との縮約をCと書くと、H=CAX。
入力backwardはdX=A^T C^T dH。同じ支持分割・基底・位相の転置を使える。
出力側もY=A_out^T C_out^T(Amp*H/D)の形で、先にatomの係数寄与をまとめ、
親から子へ係数を配り、各出力tileでsite値を再構成する。
入力値は小tile→親、出力側の係数はその逆向きであり、役割を混同しない。

出力/dXの実行tileに書き込み担当を一つ持たせる方式では、siteごとのatomicを減らせる可能性がある。
係数の集約自体のreduction・中間bufferは別に必要であり、全atomicが無料で消えるとはしない。
全atom勾配には係数・中心位相・sigma混合・離散L2の微分も含め、Torch参照と照合する。
Polarのtask VJPと、profile単体のsigma勾配の契約も区別する。

## 次の比較で決めること

優先候補は軸別矩形分割＋実行単位内のbottom-up＋最後の使用後のbuffer再利用。
対照は既存quadtreeと、同じFourier関数の小支持局所直接評価。
計算順序で性能を変え、kernelの形や中心・sigmaの自由度は変えない。

M4/8、leaf4/8/16、小sigmaとmixed、共有atom数を変え、準備・L2・集約・H・転置まで測る。
query数だけでなく、必要周波数対、X/basis/carry/Hのread/write、peakの生存bufferを記録する。
その後にGPU実通信、Y/dX/全atom勾配、完全step時間とallocated/reservedを確認する。
この設計ノートの時点でGPUを再開したり、旧Dense目標の達成を主張したりはしない。
