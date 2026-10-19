# 2次B-splineのGPU直接評価と共有累積集計

2026-10-17。branch `kernel/quadratic-bspline-prefix`。
[merge済みTorch参照](quadratic-bspline-profile.ja.md)と同じ形・離散L2・Polar更新を使う研究Algorithm。
公開Registryや既定dispatcherは変更しない。以下は初版の実装範囲と検証手順。

## 初版の範囲

- Euclidean RegularGrid、出力1座標×入力1座標。軸のsite数は2以上。
- `QuadraticBSplineSpec` revision 1のPolar profile-product。中心曲率幅sigma、支持半幅√6 sigma。
- X/sourceはFP32、chartのorigin/spacingはFP64。batch 1..64、autocast/TF32なし。
- 任意の中心位置とatomごとのlive sigma。支持は有限gridの観測siteで切る。
- 入力側・出力側のbandwidth boundsは同一。異なる設定は宣言検査で拒否する。
- 空支持、零振幅、norm floor、retained backward、forward後の中心変更を扱う。
- first-order VJP。高階微分、周期座標、入出力側に複数の座標を持つgridは初版の範囲外。

`research_cuda_quadratic_bspline`のRecipeで同じ演算の4経路を比較する。
`torch`は既存のTorch factored参照、`direct`は支持内だけの軸別GPU直接評価、
`hybrid`は各軸の支持site数により直接評価と共有prefixを選ぶ。
直接評価のsite演算枠は16/64、半幅の逆数のatom内共有も独立のRecipe設定として比較する。
初期候補の閾値32 siteは実装上の比較値で、最適な閾値やσの上限の結論ではない。
計算blockと数学的な支持／周期を混同せず、支持をblockへ丸めない。
FourierのM=4/8に相当する可変次数は使わず、次数2・3pieceの同じ山を全経路で評価する。

## 共有する値

格子indexをz_j∈[-1,1]へ正規化する。B-splineの各pieceを
`F_a(j)=c_a0+c_a1*z_j+c_a2*z_j²`と展開し、入力ごとに3本の

\[
P_{bk}(t)=\sum_{j<t}z_j^k X_{bj},\qquad k=0,1,2
\]

を作る。各pieceのindex区間[l,r)から
`Σ_k c_ak(P_bk(r)-P_bk(l))`でHを取り出す。
中心微分は同じprefixを別の係数で読む。境界siteの直接走査を追加しない。
入力が変わったstepではprefixを作り直す。sigma・中心・norm・係数も毎forward更新する。
固定index基底だけを同じdevice/shapeで再利用できる。

書き戻しは随伴で、pieceの左端へ`+c_k*w`、右端へ`-c_k*w`を加え、
site方向の累積和と基底の内積で出力を復元する。
端点の衝突にatomic加算が残る。CSRやsite×atomのindex表は作らない。
小さい支持は同じ山を支持内で直接評価し、siteへatomic加算する。

初版のbank、端点加算、係数、normはFP64。H/H'とG/G'、Y、dX、source勾配はFP32。
FP64にするだけで任意のdomain/幅比の精度を保証するわけではない。
FP32 prefixの採用や局所blockへの変更には独立oracleによる再検証が必要。

## 離散L2とVJP

各pieceのsample数をK、sample中点をm、`t_j=(2j+1-K)/K`とする。
山を局所座標で`a0+a1*t+a2*t²`と表すと

\[
M_0=K,\quad M_2=K/3-1/(3K),\quad
M_4=K/5-2/(3K)+7/(15K^3),
\]
\[
Q^{piece}=a_0^2M_0+(a_1^2+2a_0a_2)M_2+a_2^2M_4.
\]

空区間はQ=0、K=1ではM2=M4=0を使う。3pieceのQを足す。
積分への置換ではなく、実際の端点位相を含む有限離散和の恒等式。
中心微分の`S=Σ F*dF/dmu`も同じ局所momentで計算する。
`D=max(sqrt(Qo*Qi),epsilon)`としてfloorは積全体へ一度だけ適用する。

生profileによるH=Fi X、G=Foᵀ dYとその中心微分Hp/Gpを使い、batchについて

\[
d\alpha=\langle H,G\rangle/D,
\]
\[
d\mu_o=(\alpha/D)\{\langle H,G_p\rangle-\gamma_o\langle H,G\rangle\},\quad
d\mu_i=(\alpha/D)\{\langle H_p,G\rangle-\gamma_i\langle H,G\rangle\}.
\]

floor非作動時はgamma=S/Q、作動時は0。振幅のPolar pullbackは元sourceの
特異原点guardを含めて行う。幅はtask VJPでstop-gradient、更新は既存のPolar則。
forwardのmetadata、origin、spacing、H/Hpを保存し、後のforwardや更新値と混ぜない。

## 検証と測定

`tests/test_quadratic_bspline_cuda.py`は符号付き3区間式と全軸の離散normによる
独立FP64 oracleで、Y/dX/全source勾配を照合する。
公開CSTOptimizerは現在Graph captureを拒否するため、Graph検証は既存research runnerの
capturable AdamW＋Polar updateを使い、公開CSTOptimizerのeager更新と比較する。

既存`benchmarks.cuda.linear.run`へfixture、Plan catalog、Caseを接続した。
Case内の既存`widths.rho`欄はこの研究fixtureでは物理sigmaの初期値を記録する。
実際のsigma/spacingはdecode結果とspacing=.1から求め、他fixtureのrhoと混同しない。
別の計測runnerは作らない。`bspline.py`がfixtureとoracleを持ち、
結果adapter revision 9で他profileと区別する。dispatcherへの既定採用は未実施。

```bash
PYTHONPATH=src:. python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-64-a204-small.json
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-64-a204-small.json \
  --polar-update fused --phase-diagnostics --output output/bspline-small.json
python -m benchmarks.submissions check output/bspline-small.json
```

完全stepの未計装Graph時間とcapture/replay込みのallocated/reserved peakで比較する。
診断phaseの合計を完全stepの代用にせず、独立run数と計測sample数を区別する。
原本はignored `output/bspline-gpu-20261017/`および共有poolのjob directoryに保全する。


## 小さい支持のmatrix経路

`matrix`は支持内のFo*FiをWへ集約し、Y=XWᵀ、dX=dY WをGEMMで計算する。
`dW=dYᵀX`を全atomで共有し、T=Σji dWji Foja Fiiaとすると
`dalpha=T/D`、`dmu_o=(alpha/D)(Σ dW Fo_prime Fi - gamma_o*T)`、入力中心も同様。
H/Gでbatchを先に縮約する式と、和の順序だけが異なる。
生profile・global floor・幅stop-gradientは同じ。支持が広がった場合も全区間を走査する。
初版は各軸1024 site以下だけを受け付け、大きいsigmaでの高速性は保証しない。
W/dWとGEMM workspaceを含めた完全step peakを測り、小さい支持向けの候補として扱う。

5万atomを超える測定では、全sigma配列をJSONへ列挙すると受付上限5MiBを超える。
このfixtureに限り、配列をhash付き`*.widths.pt`へ保存し、JSONにはsample indexと16例、
全体のmin/max、changed-atom数を記録する。値を打ち切って実行する変更ではなく、
計測後の証拠表現だけを小さくする。両方のファイルをpoolのmanifestで保全する。

## 固定moment表の再利用

`norm_lookup=True`はK=0..max(No,Ni)のM2/M4をFP64で一度だけ作り、pieceのsample数Kで読む。
幅の量子化や係数の補間ではなく、上記の有限和そのものを再利用する。
Kと局所座標係数・norm・norm微分は各forwardで更新する。表はdevice/shapeごとに保持し、
メモリ計測に含む。`norm_lookup=False`は同じ式を毎回評価する対照。
`reciprocal=True`ではatom内のRの逆数とspacingの逆数を共有する。
widthのupper decay powerが1の場合はxを使い、他の許可されたpower (0,1)には元のpowを使う。
どちらもlive scalarを読み、学習値を固定・切り捨てない。

最終版のmetadataはatomごとに12個のFP64値を持つ。振幅・支持半幅・2中心・逆norm・
2つのnorm補正・2つのPolar振幅微分・逆支持半幅・2つの逆spacingである。
逆数もforwardごとに更新し、H/G query、端点書き戻し、matrix構築/VJPで共用する。

## 計算量と保存量

出力/入力site数をNo/Ni、batchをB、atom数をA、各軸支持site数をSo/Siとする。
directは概ね`O(AB(So+Si))`、保存するH/Hpは`2BA`個のFP32。
hybridの広い支持は各軸3本のprefixと端点scanが`O(B(No+Ni))`、
atom query/端点加算が`O(AB)`となる。広い支持の境界site走査はない。
狭い支持ではdirectと同じ支持site演算を行う。
moment表は`16*(max(No,Ni)+1)` bytesで、1024 siteなら約16 KiB。
normの算出はatomごと3piece×2軸で、支持site数には比例しない。

matrixは`O(A So Si)`のW構築/VJPと、`O(B No Ni)`の3つのGEMMを使う。
WとdWの保存量は合計`8 No Ni` bytesで、1024²なら8 MiB。
batchを全atomで共有する効果がある一方、支持が広いほど各atomの支持積が増える。
これらはtensor量の式であり、allocatorやGEMM workspace込みの実測peakとは区別する。

## L4測定記録

NVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0、IEEE FP32。
chart座標・geometry/norm/prefixはFP64。AdamW＋既存fused Polar更新を含む。
spacing=.1、smallの初期sigmaは(.1,.14,.18)、mixedは(.1,.4,1.6,6.4,12)、
wideは(3.2,6.4,12)、各群をほぼ同数でshuffleしている。
各atomのsigmaはPolar状態から毎forward/replayで算出する。

次の表はsource `c4343268776e3745c5fc44e79bd40ff0d9ce9625`の完全step中央値(ms)。
独立runは各cell 1回、各run 21 sample。小さいσのmatrixと広いσのhybridを別々に評価し、
両者を混在させたdispatcherの測定ではない。

| N, A, B, 初期幅 | Torch参照 | direct | hybrid | matrix | Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64, 204, 32, small | .4519 | .1368 | .1736 | .1117 | .0390 |
| 128, 819, 32, small | .6822 | .3521 | .3971 | .1552 | .0453 |
| 1024, 2048, 8, mixed | 11.0630 | 2.3784 | .7686 | 未測定 | .0720 |
| 1024, 52428, 32, small | 444.6252 | 18.6944 | 18.7661 | 4.9743 | .0818 |
| 1024, 52428, 32, mixed | 449.1328 | 未測定 | 31.3840 | 未測定 | .0814 |
| 1024, 52428, 32, wide | 450.2825 | 未測定 | 37.0841 | 未測定 | .0823 |

同じsource、1024/A52428/smallのcapture/replay込みpeak allocated/reservedは、
Torch=8137.09/17468 MiB、direct=33.73/98 MiB、hybrid=34.48/98 MiB、
matrix=48.63/114 MiB、Dense=49.00/106 MiB。
matrixはdirectより速いが保存量が増え、Denseよりreservedが大きい。
GPUプロセス全体のpeakは未測定。Dense並みの速度・メモリを達成したという結果ではない。

固定moment表の効果はmatrix-formula対matrixで比較した。
64では.1237→.1117 ms、128では.1670→.1552 ms、1024/A52428では5.1237→4.9743 ms。
norm自体は毎forward算出し、Kだけで決まるmomentの再評価を除いている。
phase診断は別の計装Graphで、1024/A52428/smallのmatrixは
forward/loss=1.8391、backward=3.0935、optimizer=.0451 msだった。
計装値を完全step時間の代わりには使わない。

この全6 caseはY/dX/全atom勾配の独立FP64検査と提出前検査を通過した。
原本jobは`l4job-d538de6865184a28adaf8573f6604a8e`、
source archive SHA256は`4f896f760f62bb226f310e9c9bb3b102b81713f9cf6dbcc39a1d5e09c7b6609c`、
result archive SHA256は`57eaad75e95812cd535f0924722b0eab04c7aa4f309374a96b142d719b45799d`。
manifest全118ファイルのhashを回収後に照合済み。

最終CUDAコードsource `83b2dae4`では追加stress検証45件を通過した
(`l4job-25bec8dfcd4c477080e5b4f6aa57a446`)。
direct/hybridの64..8192 site、matrixの64..1024 site、2種類のnorm floor、
極小片側支持、空支持、零振幅、異なるorigin/spacingの矩形grid、
live width-law power=1/.5/.75、非連続stride入力、retained backwardを含む。
3経路それぞれでGraphを24更新し、eager公開optimizerとParameter/moments/stepを照合し、
更新後のY/dX/全atom勾配も独立oracleで検査した。
45件のうち1件はCPU metadata検査、44件が実CUDA検査である。
CPU全suiteは1968 passed / 2820 skipped (38.57 s)。
wheel/sdistはbuild成功、全330 runtimeファイルがcheckoutとbyte一致した。
8192²・約5% atomの性能測定、周期座標、複数座標軸、実DB登録は未実施。

失敗した初期jobも同じignored evidenceへ保持した。
Triton構文とGraph検証のclock差は修正し、5MiB超過の配列証拠は上記sidecar方式へ移した。
追加width-lawテストでAPI範囲外のpower=2/1.5を指定した2件は、検証設定を修正して再実行した。
誤差閾値の緩和、支持の切り捨て、固定sigmaへの置換は行っていない。

### 逆数を保存して共有する最終版

同じsource `83b2dae4db39d7aa546725ca22f4237486812df3`で上記45件を再度通した後、
1024/A52428/B32の2 caseを再測定した。条件・21 sample・独立run数1は同じ。
全52428 atomでsigma更新を確認した。

| 初期幅 | Torch | direct64 | direct | hybrid64 | hybrid | matrix-formula | matrix | Dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 442.6622 | 41.0390 | 14.4709 | 41.0989 | 14.5629 | 4.8603 | 4.7104 | .0817 |
| mixed | 447.3258 | 未測定 | 未測定 | 47.0869 | 24.7973 | 未測定 | 未測定 | .0821 |

最終版のallocated/reserved MiBは、direct=34.93/98、hybrid=35.68/98、
matrix=49.83/114、Dense=49.00/106。逆数3個×FP64×Aのmetadata増加分がpeakにも現れる。
smallのmatrixはTorch比約94倍速いがDenseの約58倍の時間で、allocatedもDenseを超えた。
mixedのhybridはTorch比約18倍速く、Denseよりallocatedを減らす一方、時間は約302倍。
時間とメモリの両方でDenseを上回るという採用条件は満たしていない。

別計装Graphのsmall matrixはforward/loss=1.7162、backward=2.9553、optimizer=.0451 ms。
mixed hybridはそれぞれ11.9501、12.7939、.0451 ms。
backwardの個別kernelやL2単独の時間はこのphase診断からは確定しない。
FP64のprofile/係数演算・atomごとの処理・atomic書き戻しを次の診断対象とするが、
この測定だけで各kernelの支配率やFP32化の可否を結論しない。

job `l4job-d8644b2e23be4ea9b1817688f384d3cb`は両case PASS、提出前検査もPASS。
source archive SHA256は`a9c5e8087a219aa7f4d37d50b6091774df6411285ceb2deb47e5ed7080d5f53e`、
result archive SHA256は`e47ea5650542a519c6a9710deb8d41f507d70a4b3db59e4affd625235ca569df`。
manifest全43ファイルを照合済み。両caseのfull-axis/全atom FP64検査を省略していない。


## 内訳診断と次の最適化

[個別stage・レジスタ退避・launch配置の比較](research-history/cuda-linear/quadratic-bspline-bottleneck-20261017.ja.md)
に続く。v1の宣言と固定launchを保持し、追加設定はv2のBSplineTuningRecipeへ分離した。
小さいsigmaではmatrixの小patch＋axiswise VJP、混在sigmaではhybridのwarp削減を比較する。
全候補は既存runnerの`--kernel-diagnostics`で診断できる。新しい数学的profileや近似L2ではない。

[支持点列挙・atom集約・prepareの追加比較](research-history/cuda-linear/quadratic-bspline-parallel-20261017.ja.md)
ではv3 Recipeを追加し、prepareのwarp削減とVJPだけのatom集約を検証した。
全面的なatom集約と支持点のflat列挙は負の結果で、対照とともに記録している。

[live支持点数によるatomのグループ化](research-history/cuda-linear/quadratic-bspline-support-grouped-20261018.ja.md)
では、両軸の支持patch数で毎step atom IDをソートし、W構築とVJPで順序を共有する。
小さいσと大小混在を分け、並べ替え・厳密L2・更新を含む完全step、独立FP64の全勾配、
capture/replay込みのpeakを比較している。

[atomごと・stepごとのsigma振り分け](research-history/cuda-linear/quadratic-bspline-sigma-split-20261018.ja.md)
ではv4 Recipeで小支持Wと大支持Hを合算する。混在時の速度改善、全small時の余分な処理、
固定容量によるメモリ増加、閾値をまたぐ移動と全勾配の検証を分けて記録した。

[配置の継続利用とHの即時消費](research-history/cuda-linear/quadratic-bspline-stream-20261018.ja.md)
では、v5でGPU検証付きの配置再利用とregister内H/G、v6でW再構築とW/dW領域再利用を比較した。
形と厳密L2を維持したmixedでの速度/確保量削減、smallでの負の結果、閾値依存、
予約量の未達、独立run、全勾配精度を分けて記録している。

[Wの係数区間集約とその随伴](research-history/cuda-linear/quadratic-bspline-range-20261018.ja.md)
では、同じB-splineをbatch非依存でWへ集約する別研究branchを検証した。
B64 mixedで約42%短縮した一方、B32 mixedは改善せず、smallは遅く、総peak memoryも増えた。
FP64境界と小支持の安定性guardを含む全勾配gate、独立repeat、負の結果を記録し、
既定採用せずPR110に分離している。


係数面の第0面をscanの中間領域に使い、3面を次数間で再利用する候補は
[係数面の寿命・scan/atom VJPの切り分け](research-history/cuda-linear/quadratic-bspline-range-reuse-20261018.ja.md)
を参照。L2・形状・幅stop-gradientは同じで、16atom/1warpの実行構成も比較する。
block scanと空間順の負の結果を含め、公開の既定採用とは区別する。


[H方式の係数前計算と出力集約](research-history/cuda-linear/quadratic-bspline-h-precompute-20261019.ja.md)
では、v12で係数をforward/backward間で共有し、H queryとY injectionの分離測定、
終端書込の省略と局所集約、切替点eventのsegmented集約を比較する。
従来H・最速W・Denseの完全stepと総peak、可変sigmaの全勾配gateを同条件で検証し、
分離版の中間H/G保存と融合版のregister消費を区別する。公開の既定登録は変更しない。

[Hの連続配置とendpoint集約](research-history/cuda-linear/quadratic-bspline-h-coalesced-20261019.ja.md)
では、batch連続のbank、入力区間の共有、局所reduction、出力担当によるatomic削減を
同じL2・全atom勾配で比較する。研究用v13であり公開dispatcherは変更しない。

[forward H保存とbackwardの分解](research-history/cuda-linear/quadratic-bspline-h-save-20261019.ja.md)
では、H/Hμの再計算・G/Gμ生成・atom勾配・dX差分集約を分離診断し、
forwardのHのみ／Hと中心微分の保存を可変σで比較する。研究用v14で、数学と精度条件を維持する。

[G生成の共有とdX集約の比較](research-history/cuda-linear/quadratic-bspline-adjoint-20261019.ja.md)
では、区間のsort/histogram、prefix読み取り共有、局所合算、FP64差分bufferの分散を
研究用v15で比較する。完全step・capture込みpeakと全勾配を同条件で記録する。
