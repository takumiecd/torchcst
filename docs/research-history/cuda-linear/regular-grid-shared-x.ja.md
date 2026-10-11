# RegularGridのH/dH生成で重なるXを共有する比較

2026-10-11。input-owner backward融合の負の結果を受け、G16の共有・寿命を
現行のままにし、H/dH生成で重なるXを8 atom内で共有する研究候補。
公開Registry/default、RegularGrid/FlatTorus、profile-product、operator-sites L2と
一回floor、singleton、task width stop-gradient、AdamW/Polar更新は変更しない。

## 演算と共有範囲

入力profileをV、出力profileをU、振幅をAとすると、
H[a,b] = sum_i X[b,i] V[a,i]、dH[a,b] = sum_i X[b,i] dV[a,i]。
G[a,b] = sum_o dY[b,o] U[a,o]、dG[a,b] = sum_o dY[b,o] dU[a,o]を使い、
dA[a] = sum_b H[a,b] G[a,b]、dci[a] = A[a] sum_b dH[a,b] G[a,b]、
dco[a] = A[a] sum_b H[a,b] dG[a,b]を既存のsource VJPへ渡す。
profileと中心微分は全chartのL2と一回floor、singletonの契約を含む。
今回共有するのはH/dHのXだけで、G/dG・profile値の計算回数は変えない。

## 実装と固定条件

`research_cuda_regular_grid_shared_x_h` v1、`SharedXHRecipe(panel_sites=32/64)`。
`input-shared-x32-bm16-g16-h32` / `input-shared-x64-bm16-g16-h32`を明示的に比較する。
forward、prepared source snapshot、input routing/sort、hot fields、G16/slab、BM16、
dX ownerとparameter partial/source VJPは既存BM16/G16へ戻して共用する。
G/dGはatomごとに一度計算し、Gをglobal Tensorへ保存してdXへ共有する。
H/dHの支持のunionを入力座標順の8 atomで計算し、32/64 site内に収まる場合は
[BM16,panel_sites]のXを一度loadし、register gatherから各atomのH/dHを作る。
unionの外側はloadしない。周期siteへ戻すのはXのload時で、支持のunwrapped
low/highと各atomのBK8 accumulation順は維持する。空支持・無効atomをunionに含めない。
大きなunionは既存のatom別gatherへ戻す。新たなsort・全量Xコピー・global panelは作らない。
source-only/dX-onlyは現行経路を使う。live Parameterはbackwardから読まない。

まず独立FP64/floor/empty/singleton/周期境界/全axis支持/strides/端数/
retained snapshot/20capturedとpublic更新をGPU検証する。producer単体も既存のGと
parameter partial、canaryを比較する。資格確認失敗時は時間比較へ進まない。
全GPU suiteは1000秒以内。数値許容値4e-4を変更しない。

既存N2048/B32/A209715、初期rho3/8、seed41、IEEE FP32/TF32off、MSE、24更新を固定。
各rhoにprimaryと独立逆順の2回、計4cohortを実行する。各cohortはfresh processの
matrix-torch、factor、onchip-h、現行BM16/G16、共有32/64、denseの7workers、2700秒上限。
全6CSTのinitial/post24全shape/all atom FP64 Y/dX/all four dPを照合してから、
支持union/共有panelの準備も含む未計装の完全step21samplesとcapture/replayの
allocated/reserved peakを測る。phaseは別Graphで測定しprimaryへ足し引きしない。
遅い候補も全結果を保存し、結果に応じた反復・構成の追加をしない。

比較後、固定post24 snapshotの実routeを別Graphで計装して、input routing、hot fields、
G/H/dH＋parameter partial producer、dX集約、partial accumulation、source VJPを分ける。
保存済みprimary・未計装routeとY/dX/all dPの一致とParameter不変を確認する。
compiler register/spill/shared bytesと論理的X load要求量はhardware cache/DRAM trafficとは
区別する。L2/DRAM/stall counterを測らずにcache bottleneckを確定しない。

採否は現行に対する両独立runの完全stepと総peak、およびdenseの両peakで決める。
raw source/results/tensorsはignored outputとremote runに保存する。これは研究schemaで、
公開benchmark submission/DB schemaの対応を意味しない。

## 初回コンパイルの修正

source `7d77b724fc6fbf1ea750a9aeb1b1f330a4b9a6ae`のGPU資格確認は81PASS/56FAIL。
Tritonで条件分岐内だけに定義した`panel`を後続loopで参照し、SSA値が未定義として
コンパイルに失敗した。性能cohortは0で、数学gateや時間・メモリの結果には扱わない。
両分岐のSSA値を定義するゼロ初期値を追加する。global Xのloadは共有分岐だけで維持する。
失敗source/137testsのlogs/XMLをremote `cst-shared-x-ada-7d77b724-20261011`と
localのsource archive/operating notesに保持し、修正commitは別directoryで全資格確認を行う。
数値許容値・比較条件・候補構成は変更しない。


## 完了した完全step比較

修正measured source `231bc48feec4b6e170963dba3acd6f7d5185519d`。
RTX 6000 Ada Generation、Torch2.9.1+cu128 / CUDA12.8 / Triton3.5.1。
137qualification tests PASS、0skip/error/failure、21.97秒。
producer単体のGとparameter partialは現行版と全fixtureでbitwise一致し、canaryも不変。
4cohort、28fresh workers、588primary samples。全6CSTのinitial/post24に対する
48全shape/all atom FP64 Y/dX/all-four-dP gateがPASS。4e-4の許容値は変更していない。
共有panelのunion計算・load/gather、通常のsnapshot/準備・routing・hot fields・partial
accumulation・source VJP・AdamW/Polar更新を含む未計装完全stepの中央値、単位ms。

| rho / 独立run | 現行BM16/G16 | 共有X32 | 共有X64 | dense |
| --- | ---: | ---: | ---: | ---: |
| 3初回 | 1.420598 | 1.492036 | 1.493399 | 0.329214 |
| 3逆順 | 1.433887 | 1.503201 | 1.507761 | 0.329137 |
| 8初回 | 1.817912 | 1.939639 | 1.921551 | 0.329092 |
| 8逆順 | 1.833761 | 1.920622 | 1.936533 | 0.329604 |

共有X32はrho3で4.83–5.03%、rho8で4.74–6.70%遅い。
共有X64はrho3で5.12–5.15%、rho8で5.60–5.70%遅い。
どちらも両独立runで改善せず、都合のよいサイズ/runだけを選ばない。
capture/replay allocated/reservedは現行・共有X32・共有X64の全cohortで
**57.430664/104MiB**、denseは**81.752441/106MiB**。
共有panelはkernel内の値で、global Tensor容量も総peakも削減しなかった。
GPU process usageとhardware cache/DRAM trafficは別指標である。

両候補を採用せず、Gを共有する現行BM16/G16を維持する。
H/dHのX読み取り共有が有利とは仮定せず、実測された準備・gather・kernelの費用を
含めて負の結果として残す。公開Registry/defaultへ接続しない。

### 固定snapshotで増えた処理を確認

全6診断は保存済みprimaryのpost24と未計装routeのY/dX/all dPにbitwise一致し、
Parameterは不変。比較する3routeも全4cohortで互いのpost24 P/Y/dX/all dPがbitwise一致し、
各routeの独立逆順も自分自身とbitwise一致した。読み取り共有で数値を変えていない。

実routeの別Graphにexternal CUDA Eventsを入れ、5samplesの中央値を測った。
B32/G16の各phaseは2chunkの該当eventを合計した値で、全phaseを加算してprimaryを
置き換えたり、primaryから差し引いたりはしない。単位ms。

| rho / phase | 現行 | 共有X32 | 共有X64 |
| --- | ---: | ---: | ---: |
| 3 input routing | 0.053408 | 0.053504 | 0.053472 |
| 3 input hot fields | 0.014016 | 0.014144 | 0.013984 |
| 3 G/H/dH＋parameter partial producer | 0.280768 | 0.343488 | 0.347200 |
| 3 dX集約 | 0.200160 | 0.200256 | 0.199424 |
| 3 partial accumulation | 0.019904 | 0.019936 | 0.019872 |
| 3 source VJP | 0.010208 | 0.010176 | 0.010176 |
| 8 input routing | 0.053472 | 0.053344 | 0.053152 |
| 8 input hot fields | 0.013984 | 0.013952 | 0.014048 |
| 8 G/H/dH＋parameter partial producer | 0.423264 | 0.508064 | 0.517056 |
| 8 dX集約 | 0.300736 | 0.299744 | 0.299840 |
| 8 partial accumulation | 0.019872 | 0.019936 | 0.019904 |
| 8 source VJP | 0.010144 | 0.010112 | 0.010112 |

producerはrho3で22.3–23.7%、rho8で20.0–22.2%遅い。
input routing/hot fields、dX集約、partial/source VJPの時間はほぼ同じ。
Xを共有するunion/load/gatherの準備はproducer内なので、上表のproducerに含む。
G/dGの数式・要求回数は同じだが、同じkernelのregister配置・scheduleも変わるため、
増えた時間の全量をunion/gather単独へ帰属できる診断ではない。

実post24のprepared支持から数えた、batch rowあたりの論理的X load要求は以下。
26,215groupすべてが32/64 site内の共有分岐を使える。空group・fallbackは0。
これは支持区間の整数による要求数で、GPU命令数・L2 hit率・物理DRAM trafficではない。

| rho | 現行atom別要求 | 共有union要求 | 比率 |
| --- | ---: | ---: | ---: |
| 3 | 1,255,524 | 159,096 | 0.126717 |
| 8 | 3,334,006 | 421,214 | 0.126339 |

論理要求を約87.3%減らしても、完全stepは速くならなかった。
compiler metadataは両rhoで現行94register/0spill/512shared byteに対し、
共有X32/X64とも128register/0spill/4096shared byte。
新たな全量コピー/sort/global panelを作らない方式でも、thread間のgatherと局所保存の
配置・lifetimeを含めた費用がある。spillは今回0で、spillだけでは回帰を説明できない。
既存の重複loadがcacheや同一memory transactionで吸収されていた可能性は仮説であり、
hardware counter未測定のため確定しない。Xの論理的load量だけをbottleneckとする説明は
今回の結果と合わない。次の候補を決めるなら、profile/除算/縮約・register scheduleを
含むproducer内部を対象にする。新たな構成・hardware counter sweepは今回追加しない。

## 証拠と復旧

- remote `/home/jovyan/runs/cst-shared-x-ada-231bc48f-20261011/`。
  durable controller PID168877、6固定snapshot診断＋read-only auditのPID174067。
  controller/qualification/全4cohort/6診断/auditは完了。raw .ptはremoteに残す。
- local ignored `output/regular-grid-h/shared-x-results/`、
  `shared-x-fixed-source.tar.gz`、`shared-x-compact-results.tar.gz`、
  controller/validator/diagnostic/audit scripts、CPU/build logs、operating notesと
  download verificationを保持。初回コンパイル失敗版も別archive/logsを保持。
- source archive SHA256 `6bac19f24bfdccb0d5780f72f8f68742d8f405277e2449504422fbddc2dff99f`。
  compact archive `28809f317c787cb16079e3ab8830ff3cb682de171af969b82c9d0ae27703a542`。
  audit PASS_COMPLETE `486d0a0fd455109e23a4fef0163b3fad60213145a416a6ced14631320267e379`。
  download後116compact entriesのSHA、1651tracked source filesとpinned git commitのSHAを照合した。
  全48raw snapshotのSHA・shape・全atom/all-four-gradient coverageはremote auditで照合した。
- 全6CST/cohortを含む最大absolute Y/dX/all-dPは
  3.26435e-4 /2.35572e-4 /1.46414e-4、最大relative L2は
  8.25184e-7 /8.11237e-7 /6.92488e-7。4e-4許容値内。
- local CPU1753PASS/3964GPU-or-DB skip、18既存warnings、38Plan metadata、
  frozen research Cases、Ruff/diff、wheel/sdist PASS。修正runtime sourceのCPU/build CI PASS:
  https://github.com/takumiecd/torchcst/actions/runs/38122369396。
  宣言checkはcatalog、特殊research Caseは専用runnerのload_caseで確認した。
- 公開route/defaultへ採用しない。GPU computeは終了し、0% utilization/39MiBを確認した。
  ユーザーが開いたHubをreadyのまま残す。
