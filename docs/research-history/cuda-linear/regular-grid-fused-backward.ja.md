# RegularGrid＋FlatTorusのinput-owner backward融合

2026-10-11。Stripへ配置を変えず、[過去のsmall backward融合](../local-product/20261007-backward-fusion.md)
の担当規則を現在のRegularGrid profile-productへ移す研究候補。
数学・更新・担当の一意性は検証を通過したが、完全stepは遅くなり、総peakも
減らなかったため採用しない。現行BM16/G16を維持し、候補は研究対照として残す。

## 同じ部分と変える部分

`input-fused-backward-bm8-h32` / `input-fused-backward-bm16-h32`は、新しい
research Algorithm `research_cuda_regular_grid_fused_backward_h` v1の明示的候補。
grid_shape、spacing、origin、FlatTorus、profile-product、全chart L2と一回floor、
widthのtask VJP stop-gradientと既存AdamW/Polar更新を変えない。
forwardは現行`input-site-reciprocal-bm16-g16-h32`のRecipeへ明示的に戻し、
同じsnapshot、準備、H32生成、BM16出力ownerを使う。Hはbackwardへ保存しない。

backwardだけinput owner内でG/dGを作り、GをdXへ利用する。
input supportのprepared low端点を周期siteへ戻し、そのsiteを持つownerだけが
H/dHとG/dGからparameter partialを書く。low端点のprofileがゼロでも担当は省かない。
入力支持が空のatomはpartialを正確なゼロに保つ。周期routingの二つの区間は重複させない。
parameter partialはbatch tileごとに作り、元のsource VJPでcanonical Parameterへ戻す。
G用global Tensorとhot input fieldsは作らず、Y/dXへatomicを書かない。
param-owner以外でもdGを計算する第1版であり、その費用の削減は今回の比較に追加しない。
compilerのregister/shared/spillと実L2/DRAM転送量は別指標である。

partialはBM8で4×3×A、BM16で2×3×AのFP32値容量（B32）。
既存G16の4×16×A byteを省く一方、partialをstreamする既存方式より容量が増える。
B32/A209715ではpartialが約9.6/4.8MiB、G16は約12.8MiB。
forward H32約25.6MiBはそのまま。これらはTensor容量であり完全step peakではない。
source-only backwardは既存atom-owned VJPへ戻し、dX-onlyはparameter partialを作らない。
旧forwardのX/source/scalar/packed snapshotで再計算し、live Parameterを読み直さない。

## 固定する検証と比較

RTX 6000 Ada一台、既存N2048/B32/A209715、初期rho3/8、seed41、FP32 IEEE/TF32off、
MSE、AdamW+Polar、24回更新、既存caseを変更しない。公開Registry/defaultは変えない。

まず新候補とBM16/G16対照の131testsをGPUで検証する。
独立FP64 Y/dX/all dP、空・singleton・norm-floor・全axis支持・strides・batch端数、
retained snapshot、20captured/public更新を含む。9支持fixture×BM8/16は
実kernelのtest-only counterでparameter partialの担当回数を直接検査する。
productionではcounter引数Noneがconstexprで除去される。
GPUでskip/error/failureがあれば時間比較へ進まない。qualification上限1000秒。

通過後、rho3/8それぞれprimaryと独立逆順の計4cohortを実行する。
各cohortはfresh processのmatrix-torch、factor、onchip-h、BM16/G16対照、
融合BM8、融合BM16、denseの7workers、上限2700秒。
全6CSTのinitial/post24全shape全atom FP64 Y/dX/all dP gateを変えず、
準備・routing・partial初期化・source VJP・optimizer込みの未計装完全stepを21samples測る。
capture/replay allocated/reservedと同条件denseを比較する。phaseは別Graphで測り、
足し合わせてprimaryの時間へ置き換えない。失敗・未完了・遅い候補も保存する。
数値gateや比較条件を緩めず、反復や構成を結果に応じて追加しない。

採用判断ではdenseのallocatedとreservedを両方上回らないことを要求し、
matched baselineに対する時間・メモリの交換を両独立runで確認する。
kernel内のG再計算回数がowner数に比例して増えるため、融合だけで高速化するとは仮定しない。
raw source/results/tensorsはignored `output/regular-grid-h/`とGPU run directoryへ保存する。

## 完了した比較

measured source `eb48f8249335e625d72a08cc45b5341e345ca693`。
RTX 6000 Ada Generation、Torch2.9.1+cu128 / CUDA12.8 / Triton3.5.1。
131qualification tests PASS、0skip/error/failure、22.62秒。
4cohort・28fresh workers・588primary samplesと、全6CSTのinitial/post24に対する
48全shape/all-atom FP64 Y/dX/all-four-dP gateがPASS。許容値4e-4は変更していない。
全routeを含む最大absolute Y/dX/all-dPは3.2761e-4 /2.3559e-4 /1.4641e-4、
最大relative L2は8.2374e-7 /8.1086e-7 /6.9248e-7。
現行と融合2routeは、rho3/8とも独立逆順で自分自身のpost24 P/Y/dX/all-dPがbitwise一致。

未計装の完全step、単位ms。初回/独立逆順を別々に保存する。

| rho | 現行BM16/G16 | 融合BM8 | 融合BM16 | dense |
| --- | ---: | ---: | ---: | ---: |
| 3初回 | 1.420073 | 2.232005 | 1.781742 | 0.329850 |
| 3逆順 | 1.432075 | 2.273698 | 1.789081 | 0.329340 |
| 8初回 | 1.821556 | 3.583095 | 2.723330 | 0.329181 |
| 8逆順 | 1.835225 | 3.244595 | 2.653250 | 0.329224 |

融合BM16でもrho3は24.9–25.5%、rho8は44.6–49.5%遅い。
BM8のrho8には独立run間でばらつきがあるが、両方で現行より76.8–96.7%遅い。
都合のよいrunだけを選ばず、両方を負の結果として残す。

capture/replay allocated/reservedは現行・融合BM8・融合BM16の全4cohortで
**57.430664/104MiB**、denseは**81.752441/106MiB**。
G Tensorを省いたことは、完全step peakの削減にならなかった。
H32は変更しておらず、parameter partialの容量・寿命も変わる。
今回の観測だけでallocatorの特定blockがpeakを支配したと断定しない。

### backwardのどこが増えたか

別の更新Graphコピーで測ったbackwardの中央値、単位ms。
primaryの完全stepへ加算・減算しない。

| rho | 現行 | 融合BM8 | 融合BM16 |
| --- | ---: | ---: | ---: |
| 3 | 0.543–0.549 | 1.302–1.315 | 0.908–0.911 |
| 8 | 0.799–0.810 | 2.213–2.229 | 1.621–1.627 |

固定したprimary post24 snapshotで、実際の融合routeを別Graphで計装した。
4診断とも保存済みprimary Y/dX/dPおよび未計装routeとbitwise一致し、Parameterは不変。
小batch・支持・更新を変えた再チューニングやprimary再測定は行っていない。

| rho / BM | input routing ms | G＋dX＋parameter partial融合kernel ms | source VJP ms | registers / spills / shared bytes |
| --- | ---: | ---: | ---: | --- |
| 3 /8 | 0.05350 | 1.32678 | 0.01309 | 118 /0 /512 |
| 3 /16 | 0.05357 | 0.89389 | 0.01040 | 128 /2 /512 |
| 8 /8 | 0.05322 | 2.25392 | 0.01302 | 118 /0 /512 |
| 8 /16 | 0.05347 | 1.60403 | 0.01040 | 128 /2 /512 |

このkernel内で、inputの支持区間が触るownerごとにGを作り直す。
post24の連続する短い支持区間から数えた、atom×batch tileあたりのG要求回数は
rho3で平均**1.623112**（1owner79039atoms、2owners130676atoms）、
rho8で平均**2.862747**（2owners28784atoms、3owners180931atoms）。
現行producerは1回。この数字は支持区間からの論理的な要求回数であり、
GPU命令数・L2 hit率・物理DRAM転送量ではない。
parameter VJPは一意ownerだけが書くため、その重複を意味しない。

増えたG/dG縮約とowner内の処理が主要な次の診断対象になる。
通常のパラメータ順gather、profileの評価・除算、register圧力等も変わるため、
性能回帰の全量をGの回数だけへ帰属しない。
BM16にはcompiler spill2があるが、spill0のBM8もさらに遅いのでspillだけでは説明できない。
準備indexの構築やsource VJPだけを原因にする結果でもない。

現行のkernel間G共有を維持し、次は同じRegularGridの重なるX断片を共有して
H/dH生成の読み取りを減らす候補を優先する。Stripへの配置変更は必要ない。
Gの寿命を短くする設計でも、担当をまたぐ再計算を増やさない方法が必要になる。

## 証拠と復旧

- remote `/home/jovyan/runs/cst-fused-backward-ada-eb48f824-20261011/`。
  durable controller PID163209、4固定snapshot診断とread-only auditのPID168204。
  controller/qualification/各cohort/診断/auditは完了。raw .ptはremoteに残す。
- local ignored `output/regular-grid-h/fused-backward-results/`、
  `fused-backward-source.tar.gz`、`fused-backward-compact-results.tar.gz`、
  controller/validator/diagnostic/audit scripts、CPU/build logs、operating notesを保持。
- source archive SHA256 `0e7a0d0aff840239c7f0d0db9667e38ce0b5e4b7f16f22a14275f22a44fc6db4`。
  compact archive `bf841b45ce98fc8ce5fd24f1d1f17d29f6b44fd72b07a290e3552e14913ecb7d`。
  audit PASS_COMPLETE `6fd95e8f086b794a37cdb0b701cf39cfa7eec7d8c148a94fe3e1400da9856bac`。
  download後113compact entries、1647tracked source filesのSHAを照合した。
  全48raw snapshotのSHA・shape・all-gradient coverageはremote auditで照合した。
- local CPU1744PASS/3898GPU-or-DB skip、36Plan metadata、Ruff/diff、wheel/sdist PASS。
  source CPU/build CI PASS:
  https://github.com/takumiecd/torchcst/actions/runs/38120670832。
  宣言checkは既存catalogに実施し、特殊research Caseは専用runnerのload_caseで検査した。
  contributorの通常Case schemaへ特殊Caseを押し込んでいない。
- 公開route/defaultへ採用しない。GPU computeは終了し、ユーザーが開いたHubをreadyのまま維持する。
