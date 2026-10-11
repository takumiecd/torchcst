# RegularGrid＋FlatTorusのinput-owner backward融合

2026-10-11。Stripへ配置を変えず、[過去のsmall backward融合](../local-product/20261007-backward-fusion.md)
の担当規則を現在のRegularGrid profile-productへ移す研究候補。
この実装checkpointではGPUの正しさ・速度・メモリは未検証。

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
