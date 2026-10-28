# Cell-mass CUDAの実測状態

2026-10-27。Issue #120 / merged PR #121。公開接続はDraft PR #122で進める。
目的は、面積正規化の契約・Y/dX/全atom勾配を保ち、dense以下のメモリで
forward/backward/AdamWを含む完全stepの所要時間をdenseの2倍以内にすること。
2026-10-27にユーザーへ確認し、「時間半分が必須」という以前の解釈を訂正した。
PR #121のB4合成条件では、階層版が128・1024とも時間とメモリの条件を達成した。
denseと同等の時間は今後の任意の改善目標であり、今回の完了条件に追加しない。
以下の固定bank版の過去結果と、現在の階層版を区別する。

既存Polar＋今回のセル積分/面積正規化は`presets.polar_cell_mass`で接続した。
既存Polarの更新・幅stop-gradientを保ち、CUDA内で振幅と各軸のsigmaをdecodeする。
物理atom表を複製せず、入出力の別幅にも対応する。CPU全2,047 passed/5,362 skipped。
GPU全勾配/更新とMNIST3epochはjob `l4job-3b154a81ff6845b485d9a80332bb182e`で完了した。
ただし先行設定minimum=2/maximum=40はユーザー指定のσ_min/spacing<=1を満たさなかった。
2026-10-28の最新指定でσ_min=spacing、σ_max=min(100σ_min,Q/3)、birth=8σ_min、upper_floor=2σ_minとする。
128はσ_min=1/σ_max≈42.666664、1024は0.125/12.5。非表現可能な上限はFP32で下向きに丸める。
最小tileも128/1024とも4/8sitesとし、明示SITE_SCALED Planで疎sortと逆indexを使う。
訂正後source `636c890c`、job `l4job-31211e59eb5c41b08ce8813e60ab7141`で完了した。
GPU33+2,647 passed、racecheck0 hazards、全819/52,428atomのY/dX/4列勾配と3step更新を通過。
MNIST3epoch/B32/seed120で精度は128がCST97.02/dense97.11%、1024が97.36/97.84%。
平均完全stepは12.022/1.912ms（6.286倍）、20.935/1.926ms（10.867倍）。
allocated MiBは213.301/213.264、242.150/239.883、reservedは366/364、396/382。
**この実学習条件では速度・メモリ目標は未達。** 両peakは常駐データ込みのTorch allocator値。
σ/spacing中央値は4.793→1.494、7.080→1.508。1024の最小入力bankは85.82%に増えた。
radial正則化だけの解析予測にも近く、データ学習だけで獲得した偏りとは呼ばない。
raw/source/checkpointsと491source照合は`output/cell-mass-polar-unit-spacing-20261028/verified-run/`。
CPU2,050 passed/5,384 skipped、Ruff/format/build、346runtimeの両配布物byte照合とPR CPU CIも通過。
所有runtimeとsupervisorは停止済み。詳細・過去条件との区別は正本のσ範囲訂正節に残した。
別幅のbank省略と保存FP32 H/Gによる勾配誤差を先行検査で検出し、
Polar専用のFP64 register pullbackへ修正した。Hをbackwardまで保存せず、幅微分も計算しない。

2026-10-28、固定layout検証をbuffer identity/version/dtype/deviceでcacheする改善を実装した。
設定変更時は再検証し、失敗した検証は保存しない。atom値・σ分類はcacheしない。
旧/新のPolar更新則・CUDA計算は同じで、同一GPU/同一checkpointの完全step比較では
初期21.296/21.255→17.580/17.543ms、epoch3 22.431/21.758→17.885/17.845ms。
計測順を逆にした2 fresh processで17.45〜20.26%短縮。optimizer同期は204→20/step。
設定変更を含む全model Parameter/moments/stepはbit一致、初期/epoch3全atom参照も通過。
メモリpeakは変わらず、速度・allocated目標はなお未達。新たなepoch学習は行っていない。
既存のTriton Polar fusionはEuclidean用で、今回のFlatTorus中心wrapには未対応。
source f56af001、job `l4job-1a9b42fc2b96442c8a5a5915e64bcc29`、
rawと494source照合は`output/cell-mass-optimizer-cache-20261028/verified-run/`。
CPU2,062 passed/5,385 skipped、GPU scoped66 passed。所有GPUは停止済み。

1024の初期/epoch3 checkpointを使う段階別診断も完了（job `l4job-5432773eb82647e18a402c9868116fa8`）。
計測なし完全stepは20.686/21.084ms。同期診断のepoch3はprepare1.134、Zx/R2.151、HC3.191、
Y展開.393、Zy/R1.143、G+勾配1.888、C_in1.581、dX展開.442、optimizer8.277ms。
同期診断の和は完全step時間ではない。optimizerは204同期/step、実GPU作業約.654msで、
固定宣言snapshotが繰り返すscalar読み出しが優先調査対象となった。
メモリpeakはC_in生成中。Rx/Ry/G/勾配/C_in/部分和が同時に存在し、Hは既に解放されている。
平常時はCSTがdenseより8.787MiB少ないが、一時増分が約10.010MiB大きく、peakが逆転する。
実測配列・確保blockを4case全境界/peakでbyte照合した。rawと修正後解析は
`output/cell-mass-mnist-profile-20261028/`。詳細・図・再現条件は正本の段階別監査節に置く。
次は固定metadataのhost往復、G→C_inの寿命、owner分布からのgrain選択を検討する。未実装。

以下は先行した物理sigma直接学習であり、Polarの学習結果とは区別する。

公開CSTLinear接続では`presets.cell_mass`と明示Planを追加した。自動dispatchは変更しない。
これは物理`[amplitude,sigma,output_center,input_center]`の経路であり、Polar接続ではない。
以下のMNIST結果はsigma直接学習の確認で、Polarの幅制御やsigma分布の評価には読み替えない。
この先行実験の時点ではPolar接続は未実装だった。現在のPolar版の検証は上で追跡する。
CPU suiteは2,043 passed/5,344 skipped、Ruff/format/buildは通過し、配布物内のPython/native
全runtime 344ファイルを作業sourceへ照合した。
job `l4job-186f9896e1094c1a8989758e90710752`でGPU suite 2,621 passed/12 skipped、
全819/52,428atomの独立Y/dX/4列勾配・3 step CSTOptimizer更新を通過した。

全MNIST 3 epoch/B32/seed120、中央正方層だけの置換で、epoch3精度は
128でCST 96.75% / dense 97.01%、1024で96.70 / 97.59%。
平均学習stepは5.142/1.953ms、8.353/1.947msで、時間比2.633/4.290。
peak allocatedは213.633/213.264MiB、242.322/239.883MiB、
reservedは366/364MiB、392/382MiB（CST/dense、常駐データ込み）。
**この実学習条件では速度・メモリ目標は未達**で、以下のB4合成step達成から分けて判定する。
1024は計時直前allocatedがdenseより約12MiB少ない一方、計時中の一時領域で逆転する。

1024のsigma変化量中央値は0.03507、階層人数はほぼ維持され、3 epochで強い偏りは未確認。
ただし初期から全周owner一つに21.18%のatomが集中し、仕事量の偏りはある。
詳細な初期化・各epoch・bank移動・メモリ寿命は正本の公開接続節に残した。
保存と完全stepの調査を先に進め、2048/4096/8192はまだ拡張しない。
raw/source/checkpointsは`output/cell-mass-linear-integration-20261027/verified-run/`、
source/tests/training 488ファイルを照合済み。所有L4とsupervisorは停止済み。

階層版`cell_mass_hierarchical/`は既存ChebyshevSquare・sigmaの意味を維持する。
Zは2子ずつの階層、Rは同じ階層の3タイルを合算したH/G用の値である。
分類はsigma<=T_h、周期P_h=3T_h。以前のI=sigma/2は撤回した。
入力dtypeのH/Gを保存し、Z/R/C・縮約・位相・正規化はFP64を維持する。
任意でM8の多項式の値とbeta微分だけをFP32の2数表現で評価する。
atom値のコピー、全atom係数表、密W/dWは作らない。

runtime `e4370477` / snapshot `2054143d`で、有効なC部分和だけを保存する任意経路を検証した。
ownerのatom数からactive splitを求め、device prefixで詰めた`[task,batch,mode]`へ書く。
atomの担当範囲と加算順を維持し、空ownerの部分和は作らない。

各設定2 fresh process、2巡目は測定順を逆転した完全stepの再確認は次の通り。

| 条件 | 完全step（2 process） | allocated CST / dense | reserved CST / dense |
| --- | ---: | ---: | ---: |
| 128、既存compact・逆indexなし・Double値/勾配 | 0.872569 / 0.878817ms | 257 / 264.5KiB | 2 / 2MiB |
| matched dense128 | 0.690231 / 0.713909ms | — | — |
| 1024、S128/grain64・H/C融合、通常部分和 | 1.228739 / 1.242684ms | 9.602 / 16.063MiB | 12 / 22MiB |
| 同上、ragged部分和 | 1.160396 / 1.164707ms | 6.007 / 16.063MiB | 8 / 22MiB |
| matched dense1024 | 0.693952 / 0.698604ms | — | — |

検査/prepare/forward/全backward/fused AdamWを含む完全layer step。
B4/FP32/M8/tail1e-3/Q128、A=floor(.05*N²)、初期sigma linspace(.2,40)、seed120、
物理入力T_min4・出力8。TF32/Graph/workspaceなし、warm5/同期wall21。
全6 memory gateを同repeatのdenseと比較し、step.jsonから独立に再計算した。
1024は同設定の従来経路より約5.6/6.3%短縮、allocatedは10,067,968→6,299,136B、約37.4%減。
dense比約1.67の時間で、**完全stepがdenseの2倍以内という目標を達成した**。
128もdense比約1.26/1.23であり、両サイズで時間・allocated・reservedの条件を満たす。
上表はTorch allocator peakで、GPU process全体の使用量ではない。

全819/52,428atomを3step更新し、各stepの更新済み参照Parameterから独立CPU FFT/autogradで
Y/dX/全4列勾配を計算し直した。FP32 rtol3e-6/atol2e-7、Parameter rtol2e-6/atol2e-7、
AdamW moments/step rtol4e-6/atol2e-8を維持して全stepを通過した。
1024の最大勾配絶対差3.1573e-8、最大Parameter差1.5259e-5は許容差内でありbit一致ではない。
1024の全3stepはragged=True・HC=Trueだけの検証である。
fixture atoms/X/dY/初期Wは初回jobとbit一致した。

初回GPU suite2613 passed/12 skipped、選択racecheck4件・hazard0、
全819/52,428atomの独立Y/dX/4列勾配の全9設定、全9memory gateを確認した。
確認jobではsuiteを繰り返さず、全atom optimizer・完全step・両peakを確認した。
全CPU2037 passed/5342 skipped/既存警告18件、Ruff/format/buildと両配布物338runtime byte照合は通過。
GPU検証の代用にはしない。PR head a2fbe377のCPU Actionsも通過した。

128は既存設定を維持する。1024はcompact逆index・補償多項式値/勾配・shared shape再利用、
S128/grain64・H/C融合に`ragged_c_splits=True`を加える。
FP32/M8/tail1e-3・正のgrain・逆indexに限定し、Cinは分離する。
公開dispatcherと研究APIの既定値は変更しない。詳細なkwargsは正本に置く。
H/C融合だけの普遍的な優位は主張しない。旧H/C融合・G/勾配/Cin融合・Cdotの負の結果も残す。

数式・配列寿命・選択kwargs・全履歴・hashは[設計の正本](cell-mass-kernel-flow.ja.md)。
初回job `l4job-6004f86d063f4a009803a5509c49be59`、確認job `l4job-8ac49377a31246e3888e4574f5a9dcfe`。
runtime338・検査141・manifest各50/43・driver/binary・fixture bit一致を照合し、
source/results/spec/receiptをignored `output/cell-mass-opt-20261027/ragged-active-c/`と
`ragged-confirmation/`へ保全した。所有GPUの停止とsupervisor終了を確認した。
接続503で中断した追加試行03dfde6eは結果から除外し、pool recover後の確認jobだけを採用した。
PR #121の時点では公開CSTLinear未接続。現在は上のPR #122で明示接続を進める。
二階微分・全体Graph学習・自動dispatchは引き続き未対応。
追加の位相評価試作は`kernel/cell-mass-phase-prototype`（a3fe9da9）へ保存し、
現在の採用ブランチからrevertした。検証済みDouble位相とragged部分和を最終設定として残す。
追加試験も回収・監査済み。G/勾配単体は短縮せず完全step差も約0.5%の未再確認値だったため、
採用せず結果を正本へ残した。この最適化campaignの全所有GPUは停止して区切った。
公開接続・MNISTの新しい実験は上のPR #122で別に追跡する。
以下は別の幅分布・周期を使う固定bank版の記録である。

## 固定bank版：係数・微分のHorner評価とsigma分布

runtime source `7c1404c1`。数学・精度・sigma分布を変えず、係数とその微分を
Horner法で一緒に評価し、harmonic間でscalarの累乗を再利用する。
1024の完全stepは同じjobで **1.740 / 1.732ms → 1.354 / 1.354ms**、約22%短縮。
peak allocated **4.261MiB** / reserved **6MiB**で増加なし。
通常dense0.446 / 0.449msとの時間比は3.04 / 3.02で、時間2倍以内の目標は未達。
workspaceなしdense16.063MiB / reserved22MiB以下を維持する。

下の1024差分に `coefficient_method="polynomial_horner"` を加える。
既定値・公開dispatcher・128の採用設定は変更しない。
全atomの係数表を保存せず、追加するmodule定数表は3,552bytesのみ
（元の表と合わせ13,416bytes、Torch allocatorピーク外）。

初期sigmaは全atom同じではなく、**0.2〜12の52,428個の等間隔値**。
実際の学習モデル由来の分布ではない。現profileには集中半径の下限があり、
入力半径は7.59〜12.32、出力半径は15.17〜16.42である。
sigmaの範囲だけから小さな局所profileも多いと判断できない。
現経路は同じM8/tile/K3の1bankで、sigma別階層分類・小sigma直接計算は未実装。

GPU・数学1816 passed、CPU2006 passed / 4564 skipped / 既存警告18件。
全52,428atomのY/dX/全4列勾配・AdamW更新/moments/stepも独立oracleへ照合した。
最新の融合producerは約0.55msで引き続き主対象。
[係数の数式・sigma分布と半径・完全step・hash](cell-mass-horner-and-sigma.ja.md)に詳細を残す。

## 前段：1024のC_in融合と展開幅の再調整

runtime source `660cb0ea`（実装`907132b3`、差分は単一siteのテストfixture修正のみ）。
C生成とsite展開を分けて測定し、G/Parameter勾配とC_in生成を融合した。
Gのglobal表をなくし、入力係数をH/dHとC_inで再利用する。

別jobでの2fresh process確認は、CST **1.706 / 1.710ms**、
peak allocated **4.261MiB** / reserved **6MiB**。
同じjobの通常denseは0.450 / 0.441msで、時間比3.79 / 3.88倍。時間2倍以内の目標は未達。
workspaceなしdense16.063MiB / reserved22MiB以下のメモリを維持する。
従来推奨からの短縮は8〜9%、thread/site幅を揃えた融合単独の効果は約4%。

1024/B4では、下の従来settings1024に次の差分を加える。
数学のtile_sitesや周期は変えない。

```python
settings1024.update(
    sites_per_block=32, c_threads_per_block=32, fuse_backward_cin=True,
)
```

GPU・数学1468 passed。全52,428atomのY/dX/全4列勾配・AdamW更新/moments/stepも照合済み。
forwardは従来通り。backwardはZy構築後、G/Parameter勾配/C_in部分和を一つのkernelで作り、
次kernelがC_inを合計・展開してdXを書く。Gは現在のshared chunk内だけに保持する。
M8の融合sharedは6,176bytes/block。C_inの一時global部分和544KiBは残す。
CPU2006 passed / 4216 skipped / 既存警告18件、Ruff・format・build・321runtime source照合も通過。

[分離した実測・融合コードの順序・負の結果・再確認とhash](cell-mass-cin-fusion.ja.md)を正本とする。
研究入口のflagは既定False、融合はbatch<=4のみ。公開dispatcher/Graphには未接続。
128の推奨設定は従来の微分保存型を維持し、この回で128の完全stepを再測定したとは扱わない。

## 比較している条件

N×N、atom数floor(0.05N²)、batch4、FP32、ChebyshevSquare M8、tail_bound1e-3。
物理chart周期Q128、sigma初期0.2..12、seed120。
128ではspacing1 / 入力tile16 / 出力tile32、1024ではspacing0.125 / 入力tile128 / 出力tile256。
物理tile幅16/32とprofile周期48/96を保っている。別のsigma分布やbatchへ一般化しない。

固定X/dYのforward/backward/fused AdamWを同期wallで測る。
入力値検査・stable sort・begin/end生成・中間値の作成と解放を含む。
JIT・CPU oracle・初期dense Wの構築は測定外。lossや全networkのstepではない。
TF32無効、Graphなし。AdamW lr1e-4 / betas0.9,0.999 / eps1e-8 / weight_decay0.01。
CSTは4列のphysical atom、denseはWを学習するので、更新後の学習軌跡が同じとは主張しない。

時間は通常dense、メモリはworkspaceなしdenseという、別の比較点を使う。
workspaceなしは`CUBLAS_WORKSPACE_CONFIG=:0:0` / `CUBLASLT_WORKSPACE_SIZE=0`。
workspace削減をモデル圧縮と混同しない。全GPU process使用量は未測定。

## 融合前の比較（source c9b67dd4）

runtime source `c9b67dd4`、NVIDIA L4 / Torch2.11.0+cu130 / CUDA13.0。
各設定2fresh process、2巡目は逆順、warmup10回後30個の同期wall標本。
以下はrun0 / run1の中央値。時間の通常denseとメモリのworkspaceなしdenseを分けて示す。

| caseと方式 | step ms run0 / run1 | peak allocated | peak reserved |
|---|---:|---:|---:|
| 128・CST微分保存/recurrence | 0.733 / 0.730 | 200KiB | 2MiB |
| 128・dense通常 | 0.433 / 0.558 | 16.508MiB | 22MiB |
| 128・dense workspaceなし | 0.640 / 0.649 | 264.5KiB | 2MiB |
| 1024・CST再計算/owner窓/owner C/G融合 | 1.883 / 1.863 | 5.003MiB | 8MiB |
| 1024・dense通常 | 0.459 / 0.443 | 32.313MiB | 42MiB |
| 1024・dense workspaceなし | 0.630 / 0.622 | 16.063MiB | 22MiB |

128の時間比は1.69/1.31倍。denseのrun間変動も保存し、有利なrunだけ選ばない。
1024は4.11/4.20倍で時間2倍以内の目標には未達。allocator割当はdenseの約31%、reservedは約36%。
このメモリ条件を満たすことと、速度目標に到達することは別に判断する。
1024の最初の再計算方式9.64..9.65msからは大幅に短くなったが、denseより速いという結果ではない。

選んだ研究入口の設定は次の通り。自動dispatchにはまだ組み込まない。

```python
# N128 / A819 / B4、spacing1、input tile16、output tile32
settings128 = dict(
    z_method="merge", coefficient_method="polynomial_recurrence",
    sites_per_block=16, atom_splits=8, atom_threads_per_block=16,
    cache_projection_derivatives=True, double_intermediates=False,
)

# N1024 / A52428 / B4、spacing0.125、input tile128、output tile256
settings1024 = dict(
    z_method="merge", coefficient_method="polynomial_recurrence",
    sites_per_block=256, atom_splits=128, atom_threads_per_block=32,
    cache_projection_derivatives=False, double_intermediates=True,
    owner_window_moments=True, owner_coefficient_reduction=True,
    fuse_g_parameter_grad=True, c_threads_per_block=128,
)
```

1024の別event診断は、prepare0.166、Zx/Zy各0.073、H単独0.172、
G/Parameter勾配融合0.673、owner Y0.344、owner dX0.294ms。
主な残存コストは両軸の投影/係数計算とC集約。
phaseの和を完全step時間の代用にしない。
registerを減らした小batch専用化は改善せず外した。
[C thread数とprepareの切り分け](cell-mass-layout-tuning.ja.md)に負の結果も残す。

## C_in融合前の配列と寿命

1024/A52428/B4/M8、勾配再計算・owner窓・owner C・G融合経路。

| 配列 | 場所・大きさ | 寿命と用途 |
|---|---|---|
| 元atoms | global、838,848bytes | 学習するParameter。ソートしたコピーは作らない |
| atom勾配 | global、838,848bytes | 元atom順に書き、optimizerが消費 |
| AdamW moments 2本 | global、各838,848bytes | optimizerの学習状態 |
| sorted_rows | global、419,424bytes | forward/backwardで元atom行を参照 |
| key・error word | 一時global、キーは32bit | prepare内。値検査のhost確認1回を含む |
| owner begin/end | global、33個のint64 | 出力4×入力8のgroup範囲、forward/backward共用 |
| 入力Z窓 / 出力Z窓 | global FP64、4,352 / 2,176bytes | 入力窓はbackwardまで、出力窓はbackward内 |
| 全Z階層 | 一時global、各138,720bytes | 窓集約が完了したら不要。窓生成中は共存 |
| forward H | 一時global FP32、838,848bytes | Y作成後解放。backwardではdoubleで再投影 |
| backward G | 一時global FP32、838,848bytes | Parameter勾配と同時生成し、dXが消費 |
| owner C部分和 | 一時global FP64 | split128ならY272KiB / dX544KiB。各呼び出し内のみ |
| profile/投影/勾配部分和 | thread register | 各threadが別atomを担当 |
| 32atomの係数・H/G chunk | block shared | C生成中だけ共有。全atomの係数表は作らない |
| 最終tileのC | block shared | splitと隣接ownerを合計し、一度だけsiteへ展開 |

Cのglobal部分和はbounded scratchであり、永続したmodel状態でも全site×atom表でもない。
以上の配列は全てが同時に生きているわけではない。実測ピークを単純な容量和で代用しない。
mode数のみのCUDA constant table9864bytesはTorch allocatorピーク外なので別に数える。
atomが行列要素数の5%でも、1atomは4個のfloatなのでParameter数はdenseの約20%。

## C_in融合前の処理順

```cpp
// forward: 同じstream上の依存順。各kernel内は要素ごとに並列。
rows, offsets = validate_and_sort_original_atom_rows();
Zx = build_hierarchy_and_sum_owner_windows(X);
H = project_atoms(Zx);                    // 1thread = 1atom
Y = reduce_owner_C_and_expand(H);         // 32atom shared chunk → 小C scratch → Y
release(H);                              // 再計算型の経路

// backward
Zy = build_hierarchy_and_sum_owner_windows(dY);
G, atom_grad = project_both_axes_and_pull_back(Zx, Zy);
dX = reduce_owner_C_and_expand(G);
```

kernel境界を同じstreamの順序でつなぎ、block間の自作spin待ちや出力atomicを使わない。
128で採用した微分保存型は、H/Gと4本の投影微分を保存し、Parameter勾配の再投影を省く。
保存と再計算は切り替え可能。batch16では微分保存がdenseより大きくなったため、
[メモリと時間のfrontier](cell-mass-memory-frontier.ja.md)で別扱いにしている。

## 数学と実装の入口

- [最新Horner評価とsigma分布・集中半径の注意](cell-mass-horner-and-sigma.ja.md)
- [面積正規化・whole-tile窓・近似誤差](cell-mass-tile-window.ja.md)
- [全段階の擬似コード](cell-mass-kernel-flow.ja.md)
- [ownerごとのC・範囲・chart端の扱い](cell-mass-owner-coefficients.ja.md)
- [GとParameter勾配の融合・H解放](cell-mass-fused-pullback.ja.md)
- [mode定数とsigma依存係数](cell-mass-polynomial-constants.ja.md)
- [共有Z窓とsin/cos漸化式](cell-mass-owner-windows.ja.md)
- [1024拡張と微小勾配の丸め・除外した実行](cell-mass-1024.ja.md)

実コードは`src/torchcst/_backends/cuda/algorithms/linear/cell_mass/`。
`executor.py`がprepare/forward/backwardと保存寿命を決め、`bindings.cpp`が入口を公開し、
`kernels.cu`がZ・投影・C・展開・Parameter勾配を実行する。
物理atomの並びは自由で、勾配は元の行番号へ戻す。

## 融合前sourceの検証と再現

CPU2006 passed / 4041 skipped / 既存JIT警告18件（43.08秒）。
CUDA・数学1293 passed（40.56秒）、failed/skippedなし。
32/64/128thread C、FP32/64、direct/merge、DFT/各polynomial、M2..16、
周期端・合計1・端数/空batch/atom・retained backward・元Parameterのmutation検査を含む。

全52,428atomは独立CPU float64 oracleを256atom単位で構築し、全4列の勾配を照合。
CSTの最大絶対差はY/dXとも1.49e-8、atom勾配2.91e-11。
同一GPU fused AdamWのParameter更新delta/stepの最大差は0、
exp_avg3.64e-12、exp_avg_sq2.78e-17。許容差は緩めていない。
128の同source確認もY/dX/全atom勾配とoptimizer検査を通過。

Ruff・clang-format・差分検査、wheel/sdist buildを通過。
両packageとGPU snapshotと現runtimeの321source filesをbyte照合した。

job `l4job-144c58b74b754deea6efc6f89b95110b`:

- source SHA256 `5c89ce1a8ba82d062be9c9f8b57761e639219a3670a6cc7fa2ba08526ffbae81`
- result SHA256 `4fc278e0e59657927efb68a5879732651a27092e50f7bdf41a179954ec5e9fdf`
- driver SHA256 `4fb18f73e476c694159bf9f0ee429079db28e9abe8c58768068ca3d9d32779b1`
- binary SHA256 `f439c3bf15a6e6d3996651a1e77880d4600685d08530de3cc139e77eb1a61631`

132manifest files、driver、128helper、binaryを照合。
rawはignored `output/cell-mass-1024-20261027/confirmation/`。
過去の失敗・未回収・比較候補のarchiveは同じ親directoryに保存し、削除していない。
共有L4 poolから結果を回収し、所有runtimeの停止を確認した。

ローカルCPU再検証は`python -m tools.kernel_dev test --suite cpu`。
GPU環境では`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。
性能を再現する場合は、固定source archiveの`__pool_driver__.py`へsrc/rootのPYTHONPATHと、
空directoryの`CST_JOB_OUTPUT`を渡す。共有L4ではpoolのsubmit/serveを使う。
archive/spec/receipt/manifest/verification.jsonも一緒に保持する。

## 残っている範囲

現在の実行は1軸1bank、ChebyshevSquare、1024以下の2のべき乗site数、65536以下のatom。
公開CSTLinear / Parameterization / optimizer / dispatcherとは未統合。
多軸・複数bank・階層別sigma分類・小sigmaの直接処理は今後の設計範囲。
CUDA Graphはerror wordのhost確認が残るため未対応。二階微分も未対応。
現在の時間値をGraphや公開経路の完全学習stepへ置き換えてはいけない。
