# Cell-mass CUDAの実測状態

2026-10-27。Issue #120 / Draft PR #121。
目的は、面積正規化の契約・Y/dX/全atom勾配を保ち、dense以下のメモリで
CSTの1step速度をdenseの約2倍、つまり時間を約半分にすること。
過去の「時間をdenseの2倍以内」は中間目安であり、最終目標ではない。
固定bank版128では中間目安の「時間2倍以内」とメモリ条件を満たした。
固定bank版1024はメモリ条件を満たすが、時間は未達。階層版は下記で区別する。

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
dense比約1.67の時間で、**最終目標のdenseの2倍の速度＝時間半分は未達**である。
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
公開CSTLinear/dispatcher、二階微分、全体Graph学習は未接続。
以下は別の幅分布・周期を使う固定bank版の記録である。

## 固定bank版：係数・微分のHorner評価とsigma分布

runtime source `7c1404c1`。数学・精度・sigma分布を変えず、係数とその微分を
Horner法で一緒に評価し、harmonic間でscalarの累乗を再利用する。
1024の完全stepは同じjobで **1.740 / 1.732ms → 1.354 / 1.354ms**、約22%短縮。
peak allocated **4.261MiB** / reserved **6MiB**で増加なし。
通常dense0.446 / 0.449msとの時間比は3.04 / 3.02で、約2倍目標は未達。
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
同じjobの通常denseは0.450 / 0.441msで、時間比3.79 / 3.88倍。約2倍目標は未達。
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
1024は4.11/4.20倍で約2倍の目標には未達。allocator割当はdenseの約31%、reservedは約36%。
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
