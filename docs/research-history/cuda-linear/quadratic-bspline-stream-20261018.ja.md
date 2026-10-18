# B-spline: 継続配置とHの即時消費（2026-10-18）

## 契約と比較の目的

v4 sigma splitは、混在幅の完全stepを8.79から5.83msへ短縮したが、H/HP/G/GPを
それぞれ全atom数の固定容量で確保し、allocated/reserved peakが74.18/140MiBに増えた。
この記録は同じ数学のv5で、配置の再利用とHの保存寿命を変更する比較である。
公開dispatcherは変更しない。旧v1-v4のPlanと意味も維持する。

profileは二次B-spline、半径はR=sqrt(6)*sigma、Polarの幅stop-gradient、
operator全sitesの離散product L2とfloorを一度だけ適用する規則を維持する。
sigma/center/ampはatomごと・stepごとに現在値から準備する。
小支持のWと広い支持の累積momentをsigma cutoffで分け、各atomを一方にだけ所属させる。

\[
Q_{d,a}=\sum_{j\in\mathrm{sites}_d}f_{d,a}(j)^2,\quad
D_a=\max(\sqrt{Q_{o,a}Q_{i,a}},\mathrm{floor}),\quad
W_{ji}=\sum_a A_a f_{o,a}(j) f_{i,a}(i)/D_a.
\]

Qは既存の有限和momentで求める。幅の近似、部分区間ごとのnorm、軸別floorは導入しない。

## 継続するのは配置であり、古い数値ではない

各軸のspacingをh_d、matrix patch幅をPとして、配置用の幅bucketを

\[
p_d=\min(\lceil 2R/(P h_d)\rceil,\lceil N_d/P\rceil)
\]

とする。smallには両軸のbucketの組、broadには一つのbucketを使う。
両軸のp_dは同じRに対して単調なので、配置keyはp_o+p_iで表せる。
到達可能なpatch数の組が変われば和も増え、直積サイズのhistogramは不要である。
N=1024/P=4ではsmall key 0..512とbroad一つの計514binを使う。
このbucketは並びを決める実行上の情報であり、支持を丸めたりsigmaを量子化したりしない。
Wと勾配の計算時には現在の中心と幅から厳密なclipped支持を求め直す。
中心がsite境界をまたいだだけでは配置bucketを変更する必要がない。

毎forward/replayで全atomの現在のkeyをGPU上で検証する。全keyが前回と等しければ、
ID順序・bucket histogramを維持する。一つでも変わればGPU上のhistogram/scan/scatterで
全IDを再配置する。変更atomだけを移すincremental allocatorではない。
固定長histogramのscanと検証launchは、keyが不変でも残る。CPUへのcount読み戻しはない。
`reuse_layout=False`は同じ幅keyを毎回argsortする対照である。

persistent workspaceはAlgorithmStateが所有する。eagerとGraph captureのmode変更では
配置を保持する。shape/device/recipeが変われば作り直し、現在値のkey検査は常に行う。forwardごとにorder/countをsnapshotし、
後のforwardで配置が変わっても以前のbackwardは自身の配置を使う。
準備済みMetaをその順に物理packし、W・H側の計算では連続する実行rowを読む。
最後にatom勾配を元Parameterの行番号の順へ戻す。ここでのorderは実行用のrow indexであり、
AtomStateのstable ID対応、Parameter本体、Adam momentsの行順は移さない。
従ってAtomState.relayoutのCUDA Graph対応を追加したものではない。

## Hをregister内の小分けにする

入力のFP64 prefix momentをP_k(t)=sum_{j<t} X_j z_j^kとする。二次profileの三つの
多項式係数をc_0,c_1,c_2、四つのclipped開始/終了indexをt_0..t_3とし、

\[
d_0=c_0,\quad d_1=c_1-c_0,\quad d_2=c_2-c_1,\quad d_3=-c_2
\]

と置く。有限site列でのprofileは

\[
f(j)=\sum_{q=0}^3(d_q\cdot[1,z_j,z_j^2])\,1_{j\ge t_q}
\]

であり、sum_q d_q=0なので、

\[
H=\sum_j X_j f(j)=-\sum_{q=0}^3 d_q\cdot P(t_q)
\]

となる。中心微分も同じ四境界で係数を微分する。これは追加窓やprofileの変更ではない。
以前の三区間それぞれの左右端（六端点）の処理を、共有境界で結合した表現である。
空支持は明示的に寄与ゼロとし、大きい係数の数値相殺に依存させない。
境界位置の丸めとFP64の加算順序の違いは独立oracleで検証する。

Hを求めたCTAは直ちに出力のFP64差分配列に4境界×3係数を加える。
全atom分のH/HPをglobal Tensorとして保存しない。小分けはaxis_atoms×batch、
batchは最大64。scanで差分を復元し、small Wの出力へ加える。

backwardではforward時のMeta/Xから入力bankとH/HPを再計算し、adjointからG/GPを求める。
同じCTAで全batchを縮約してatom勾配を書き、Gを直ちにdXの差分配列へ反映する。
H/HP/G/GPの全atom容量はゼロ。register圧力とspillは別途compiler reportで確認する。
H/GをFP32へ丸めてからFP64演算へ戻す精度契約は旧経路と同じ。

v5の保存量はMeta=12A FP64、X、small W、order=A int64、count、origin/spacing。
v6はsave_w=FalseならWを保存せず、backwardのdX直前にforward MetaからWを再構築する。
dXのGEMMが終わったらそのWの領域にdW=dY^T Xを上書きし、small atom VJPへ使う。
保存Tensorを上書きしないので、retained backwardでも同じforward状態から再構築できる。
これは4MiBのW/dW併存を避ける代わりに、追加のW構築時間を払う方式である。
v1/v2/v3/v4/v5のserialized field数5/9/13/14/15は維持し、v6だけ16番目のsave_wを持つ。
cacheにはkey/order、dirty flag、histogram/end/cursorと診断counterがある。
prefix bank、差分、W/dW、勾配、optimizer、Graph pool等も実測peakに含める。
Tensor payloadの式だけをGPU peakやDRAM trafficと解釈しない。

## 検証・実測

CPU全suiteは最終sourceで1968 passed / 3041 skipped / 18 warnings、38.10秒。
初期sourceのGPU全suiteは240 passed（うちmetadata 1、CUDA 239）、234.71秒。
Graph境界の修正とv6追加後のstream専用検証は64 passed / 202 deselected、82.37秒。
最終sourceの全GPU suiteは266 passed（metadata 1、CUDA 265）、1 warning、181.01秒。
独立FP64 oracleのY/dX/全atom勾配、4e-4のmax absoluteとrelative L2両方の基準を維持した。
空/単一/広い支持、積normのfloor、異なる原点/spacing、batch1/3/32/64、
4/8/16 atom小分け、幅の強制切替、配置keyが不変の数値変化、
後のforwardで再配置した後のretained backward、個別gradient要求を検証した。
既存の24-update Graph対公開optimizer比較ではParameter/Adam moments/stepを確認している。

wheel/sdist buildは成功し、runtime package全336ファイルがcheckoutとbyte一致した。
変更pathのRuffと宣言/JSON往復検査はPASS。全repoの既存lint指摘の修正はこの変更に含めない。

## 初期比較と発見した問題

初期source `102c24f1`、job `31e39cddc3eb42d9b937ffdfbeec46de`では、
3条件とも全shape FP64 Y/dX/全atom勾配と提出前検査を通過した。
N=1024x1024、A=52428、B=32、spacing=.1、21 samples、AdamW+fused Polar更新である。
smallはsigma=.1/.14/.18、mixedは.1/.4/1.6/6.4/12、
switchは.1/.4998/.5002/.501/1.6をそれぞれatom間で混在させる。初期配置はshuffleする。

| 条件 | 従来の適切な最速 ms | stream sort8 ms | stream reuse8 ms |
| --- | ---: | ---: | ---: |
| mixed | split .5: 5.828418 | 4.353720 | 4.368305 |
| small | grouped W: 1.211136 | 1.496139 | 1.513650 |
| switch | split .5: 5.192113 | 5.264108 | 5.272774 |

streamのallocated/reserved peakは52.438/118MiB、splitは74.181/140MiB、
small grouped Wは50.231/114MiB、Denseは49.002/106MiBだった。
4/8/16 atom小分けのmixedは4.417370/4.368305/4.347538msで、
この範囲の差だけから特定の小分けサイズを普遍的な最適値としない。

ただし初期版はeager→captureでAlgorithmStateのprepareが配置を捨て、
capture内にcache初期化を記録していた。Graph replayでも毎回key=-1とcounter=0へ戻るため、
診断counterはcalls=1/rebuilds=1で、**配置の継続利用は実現できていなかった**。
この初期結果はHの即時消費の効果を示すが、配置再利用の効果を示す結果ではない。
修正版ではmode変更でも配置を維持し、実Moduleのeager→capture→replay→eagerで
counterが継続し、key不変時に再構築されないことを追加テストした。

初期のmixed stream8の別計装Graphではprepare=.162、group=.065、Meta pack=.041、
small W=.279、small VJP=.585、stream forward=1.042、stream backward=1.973ms。
小さいsigmaだけではsmall W=.327、small VJP=.680に加えて不要なbank/scanが残り、
純grouped Wに届かなかった。switchではsmall W=1.024、small VJP=2.735msが支配する。
4境界の書き込み・読み出しやW側の支持面積に費用が残る。物理DRAM/cache/atomic counterは未測定。

## 最終比較

source `c6028d57`で配置再利用を修正し、直積histogramを単調keyへ縮め、
v6のW再構築を追加した。mixed/switchではcutoff .2と.5も比較する。
速度は未計装の完全stepで判断し、stageを合算したりnorm_replayを主測定から差し引いたりしない。
同じ条件の最終比較（各候補21 sample、各sampleは独立runではない）：

| 条件 | 従来最速 ms | stream sort8/.5 | stream reuse8/.5 | memory8/.5 | reuse8/.2 | memory8/.2 | Dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed | split: 5.821945 | 4.346689 | **4.313107** | **4.591001** | 4.636089 | 4.687329 | .082270 |
| small | grouped W: **1.209370** | 1.491939 | 1.457982 | 1.785345 | 未測定 | 未測定 | .081903 |
| switch | split: 5.190348 | 5.255552 | 5.221677 | 6.248893 | **4.631335** | **4.694472** | .081685 |

mixedのreuse8/.5は従来split比25.92%短縮、memory8/.5は21.14%短縮。
switchは.5のままでは速度の改善がなく、.2へ変えるとreuse/memoryで10.77%/9.55%短縮する。
smallだけなら純grouped Wが依然として最速で、H経路を併用する利益はない。
新候補を全caseの無条件な最速版とはしない。

### メモリ

全caseで同じshapeと固定容量のため、peakは以下のとおりだった。

| 方式 | allocated MiB | reserved MiB |
| --- | ---: | ---: |
| 従来split | 74.181 | 140 |
| 従来small grouped W | 50.231 | 114 |
| stream sort8 | 51.279 | 116 |
| stream reuse8（W保存） | 51.688 | 116 |
| stream memory8（W再構築） | **47.688** | **116** |
| Dense | **49.002** | **106** |

memory8は従来split比でallocatedを35.71%、reservedを17.14%削減した。
W保存版からのallocated差4MiBはW/dW併存を避ける効果に対応する。
allocatedはDenseを下回るが、reservedは10MiB上回るため、両peakをDense以下とする
採用目標は満たしていない。総GPU process usage、物理DRAM traffic、cache hit率は未測定。
確保量と予約量を混同しない。

### 配置の再利用と残る費用

mixed/smallの.5、mixed/switchの.2は52 call中1 rebuild、51 reuse（98.08%）。
switch/.5は52 call中3 rebuild、49 reuse（94.23%）だった。
このcounterの範囲はwarmup/eager/replayを含む主測定全体であり、timed replayだけではない。
switch/.5では20971 atomのrouteが移動した（同じ全測定windowのinitial/final比較）。
.2ではそれらを最初からHへ置くのでroute移動はなく、全52428 sigmaの数値は更新される。
3条件・全候補でinitial P/X/target hashが一致し、全atomの幅更新を確認した。
これは幅の固定bankや量子化による高速化ではない。

別計装Graphのmixed/.5では、分類/配置確認は約.020ms、毎step sort版は約.055ms。
完全stepの差は4.346689→4.313107ms（約0.77%）で、配置再利用単独の効果は小さい。
主要な改善は全H/G保存をやめ、query/注入/VJPの間の転送とlaunchをまとめたことにある。
Meta pack=.041、prepare（厳密L2込み）=.162、small W=.279、small VJP=.584、
stream forward=1.043、stream backward=1.972msで、W再構築版は追加約.279msを払う。
H/Gのglobal Tensorはゼロで、8 atom×32 batchの小分けをregister内で処理する。
選択したstream kernelのcompiler reportはspill_slots=0、local load/storeなし。
register数はforward56/backward128。これは実測のDRAM/cache counterではない。

switch/.5のsmall W/VJPは1.024/2.735ms。cutoffを.2へ下げると.056/.110msになり、
代わりにstream forward/backwardが1.394/2.636msへ増える。
閾値は支持面積とbatch再利用の兼ね合いで選ぶ必要があり、.2や.5を普遍的な最適値とはしない。

### 誤差と測定範囲

全shape・全atomの独立FP64 Y/dX/DPを三つのlive stateで照合した。
最終比較の新候補の最大絶対誤差はY<=2.40e-6、dX<=2.70e-6、DP<=1.89e-5で、
max absolute/relative L2とも4e-4の既存gateを変更していない。
unitでは空支持、floor、極小寄与等の別条件も同じgateで確認する。
FP32 source/X/H丸め/W/DPとFP64 geometry/profile/L2/reduction、TF32無効を維持する。

主時間は未計装CUDA Graphのforward/loss/backward/AdamW/Polar更新を含む完全step。
毎stepのkey検証・必要時の再配置、Meta準備/pack、W構築、L2、bank/scan、
H再計算、W再構築も含む。初期コンパイル/capture時間はstep中央値に含めず、
peakはcapture/replay込み。配置不変のwarm状態では初回rebuildを毎step課さない。
幅が大きく往復してkeyが変わるたびに全IDの再配置は必要になる。
Denseは同じshape/input/target/精度/optimizerの通常の学習行列で、CSTの再パラメータ化ではない。

### 独立runでの再確認

別のL4 VM、job `34ed6de9bb9f491ebffe028c212d31d3`で選択した高速/省メモリ候補を再測定した。
sourceは同じ`c6028d57`。21 sample、同じfixture/初期P/X/target/optimizer/精度/stepである。
repeatのCaseは従来最速をbaselineにし、Torchの時間測定を省いた候補subsetで、
各候補の独立FP64照合は三つのlive stateで再実行する。

| 条件 | 従来最速 ms | stream高速 ms | stream省メモリ ms | Dense ms |
| --- | ---: | ---: | ---: | ---: |
| mixed（cutoff .5） | 5.829197 | 4.315576 | 4.596603 | .082311 |
| small（cutoff .5） | **1.207894** | 1.459402 | 1.789326 | .081784 |
| switch（cutoff .2） | 5.190531 | 4.630579 | 4.686614 | .082054 |

ピークは初回と同一で、各stream候補の配置counterも52 call / 1 rebuild / 51 reuseだった。
全候補で52428 atomの幅更新を確認し、initial P/X/targetのhashは候補間・job間で一致した。
新候補の二つの最終jobを合わせた最大絶対/relative L2誤差は、
Y=2.3982e-6/1.7089e-7、dX=2.6910e-6/1.7278e-7、DP=1.8841e-5/1.0639e-7。
この結果は選択した候補について独立run数2であり、21 sampleを21 runとは数えない。

mixedは同じkernelのまま速度とメモリをともに改善できたが、Dense時間の約52倍は残る。
smallでは純grouped Wが適切であり、switchはcutoff .2がこの比較では有効だった。
reservedも含めたDense以下という採用目標は未達で、公開dispatcherへの採用は行っていない。

CUDAの検証範囲はEuclidean RegularGridの入力/出力それぞれ1座標、batch 1..64。
Wを使う経路は各軸1024 siteまでで、性能の比較は上記1024x1024/B32のみ。
8192x8192完全step、周期/多座標CUDA、複数stream同時実行、実DBへの登録、公開採用は未実施。

## 証拠と再現

runtime source checkpointは`c6028d5704e3f8a9b5b3b53bad50b5d1add6dd05`。
NVIDIA L4、driver580.82.07、Python3.13.15、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。
最終比較GPU UUIDは03354ec6-e32f-7e54-8448-0ad45d4e610a、repeatは
0188989c-9127-42dc-4168-8f7060c93316。全owned pool runtimeは停止確認済み。

全archiveのSHA256、manifest内全ファイルを照合し、初回最終比較76ファイルとrepeat45ファイルを確認した。
最終smoke/全比較/repeatのruntime package全336ファイルとtests/runner/manifest/fixtureは
checkoutとbyte一致する。wheel/sdistも同じ336ファイルを含む。
raw source/result archive、receipt/spec、driver、log、width tensor、完成runner JSONは
ignored `output/bspline-stream-20261018/verified-jobs/`に保全した。
JSONの`benchmarks.submissions check`は全比較PASS。

初回smoke `d40ebe7d...`はmatrix端数groupのmask不足で失敗し、修正後再検証した。
その結果を速度改善や最終精度の証拠には使わない。初期prototype/失敗も含めて保全している。

| job（l4job-以下） | 用途 | source archive SHA256 | result archive SHA256 | files/exit |
| --- | --- | --- | --- | --- |
| `d40ebe7d898d40dca41003fcc242d334` | initial smoke failure | `1f0d02710acafc04c624e4f1a9ec0d1c6f2635d97cdce08df2c62fcdc28507cc` | `88eddbebbd6e3d9c759116f13ba49458cc54d4f71141657d0af3884368658404` | 4/1 |
| `e144bf6b8d0e4fe0956e0cdc3f7ee8e8` | corrected smoke | `f0e4464523c39ef84c99da1d61bac8fbb8f61f7425900524a4c1b01eedf57bcc` | `5d7935bfb606794f3a4554708428e9f139d640e9599df3f405d77d923635bfff` | 4/0 |
| `d56fd029e3c1424796ac67a9ea714ecf` | prototype qualification | `6b1b3bf14051bbced7eb0f2963933d1dee13ae7be85eb747be9b6300213075fb` | `c62f9dad55048f95159e8132b698b01cd75a520c1db1b1f89468b5f433f533e9` | 4/0 |
| `31e39cddc3eb42d9b937ffdfbeec46de` | prototype comparison | `2ec9c3f98a2fcefb0dbd3c4f2d18bae12a7e8dd52922aac6057cff998e63f90a` | `9b812eef438bd1e2dd96b943ca8b1dfbff0b08d4d1556e795c81feaf794510de` | 72/0 |
| `b453521f45d74e59a1bd397ff6507337` | persistence/memory smoke | `6cf56e10c93d44bf00960ba698fec01a9a155d556563b9686209abe9bed7bb8c` | `86c9c493baad7e30a9ba656ec9e922737bdd030d787c59e7bf12e8ad316eada7` | 4/0 |
| `78f4e24c5af94ac89b9736c32f6dacb4` | final qualification+comparison | `fd3237927697d47bd0879b7abfbf24707d0fde72d0765362feef662d3fdca6c8` | `1a8152d0c50d42e10252c0b5d36c6e816833d64ebc93d2d45ee5c3454c6fdd9f` | 76/0 |
| `34ed6de9bb9f491ebffe028c212d31d3` | independent repeat | `c64e17edf108f5ace02704658565d241e729e72c94f47957856bbead6c0ac095` | `851f7c53482cfde358f636771b57ce61a4fb20a633ab067ec5ec20f91f4189ff` | 45/0 |

再現（GPUの割当は共有pool skillに従う）：

```bash
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-stream.json
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-stream.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --source-commit c6028d5704e3f8a9b5b3b53bad50b5d1add6dd05 \
  --output output/bspline-stream-reproduce.json
python -m benchmarks.submissions check output/bspline-stream-reproduce.json
```

同じcatalogで`small-stream.json`/`switch-stream.json`のCaseも実行する。
歴史上の完全なsource再現には上記のfrozen archiveかcheckpointを使い、今後のHEADと混同しない。
