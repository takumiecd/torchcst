# B-splineの同一入力event group：構築のscreen

完全step <=2倍Dense、capture/replayのallocatedとreservedの両方<=Denseという目標を維持する。
これは入力groupの索引を作る実験で、grouped H/G executorの完成・性能・採用の記録ではない。
profile、exact sampled product L2の単一floor、task-width stop-gradient、全source勾配と4e-4 gateは変えない。

## 同じprefix点を読むatom

入力側のdyadic sizeと4 knotが同じなら、4 knotと最大2 boundaryの全prefix点が共通になる。
入力側の係数と振幅・正規化はatomごとに異なるまま保持する。
prefixのplaneをp、eventをe、batchをb、atomをaとして、同一group gのatomのHは

\[
H_{ab}=\sum_{e,p}s_e c_{aep}B_{j_{ge},p,b},
\qquad s_e=-1\ (e<4),\quad s_e=1\ (e\ge4)
\]

で、共有できるのはBの読み取り。center微分は同じBと既存の係数微分から計算する。
backwardではatomごとのGとsource VJPを保ち、入力eventへ注入する値を

\[
D_{gep b}=\sum_{a\in g}w_{ab}\widetilde c_{aep},
\qquad w_{ab}=G_{ab}\,\mathrm{amp}_a/D_a
\]

と先に足せる。tildeは既存のboundary座標shiftを含む。和の順序は変わるため、
独立FP64 gateを維持する。これはまだ統合していない設計式であり、GPU性能の証拠ではない。

CPUの初期mixed入力groupは13888、平均3.78 atom。38673 atomが4個以上のgroupに入り、
15057 atomは8個以上のgroupに入った。syntheticなPolar activity乗算後の平均は約3.2〜4.3。
16行単位へpaddingすると仕事が増えるので、そのまま大きいGEMMへ置換する根拠にはしない。
CPU原本はignored `group-density.py/json`。実optimizerをGPUで走らせた再利用率とは区別する。

## Torchのsort構築は重い

job `l4job-2bd451acdf364ec39380b4d577d4e484`、source
`a3c7670cb28dbb6e7ea22402098da3da7fadbc52`。N1024/A52428の入力size/4 knotから
完全な整数keyを作り、argsort、group boundary、offset/count/chunk表を構築した。
group countとboundaryはCPUと厳密一致。L4/Torch2.11.0+cu130/CUDA13。

| 初期分布 | key ms | sort ms | table ms | 全構築 ms |
|---|---:|---:|---:|---:|
| mixed |0.017408|0.083968|0.044032|0.139264|
| small |0.016384|0.082944|0.047104|0.139264|
| switch |0.016384|0.083936|0.047104|0.140288|

全構築の増分allocated peakは各2.433MiB。これは索引構築のみのピークで、完全stepやDense比較ではない。
keyの元となる整数intervalは計時外に用意した。独立phaseを足してfused構築時間と同一視しない。
現在のH fused forward約0.132msを上回り、読み取り共有をただ追加するだけでは回収しにくい。
raw archive・source・manifest・return0・cleanup ackを検証し、primary性能結果とは別の
`verified-diagnostics.json`へ保存した。

source SHA256 `cc1897b1acc4105f3aae05a6183b0c8737c6c83e1e19f2d421d42f97e0b28527`、
result SHA256 `626116c6d56b126e71e02b974a377d602f9284ac9c8b89b9e551f79d6bd0d746`。

## ハッシュ索引候補

入力4 knotのglobal local-prefix位置を各16bit以内で正確にpackし、64bit CASの辞書へ入れる。
hash衝突時は元の64bit keyを比較して線形probeし、異なるkeyを同一groupへ入れない。
辞書容量は2A以上の2冪、Count未満のnative slotは除外する。sortは使わず、countのscanと
offset/IDsの圧縮表（CSR型）および小さいquery chunkの対応表を作る。
新しいhelperはprivateな索引primitiveだけで、現在のruntime dispatcherからは使わない。
Parameterとoptimizer stateを並べ替えず、各forwardに独立した所属表を返す。

0/一部/全native、0 atom、同一key、多数の異なるkeyと衝突、16bit pack上限、Graphの動的Countと
位置、retained tableの独立性をGPUで確認し、同じTorch構築との時間・メモリをscreenする。
このprimitiveだけのGPU検証を、H/Gや全atom勾配の照合・完全stepの速度改善とは扱わない。


## ハッシュ構築のGPU screen

source `5fc5ccff2620f4a09cd130415fa467d2aade2626`、job
`l4job-1dcb77c9731943f1b287fa7dd37378a0`。26 checks PASS、7.61秒。
0/一部/全native、空atom、同一key、4096-siteの16bit pack、Graphの動的Count/位置、
retained所属表を確認した。実際のmixed/small/switchについてもCPU位置と全要素一致、
group数・count・全atom IDの被覆が一致した。L4/Torch2.11.0+cu130/CUDA13/Triton3.6.0。

| 初期分布 | Torch全構築 ms | ハッシュ全構築 ms | 構築の増分allocated MiB（Torch / hash） |
|---|---:|---:|---|
| mixed |0.139264|0.034816|2.433 /4.103|
| small |0.139264|0.034816|2.433 /4.103|
| switch |0.140288|0.034816|2.433 /4.103|

辞書初期化、membership、count scan、ID配置、chunkのscan/mapまでを含む。
元の位置は計時外。約4倍速いが構築時の増分memoryは増えた。これは索引primitiveのみの
screenであり、完全stepの速度・メモリやY/dX/全atom勾配の照合結果ではない。
large primaryでgroupの全memberと全6点が一致する検査を追加し、独立再測定してから
H/G実行方式へ統合する。dispatcherへの採用はしない。

source SHA256 `c32112dc64434e17797ce296b787c7576b2626e62d2eb5e0d1d1fb1211649a09`、
result SHA256 `071e11eaf5c31b3746791a96ce92137a79f476c618601d094fd4724879aa4182`。
source byte一致1579 files、manifest8 files、return0/cleanup ackを確認。
CPUは1995 PASS /4269 skipped /既存warning18、38.46秒、Ruffとisolated sdist/wheelの
363 runtime-file byte一致を確認。CPU skipをGPU成功と数えない。

初回job `l4job-7c63b79cf747473c9b7b7b7080d51507` は26単体検査を通った後、
測定driverの位置変換検査で失敗。次の `l4job-ad99c9c7ab6d48198fa7ea8fb76c5bf2` で、
smallのsize8だけdriverがGPU log2を整数へ切り捨て、prefixのlevelを一つ下へ置くことが判明した。
GPU log2の丸めを整数変換前に行い、CPU位置との全要素照合を追加した。runtimeのhashコードと
精度基準は変更していない。7c63はsource/archive/manifest/cleanup ackを検証してfailed-jobsへ保存。
ad99は結果download後のremote cleanupがWebSocket切断で失敗した。source/archive/manifestは照合したが
cleanup ackがなく、incomplete-jobsへ区別して保存した。owned VMの停止はpoolが確認済み。
両方とも性能の根拠から除外する。


## 独立再測定と全memberの照合

source `bb59beb8d7d3076436afbec537bf5fcea54386df`、job
`l4job-cb3638e528864e88a6f2f46f519aba48`。27 checks PASS、8.66秒。
A52428の実際のprimary形状についても、各groupの全memberと全6 prefix点を照合した。
測定driverでもmixed/small/switchの全memberが同じ4 knotに属すること、group重複がないこと、
全52428 IDを一度ずつ被覆することをCPUと厳密に照合した。hash runtimeは5fc5ccffと同じ。

| 初期分布 | Torch全構築 ms | hash全構築 ms | 4-atom chunk数 |
|---|---:|---:|---:|
| mixed |0.139264|0.034816|19644|
| small |0.140288|0.034816|17773|
| switch |0.141312|0.034816|17881|

初回と同じ約0.035ms帯を再現した。両runの増分allocated peakはTorch2.433 /hash4.103MiB。
hashの返却表payloadは1,992,288 bytes、約1.900MiB（count/ends/IDs/chunk ends/map）。
構築時の辞書と一時bufferはpayloadとは別。完全stepのpeakをこれらの足し算で推定しない。
H/Gへは未統合なので、現時点の完全step最速はv32 bundleの約0.796 /1.299msのまま。

source SHA256 `74fcf67319b31a47ec0ced9e3fa96b0f5aff0e561b23ead073d2f5abd99cee42`、
result SHA256 `00703e56710043a7913544e0571aa8510d3e3f303a931b539d925fec684e1225`。
source byte一致1579 files、manifest8 files、return0/cleanup ackを確認。
次は共有input H queryと、同じinput eventへ戻すdXのgroup内reductionを同じframeで試す。
個別G/Gmuと全source VJP、native guard、exact L2、完全stepと両peakのgateは維持する。


## v34候補：共有H queryとgroup内dX reduction

新しいresearch Algorithm v34はv32 bundle4/mid1を継承し、forwardの係数とexact sampled L2を
そのまま使う。hash索引は各forwardで一度だけ構築し、ends/IDs/chunk ends/mapの4 Tensorを
save_for_backwardで保持する。countはqueryに不要なので構築後に解放する。A52428では
保持容量1,468,000 bytes（約1.400MiB）。構築時の一時確保とは区別して実測peakを確認する。

1 CTAは同じ入力6点の最大axis_atoms個のslotを担当し、共通のBankを読み、各slotの係数でHを
評価する。backwardのG/Gmuは従来のatom別query、source VJPも同じ関数を使う。dXの注入は
既存と同じFP32 weight×coefficientを計算してgroup内で足し、各point/plane/batchへ一度だけ加える。
boundaryのFMAによる座標shift、end node skip、origin/boundaryのshard planeも保つ。
source/optimizer行は移動せず、共有するのはprefixの読み取りとdXへの同じeventの加算だけ。

chunk数はgroupごとの端数でceil(A/axis_atoms)を超えるため、各CTAはgrid strideでchunkを
担当する。少数CTAへ固定するv33とは異なり、起動gridはceil(A/axis_atoms)を維持する。
全atomは一度ずつ所有し、全batchを同じCTAでreduceするのでsource DPのatomicは追加しない。
B64などは従来どおりbatchを小分けにし、同じforward snapshotとgroup表を使い、DPは
既存のserialized ADD_DPで足す。native prefixはgroupから除外する。

共有Hだけ、dX reductionだけ、両方の3候補を、v32 controlとDenseで比較する。
浮動小数点和の順序は変わるので、独立FP64 Y/dX/全4 source勾配gateを緩和しない。
同じ入力点でもσ/係数が異なる257 atom、B17/33、動的Count全/一部/ゼロ、retained、
個別勾配と共通のtiny/floor/dyadic条件を検査する。新方式は未検証のresearch候補で、defaultは変更しない。
CPU1996 PASS /4312 skipped /既存warning18、38.53秒、Ruff/diff checkを確認。GPU成功とは数えない。


v34初回job `l4job-29da1475c0b5476e8ec12fc4e8bec119`、source
`f27805fa892a87a4b62b9f60017dc93f9d23fe0e` は71 checks PASS /1494 deselected、422.40秒。
B32大規模caseの全5 plan・3状態の独立FP64 Y/dX/全source勾配gateも通った。
その後、従来v32 controlのphase診断で未登録`input_group_prepare`へのstampが起き、
KeyErrorで測定が止まった。groupを使わないrecipeでは早期returnし、stampを発行しないよう修正する。
GPU計算kernelは変更しない。性能比較は完成していないので、partial測定を採用の根拠から除外する。
source SHA256 `e7cb12cc603b00b441cbb7e3e978a30f22e93e7985e33df64ce17235f11ca4a8`、
result SHA256 `13eb28244ffabf3e639ff798887973b5f159ab858a47dc3f3b37bd9d86cded39`。
source byte一致1584 files、manifest16 files、return1/cleanup ackを検証しfailed-jobsに保持。
既存の71検査を緩めず、同じkernelの完全stepと両peakを修正後sourceで再測定する。


## v35候補：FP64座標積の共通化

v34とは独立に、v32 bundle4をcontrolとする準備処理の候補を追加する。原点0の軸で、
局所tile中心indexをk、spacingをh、中心をmu、半径をRとして

\[
 z=(kh-\mu)/R=(-\mu/R)+k(h/R)
\]

を使う。h/Rと−mu/Rを共通項にし、FP64のまま計算した後、既存のFP32 coreへ同じ段階で変換する。
係数式、支持区間のFP64判定、moment、source pullbackは変えない。原点が0でない軸は従来の
計算順へ戻す。L2でもpieceの中央indexを使う同じ共通化を試し、正の離散moment式と
FP64 tiny/floor guard、product全体の単一floorを保持する。新しい窓・積分近似は入れない。

CPUサンプルのN64/1024/8192・各52428 atomでは、tileのFP32係数への入力は従来と一致した。
L2の3 pieceのnormalized inputsもN64/1024では一致し、N8192の9 pieceでzの差が
最大3.726e-9だった（wは一致）。有限サンプルであり、全条件のbit一致とは主張しない。
原本はignored `geometry-screen.py/json` と `norm-coordinate-screen.py/json`。

係数側だけ、L2側だけ、両方の3候補を完全stepで比較する。GPUでは共通の動的σ/Xと全勾配gateに加え、
原点非0のfallbackでMeta正規化値と全係数・整数eventがbit一致することを確認する。
CPU1997 PASS /4356 skipped /既存warning18、38.79秒を確認済み。v34の診断stamp修正を取り込んだ
状態でもCPU確認し、GPUへ提出する。defaultは変更しない。


## v34完全step結果：共有queryとgroup reductionは遅い

診断stamp修正後source `e9a85ad6547e6ed49a26d850aa4d47d7d9b2e9e0`、job
`l4job-a16c0570f73545978d0e6cb8bd4a6f29`。計算kernelは初回f27805faと同じ。
初回71 checksの範囲を保持し、再runで両primaryの全plan・3状態の独立FP64 full-shape gateと
全52428 σ更新、同一初期X/P/targetを通過してから完全stepを測った。

| plan | B32 ms | B64 ms | B32 allocated/reserved MiB | B64 allocated/reserved MiB |
|---|---:|---:|---|---|
| v32 control |0.785760|1.275836|34.153 /98|34.653 /100|
| 共有H+dX reduction |1.114874|2.028886|35.554 /102|36.054 /104|
| 共有Hのみ |0.977199|1.634890|35.554 /102|36.054 /104|
| dX reductionのみ |1.115971|2.006767|35.554 /102|36.054 /104|
| Dense |0.082101|0.105121|49.002 /106|49.502 /106|

全candidateは両peakがDense以下だが、時間は悪化。索引構築約0.0338ms、BT32 forwardは
control約0.130msに対し共有群約0.174ms。backwardはcontrol約0.231ms、共有Hのみ約0.349ms、
dX reduction込み約0.550ms。compiler reportはcontrol register40/shared16bytesに対し、
共有Hのみ56/16、dX reduction込み72/512（B64）で、spill/local load/storeはない。
メモリ読み取りを論理的に共有しatomicを減らしても、group内の端数、間接参照、reductionの
実行コストを回収できなかった。hardware counterで原因比率を切り分けた結果とは扱わない。
source/勾配精度を維持した負の結果として残し、defaultへ採用しない。2倍目標は未達。

source SHA256 `add899130198de6bd4c59fcc33438ca5b97d024ea6d2ed04b1193b851c290b57`、
result SHA256 `390328c318bb2ede9ca0ec699e443ea5ec887a210d26882dfd8dad92bfcd9884`。
source byte一致1584 files、manifest42 files、return0/cleanup ackを確認。

## v35完全step結果：座標共通化の差は小さい

source `8875a7a979cbb95c2e4c7128f33818075a409021`、job
`l4job-39ae5ab841924919b8ab837b948e6be2`。46 checks PASS /1537 deselected、306.11秒。
原点非0でのMeta正規化値・係数・整数eventの一致、動的σ/X、個別/全勾配、retained、Graphを確認。
両primaryの全plan・3状態の独立FP64 full-shape gateも通った。

| plan | B32 ms | B64 ms |
|---|---:|---:|
| v32 control |0.789481|1.286653|
| 係数+L2の座標共通化 |0.788440|1.279037|
| 係数側のみ |0.788681|1.283025|
| L2側のみ |0.789366|1.283044|
| Dense |0.082565|0.105272|

両peakは全candidateでcontrolと同じ34.153 /98、34.653 /100MiB、Dense以下。
係数phaseは約1µs短縮する一方、norm_initは約0.073msのまま。B32は約0.1%、B64は約0.6%の差で、
この1runだけを恒常的な速度改善と解釈しない。同じ速度帯として保持し、繰り返しの微調整を優先しない。
source SHA256 `ff11bee8c048d7d7de389910981a23aaeb5e4c3d7fe43228705cbc84bb1ca4f5`、
result SHA256 `bffb33d4e74ca41e1a45fa6eed9749f1e9f21a736f3228bbf03232373bf537ff`。
source byte一致1588 files、manifest42 files、return0/cleanup ackを確認。
isolated sdist/wheelの364 runtime filesと同sourceのbyte一致も確認。

次は既存recipeのprepare_warps=1/2/4だけを比較する。64 atomを1 warpで処理する現在の
decode/normはregister80/180（各spill0）で、複数warpに仕事を分ける影響を調べる。
式・保存量・gridのatom単位は変更せず、Meta全体の一致と完全step・両peakで判定する。


## prepare_warps比較：並列度だけでは改善しない

source `b652a12b2a1784bb8d3a11e86ca17997c9510b91`、job
`l4job-f322ac4835634976b86608579f19322a`。5 checks PASS、Meta全体のwarp1/2/4 bit一致と
両primary全planの独立FP64 full-shape 3状態gateを確認。

| prepare warp | B32 ms | B64 ms |
|---|---:|---:|
|1 control|0.790743|1.286343|
|2|0.794908|1.290294|
|4|0.917073|1.424483|
|Dense|0.082198|0.107521|

全CSTのallocated/reservedは34.153/98、34.653/100MiBで同じ。Dense49.002/106、49.502/106MiB以下。
warp2ではdecodeは約2µs減るがnorm_initが約3µs増え、完全stepは速くならない。warp4は64 atom CTAに
対し仕事のないlaneが増え、decode/norm_initとも遅くなる。register減少のみでは速度を推測できない。
なお`norm_replay`は診断時だけの独立測定で、未計装完全stepの実行項目ではない。phaseを単純合算しない。

source SHA256 `cb67cf52cae7d3c755aa76a25f17c74b5d8de84e47e62bb4dccfb07697f4b20a`、result SHA256 `379004cc60ea30c43b374b96563664bb35394ab3f195bf578dbe987f68b52bcb`。
source byte一致1592 files、manifest30 files、return0/cleanup ack確認。

## v36：atom laneを内側に置くH/G tile（検証前）

v32 SoAを基準に、query/injectの論理tileを[atom,batch]から[batch,atom]へ転置する。
通常slot順のまま処理し、group hash・並べ替え・paddingは追加しない。係数・prefix bank・H保存形式、
正規化・source pullback・optimizerは同じ。各eventのFMAと加算順は変えず、batch方向の3つの
勾配内積だけが転置後のaxis0 reductionになる。4/8 atom per CTAを、既存4/8 atom layoutと比較する。
係数のatom方向の読み取りをまとめ、bankのbatch方向の連続読み取りとの兼ね合いを測る。
新しいtensorはないが、register/shared、実測両peakと完全stepを確認する。
動的X/σ、dyadic境界、floor/tiny、個別勾配、retained、Count all/partial/zero Graphを既存の
独立FP64基準で確認してから性能を評価する。目標は引き続きDense2倍以内かつ両peak Dense以下。

実際のlane割り当てはTritonの決定にも依存するため、論理axisの転置だけから物理的な
coalescing改善とは主張しない。GPU jobで対象forward/backwardのTTGIR/PTXも保存して調べる。
提出前CPU1998 PASS /4389 skipped /既存warning18、38.62秒。Ruff・両case宣言検査PASS。


## v36結果：論理転置だけでは物理配置は変わらない

source `904cf8a9b4a8c891f75441962ac8ba5a639f6bdf`、job
`l4job-405c7ba7047849a594067964cb7d5c6c`。31 checks PASS /1586 deselected、256.30秒。
両primaryの全plan・3状態の独立FP64 full-shape Y/dX/all4P gate、全52428 σ更新と同一初期条件を確認。

| plan | B32 ms | B64 ms |
|---|---:|---:|
| v32 atom4 control |0.799208|1.296216|
| v32 atom8 |0.907247|1.531436|
| v36 論理転置atom4 |0.797977|1.300156|
| v36 論理転置atom8 |0.908609|1.529766|
| Dense |0.082785|0.103024|

両peakは全CST同じ34.153/98、34.653/100MiBで、Dense49.002/106、49.502/106MiB以下。
atom4のforward約0.134ms/backward約0.234msは同じ帯、atom8はforward約0.22msで遅い。
register40、sharedはforward0/backward16(atom4)・32(atom8)、spill/local load/storeなし。

jobのTTGIR/PTXを52組保存し、primaryの12ファイルを抽出すると、controlと転置candidateの
対応する6組すべてでPTX命令列が一致した（debug/location等の非命令行を除外、register/address operandsは保持）。
atom4 controlは[atom,batch]のthreadsPerWarp=[1,32], warpsPerCTA=[4,1]、
転置後は[batch,atom]の[32,1],[1,4]で、物理的に同じbatch連続配置になっている。
したがってこの結果は、論理axisを変えるだけではcompilerが同じ配置を選ぶことを示す。
実際にatom laneを内側に固定した別実装が遅いと結論する根拠にはしない。
比較には追加tensorもなく、既存の数式・勾配契約は通過した。速度改善として採用しない。
原本はignored `compiled-tiles/`、抽出表は`transposed-primary-asm.json`と`transposed-asm-summary.json`。

source SHA256 `fc1c6a834561f50e10b5c4621b7a6f940517e136c6a803949cb291fb3d8e3c3d`、result SHA256 `f36513ded91a8af7d982087426f43ea1d81b4013ec9eee2dd4091ee8842561b9`。
source byte一致1595 files、manifest192 files、return0/cleanup ack確認。

提出sourceのCPU1998 PASS /4389 skipped /既存warning18、38.62秒、Ruff、case宣言、CPU CI PASS。
isolated wheel/sdistは365 runtime filesが同sourceとbyte一致。目標は依然未達。
次は論理転置の繰り返しを避け、明示的なblocked lane layout（例：Gluon）の可否を調べる。
実API・実際のTTGIR/命令と独立oracleを確認し、同じ完全stepと両peakで評価する。
小さいσのprofileを既存native Co領域へ保存する案も別候補で、現時点で未実装・未測定。


## v37：Gluonで物理的なlane配置を指定（検証前）

論理転置をcompilerが元の配置へ戻したv36を受け、Gluon版のquery/inject/source pullbackを用意した。
APIは[Triton v3.6.0のBlockedLayout/SliceLayout](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/experimental/gluon/language/_layouts.py)と
[arange/full](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/experimental/gluon/language/_core.py)を確認した。
Gluonと通常Tritonのtensor helperを混用せず、元のFMA・cast・event加算・source pullbackの式を
Gluon演算へ移している。通常のprefix bank、係数、Meta正規化、native tiny guardとFP64 destination、
Parameter/optimizer orderは変えない。追加tensorはない。

[atom,batch] tileのthreadsPerWarpを[1,32]・[2,16]・[4,8]、warpsPerCTAを
[min(AB/lane_atoms,4),4/min(AB/lane_atoms,4)]に固定し、4 atom/CTA・4 warpを基準に比較する。
1 atom/warp版は旧配置とのlibrary差分のcontrol、2/4 atom版で物理的な読み取り割り当てを変える。
係数はSoAだけ、axis_warps=4だけに限定してrecipeで厳密に拒否する。

新Algorithm revision37はopt-in研究routeで、Gluon moduleは実行時だけimportする。
公開/default selectorは変更しない。動的X/σ・floor/tiny・dyadic境界・個別勾配・retained・
Count all/partial/zeroのGraphを独立FP64基準で確認する。主条件はB32/B64 mixed、N1024/A52428、
全52428 σ更新・構築/L2/optimizer込みの未計装完全step・両peakで判断し、TTGIR/PTXも保存する。
実測前にcoalescingや速度改善とは主張しない。

提出前CPU1999 PASS /4429 skipped /既存warning18、38.81秒。Ruff・両case宣言検査PASS。
GPUでは40 CUDA checksと2 metadata checksを実施後、主条件の全plan精度gate・完全stepを測る。


## v37：配置を固定した結果と1 atom版の候補改善

source `ecbe66c715949ec5faa43c5b08ace6ac530b1289`、job
`l4job-351462f09071495d82116d6800eb1f12`。42 checks PASS /1616 deselected、150.28秒。
全plan・両primaryで3状態の独立FP64 full-shape Y/dX/all4P gateと全52428 σ更新を確認。
primary TTGIRにforward/backwardそれぞれ[1,32]/[2,16]/[4,8]の指定layoutが残ることも
job内でassertしており、v36の論理転置とは別の実験である。

| plan | B32 ms | B64 ms |
|---|---:|---:|
| v32 control |0.801867|1.308898|
| Gluon atom lane1 |0.783794|1.269657|
| Gluon atom lane2 |0.831758|1.379352|
| Gluon atom lane4 |0.930261|1.579615|
| Dense |0.082619|0.104033|

両peakは全CST同じ34.153/98、34.653/100MiB、Dense49.002/106、49.502/106MiB以下。
2/4 atomを同じwarpへ置く配置は遅い。lane1は2.25%/3.00%の候補改善で、次の独立repeatと
small/switch条件で確認する。1 runだけを恒常的な改善とは扱わない。

lane1 forwardはcontrolと約0.133msで同じだが、backward BT32は約0.236–0.240から
0.218–0.219msへ短縮した。Gluon lane1ではbatch reduction後のatom vectorのSliceLayoutを
保持でき、source pullbackへのshared conversionが消えている（control shared16、candidate0）。
registerは40から48–52へ増えるがspill/local load/storeはない。lane2/4のbackward sharedは
32/64bytesで、実際の配置変更が改善するとは限らない。hardware counterによる比率の因果分解ではない。

source SHA256 `29444b52b8a9ab9f072c390eb8948fcb3ef7b61008feff3ed05219c0762af77f`、result SHA256 `de12adaa7b8b28b60cae1d1b0a8f790f708122707cacf55a5ee15e132edf523c`。
source byte一致1598 files、manifest235 files、return0/cleanup ack確認。

## 追加の成分診断：中間bufferの書き出しは別コスト

同source、job `l4job-0996a5ef8ce44c66b2b3d5cc856e05b5`、N1024/A52428/B32 mixed。
σ・中心・Xを更新する3状態で、既存v32 helperを別kernelへ分けた診断のみを行った。
全atom勾配の分割/融合差は0、Y/dXの差は最大2.480e-5/1.979e-5、relative L2<1.079e-6。
これは融合版との照合であり、独立FP64照合の追加結果とは区別する。主条件の独立検証は上記jobを使う。

| 診断kernel | 3状態のmedian ms範囲 |
|---|---:|
| fused forward |0.140–0.161|
| H query +H store |0.055–0.069|
| H/Hμ query +2 stores |0.069–0.074|
| Y injection +H load |0.144–0.152|
| fused backward |0.312–0.327|
| G/Gμ query +2 stores |0.071–0.072|
| source pullback +H/Hμ/G/Gμ loads |0.189–0.194|
| dX injection +G load |0.110|

各kernelは21 graph replay×20 callsで測り、分割版では4枚の[A,B] bufferを追加している。
これらの時間は余分なstore/load・cache状態・個別launchを含み、単純合算や完全step内の専有時間の
比率として解釈しない。診断のfused backwardも主条件の0.24msと測定状態が異なる。
queryだけを軽くしても書き戻しと中間buffer負担は残り、全H/Gを外部化する案の判断には
完全stepでの再測定が必要である。追加26MiB前後の診断bufferをDenseとのmemory比較には使わない。
raw結果は`local-split-probe.json`、driverと照合記録もignored outputへ保持する。

診断source SHA256 `97bf850e9f9bdb991e6b9ffd494b6fe06b5301af91b3daf00fc4bb5f130cb922`、result SHA256 `c6335aae6b80c41939a540eabd7e41e7846f2f649111387317070c625e272749`。
source byte一致1598 files、manifest4 files、return0/cleanup ack確認。
