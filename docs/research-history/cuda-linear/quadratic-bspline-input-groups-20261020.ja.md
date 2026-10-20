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


## v37独立repeat：小さい改善は再現、Dense2倍は未達

source `744e937c9a142f5e2c74738cf476bb790ee09aae`（runtimeはecbe66c7と同じ）、job
`l4job-9d73003dfaaf423ea0bb31be8715e520`。catalog metadata1 check PASS、4 caseすべての
全plan・3状態の独立FP64 full-shape gateと全52428 σ更新を確認した。

| case | v32 ms | Gluon lane1 ms | Dense ms |
|---|---:|---:|---:|
| B32 mixed |0.787373|0.770304|0.083208|
| B64 mixed |1.281540|1.243155|0.105507|
| B32 small |0.786843|0.768025|0.083191|
| B32 switch |0.792478|0.773918|0.083486|

mixedの2.17%/3.00%短縮が独立runでも再現した。small/switchは各1runで2.39%/2.34%短縮。
両peakは変わらず34.153/98、34.653/100MiBでDense49.002/106、49.502/106MiB以下。
時間はまだDense約9.26/11.78倍で、2倍以内の目標を達成したことにはならない。
主要backwardのPTXも確認し、同じwarp配置のcontrolにあったbar.sync7箇所がlane1では0に
なっている（primary2組、debug行を除いた静的な命令検査）。raw抽出は`gluon-backward-barriers.json`。

source SHA256 `f4d6ae035bee84c870c419652a3803dd1a37ba67945624358184c1600cf5b6a7`、result SHA256 `c03e959c27d02718116c6d4efe0a8efd9aacb4fb6ea58ae71dbbeddc19f6320b`。
source byte一致1602 files、manifest44 files、return0/cleanup ack確認。

## v38：nativeの軸profileをforwardごとに共有（検証前）

小さいσのf/dfを既存Co[2,18,A]のnative処理slotの先頭16 fieldへ入れる。
regular slot>=Countはこれまでの6 event×3係数を保ち、領域は重ならない。
native Pos[2,6,A]のfield0/1へ軸別left/rightを保存する。追加tensorはない。
canonical Parameter/Meta/optimizer orderは保ち、Orderでsource rowへ対応する。
forwardごとのCo/Pos snapshotをそのbackwardで再利用し、step間のX/Hは再利用しない。

半径R=√6σの同じB2 shape、同じFP32局所距離からf/dfを作る。
各native CTAは、両軸で全atomの支持点数が3以上かつ8以下の場合だけcacheを使う。
どれかが2以下なら従来のCTA単位FP64 cold pathへ、8より多いなら従来hot pathへ戻す。
σだけから8点を仮定せず、spacingとclip後の実際のleft/rightで判定する。
FP64 native Y/dX destination scratch、単一cast、exact sampled product L2、floor/tiny guard、
全source pullbackは維持する。cache値のsumはFP32で、従来のH保存精度境界でFP64へ戻す。

従来の3D支持点reductionを保つ候補と、8点を2D tileへ展開する候補を分ける。
8点の加算順の差については独立FP64の同じ4e-4 gateを要求する。
両候補はσ cutoff0.15、比較にはcacheなしの同cutoff0.15と、既存最速Gluon lane1 cutoff0.05を入れる。
native_cache_prepareもphaseへ含め、mixed/B64/small/switchで完全step・両peakを測る。
間隔0.025のoversize、0.3のtiny、非0原点、strided、retained、Graph、coincident nativeに加え、
cacheのsource slot対応とregular fieldが上書きされないことを検査する。公開/defaultは変更しない。

提出前CPU2000 PASS /4472 skipped /既存warning18、38.51秒。Ruff・4 case宣言検査PASS。
GPU43 checks（共通native cache29とspacing/fallback/profile14）およびmetadata2件を先に確認する。


v38初回source `92d83dc173d038172c0821a82a0e743a0650c078`、job
`l4job-d58e82c9fa864b0b9acd50cab3d709ff`は43 GPU checksが同じcompile errorで失敗し、
metadata2件だけが通った（65.24秒）。static_range(2)のdimループ内で`n: tl.constexpr`を
再宣言していた。既存lk.prepareと同じ通常のconstexpr値代入`n = NO if dim == 0 else NI`へ
修正する。profile式・支持・dtype・cache配置は変えない。数値検証や性能結果はなく、
失敗archive/source/manifest/cleanup ackを保存してprimary表から除外した。

failed source SHA256 `6d6a2faa1b457162afe05825f7798b069c82c70d952c949ab289a026f6c7f36a`、result SHA256 `8b08df6dea7e74d0b164163dc95a5bfad4e96c413ef357be624116961a568a1c`。
source byte一致1607 files、manifest4 files。


## v38 tail stress diagnosis and guarded FP64 contractions

The f6ae6379 GPU run 5b88cf3c4ea84f61be732b8c6e62d49b observed33 passing checks and12 failing extra unscaled tail cases, with no primary timings. Remote cleanup timed out after archive download. Source/archive/manifest hashes match, and the pool-owned VM was stopped, but POOL_CLEANED is absent: this run stays under incomplete-jobs and is excluded from all accepted tables. Recovery completed before the next owned job.

A separate fully verified diagnostic c211933039d74aaf907636948e9ccc87 at f6ae6379 compares the same P/X/dY for the old Gluon native path, cache3D and cache2D against the independent FP64 oracle. Cache3D equals old output/all gradients; cache2D differs by at most1.526e-5. The failures are shared with the previous path.

At spacing.025/B33, a center gradient should be3.469e-12 but is-0.001616; the fixture's nearest-FP32 error is only1.822e-6. The guarded cold query accumulates in FP64, rounds H/Hmu/G/Gmu to FP32, then promotes them back before subtracting the L2 derivative. Commit c5c7ad68 removes this intermediate rounding for the cold branch only (KEEP64=True). Hot profiles, normalization mathematics, final gradient dtype and acceptance threshold are unchanged. This is being diagnosed before adoption.

Other extreme floor cases have an unavoidable final FP32 precision limit: the FP64 gradient-17069.6205645 (B3,spacing.1) has minimum FP32 absolute error0.0005292;34530.0461711 (B33,spacing.3) has minimum error0.0007039. Both already exceed4e-4 even for the nearest representable value. Original unscaled fixtures and dtype limits remain as diagnostic evidence. The primary N1024/A52428 workloads and4e-4 gates are unchanged. A separate amplitude.125 diagnostic distinguishes representable stress inputs from this dtype limitation; it is not an accepted full-step run.

Task-owned v39 work was preserved in stashes9ffbbaabb991cfa0af6e2473f4c000b40a79ad91 and9fc60dc58f6cd568c08398021376b5e2594793d6 to freeze exact f6/c5 sources. It was restored with byte checks after submission; these backups are retained.

## v39：係数構築と使用尺度の記録をまとめる（検証前）

係数の構築時に既に求めている各atomのlevelを、軸別の整数bitmaskへORする。
新しいmaskはforwardごとに0初期化し、128 atomごとに2回のatomic ORで記録する。
従来の独立した分類scanは実行しない。使用しないlevelのprefix bankとrestoreを省くが、
bank/Deltaのshape、全係数・位置・L2・H/G・source pullbackの数式は保つ。

使用尺度の集合は \(m_d=\bigvee_{a\ge C}(1\ll \ell_{a,d})\) とする。
ここでCはnative atomの処理slot数で、\(\ell\) は既存の係数構築と同じ整数level。
levelが有効なのは \((m_d\mathbin{\&}(1\ll\ell))\ne0\) の場合だけである。
nativeだけのforwardではmask=0となる。maskは既存Active保存欄に保持し、
後のforwardで古いbackward用maskを書き換えない。

Deltaを全体0初期化する候補と、有効levelだけを0初期化する候補を分ける。
後者の追加launchが損になる可能性もあるため、両方を同じ完全stepで測定する。
Gluon lane1を基準とし、v38のnative cacheとは分けて効果を検証する。
既存recipeは新引数のdefaultで旧処理を保ち、公開/default selectorは変更しない。

旧classify_levelsとのmask一致、元の係数・位置・half/BP/Injectionのbit一致を確認する。
Graph replayでS4→S64→S16→S256→S4へ切り替え、新しい尺度の有効化と初期化を検査する。
Count all/partial/zero、retained、入力/全atom勾配、floor/tinyも既存の独立FP64基準で評価する。
CPU2001 PASS /4509 skipped /既存warning18、38.87秒。Ruff・4 case宣言検査PASS。

## cold FP64保持の診断結果と追加stressの表現可能域

source c5c7ad68、正常回収・cleanup済みjob
`l4job-4ed8a67f9baa4e99b481bb4e2a081ebf`で、未scaleと振幅0.125を分けて72状態を記録した。
spacing.025/B33の未scale全atom勾配差は0.001616から1.857e-5へ減り、B3も0.000802から
3.970e-6へ減った。ほぼ0になるはずの勾配の問題は、入力を弱めずに改善した。

未scaleの残りの大勾配差は最近傍FP32丸め誤差に一致した。B3/spacing.1は0.0005292、
B33/spacing.3は0.0007039で、どのFP32結果でも4e-4を満たせない。
この2組だけ追加stressのXとdYを0.125倍にし、test IDをfp32-envelopeと明記する。
他の4組は未scaleのままにし、特にspacing.025は旧中間丸めを検出する回帰条件として残す。
元の未scaleデータとdtype限界は診断archiveに保持し、成功した精度gateとして数えない。
主要N1024/A52428の全4 case・3状態・入力振幅・4e-4基準は従来と同じで、
この修正によってそれらを簡単な比較へ置き換えない。

新しい振幅域を含む追加spacing testsも含め、改めて全cache回帰とprimary性能を確認する。
診断jobだけでは完全step速度・メモリの採用根拠にしない。

## v39完全step：小さい候補改善、独立repeatへ

source `9e928c3fc08e48af9c828c0e7ff3e561df1bea61`、job
`l4job-26001f45364b4d98aabbcb8d22c5e76b`。39 checks PASS /1701 deselected、160.11秒。
maskと旧分類の一致、係数・位置等のbit一致、Graphの新尺度有効化を通過した。
4 caseの全plan・3状態の独立FP64 full-shape Y/dX/all4P gateと全52428 σ更新も確認した。

| case | Gluon lane1 ms | mask+全zero ms | mask+有効level zero ms | Dense ms |
|---|---:|---:|---:|---:|
| B32 mixed |0.773283|0.755915|0.765538|0.082318|
| B64 mixed |1.247125|1.211925|1.238853|0.104683|
| B32 small |0.772651|0.695009|0.695598|0.082892|
| B32 switch |0.774559|0.722699|0.726222|0.083310|

mask+全zeroの候補改善は2.25%/2.82%、small10.05%、switch6.70%。
有効levelだけのzeroは追加launchを回収できず、全zeroより遅いか同じ帯だった。
allocatedはmaskのallocator丸め分512bytesだけ増え34.154/34.654MiB、reserved98/100MiBで同じ。
Dense49.002/49.502MiB・106MiBに対し両peakは小さいままである。
source/result/manifest/cleanup ackの検証済みだが、速度は単発なので独立repeatで確認する。

B32 mixedのbank_inputは約0.0256→0.0205ms、scan_outputは0.0266→0.0246ms、
smallではそれぞれ0.0092/0.0123msへ減った。係数構築は約1µs増え、H/G本体はほぼ同じ。
独立分類scanを再投入せずに不要な集約を省けたが、Dense2倍の速度目標は依然未達。

source SHA256 `e1d0254377823a6af07267a4831c054764f43c1a75d1f1e5a9939461f7142852`, result SHA256 `8042ac6ea82e8e9a91fc738818fa25aedd75a2bd23e64fc49fbb9c931a1ee61e`; source files 1611, manifest 56, return0/cleanup ack verified.


## v38 complete steps: cached direct native is still slower

Job d7589d0593064b129b85623ea416d5b6 at4c456877 passed45 checks (1695 deselected,222.18s), then all four primary case3-state independent FP64 gates,52428 sigma updates and paired initial hashes. The primary protocol and inputs are unchanged; only the two named supplemental stress fixtures use the documented representable amplitudes. All variants have allocated/reserved34.153/98MiB(B32) or34.653/100MiB(B64), below Dense49.002/106 and49.502/106MiB.

|case|Gluon lane1 ms|old native cutoff.15 ms|cache3D ms|cache2D ms|Dense ms|
|---|---:|---:|---:|---:|---:|
|B32 mixed|0.773595|1.558422|0.914016|0.897113|0.082192|
|B64 mixed|1.248735|2.252022|1.535784|1.496527|0.103639|
|B32 small|0.772757|3.359884|1.206931|1.174840|0.082359|
|B32 switch|0.774758|1.554710|0.915397|0.899675|0.082914|

Cache reuse improves the old direct implementation substantially, but every case remains slower than the moment path. It is negative speed evidence, not an adopted improvement. In small-only, cache preparation is~0.0143ms; cache2D native forward/backward~0.289/0.377ms versus old direct1.195/1.664ms. Despite all atoms being native, regular stream forward/backward still cost~0.049/0.078ms: the static regular grid launches ceil(A/4)=13107 CTAs which return without useful regular work. Native launch grids remain capped at128 and loop over live groups. Forward/backward registers66/123(no hot spills), compared with64/109 old direct; cold reports4 spill slots but no local load/store and does not establish a primary hot-path cause.

These observations motivate a future unified per-atom native/regular executor with cached class flags and isolated fallbacks. It has not been implemented or measured. Do not infer its speed from empty-grid times or occupancy assumptions.

## v39 independent repeat

Job910e94941d6f455fb89d4a543cc302e9 at7bbbc1f9 has runtime identical to9e928c3f. Catalog1 check passed(2.57s), followed by all four-case paired full-shape gates and sigma updates.

|case|Gluon lane1 ms|fused mask/full-zero ms|Dense ms|
|---|---:|---:|---:|
|B32 mixed|0.771001|0.755559|0.083025|
|B64 mixed|1.247267|1.211924|0.105716|
|B32 small|0.769919|0.692880|0.082978|
|B32 switch|0.774750|0.724852|0.082459|

Mixed gains2.00%/2.83%, small10.01%, switch6.44% repeat independently. Peaks remain34.154/98 and34.654/100MiB for the mask version, below Dense in both allocated and reserved. The best validated ratios are still about9.10x/11.46x Dense, not the2x target. No public/default dispatcher is changed.

l4job-d7589d0593064b129b85623ea416d5b6: source SHA256 `e81aa45d52b62e68b391b03e9de567ced116c3b0ea3ff9db938a5248ce80069b`, result SHA256 `31d07d8d24611efe0e31e5da26676741d3a29ce63a382898f46aafc27cd1c4e5`, source byte-matched files 1611, manifest 68, return0/cleanup ack verified.

l4job-910e94941d6f455fb89d4a543cc302e9: source SHA256 `48eb4638e3194ad8b4439c5badfa717e94b8eb6b7475f7c3a1c8d77e0d5dc511`, result SHA256 `21bad0eb0e9a3139472b1e615e71f7f872d68a51915277df4e6607fc59786842`, source byte-matched files 1615, manifest 44, return0/cleanup ack verified.

l4job-c211933039d74aaf907636948e9ccc87: source SHA256 `d869cda997f2b00e5ef506f7b7a822a8cdb7d57bcf796c0797eca36164862439`, result SHA256 `51bf19679ef5da571c677b4cd446052d3434bb0fecc3f4e35c1ebe9365372477`, source byte-matched files 1607, manifest 4, return0/cleanup ack verified.

l4job-4ed8a67f9baa4e99b481bb4e2a081ebf: source SHA256 `ce6d7d17277727223fd498d7b221b987c5c13de067881888ad8d4cd5770c46c7`, result SHA256 `2d58bfd36654077c6edbe30af35ef03fed07f8f4eede418e0b84d6f5434ad3fc`, source byte-matched files 1607, manifest 4, return0/cleanup ack verified.

## v40: native hotとmomentの固定slot grid統合（検証前）

v38 small診断では、全atomがnativeでもregular側13,107 CTAが空処理で起動され、
native側は128 CTAの反復となった。v40は`ceil(A/AB)`個の共通gridで、
slotがnative prefixなら保存済み8点profile、regular suffixなら既存momentを評価する。
CountがCTA途中にあれば双方をマスクして処理する。元Parameter/Meta順序は保ち、
H/Gの全体Tensorや追加のatom分類Tensorは作らない。

native Coの未使用field16にCTA分類を置く。両軸の実支持点数の最小が2以下なら
従来FP64 cold、最大が8超なら従来hot、それ以外はcached hotである。
fallbackの存在を既存ActiveMaskの上位bitに記録し、該当群がなければfallback反復を省く。
下位bitのbank/restore尺度はv39と同じである。fallbackは従来の全batchを処理し、
cached hotとregularは既存batch chunkに従う。

FP64 native destinationは統合gridの後に一度だけY/dXへ加え、regular restoreへ進む。
その寿命は旧版より長くなるため、メモリ減少を仮定せずcapture/replayの両ピークを測る。
小batchごとのatom勾配加算とborder shardの担当順序は変わるので、同じ独立FP64 gateで判定する。
profile、支持、exact sampled product L2の単一floor、task-width stop-gradientとPolar則は変えない。
cutoff .05/.15をv39最速版とDenseへ比較し、主混合B32/B64とsmall/switchの全stepで評価する。
現段階ではGPUの正しさ・速度・メモリの主張はしない。

### v40実測: 精度PASS、速度は負の結果

job `l4job-1ef48d3105f7495f83c70150c5be9581`、source
`8a1b77ead43ae3acc1f28e524fa39efeadffa593`。
46 targeted checksがPASS（224.73s）。全4条件で3状態の全shape独立FP64 Y/dX/全4atom勾配、
52,428個すべてのσ更新、初期X/P/target hash一致を確認した。

|条件|v39 mask ms|統合cut.05 ms|統合cut.15 ms|Dense ms|
|---|---:|---:|---:|---:|
|B32 mixed|0.760142|0.759936|0.808702|0.083488|
|B64 mixed|1.218178|1.225800|1.340873|0.110982|
|B32 small|0.697337|0.698629|0.980522|0.083350|
|B32 switch|0.725063|0.728123|0.779888|0.083831|

統合版のpeak allocated/reservedはB32 **34.404/98 MiB**、B64 **35.154/100 MiB**。
FP64 native scratchの寿命延長によりv39からallocatedが0.25/0.5MiB増え、
reservedは同じ。Dense49.002/106、49.502/106MiB以下は保つが、速度改善はない。
cut.05は同帯、cut.15は全条件で遅いので採用しない。
small診断のstream forward/backwardは0.289792/0.368640ms、
v39の0.131072/0.212992msより重い。native fallback自体は約0.005/0.006msまで減るが、
cache構築0.014336msと直接集約側の実作業を相殺できない。
統合hot backwardは56register、shared0、spill0である。
ここからFP64 atomicだけを原因と断定はしない。

source SHA256 `6838fd396320c4bfd53620b08ac8e1fe4502bab32ea63d306997b88475f1333f`、
result SHA256 `a57c92ef42b17ec798eafda76d9c5cebc8bc58903913e1c8f8a45409e4d45403`。
source1621 files、manifest56 files、return0、timeoutなし、cleanup ackを厳密確認。
CPU2002pass/4554skip/18既存warning、Ruff/casesと369 runtime filesのwheel/sdist一致も確認。
後続c93fd513は上位bit16/17の予約をsupport metadataへ明示し、16尺度を超える入力を拒否する。
そのmetadata境界2件とroundtripの3CPU checksはPASS。主条件のkernelは変えていない。

## v41: bounded native hot partial sums（検証前）

v40のcached hotだけを4または16個のFP32部分和に分け、CTA番号で書き込み先を選ぶ。
最後に各部分和をFP64へ変換して合計し、既存FP64 cold scratchと合わせてY/dXへ戻す。
profile・H/G・全atom勾配の式や演算精度はv40と同じで、変わるのはcached hotの
出力加算の順序と部分和のdtypeである。FP32部分和に丸め誤差が入るため、
従来と同じ4e-4 gate・重なり支持・Graph切替・個別勾配を通るまで採用しない。
cold/tiny/oversize fallbackは従来FP64 destinationのままとし、近似的な支持打切りはしない。

部分和は`[shards, min(B,batch_chunk), N]`とし、全Bでは増やさない。
finish kernelでFP64合計と使用済み行のzero resetを融合し、次chunkへ同じ領域を再利用する。
B32/N1024では4shardsが0.5MiB、16shardsが2MiBの追加payloadだが、
これは実測peakではない。初期zero、finish/reset、capture/replayを含め速度と両ピークを測る。
v39最速版・v40cut.15・v41の4/16shards・Denseを同条件の完全stepで比較する。

### v41実測: 主条件PASSだが速度改善なし、4shardsは追加重なりgate失敗

main job `l4job-a1eb7bd1cac74d1ca3dbc970d49eeaa9`、source
`2cad75b693e56682ca6bb02b817edc12d15f46e1`。
40 targeted checks PASS（221.60s）、全4条件の3状態全shape独立FP64 gatesと
全52,428σ更新・初期hash一致を確認した。完全stepは以下。

|条件|v39 mask ms|v40 cut.15 ms|4shards ms|16shards ms|Dense ms|
|---|---:|---:|---:|---:|---:|
|B32 mixed|0.752962|0.808455|0.812437|0.835889|0.084348|
|B64 mixed|1.213115|1.339285|1.350247|1.381633|0.109249|
|B32 small|0.691202|0.980342|0.974096|0.990540|0.083727|
|B32 switch|0.719049|0.776596|0.782599|0.797327|0.084225|

4shards peak allocated/reservedはB32 **34.904/98 MiB**、B64 **35.654/102 MiB**。
16shardsはB32 **35.901/98 MiB**、B64 **36.651/100 MiB**。
Denseの両ピーク以下だが、v39/v40に対する速度改善はない。
smallのstream forward/backwardはv40が0.289792/0.367616ms、
4shardsが0.286720/0.364544ms、16shardsが0.287744/0.366592msとほぼ同帯。
4shardsのzero約0.004ms/方向、finish約0.007ms/方向、16shardsは約0.005/0.012ms。
これらは別の診断Graphの時間であり、主時間から差し引いたり足して割合を算出しない。
FP64 atomicをFP32の分割先へ変えても、直接側全体の重さはほぼ変わらなかった。
この実験だけで個々の読み取り/atomicのハードウェア待ち時間を同定したとはしない。

source SHA256 `884f797817f51091e6281597d525193015587014910b136fa0dd0caaecbbc719`、
result SHA256 `d2c28be45a453c7c1e2e8be49f6851056cb4aa2c8e61d7140e8786bb9288ff8f`。
source1626 files、manifest68 files、return0、timeoutなし、cleanup ack確認。
CPU2005pass/4593skip/18既存warning、Ruff/cases、370 runtime filesのpackage byte一致確認。

追加job `l4job-f58f02ede60f44e08ee8cd433fae0e2c`、tests-only source
`4c921b316dbaa021d911b04daebeaa54c78849b1`（runtimeは2cadと同じ）では、
513個のcached-hot支持を同じ場所へ重ね、B3/B33、3つのlive状態を照合した。
**4shardsは両batchでmax誤差約0.00083844となり4e-4を超えて失敗、16shardsの2件はPASS**。
入力振幅やgateを変更せず失敗を残す。4shardsは主条件の速度表がPASSでも採用不可。
16shardsも速度改善がないので採用しない。
追加jobはsource SHA256 `a811302d141f626d46630f7cbfe5ed0a9577200d55f153c830014719458d89cc`、
result SHA256 `935b1b0c4ccd80446c004dc6015218921877b2044320d1cc753477f47bb88c0f`、
source1626/manifest4、return1/timeoutなし/cleanup ackまで確認し、failed-jobsへ隔離した。
主条件PASSと追加gate失敗を混同しない。速度目標は依然未達。

## v42: nativeのbatch方向連続配置（検証前）

v40/v41のhot atomはwarp内laneがbatch方向を担当するが、RawXとY/dXは`[B,N]`であり、
隣のlaneのaddress差が`N * sizeof(value)`となる。一方moment bankは既にsite-majorである。
この配置上の違いを次の比較対象とする。ハードウェアのtransaction数や待ち時間の実測ではない。

v42はnative用のX/dYをchunkごとに`[N,B]`へコピーし、hotの出力も
`[1,N,min(B,batch_chunk)]`のFP64 scratchへ蓄積する。隣接batch laneの要素を連続配置し、
finishで元の`[B,N]`へ戻す。端数chunkは物理row capacityをstrideとして使い、
利用した行だけzero resetして再利用する。cold/oversize fallbackのFP64 scratchは元配置のまま。
hotのFP64加算を保ち、v41のFP32部分和は使わない。
profile/H/G/atom勾配式はv40と同じで、copy・zero・transpose finish込みで完全stepを測る。
追加payloadはB32/N1024でFP64 hot256KiBとX/dYコピー各128KiBであり、実測peakとは区別する。
v39最速版・v40cut.15・v42・Denseを4条件で比較し、513重なりも同じgateで確認する。

### v42実測: 直接側の配置改善は効くが、最速momentを超えない

job `l4job-2425cfa75f814786bcc1f5bbcff01d7d`、source
`f5e2d4d07064ed37d40ba3a9442ee82a162e3778`。
41 targeted checks PASS（231.59s）、513重なり・端数chunkの物理stride/zero再利用、
全4条件の3状態全shape独立FP64 Y/dX/全4atom勾配、全52,428σ更新と初期hash一致を確認。

|条件|v39 mask ms|v40 cut.15 ms|v42 point-major ms|Dense ms|
|---|---:|---:|---:|---:|
|B32 mixed|0.764634|0.812698|0.765297|0.083503|
|B64 mixed|1.234458|1.353722|1.250661|0.107560|
|B32 small|0.703778|0.974817|0.797884|0.082827|
|B32 switch|0.731340|0.783509|0.733314|0.082374|

v42 peak allocated/reservedはB32 **34.904/98 MiB**、B64 **35.654/102 MiB**。
両方ともDense49.002/106、49.502/106MiB以下。
しかしv39に対してB32 mixed/switchは同帯、B64とsmallは遅いので最速版へ採用しない。
smallはv40比18.15%短縮するが、既に不採用の直接方式に対する改善であり、
最速momentを超えたとは扱わない。

smallのstream forward/backwardはv40 **0.289792/0.366592ms**から
v42 **0.208896/0.270336ms**へ減る。ここにはX/dYのtranspose copyを含む。
hot zero約0.003ms/方向、finish約0.006ms/方向も完全stepに含む。
streamのregister40/56・shared0・spill0は同じ。finishは26register/shared2048bytes/spill0。
配置変更は直接側の実行時間を減らした一方、FP64 atomicをFP32分割したv41だけでは減らなかった。
これは実装の比較証拠であり、個別のhardware transaction/待ち時間を計測したという主張ではない。

source SHA256 `9e4b3ac23164c32e9ed837148a9f541f8c18702c5e5c06e5222fa830e3352599`、
result SHA256 `2b04b5e331ae9ef9d8f706160f12548584d0141bb0d9fe4fe991c85afdd7e61f`。
source1630 files、manifest56、return0/timeoutなし/cleanup ackを厳密確認。
CPU2006pass/4637skip/18既存warning、Ruff/cases、370 runtime filesのsdist/wheel一致、
source f5e2d4d0のGitHub CPU validationもPASS。
source内のv41 4shardsに対する追加重なりテストは失敗を再現するものであり、
全CUDA routeの回帰suiteがPASSしたという意味ではない。v42 targeted PASSと区別する。

目標は引き続き**同条件Denseの2倍以内かつ両memory peakがDense以下**。
最速v39の今回再測定はDense比約9.16x/11.48xで、速度目標は未達。
v40/v41/v42のいずれも公開/default dispatcherへ反映しない。

### 次の準備改善候補（未実装）

現decoderの幾何補間
`sigma=exp((1-alpha)*log(lower)+alpha*log(upper))`は、数学的には
`sigma=lower*exp(alpha*log(upper/lower))`と同じである。
有限ratioの場合はFP64 logを2回から1回へ減らせる。ratio overflowでは元式へ戻す候補とする。
CPU-only design probeではprimary範囲20万例のmax relative差1.236e-15、
極端な幅範囲20万例では29,513例のfallback込みで2.025e-13。
これはGPU精度gateや速度証拠ではない。別revisionとして動的σ・全勾配・境界・両peakを検証する。
CPU式の確認だけで現decoderを書き換えたり、速度達成扱いにはしない。

## v43: 1回のFP64 logによる幅復元（検証前）

最速v39のH/G・routing・coefficient・exact sampled L2・pullback・Polar更新を保ち、
split decoderの幾何補間だけを別recipeで切り替える。

\[
\sigma = \exp((1-\alpha)\log l+\alpha\log u)
       = l\exp\!\left(\alpha\log\frac{u}{l}\right),\quad
0\le\alpha\le1,\quad 0<l\le u.
\]

`u/l`が有限で正なら右辺をFP64で評価し、従来と同じ上下clipを適用する。
CTA内の有効atomに非有限ratioがあれば、そのCTAは従来式で評価する。
無効tail laneはfallback判定から除く。これにより有限の幾何平均を表せる極端な幅でも
中間ratioのoverflowで結果を壊さない。runtimeのdevice scalarからl/uを計算し、
hostに値を戻したり固定幅と仮定したりしない。

既存recipeは`RATIO=False`のまま。新しいv43 `BSplineLocalDecodeRatioRecipe`は
v39を継承し、native cache/統合/FP32部分和/point-majorの負の候補は組み込まない。
Tensorを追加せず、Metaのdtype/形状とsnapshot寿命も変えない。
数学的に同じ補間でもFP64加算順序は変わるので、全shape独立FP64の既存4e-4 gateを維持する。

追加検証は、2種類のpowerと4種類のlive幅境界でのdecoder比較（8件）、
Graph replayで通常幅→ratio overflow→通常幅へ切り替える1件、
通常の動的σ・retained snapshot・個別勾配・floor・513重なり・非単位spacingを含む。
radius/inverse以外のMeta fieldは旧decoderとbitwise一致、overflow fallbackもbitwise一致を要求する。
radiusは独立Torch FP64式と比較する。この局所検証と全modelの誤差gateは別々に行う。
B32/B64 mixedとsmall/switchをv39最速版・v43・Denseで比較し、準備/構築/L2/全勾配/更新と両peakを含める。

初回job `l4job-d87b433ab7cc4f9393e33f588c55c4b9`、source5e684fedは29 checks PASSの後、
共有spacing fixtureがnative専用`cache_unroll`をv43へ渡したTypeErrorで停止した。
性能測定には進まず、数値/速度の採用証拠にはしない。幅・入力・gateを変えず、
テストの引数だけをrecipeに応じて渡す修正で再実行する。
failed archiveはsource/result/manifest/git source bytes/return1/cleanup ackまで確認して保全した。

### v43の修正後実測: decode短縮、完全stepは小差なので再測定中

job `l4job-df8cd34450dd42ee944322805bd2bec6`、source
`18c2ff94db2a71809dadacbd867e58030d652a75`。45 checks PASS（197.21s）。
overflow時のbitwise fallback、Metaの非radius field一致、独立Torch FP64式、
live scalarのGraph変更に加え、全4条件の3状態の全shape独立FP64 Y/dX/全4atom勾配がPASS。
case原本はN1024/A52428、B32/B64を明示し、oracleはcase全体のP/Xから計算することを再確認した。
全52,428σ更新と初期hash一致も確認した。

|条件|v39 ms|v43 ratio ms|Dense ms|
|---|---:|---:|---:|
|B32 mixed|0.758281|0.752114|0.083489|
|B64 mixed|1.216306|1.215121|0.103322|
|B32 small|0.696890|0.692370|0.083035|
|B32 switch|0.724995|0.721933|0.082741|

両版ともpeak allocated/reservedはB32 **34.154/98 MiB**、B64 **34.654/100 MiB**で同じ。
Dense49.002/106、49.502/106MiB以下を保つ。decode診断は約0.0635–0.0645→0.0594ms、
register80→78、shared0/spill0。正規化・係数構築・H/G本体は同帯。
B32 completeの0.81%、small0.65%、switch0.42%、B64の0.10%は小さい差であり、
この1回だけで確立した高速化とはしない。次のv44比較にv43を含めて独立再測定する。

source SHA256 `7874ba78afcd8e7787f8b6086027e3827699c5d05aafc2e9f5d1a2a5d7b3f703`、
result SHA256 `da75b4b02942e45e67e94500f2a74b70cb91caee2fab09847037e12dd98030ec`。
source1634 files、manifest44、return0/timeoutなし/cleanup ackを厳密確認。
初回failureはsource SHA256 `892cf77d1fb4d90b0246c8ff23dfe75a905943717d13e88ead9611c6152b31ac`、
result/manifest等は`failed-jobs/l4job-d87b433ab7cc4f9393e33f588c55c4b9/exclusion.json`に保全し、
採用表へ混ぜない。入力振幅や4e-4基準の変更はない。

## v44: FP64 Metaの隣接pair書き込み（検証前）

decoderが書く9個のFP64 fieldについて、(0,1)/(2,3)/(8,9)/(10,11)をpair、
field7をscalarで保存する。物理Metaは従来の`[A,12]`のまま、field4/5/6は既存normalizeが書く。
Gluonの2D layoutをsize-per-thread `[1,2]`、threads `[32,1]`、warps `[W,1]`とし、
field方向の2値を同じthreadへ置く。1D atom計算はそのSliceLayoutを使い、
隣のfieldを別threadへ割り当ててPolar計算を重複させる配置は使わない。

FP64の元演算をそのまま移し、通常式とv43 ratio式の2設定を別々に比較する。
新Tensor・Meta並べ替え・正規化変更はない。旧decoderとの全Meta bitwise一致、
生成PTXでの128bit (`v2.b64/f64/u64`) storeの存在、通常→極端な幅→通常のGraph replayを要求する。
実際にその命令が生成されるまではvector化できたと主張しない。
NVIDIAのメモリtransaction数・占有率・速度改善もソースだけから推定確定しない。

Gluon/libdeviceのAPIは実行環境と同じ[Triton v3.6.0の公開ソース](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/experimental/gluon/language/extra/__init__.py)
で確認した。CUDA importをCPUの宣言/recipe検査へ持ち込まず、実行時だけimportする。
CPU・GPUの入口検証後、v39・v43・v44通常式・v44ratio式・Denseの4条件比較を行う。

### v44実測と独立再測定

主job `l4job-adbd4113eb194458afe6d4de2a02bc8b`、source
`b569bd449fc9f271d651cd33a1d080828033f2da`。50 targeted checks PASS。
12 fieldのbitwise一致、Graphの極端な幅への切替、実PTXのpair storeを確認した。
全4条件の3状態全shape独立FP64 Y/dX/全4atom勾配、52,428σ更新、初期hash一致もPASS。

|条件|v39 ms|v43 ratio ms|v44 pair通常式 ms|v44 pair+ratio ms|Dense ms|
|---|---:|---:|---:|---:|---:|
|B32 mixed|0.758131|0.753272|0.752920|0.747158|0.081924|
|B64 mixed|1.227714|1.215445|1.216528|1.210881|0.103547|
|B32 small|0.698458|0.693805|0.693803|0.687706|0.083080|
|B32 switch|0.727817|0.722786|0.724165|0.717028|0.082975|

decode診断は約0.064→0.052–0.053ms、prepare全体は約0.141–0.144→0.130–0.133ms。
正規化・係数・H/Gは同帯。pair+ratioはregister80、shared0/spill0。
新Tensorはなく、全候補のpeak allocated/reservedはB32 **34.154/98 MiB**、
B64 **34.654/100 MiB**。Dense **49.002/106、49.502/106 MiB**以下を保つ。

独立再測定job `l4job-d507bf5317f24d028ae17c35b0d43e20`、source
`ee28e4495e3ea0c7e57f8a7dc7930f63cc2c77ce`（runtimeはb569と同一）も全gate PASS。

|条件|v39 ms|v44 pair+ratio ms|Dense ms|
|---|---:|---:|---:|
|B32 mixed|0.768294|0.759292|0.084380|
|B64 mixed|1.239294|1.231679|0.108056|
|B32 small|0.706606|0.695316|0.084186|
|B32 switch|0.735408|0.722767|0.084078|

再測定も両peakは同一。v39比の短縮はB32/B64 mixedで1.17%/0.61%、
small/switchで1.60%/1.72%。小さい改善は反復するが、Dense比は約9.00x/11.40xで
2x目標には届かない。default dispatcherは変更しない。
CPU2008pass/4730skip/18既存warning、Ruff/cases、371 runtime filesのwheel/sdist一致を確認。
ee28のGitHub CPU validationもPASS。既知のv41追加テスト失敗とtargeted PASSは区別する。

主job source SHA256 `70050d67a52bba00f4eec7a22528fda9f95d73444fd07961257aad40542d08ce`、
result SHA256 `2263ae2b40dfdfb11a3cfe1ceade507615422076dcca5bfc161f5678f5bf6293`。
両jobともarchive/manifest/sourceのGit bytes/return0/timeoutなし/cleanup ackまで照合し、
正確な再測定hashは`verified-jobs/l4job-d507bf5317f24d028ae17c35b0d43e20/receipt.json`と
`verified.json`へ保全した。全所有slotの停止も確認した。

### G4比較の追加承認

2026-10-20のユーザー指示により、G4でも同条件Denseの完全step時間2倍以内かつ
capture/replayのpeak allocated/reservedが両方Dense以下なら、今回の目標達成とする。
精度gate、σ/入力の動的条件、全4atom勾配、optimizerを含む比較契約は維持する。
デバイス間で速度表を混ぜず、L4上のアルゴリズム改善とは区別する。
検証済みv44とv39/Denseを同じsource/caseで短く比較し、実GPU・runtimeを記録する。
G4結果は検証前であり、目標はactiveのまま。

### G4実測: 差は縮むが、速度目標は未達

job `colabjob-4d6b74524c224b71b4cc5fb3d202c083`、source ee28。
実機は **NVIDIA RTX PRO 6000 Blackwell Server Edition**、表示メモリ97,887MiB、
driver580.82.07、Torch2.11.0+cu130/CUDA13.0/Triton3.6.0。
28 targeted checksと両primary条件の3状態全shape独立FP64 gateがPASS。
全52,428σ更新・同条件の初期X/P/target hash一致も確認した。

|条件|v39 ms|v44 pair+ratio ms|Dense ms|v44/Dense|
|---|---:|---:|---:|---:|
|B32 mixed|0.345440|0.343189|0.066210|5.18x|
|B64 mixed|0.527840|0.525179|0.074770|7.02x|

v44のpeak allocated/reservedはB32 **34.154/98 MiB**、B64 **34.654/100 MiB**。
Dense **49.002/106、49.502/106 MiB**以下。Y/dX/全atom勾配のmaxabsは
3.300e-5/2.855e-5/1.102e-4、relL2は全て1.686e-6以下で4e-4 gateを満たす。
速度2xは未達なので達成扱いにしない。L4とは別の`verified-g4.json`/`audit-g4.json`へ記録。

診断のB32 stream forward/backwardは約0.0467/0.0692ms、
B64はchunkごと約0.0467/0.067–0.0692ms。decode約0.0196–0.0198ms、
norm_init約0.0236–0.0247ms、係数約0.0226–0.0232ms、group約0.018ms。
bank各約0.0139ms、scan各約0.0160msも残る。
個別イベント診断を完全stepへ単純加算した内訳とは扱わない。
G4上で絶対時間とDense比が変わることは確認できたが、キャッシュmissやFP64待ちを
hardware counterで測定したものではなく、その機序までは断定しない。

跨VM照合ではcase/seedとX/targetのhashは一致する一方、初期Pのhashは非一致だった。
Pは`bspline.initialize`のCPU計算で生成されるが、差の発生原因や大きさは今回確定していない。
各GPU内のCST候補同士は初期Pも一致し、DenseともX/targetが一致する。
したがって各デバイス内の目標判定には使うが、Pまでbitwise同一に固定した
GPU世代間アブレーションやcache効果の証明とは扱わない。

source SHA256 `d68b2ce8390e6b9a64f7322e3a23b498bc3c6201594f188b6dd9fb4147913ff2`、
result SHA256 `29526154becc28d680c45f8eeb92180a4fb1249f8658ce11ba7deed4376cbeeb`。
source1643/manifest24、archive/manifest/Git bytes/return0/timeoutなし/cleanup ackを照合。
supervisorは正常終了し、所有VM停止を確認した。default変更なし。

## v45: 古い空間順序の継続利用と、厳密なnative区分（検証前）

v44の演算を保ち、broad atomを幅level・出力support開始tile・入力support開始tileで
近い順へ並べる。tile幅は実行順のヒントであり、supportやσを量子化しない。
Parameter/optimizer行は移さず、処理slotから元atom IDを参照する。

順序を毎step再構築せず、native/broadの所属が一つでも変われば即時再構築する。
所属が同じなら、空間keyが変わっても古い順序を使い、1/64stepの周期候補で更新する。
幅cutoffに加えて、domain clip後のtiny/empty supportによるnative所属も毎step検査する。
古い順序の利用は数学的な近似ではない。両区分内の順序は正しさの条件でなく、
Meta・L2・係数・支持区間・全勾配は毎forwardの実際の値で計算する。
order/countはforwardごとimmutable cloneを保持し、retained backwardを壊さない。

追加histogram容量は幅keyの最大値`floor(log2(max_patch_count+1))`を含める。
従来のrange配置でmax_patch_count=31等のときに一bin不足し得た計算も修正し、
124sites/4site-patchの全幅を検査する。N1024の既存容量は変わらない。
構築/clone/更新周期の費用と追加histogramメモリを含めてv44/Denseと比較する。
空間局所性が改善してもatomic競合が増える可能性があり、実測前の速度改善は主張しない。

初回job `l4job-87d6cde1b58c4b35b7c6f9b968417ce6`、source626a8837は36 checks PASS後、
tile64のテストで停止した。64site全体をtile64一個で覆うfixtureなのに、
領域内移動でspatial keyが変わることを要求したテスト条件の誤りだった。
N128/中心1–8へ変更し、最大tileでも実際の内部境界を跨ぐように修正した。
入力振幅・数式・4e-4 gateに変更はない。修正00f2c3baのruntimeは626a8837と同一。
初回は性能測定なし。archive/source/manifest/Git bytes/return1/cleanup ackを照合して
failed-jobsに保全し、採用証拠に混ぜない。

period64の短いcase（21rounds）は、warmup/eager/capture/replayを合わせても52forwardである。
周期再構築を含む長期平均が必要なら64を超える測定で別途確認する。
この短期中央値をperiod64再構築の償却費込みと解釈しない。

### v45 L4実測: 空間順序を維持してもforwardが遅い

修正後job `l4job-e72717e064434fea830ff2f123befed8`、source
`00f2c3ba8be774510f09cb9dc9c96f0a13dab8d3`。
38 targeted checks PASS（176.16s）。native所属変更・中心だけのtiny所属変更・
advisory keyの変化時の順序再利用・周期更新・過去snapshot保持・124site容量を確認。
全4条件の3状態全shape独立FP64 Y/dX/全4atom勾配、全σ更新、同条件初期hash一致もPASS。

|条件|v44 control ms|tile32/毎step ms|tile32/64step ms|tile16/64step ms|Dense ms|
|---|---:|---:|---:|---:|---:|
|B32 mixed|0.750181|0.775475|0.763648|0.765992|0.083000|
|B64 mixed|1.219921|1.271198|1.261517|1.257495|0.104395|
|B32 small|0.693156|0.729895|0.719113|0.722609|0.082987|
|B32 switch|0.722246|0.756508|0.745214|0.749533|0.083073|

tile32のallocatedはB32/B64 **34.259/34.759 MiB**、tile16は **34.576/35.076 MiB**。
reservedは全候補 **98/100 MiB**で、Dense49.002/49.502 allocated・106reserved以下。
64step候補は52forward中1回の構築で51回再利用、毎step候補は52回とも再構築した。
比較controlも52回中1回の構築なので、再利用を追加するだけで費用が消える状況ではない。

groupはcontrol約0.0287ms、毎step候補約0.039–0.041ms、64step候補約0.029ms。
しかしB32 mixedのstream forwardが0.1331→0.1577ms、smallは0.1311→0.1618msへ増え、
backwardの0.2150→0.2130ms、small0.2181→0.2109ms程度の短縮を上回る。
係数準備は約0.058–0.059msで同帯。正しい空間順序でもforwardが遅くなる実測であり、
cache局所性の改善を自動的な速度改善とみなさない。atomic競合という原因の断定も避ける。

source SHA256 `eed882219d0cdd068f613239eb622b3c491bd6a96ebe27ccbfd2957401243e7b`、
result SHA256 `e200027970ccc5bba9456dea4465bfd65ef5ff897ce9ac5a2492f872c653f833`。
source1647/manifest68、archive/manifest/Git bytes/return0/timeoutなし/cleanup ackを照合した。
CPU2009pass/4767skip/18既存warning、Ruff/4caseと371 runtime-file package一致、
00f2c3baのCPU CIもPASS。L4のv45は不採用。

世代差の確認は、数値検証済みtile32/64step一設定に絞ってG4で行う。
129roundsへ延長し、period64再構築を測定区間へ含め、v44とDenseも同じroundsで測る。
この比較の準備・snapshot clone・再構築・L2・optimizer費用は除外しない。

### v45 G4長期比較: 周期再構築を含めても不採用

job `colabjob-195a662be03e4210a749ec8cd68f77a8`、source
`e1102c645df2625fac09ab2808d30892ec05e6c6`（runtimeは626a8837/00f2c3baと同一）。
実機・runtimeは前G4比較と同種。5件の配置検査と、B32/B64の3状態全shape独立FP64
Y/dX/全4atom勾配、全52,428σ更新、同条件初期hashが全てPASS。

|条件/方式|中央値 ms|平均 ms|p95 ms|最大 ms|allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|
|B32 v44|0.353870|0.352733|0.356340|0.393910|34.154/98|
|B32 spatial32/64step|0.363810|0.364554|0.368039|0.424749|34.259/98|
|B32 Dense|0.066380|0.066480|0.067240|0.087090|49.002/106|
|B64 v44|0.543549|0.540480|0.546189|0.585640|34.654/100|
|B64 spatial32/64step|0.560740|0.561102|0.568010|0.613189|34.759/100|
|B64 Dense|0.075140|0.075303|0.076050|0.095430|49.502/106|

129roundsのGraph測定は周期更新を含む。warmup/eager/replay計268forward中、
v44は構築1回、spatial候補は5回であり、後者は測定replay区間内でも再構築した。
中央値で2.81%/3.16%、平均でも3.35%/3.82%遅い。最大値の個々の原因は未特定で、
全ての長いsampleを再構築へ帰属させない。両memory peakはDense以下だが、候補は不採用。

B32 stream forwardが約0.0508→0.0630ms、B64 chunk0は0.0508→0.0630msで、
groupの中央値約0.018msは同帯。並べ替え費用を周期化するだけではforward悪化を解決しない。
この結果は今回の出力tile優先・入力tile次順位の配置への負の証拠であり、
あらゆる配置やCST全体の達成不能を示すものではない。
最速controlもDense比5.33x/7.23xで速度2xは未達。目標はactiveのまま。

source SHA256 `03604479957ed360664c3dc6dc075d2ee87a1f0c6fa772d8b855873821ea59c1`、
result SHA256 `53f81bb83e4d42480921dd7a7ec5b92ecb42bdfec9550ebcd93b25ed1a6401d9`。
source1649/manifest24、archive/manifest/Git bytes/return0/timeoutなし/cleanup ackを照合。
G4記録は`verified-g4.json`/`audit-g4.json`、sample統計は`spatial-g4-summary.json`へ保全。
追加metadataのCPU60checks、長期2case宣言、371 runtime-file package一致も確認した。
所有supervisor正常終了とVM停止を確認し、公開/default dispatcherを変更しない。

## v46: forward入力bankの不変snapshotをbackwardで使う（検証前）

v44を基準にし、v45の空間順序は使わない。block幅s、開始site t*sに対して、

\[
B_{p,t,k,b}=\sum_{r=0}^{k-1}X_{b,ts+r}z_r^p,\qquad
z_r=2r/s-1,\quad p\in\{0,1,2\}.
\]

H/Hμのqueryはこの3moment bankとforward時の係数を読む。
backwardで同じX・同じbasisからbankを再構築していた部分を、そのforwardのbank保存で置き換える。
bankの数値計算・dtype・query・注入・L2・全4atom pullbackは変更しない。
新stepではX/σから新しいframeを作り、古いbankを再利用しない。
active levelもframe固有のままで、過去VJPは過去Meta/係数/order/count/bankを使う。

B≤batch_chunkではforward bankをそのままsave_for_backwardへ追加する。
B>batch_chunkでは各chunkに個別のbankを確保し、後続chunkが上書きしないよう全て保持する。
端数chunkの物理strideは従来と同じbatch_chunkであり、有効rowのみqueryする。
atom勾配不要時は保持しない。dY bankと差分配列は別領域であり、query中の入力bankを上書きしない。
retained backwardの再実行にもbankの不変性を維持する。

N=1024/min_block=4ではlocal point数P=9727。保存payloadはB32で3*P*32*4byte=
**3.562 MiB**、B64/chunk32で **7.124 MiB**。H/Hμを両方保持する12.800/25.600MiBより小さい。
従来backwardの入力bank allocationも省くので、このpayload全量がpeakへ追加されるとは限らない。
一方B64では従来の一個の使い回しから二個の保持になるため、総peak増加をGPUで検査する。
payload式をallocated/reserved peakの達成証拠には使わない。

`save_input_bank=False`は従来の再構築と同じ経路。新v46はTrueを標準にし、公開/defaultは変更しない。
CPU2010pass/4808skip/18既存warning、Ruffを確認した。GPUでの42 targeted checksは、
既存の動的σ/全勾配/Graph/tiny/floorに加え、3/33/64batchのbank一致・保存寿命、
atom勾配なしでの非保存を含む。B32/B64 mixed・small/switchでv44/Denseと完全step/両peakを比較する。

### v46 L4主測定: 再構築を省き2.4–2.7%短縮、メモリはDense以下

job `l4job-773ac2292a2249cba0912f6e2c9389ea`、source
`f26c4566`。42 targeted checks PASS（201.35s）。有効level/rowのbankが再構築値とbitwise一致し、
別forward後の保存寿命・retained VJP、端数chunk、非atom勾配での非保存も確認した。
全4条件の3状態全shape独立FP64 Y/dX/全4atom勾配、全52,428σ更新、初期hash一致もPASS。

|条件|v44再計算 ms|v46保存 ms|Dense ms|
|---|---:|---:|---:|
|B32 mixed|0.751303|0.733227|0.082376|
|B64 mixed|1.214298|1.181831|0.105295|
|B32 small|0.689469|0.682789|0.083200|
|B32 switch|0.719430|0.709110|0.083120|

mixedの短縮は2.41%/2.67%。B32 small/switchは0.97%/1.43%。
B32 allocated/reservedは **34.154/98 MiB**でcontrolと同じ。
B64はallocatedが34.654→**38.216 MiB**（一bank分3.562MiB増）、reservedは **100 MiB**で同じ。
Dense49.002/106、49.502/106MiB以下を保つ。
入力bank再構築の診断約0.0205–0.0215ms/chunkが空のevent marker区間へ変わり、
H/Gのstream区間は同帯。B64の追加payload全部ではなく、一bank分がpeakへ加わる実測だった。

source SHA256 `d8c6938a66f91e1cef8534daafa5e44a2b1942f3746e4841435995dff0a6f080`、
result SHA256 `49c081c06522743a96586f2cb67ded405b0927e60721b940d4cbd330ae206767`。
source1653/manifest44、archive/manifest/Git bytes/return0/timeoutなし/cleanup ackを照合。
371 runtime-file package一致とf26c4566のCPU CIもPASS。
単回のcandidate改善なので、129roundsの独立L4再測定と、検証済み同候補の短いG4比較を行う。
速度2x目標には依然未達であり、default選択器は変更しない。

### v46の129round再測定: G4改善、L4は条件依存

source `428b5bd0b6d848abfff7dc89165ff29302bb9a67`（runtimeはf26と同一）。
G4 job `colabjob-a8abc16c5eaf46f4b4206ccabeba606f`は9件の保存bank検査と
両primary条件の全shape/3状態FP64 gate、全σ更新、初期hash一致がPASS。

|G4条件|v44 ms|v46保存 ms|Dense ms|保存/Dense|
|---|---:|---:|---:|---:|
|B32 mixed|0.352320|0.342510|0.066420|5.16x|
|B64 mixed|0.541890|0.523340|0.075320|6.95x|

2.78%/3.42%短縮。peak allocated/reservedはB32 **34.154/98 MiB**、
B64 **38.216/100 MiB**で、Dense49.002/106、49.502/106以下。
再構築約0.0119ms/chunkが空event markerへ変わる。精度基準4e-4は維持する。

独立L4 job `l4job-871b3dfb61f84330a85f18a8d7fc5b84`も全4条件の全shape/3状態gateがPASS。
しかし長い測定の中央値は以下であり、初回の短い測定と一様には一致しない。

|L4再測定条件|v44 ms|v46保存 ms|Dense ms|
|---|---:|---:|---:|
|B32 mixed|0.760738|0.821187|0.082184|
|B64 mixed|1.447772|1.419492|0.104899|
|B32 small|0.780025|0.755625|0.082031|
|B32 switch|0.714776|0.794863|0.082018|

両peakは初回と同じ。B32 mixed/switchは保存版が遅く、L4の無条件な最速とは扱わない。
両方式とも時間系列が非定常だった。mixedの32sample区間中央値はcontrol
0.745/0.760/0.762/0.845ms、保存0.727/0.786/0.844/0.856ms。
B64もcontrol1.204→1.449ms、保存1.174→1.430msへ推移した。
最終native countは全条件0で、stream診断はmixed両版約0.140/0.219msで同帯。
ホスト/デバイスのclock・電力・OS待ちの同時記録はなく、原因は確定しない。
都合のよい初期sampleだけを採用したり、L4の負の中央値を除外したりしない。
G4での候補改善と、L4での再現性の制限を分けて記録する。速度2x目標は未達。

G4 source SHA256 `9b41971ffcc65a8fe02e51e3f9db5ce0be8c859134ffeea9e25a31185ac22c05`、
result `3b90013a6e767d613318c98eaca7f25cf626254c2eb0bfb93a179df8115f2c56`。
L4 source `c6004315949bcab3d197ad5dcc73a44ad97506380d2622bacabb4641e013cf94`、
result `ea5b8364de4268f4a1273a8caf0d8f0890aa5fd9a05ffbafe689f7978405b1c9`。
各source1657、manifest24/44、archive/manifest/Git bytes/return0/timeoutなし/cleanup ackを照合した。

## v47: bank構築と独立scratch初期化の起動融合（検証前）

最初のbank groupのCTA群と、delta/borderのzero CTA群を一つの1D gridへ置く。
bank CTAは従来と同じ2Dのtile/batch番号へ復元し、既存のscan関数を呼ぶ。
zero CTAは別の出力領域をflatに全初期化する。bankとdelta/borderは別allocationであり、
CTA間のデータ依存やgrid同期は追加しない。全bank levelが無効でもzero CTAは実行する。
異なる入出力site数・物理batch padding・partial tileを含める。

従来のbank算術、dtype、query、L2、注入式は維持し、起動を2回減らす候補である。
zero費用は最初のbankの診断区間へ含め、deltaの空markerを別のzero実行時間とみなさない。
追加の保存Tensorはない。bank保存あり/なしを別Planにし、v44/v46も同じ条件で比較する。
元のbank関数のPID overrideは未指定なら従来のprogram_idを使う。
公開/defaultの選択は変更しない。

CPU2011pass/4849skip/18既存warning、Ruffを確認した。GPUでは有効bankのbitwise一致と
全scratchのzeroを、全level無効・異なる軸サイズ・端数batch・部分bundleでも確認する。
全shape/3状態FP64 Y/dX/全4atom勾配と、構築/L2/optimizer込みの両peak・完全stepも必要である。

### v47 L4実測: 初期化融合の候補改善、追加保存領域なし

job `l4job-d68a681af8d44d5c9d620a2d9b56e7c4`、source
`6aacb1d46e265e3ffd7fe05c8e6ab51eb243d57e`。
42 targeted checks PASS（182.81s）。bank値のbitwise一致、全scratch zero、
無効level・異軸サイズ・padding・部分tileを確認し、全4条件の3状態全shape独立FP64 gate、
全σ更新、初期hash一致もPASS。最大誤差は従来の4e-4 gate内に収まる。

|条件|v44 ms|v46保存 ms|v47融合/非保存 ms|v47融合/保存 ms|Dense ms|
|---|---:|---:|---:|---:|---:|
|B32 mixed|0.752383|0.728222|0.734732|0.717532|0.083635|
|B64 mixed|1.206005|1.176789|1.203504|1.161727|0.107769|
|B32 small|0.688394|0.682872|0.684964|0.676839|0.083088|
|B32 switch|0.715669|0.704332|0.706898|0.697441|0.083167|

v47融合/保存は同条件v46比1.47%/1.28%短縮（mixed）。
融合単独の非保存はB64で同帯であり、一律な高速化と扱わない。
全候補のB32 allocated/reservedは **34.154/98 MiB**。
B64非保存 **34.654/100 MiB**、保存 **38.216/100 MiB**でv46と同じ。
Dense49.002/106、49.502/106MiB以下。新しい保存payloadは増えていない。

bank区間はzero CTAを含むため約0.0205→0.0246msへ増え、
別のdelta/border zero約0.0082msが空markerへ変わる。費用を単に除外したものではない。
完全stepは全起動・構築・L2・optimizerと、Graph capture/replay peak込み。
前述のL4長期非定常性を踏まえ、短期主測定だけで普遍的な改善とはしない。
検証済みの保存+融合一設定をG4で129rounds比較する。

source SHA256 `4446fb6c50681169d38d9a236169aa191b641c5762b3b935169f205c987a3413`、
result SHA256 `10d24afdba089346a5bb7f93221ca69de00b608fb5c7450e3a5dc7f7652f756f`。
source1662/manifest68、archive/manifest/Git bytes/return0/timeoutなし/cleanup ackを照合した。
372 runtime-fileのsdist/wheel byte一致も確認。公開/defaultは変更しない。


### v47 G4 129round比較: 小幅な追加改善、目標は未達

job `colabjob-5d4ff77cc01b47ccab5a87889dbfed6d`、source `ac06d717`。
9件のraw bank/zero検査と、両primaryの3状態全shape独立FP64 Y/dX/全4P gate、
全52,428 sigma更新、同条件初期hash一致がPASS。

|G4条件|v44 ms|v46保存 ms|v47保存+融合 ms|Dense ms|v47/Dense|
|---|---:|---:|---:|---:|---:|
|B32 mixed|0.351690|0.341910|0.339690|0.066160|5.13x|
|B64 mixed|0.541630|0.522770|0.515100|0.074940|6.87x|

v46比0.65%/1.47%短縮の候補改善であり、大きな構造改善ではない。
allocated/reservedはB32 34.153809/98MiB、B64 38.215942/100MiBで、
Dense 49.002441/106、49.502441/106MiB以下。速度2x目標は未達、goal active。
bank spanはzero CTA込み0.0119→0.0139ms、独立delta zero spanは約0.0057msから
空marker約0.000448msへ移る。spanの包含変更と未計装完全stepを区別する。
G4全jobのmaxabs最大はY 3.300e-5、dX 3.330e-5、全4P 1.309e-4。
relL2最大は1.731e-6/1.774e-6/6.385e-7で、4e-4基準を維持。
source archive SHA256 `1ec2e505da4debc32e07232a848c4a64309b33535bfb8944e1df2ba4875e8334`、
result archive SHA256 `9334140a657f9803a81a160a5ad5190a8585cd80b24c2188e110e61d1546b774`。
1664 sourceファイル、manifest30件、git byte一致、return0、cleanup ackを検証した。
全所有GPU停止、CLI active sessionなし。CI ac06 CPU PASS。

## v48: batch4物理配置とSM90以降のvector RED（検証前）

v47保存+初期化融合を継承し、H/G tileを16 atoms ×32 batch、
BlockedLayout sizePerThread=[1,4], threadsPerWarp=[4,8], warpsPerCTA=[4,1]へ変える。
実batchが32未満でも論理BBを32以上にし、各threadが異なる4要素を担当する。
旧v37 lane4のsizePerThread=[1,1]とは異なる。動的native分岐やsigma値は量子化しない。

同じ配置のscalar atomic対照と、SM90以降でvector REDを有効にする候補を用意する。
inline asmのpack順を仮定せず、4アドレスの連続性・16byte整列・全mask trueを
実行時に確認する。成立時だけred.relaxed.gpu.global.add.v4.f32を使用し、
それ以外は各要素のmaskに従ったscalar REDへ戻す。古いSMでは全てscalar。
FP32各成分の加算、FP64 metadata/最終pullback、単一floorのexact sampled L2、
支持・task幅stop-gradient・入力/全atom勾配の4e-4 gateは維持する。
新規GPU tensorや追加scratchはないが、register数・spillと実ピークは実測前。

raw binary-rational衝突/端数/アドレス不揃い/重複アドレス/Graph再実行を検査し、
vector経路の実行フラグとPTXを確認する。続いて独立FP64勾配・動的sigma・retainedを検査。
完全stepはG4の両primary一設定だけで、v47・batch4 scalar・batch4 vector・Denseを比較する。
公式仕様: https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red

### 別案: 3階差分からのW構築（CPU数値設計のみ、GPU未実装）

同じsampled quadratic B-spline qは区分2次多項式なので、d=Δ³qは各節点付近のみ非零。
因果prefix Pに対しq=P³d、従ってW=P_o³[sum_a A_a d_o d_i^T /D_a](P_i³)^T。
D=max(||q_o||2 ||q_i||2, eps)の単一floorを保ち、U=q_o/sqrt(D), V=q_i/sqrt(D)とすると
||U||2||V||2<=1である。差分係数の符号打消しを浮動小数点scanで失わないよう、
52bit係数、32bit振幅、3 signed64 words・基数2^48の整数表現をCPUで試した。
下位2wordをatomごとにbalanced carryしてから加え、各prefix段でもcarryを正規化する。
重複impulse位置を統合し、tiny/危険な値は元の直接式へ戻す。

ignored `integer-splitnorm-difference-cpu.py/.json`の843軸検査とN32/A64/B3で、
整数word scanは任意精度整数scanと一致。W maxabs 9.017e-11、Y 4.378e-10、dX 2.628e-10、
物理振幅/中心勾配2.246e-12。ただし全Polar4勾配の大規模gateではなく、GPU性能/peakの証拠でもない。
一枚N1024の3word payloadは24MiBで、独立測定したpeakではない。
中間overflow上限の検証、整数積/逆prefix VJP、fallback、全shape独立FP64 gate、実速度は未実装。
source SHA256 `90d59892588e5a0e946cb74916158393e4c5e7519a9c40e266133a0c4768d855`。

CPU validation: 2012 passed /4890 skipped /18既存warning (39.44s)。両case宣言と変更対象Ruff PASS。
全src Ruffには既存torus_profile_productの15件があるため、全tree Ruff PASSとは記さない。
wheel/sdist build成功。GPU検査はraw8件+独立参照等34件、primary2条件を予定。


### v48 G4実測: vector REDは精度を保つが遅い（不採用）

job `colabjob-c7f9175217ac4b5ab936251ed75ebf28`、source
`c4c69a837627bbe0b7cc5d47ef8f542de1fc4f56`。
raw8件(4.42s)、参照/動的sigma/Graph/Count all-partial-zero/retained等34件(62.34s)がPASS。
両primary・全planの3状態全shape独立FP64 Y/dX/全4P、全52,428sigma更新、
同条件P/X/target hash一致もPASS。旧4e-4基準・主入力振幅・optimizerを保持する。
G4実機/ドライバ580.82.07、Torch2.11.0+cu130/CUDA13/Triton3.6.0。

|G4条件,65rounds|v47最速対照 ms|batch4 scalar ms|batch4 vector ms|Dense ms|
|---|---:|---:|---:|---:|
|B32 mixed|0.336390|0.343080|0.347980|0.066580|
|B64 mixed|0.504810|0.517280|0.523500|0.075490|

batch4配置だけで1.99%/2.47%遅く、vector化はさらに遅い。
v47対照が約5.05x/6.69x Denseで、目標未達のまま。v48は最速版へ採用しない。
CST全候補のallocated/reservedはB32 34.153809/98MiB、B64 38.216309/100MiBで一致。
Denseは49.002441/106、49.502441/106MiB。新規tensorを作らないことと実peakの一致を別々に確認した。

コンパイル結果でsizePerThread=[1,4]を確認し、vector版のstream両方向にv4 RED18命令がある。
raw検査では整列・全maskのquadでvectorフラグが立ち、offset/非連続/alias/短い端数はscalarへ戻る。
従って未使用vector実装を測った結果ではない。ただし実streamのvector実行率counterは採取していない。
stream forward register数はv47 39、batch4 scalar48/vector44、backwardは54→168、全てspill0/local load-storeなし。
B32診断spanはforward0.047840→0.053984ms、backward0.070368→scalar0.074464/vector0.078560ms。
register増加と時間悪化は観測できたが、占有率counterやキャッシュmissを直接測定した原因確定ではない。
scanは両方向各0.015968msで変わらず、vector命令だけでは全経路を軽くできない。

source archive SHA256 `9be0dd7b930943d9113b009f5ccec3982fa927e6386aa8156fdc7bbba46fd782`、
result archive SHA256 `f382ccbb72bedfae1fa0f1e0095ec26535e735a7b294a1b66236798f690bdf25`。
1667sourceファイル、manifest31件、git byte一致、return0/no-timeout、cleanup ackを検証。
`verify-g4.py`と全G4 job対象の`audit-g4.py`を実行し、L4 registryを上書きしていない。
supervisor52725 exit0、所有GPU全停止・CLI active sessionなし。

### 次の数値設計: int64の3回prefixでH/G bankを縮める（CPUのみ）

W全体を作る前節の差分案とは別に、H生成だけを同じ代数で置き換えられるか調べた。
q=P³dなのでH=q^T X=d^T(P^T)³X。d=Δ³qとその中心微分を区分多項式から求める。
qの係数を52bit整数へ、各batchのXをmax|X|に応じた34bit尺度へ丸め、
後ろ向きprefixを3回int64で実行する。queryはsigned128積/和を完了してからFP64へ戻す。
値が大きいprefixを先に浮動小数点へ変換しないことで、境界の大きい符号打消しを保つ。

N<=1024ではmax|X_int|<=2^34、全prefix段の保守上限は
2^34*C(N+2,3)<2^63。scan内部部分和も絶対値和の上限で覆う。
係数と各積/部分和について別途signed64/signed128内のassertを入れた。
Tiny support(<3)は直接式へ戻す。L2は真のsampled norm productと単一floorのままで、
積分近似やsigma量子化は行わない。Y/dXの注入を変更しない想定で、入力bankの読み取りのみを代える。

ignored `integer-h-cpu.py/.json`はN128/A128/B3およびN1024/A128/B32、
各3step・floor1e-6/1000、atom別/step別sigma、空/端/zero ampを含む12状態。
独立Torch FP64 autogradに対しY maxabs4.154e-11、dX2.848e-11、全Polar4P4.865e-10。
H query maxabs3.518e-10、中心微分1.575e-9。最大整数積/部分和113bitだった。
入力尺度0/1e-15/1e-6/1/1e6/1e15のquery relL2は最大5.811e-11。
N1024/B32のbank payloadは262,272bytes(値256KiB+尺度128bytes)で、
従来の3.562MiB input bankより小さい見積もり。ただしGPU peak/速度の証拠ではない。

このCPU検査は主条件A52,428の独立GPU gateを代替しない。GPU実装、128bit積/和のコスト、
係数生成/保存とnative fallback、Graph/retained寿命、完全stepとallocated/reservedは全て未検証。
次はH/G bankとquery単体を実装し、数学・精度を確認してから同条件完全stepへ進む。
source SHA256 `27c47dfca1b2b7629f18d356bdac6c61a4010caa1bb62ecda6b49aefe84988eb`、
再利用軸式source SHA256 `90d59892588e5a0e946cb74916158393e4c5e7519a9c40e266133a0c4768d855`。


## v49: 整数H/G queryと小さい共有bank（検証前）

v47の注入・FP32 dyadic delta/restore・Meta/L2・native tiny経路は保持し、
X/dYからのquery bankのみint64 reverse triple prefixへ変える。N<=1024の研究recipe。
最大絶対値からFP32の指数bitsを読み、normal/subnormal/zeroを含めてbatch別の整数尺度を決める。
max|X_int|<2^34を保ち、各scan段の絶対値和上限2^34*C(N+2,3)<2^63を満たす。
値と尺度を同じint64 tensorの(N+1)×physical_batchへ保存する。

係数をqueryごとにFP64計算し直さず、Metaのradius/centerから52bit整数の区分2次係数を
前計算してforward snapshotへ含める。sigmaは量子化せず、stepごとに更新する。
元Coの先頭36A FP32要素の後に18A int64係数と8A int32 cutsを置く。
Co全体は80A FP32 storage、追加payload176A bytes(約8.8MiB/A52428)。
新規ctx slotは増やさず、1つのimmutable Co snapshotへ含める。
中心微分の係数は同じ整数多項式から−c1/h,−2c2/hと導く。

H=(Δ³q)^T reverse_prefix³(X)をsigned128の積/和で評価する。
負の小さい残差はhigh/lowを別々に浮動小数点へ変換すると失われるため、
整数の絶対値を取ってからFP64へ戻す。次数ごとの開始/終了impulseを正確に足す。
各pieceの係数・区間長から保守上限を検査し、unsafeならclampした別のkernelへ変えず、
Raw X/dYに対する同じprofileのFP64直接queryへ戻す。tiny native routingも保持。
入力bankは同一forwardのbackwardまで保存でき、chunk間/retained間で上書きしない。
新forwardの数値を前stepから再利用することはない。

raw GPU予定: int64 bankのbit一致(subnormal/large/zero/padding含む)4件、
signed128正負の小さい残差1件、query/prime/unsafe fallback8件。
共有の全勾配・動的sigma・Count all/partial/zero・retained検査33件とCPU metadata1件を合わせる。
その後、主N1024/A52428/B32/B64 mixedをv47/Denseと同条件で比較する。
準備・係数構築・L2・更新込みの時間とallocated/reservedが必要であり、
bank payloadだけを省メモリ達成とは扱わない。GPU速度/全shape精度はまだ未検証。

CPU2013 passed /4936 skipped /18既存warning (40.01s)、変更対象Ruff・両case宣言・wheel/sdist build PASS。
GPUは47件中metadata1件を除く46件と、主条件2caseを同じsnapshotから検証する。


v49初回job `colabjob-a63b95bf20ca46d88dd7a8aa1feb71ad`、source9de1a0abはraw最初でコンパイル失敗。
bank/zeroの別branchで同じj名をN要素/1024要素へ使い、Triton SSA型が一致しなかった。
演算検証/性能測定には到達していない。独立のzero index名へ分離し、zero要素数0の静的guardも追加する。
source/result/manifest/git bytes/return1/no-timeout/cleanup ackをfailed-jobsへ厳密保存、primary表から除外した。
supervisor90362 exit0、修正を明示的な新snapshotで再提出する。精度基準の変更はない。


job `colabjob-bbfef58a5d454a20b603c49c5485217b` (8c3767ec)でbank4件とsigned128残差1件PASS。
queryのfallback内でint64のx名をFP64へ再代入したため、loop-carried SSA型エラーで停止。
fallbackの入力値とsite indexを別名へ分離する。数式・量子化bits・基準は変えない。
5PASS/1compileFAIL、性能なし、source/result/git/manifest/cleanupを厳密保存して除外。
supervisor78435 exit0。次snapshotで同じ全検査を再実行する。


### v49 G4: bank構築は短縮、query/係数構築増で不採用

job `colabjob-2cff4e58e91d4055865d3360ec0ff69c`、source
`caca65127feaf7f2b8b820504313bb7fb58f6501`。raw13件(8.68s)、共有参照33件(84.52s)PASS。
両primaryの全plan・3状態全shape独立FP64 Y/dX/全4P、全52,428sigma更新、初期P/X/target hash一致もPASS。
同じ4e-4 maxabs/relL2基準、sampled単一floor L2、task幅stop-gradient、AdamW/Polar更新を保持した。

|G4,65rounds|v47 ms|v49整数query ms|Dense ms|v49 allocated/reserved MiB|
|---|---:|---:|---:|---:|
|B32 mixed|0.333270|0.597080|0.065810|35.790039/90|
|B64 mixed|0.502040|0.978420|0.074490|36.540527/94|

v49はv47比1.79x/1.95x遅い。v47自体もDense比約5.06x/6.74xで、速度目標は未達。
v47 allocated/reservedは34.153809/98、38.216309/100MiB。
v49はB64で両指標が下がるが、B32 allocatedは1.636MiB増える（reservedは減る）。
Dense 49.002441/106、49.502441/106MiBよりは両方低いが、速度とメモリの同時改善ではない。
旧最速v47を保持し、v49は採用しない。bankだけを速く/小さくしても全stepは改善しないという負の結果。

B32計装span(ms): input bank 0.013920→0.007776、adjoint bank0.015072→0.008928。
一方、係数準備0.022624→0.071776、forward query+注入0.047840→0.068288、
backward query+全atom勾配+注入0.070368→0.273120。scanは両方向とも0.015968で同じ。
backward registersは54→168、forward39→96、spill0/local load-storeなし。
これは計装spanであり、未計装の完全stepと分けて記録する。
整数積・読み取り・低占有率のどれが主因かを個別に特定するcounter/単体ablationは未採取。

整数query候補の独立参照誤差はB32 Y2.381e-5/dX2.020e-5/全4P1.474e-5、
B64 Y2.290e-5/dX2.136e-5/全4P1.953e-5(maxabs)。
全4P relL2は約1e-7で、速度は負でもこの数値精度は確認できた。
rawbankのsubnormal/large/zero/padding、正負signed128小残差、直接query fallbackも検査している。

source archive SHA256 `6d0d903cfdc056875cdb8dcf96b4c04c98bdf3f57afffd3a364310f2605896f1`、
result archive SHA256 `b193cb24243b453e13601fb94089a580f6ee3c3323cbbeb611d9c5a65618518c`。
1670source files、manifest25件、Git byte一致、return0/no-timeout、cleanup ackを厳密検証。
全6件G4 jobをauditし、L4 registryは変更していない。CPU CI caca PASS。
supervisor16401 exit0、全所有GPU停止、CLI active sessionなし。

### 次の圧縮の数式確認（CPUのみ）

現v49 queryは3piecesの両端各3impulse、計18点をHにもHmuにも使う。
同じ節点の開始/終了を合算すればqの3階差分は最大12点。
qmuは区分1次なので、その2階差分は最大8点であり、
Hmu=(Δ²qmu)^T reverse_prefix²(X)とすれば3階prefixを使う必要がない。
H/Hmuを合わせると36→20積和だが、これはGPU速度の証拠ではない。

`integer-query-events-cpu.py/.json`ではN64/128/1024の384軸例で整数の再構成が完全一致。
H maxabs1.863e-9、Hmu1.394e-9。係数整数はv49と同じq polynomialから微分している。
2階/3階bankのpayloadはN1024/B32で524,544bytes、保存event係数を全部持つ案は
追加352A bytes(17.6MiB/A52428)となる。現v49より係数payloadが増えるため、
単純に36→20積和だけで採用してはならない。GPU未実装、全shape/all4P/完全step未検証。
source SHA256 `94137a89377c5c2b277dd77863f11cba39959fee7b6f8faf4a1be4380c2755ee`。

さらに、整数の128bit積和をqueryの必須条件とせず、FP32二成分でprefix/queryの精度を保つ案もある。
出力注入は成分別FP32 atomicだけでは和の丸め誤差を補えないため、単純なhi/lo分割では代用できない。
unit-axis factorization U=q_o/||q_o||, V=q_i/||q_i||,
λ=A ||q_o|| ||q_i||/max(||q_o|| ||q_i||,eps)なら同じ単一floor演算W=ΣλUV^Tとなる。
入力側H生成、係数構築、出力の正確な加算を一緒に考える必要があり、
unit profileの差分とbatch別尺度の重みを2word整数へ足してprefixで戻す設計は未検証の候補。
現時点でこの案のGPU実装や速度・peak・精度の主張はしない。


## v50: unit軸のFP32 pair queryと整数deltaによる出力集約（検証前）

q_o,q_iをそれぞれsampled L2 norm n_o,n_iで割ったU,Vへ分け、
λ=A*n_o*n_i/max(n_o*n_i,eps)とすると、元の演算はW=ΣλUV^Tのまま。
軸ごとにfloorを入れない。既存Metaのinverse denominatorと条件付きGammaを保持し、
norm kernelが既に求めたn_o,n_iをsuffixへ保存する。Metaは14A FP64 storageで、
先頭12Aは従来のstride12 layout、末尾2Aはnorm用SoA。旧recipeは従来12Aのまま。
振幅/幅・中心・task幅stop-gradient・単一floor・Polar更新は同じ定義を使う。

CPU `unit-two-word-cpu.py/.json`ではN128/A128/B3、N1024/A128/B32、
各3step・floor1e-6/1000の12状態を独立Torch FP64 autogradへ照合した。
2成分FP32 prefix/queryと、52bit unit多項式×batch尺度34bit重みの正確な整数積/scanを使い、
最終Y/dX/全4PをFP32へ丸めてmaxabs9.105e-8/1.031e-7/1.818e-6。
H/Hmuは2.894e-7/3.899e-9。2word scanはPython任意精度整数と完全一致。
これはA52428のGPU gateではなく、GPU時間/peakを示さない。
source SHA256 `b06335aa581122dc82b683e865a8423353b82b34bcbf41545fb8fcb2c4eaaa14`。

新GPU候補は入力X/dYをbatch別の2冪で正規化し、2階/3階reverse prefixを
FP32二成分で保存する。TwoSum/FMA残差で精度を保ち、Hは12点、Hmuは8点でqueryする。
出力は単純なFP32 hi/lo atomicにせず、各atomの重みを一度だけ整数化し、
unit profileの差分との正確な積を、balanced base2^48のsigned64 low/highへ足す。
3回prefixの各段でcarryを戻してからFP64へ変換し、元のbatch尺度へ戻す。

同じatomの重複位置は係数構築で合算し、各pointへatomあたり一度だけ加える。
これにより低wordの加算上限を保つ。N<=1024、A<=60000を数値支持範囲とする。
低wordはatomごとに[-2^47,2^47)へbalanced化するため、atomic和とscan内部部分和がsigned64内。
高wordはunit vectorの差分のL1上限とbatchの入力normから上限を取る。
細かい係数の独立丸めでcompact supportを壊さず、連続sigmaを量子化しない。

source順の全atomを処理し、nativeはcutoff・<=2支持点・unsafe係数だけのfresh compactionにする。
full width sort/parameter permutationは行わず、nativeIDs/countもimmutable forward snapshotへ含める。
unsafeなatomは従来の同じFP64 native式で計算する。Coは44A int64 storageで
40A event係数+8A int32 cuts (352A bytes)。新規Meta norm suffixや整数delta/shardsもpeak対象。

forward input bankは必要なら同じframeのbackwardまで保存し、chunk/retained間では上書きしない。
構築・fresh compaction・L2・query/注入・整数restore・全4P・optimizerを全て含めて評価する。
CPU2014 passed /4974 skipped /18既存warning (40.38s)、変更対象Ruff、両case宣言とbuild PASS。
GPU raw5件+共有全勾配/Graph/retained等33件の後、主B32/B64をv47/Denseと比較する。
速度とallocated/reservedはまだ未検証で、採用や達成の主張はしない。


v50初回job `colabjob-658a7b8162724c4fa21568a169a2a9f6` (78c05c9d): raw5件PASS(10.69s)。
モデル側で共通_prepareの4戻り値を3変数へ受けた接続エラーで停止し、性能測定なし。
従来と同じ4値の受け取りへ修正する。計装norm_replayは共通_prepareが記録済みのため、
新executorの空markerで上書きしない。数式/精度/入力量/基準を変えず再検証する。
source/result/manifest/git bytes/return1/no-timeout/cleanup ackをfailed-jobsへ厳密保存。
supervisor29380 exit0で旧jobはterminal。次snapshotを明示的に再提出する。


### v50精度gateの失敗とquery/縮約の切り分け

job `colabjob-f43395cab66a4586aedc27ccab0097c8`、source3ef46929でraw5件と共有12件がPASS後、
N128/A33/B3のP-only勾配で停止。torch assertが示したfield20,2の差は0.000601446。
診断で確認した全要素maxabsはfield20,3の0.000994889で、4e-4を超える。
primary性能へ進めず、基準・入力を変えずfailed archiveを保存した。

同sourceの診断job `colabjob-fc769c3d6e12446abc0d41d0e4014520`は、
H/Hmu/G/GmuをFP64のtensorへ書き出すと、Metaから独立に評価したdotとの誤差が
2.97e-11/3.15e-12/6.51e-11/1.80e-11だった。
その値を保存する診断kernelの全4Pはmaxabs4.316e-7で、通常kernelとの差は0.000994921。
Y自体は5.208e-8。消費側が直ちにFP32へcastする場合のprecision loweringを疑う証拠であり、
特定のLLVM passやPTXAS変換を原因と確定はしていない。
診断はreturn0だが、primary性能証拠ではない。厳密hash/source/manifest/cleanup検証後、
`verified-g4-diagnostics.json`へ分け、accepted `verified-g4.json`と混ぜない。

queryのF64戻り値にbitwise mov.b64のidentityを置き、消費側から精度を保持させる候補を用意する。
数値演算やFP32縮約式を変えず、同じ失敗testと全gateで再検証する。まだ修正成功の主張はしない。
同時に、整数重み尺度を入力normだけでなくlive amplitude_maxにも合わせる。
amp上限を既存decoderが同じlaunch中にsnapshotへ書き、保存bank/再計算bank/adjoint bankが
同じforwardのamp snapshotを使う。追加clone kernelを避け、live scalar変更後のretained VJPも検査する。
旧recipeのoptional pointerはNoneで、元の出力は変えない。

CPU2014 passed/4978 skipped/18既存warning (39.51s)、変更対象Ruff PASS。
新しいraw bank amplitude1/9とlive amplitude/retained検査を追加し、raw7件+共有35件を予定する。
追加の高amplitude検査はFP64 norm/contraction設定を使う。標準のFP32 core検査/primary入力・4e-4基準は維持する。

job `colabjob-473a909761044c4397b0c7a38ffb731a`、sourcef43d7decでraw7件PASS(11.32s)、
共有18件PASS。以前失敗したP-only勾配も同じ基準で通過した。
次のcutoff Graph検査では最初のreplayのY/dX/全4P照合を通過した後、
旧stream_layoutのcount参照がKeyErrorとなった。新方式はfresh compactionなので、
テストのcount参照だけをunit_native_countへ対応する。期待native数のassertと精度gateは維持する。
性能測定には到達していない。return1/no-timeout、manifest/source/result hashとcleanup ackを
failed-jobsへ厳密保存し、成功/性能証拠に含めない。runtimeはf43d7decと同一のまま再提出する。

### v50 G4: 精度・メモリを満たすが速度は悪化、不採用

source `0fe9cd0c0bbbb98382633c30f68b9355e535d5b8`、job
`colabjob-0d852c2f3a60453ea86b01a27d532206`でraw7件PASS(10.82s)、共有35件PASS(90.88s)。
B32/B64それぞれ全planの3状態full-shape独立FP64 Y/dX/全4P照合、全52,428sigma更新、
P/X/target hash一致がPASS。v50全4P maxabsはB32 2.049e-5/B64 2.342e-5。
元の4e-4 maxabs/relL2 gateを維持した。

| G4,65rounds | v47 ms | v50 ms | Dense ms | v50 allocated/reserved MiB |
|---|---:|---:|---:|---:|
|B32 mixed|.337040|1.081730|.066180|34.120/94|
|B64 mixed|.505650|1.969150|.075070|35.120/96|

v50はv47より3.21x/3.89x遅い。両メモリ指標はDense49.002/106、49.502/106MiB以下だが、
速度目標に反するので採用しない。v47の同run対照もDense5.09x/6.74xで目標未達。
B32計装span(ms)は係数.022624→.064448、forward.047808→.183840、
backward.070336→.648800。scanは片側.015968→約.0098へ短縮したが、他の増加を補えない。
v50 forward/backward registers40/48、spill0、local load/storeなし。特定のALU/cache stallの
ハードウェアcounterは測っていないため、個々の命令を単独原因と確定しない。

source archive SHA256 `91d019c522f7dc9473da6c4d3cdb6ab1285c9b9b42d527940b70b63bfe10bf61`、
result archive `0158bedc2d13b1af86f0bb5e1a516c171df33b05ccd09cb5501f66dd8078cd99`。
1675source files/25manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4 registry/auditは7job、診断専用とL4は別。supervisor14722 exit0、所有GPU停止済み。
source CPU CI PASS(2m15s)。配布artifact377runtime filesも同sourceへbyte一致。

## v51: 固定表によるFP64幅復元のrange reduction（検証前）

v47のH方式・局所bank・全4P・sampled L2を保ち、毎stepのPolar幅復元
`sigma=lower*exp(alpha*log(upper/lower))`の数値評価だけを固定表へ置き換える候補。
表は128個の `v_i=1/(1+i/128)`、`-log(v_i)`、`exp(i*log(2)/128)`、計3072bytes。
各値は初回にGPUのFP64 libdeviceで作り、atom/入力/幅によらず使えるimmutable定数とする。
CPU由来tensorのcapture中コピーはしない。更新するsigma自体のcacheや量子化はしない。

正規ratioを `ratio=2^e*m` として `f=fma(m,v_i,-1)` を取ると、
`log(ratio)=e*log(2)-log(v_i)+log1p(f)`。
丸めた逆数そのもののlogを表へ入れるので、逆数丸めの偏りを残さない。
`|f|<=1/128+ulp`でdegree7のlog1p剰余は2e-18未満。
`x=alpha*log(ratio)/log(2)`、`k=round(128*x)`、`j=k mod128`、`n=floor(k/128)`、
`r=(x-k/128)*log(2)`として `exp(alpha*log(ratio))=2^n*tableExp[j]*exp(r)`。
`|r|<=log(2)/256`でdegree6 exp剰余は3e-22未満。
演算はFP64/FMAを使い、normal finite ratioが[2^-64,2^64]以外なら従来libdevice式へ戻す。
元のratio overflow fallback・live scalar読み取り・clip・amplitude Jacobian・width stop-gradientは保持する。

CPU601,935条件（random ratio/alpha、bin境界、alpha0/1等）でFP64元式との最大相対差1.022e-14。
これはGPUの精度/速度証拠ではない。raw範囲・live Graph bounds/fallback・旧decoderのbitwise契約、
共有全4P/retained/動的sigma gateの後、同じ主B32/B64条件でv47/Denseと完全step比較する。
公開dispatcherは変更しない。

v51初回job `colabjob-2cdb8f5dff054be49d7bb0a4860a05eb`、sourcec73b64c8で旧decoder12件PASS後、
新raw幅Graph照合が相対差4.976e-9となり、3e-13基準で停止した。モデル/性能測定には進めない。
CPUとの差はGPU式の非2冪定数の型推論を疑い、表初期化・多項式係数・log2定数を明示FP64 fullへ統一する。
原因の確定や修正成功は再検証前に主張しない。gateは変更しない。失敗archive/hash/source/cleanupを厳密保存した。

sourcee5052b44のjob `colabjob-f080d58a62874d818f8068ba9de6e4da`でraw13件PASS。
明示FP64定数にした幅Graph照合は元の3e-13基準で通過した。
最後のraw power範囲testは存在しないGluon cdiv helperを呼んだテストコードのミスで停止。
Pythonの整数切上げ除算へ直す。runtimeは変更せず、全検査を再実行する。
return1/no-timeout/source/manifest/hash/cleanupをfailed archiveへ保存、性能証拠に含めない。

### v51 G4: 全精度PASS、速度差は0.1%前後で採用根拠なし

source `8b5cd1967e715c0e46bf49ee9c1b330b02f058a4`、job
`colabjob-180eaaa2a61445db91061269c78631f8`でraw14件、共有33件PASS。
両primary全planの3状態full-shape独立FP64 Y/dX/全4P、全52,428sigma更新、初期P/X/target hash一致PASS。
全4P maxabs1.362e-4、relL2 6.380e-7で元の4e-4基準以内。

| G4,65rounds | v47 ms | v51 table ms | Dense ms | v51 allocated/reserved MiB |
|---|---:|---:|---:|---:|
|B32 mixed|.332249|.332010|.066330|34.157/98|
|B64 mixed|.498930|.498340|.075310|38.219/100|

差は0.072%/0.118%で、明確な高速化の根拠としない。メモリは固定表3072bytes分だけ増える。
B32 decode spanは両方.019616ms、prepareも両方.045664ms。元のlog/expだけを置き換えても
全stepの課題を解かない。v51は採用せず、v47を比較基準として保持する。
両メモリ指標はDense以下だが、速度は約5.00x/6.62xで目標未達。
source archive `5b1a9ec424165b88e039e5a95c6685c98b0df256059dfa6e1a746cb8e81b2508`、
result archive `f9489da49252be969e4d1be0cd5cb294fe36f3b969c1380d3513154b535ed35b`。
1678source files/25manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4は8job、失敗/診断/L4は別。所有GPU停止済み。runtime配布378filesのbyte一致もPASS。

4箇所のjumpだけでqueryする重複局所block案もCPUで数値screenした。
block幅>=4radius、stride=幅/2にすると支持域全体を一つのblockへ入れられ、内部の補正2箇所を省ける。
ただし広いblockに対する係数/FP32 prefixの相殺誤差が増える。
N128/B3とN1024/B32、各A128の6状態で、FMA queryでも全4P maxabs6.056e-4、
FP64でjump係数を作ってFP32保存する案でも5.258e-4となった。元の4e-4基準を満たさないためGPUへ進めない。
このscreenはqueryを調べ、出力側は直接乗算しており、atomic restoreや速度/peakを示さない。
独立Torch oracleはvector_norm積のone-floorを使い、空支持での有限性もassertする。

## v52: 幅バケットを作らないfresh二分割（検証前）

H/G、局所bank、係数、native計算、L2、全4Pはv47と同じまま、振り分けだけを変更する候補。
同じFP64 cutoffとclipped支持点<=2の判定を用い、CTA内prefixでnative/broad順位を求める。
各CTAはnativeをID配列の先頭側、broadを末尾側から予約する。両総数の和がAなので予約域は重ならず、
総native数を待つ二回目のscatterを不要にする。幅のsort/histogram/cursor/persistent order cloneを省く。
新規ID/countはforwardごとに確保し、retained backwardへimmutable snapshotを渡す。
Parameterやoptimizer状態の行は動かさない。新しいsigmaでは毎回分類し直す。
原子の実行順は変わるので、元のfull-shape gateと完全step/peakを通過して初めて性能候補と扱う。

v52 CPU2016 passed/5049 skipped/18既存warning(39.62s)、変更対象Ruff、両case宣言、build PASS。
raw3形状のlive分類/permutation/Graph再生と共有33件の後、主B32/B64をv47/Denseと比較する。

### v52 G4: 振り分け時間とallocatedが減少、速度目標は未達

source `81734c29b9c14bcfe06dd6fc90a3e6e45ab1b6e2`、job
`colabjob-ed2fadb322624440ba8aa839de1e3c91`でraw3件PASS(4.60s)、共有33件PASS(55.63s)。
両primaryの全plan・3状態full-shape独立FP64 Y/dX/全4P、全52,428sigma更新、P/X/target hash一致PASS。
全4P maxabs1.077e-4、relL2 6.366e-7で元の4e-4基準以内。

|G4,65rounds|v47 ms|v52 ms|Dense ms|v52 allocated/reserved MiB|
|---|---:|---:|---:|---:|
|B32 mixed|.335100|.326180|.066830|33.544/98|
|B64 mixed|.504560|.495039|.075630|37.607/100|

同run対照より2.66%/1.89%短縮、allocatedは各.609375MiB減り、reservedは同じ。
B32group span .017984→.007712ms、main forward/backwardは.047840/.070368msで同じ。
振り分け改善の候補だが、1回の比較であり129round再測定で確認する。
Dense比4.88x/6.55x、両peakはDense以下だが速度目標は未達。公開dispatcherは変更しない。
source archive `dc9c6bef0eb557cb239be861f96fea6cc83406e22e93b23b9376d87644568d67`、
result archive `8d7f2e9835a0503934beadf74c527a9ac19a036fec691cce2c47af2a7b1c3cd3`。
1681source files/25manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4は9job、失敗/診断/L4は別。所有GPU停止済み。配布379runtime filesのbyte一致PASS。

再測定では同じv52のB64 chunk64も1案だけ比較する。現在のB64 peakとDenseの差が約12MiBあるため、
大きい作業域でchunk境界のkernel起動を減らせるかを調べる。数学/勾配/幅の更新は同じ。
B32はv47/v52、B64はv47/v52/chunk64を各129round、元の全shape3状態gateを通して比較する。

### v52再測定とB64一括処理: 高速化とreserved制約の分離

source `5d07445256c50dab09964d625d805e9f6e59f4e7`、job
`colabjob-ef317beda8bd4b4db353f735430d0ea0`でraw3件、共有43件PASS(57.51s)。
全planの3状態full-shape独立FP64 Y/dX/全4P、全52,428sigma更新、初期hash一致もPASS。

|G4,129rounds|v47 ms|v52 chunk32 ms|v52 chunk64 ms|Dense ms|
|---|---:|---:|---:|---:|
|B32 mixed|.338100|.328440|—|.066240|
|B64 mixed|.510630|.500650|.409600|.074910|

fresh振り分けの短縮は2.86%/1.95%で再確認できた。
chunk64はB64をchunk32より18.19%短縮したが、allocated/reserved45.688/140MiBとなった。
Dense49.502/106MiBに対しallocatedは下回るがreservedを満たさないため、条件達成として採用しない。
chunk32は33.544/98と37.607/100MiBを維持。chunk64の全4P maxabs1.281e-4、relL2 6.165e-7。
source archive `6bcdce31e905795302377e4cafb26cf8a04b0872df34187758f471cfca75203f`、
result archive `476675bcb24f6c3b196ec79a54c103804e1794e18b2f6fae5e96fdf751a3e663`。
1683source files/28manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4は10job、診断/失敗/L4は別。所有GPU停止済み。

## v53: 同じ寿命のforward状態を一つのarenaに確保（検証前）

v52と同じ数式/係数構築/読み取り/atomicsを使い、immutable Meta、ID配列、Pos、Co、half、
border位置、injection code、active maskを256byte境界で一つのuint8 storage内へ配置する。
各viewのdtype/shape/配置は同じ。フレームごとに所有し、後続forward/retained backwardと共有しない。
X/dYの数値bank、delta、勾配、optimizer状態は別のbufferを保つ。

診断用native countはbackward後もroutingに残るためarenaへ入れない。
countがarenaのviewだと古いframe全体を保持して次のframeと重なり、逆にpeakが増えるためである。
長く残るcountは従来どおり独立8bytesで持つ。
固定payloadが同じでも、allocatorのsegment/graph pool再利用でreservedが減るかは未検証。
CPU/CUDAでviewの型・alignment・非重複・別frameとの独立性を確認し、共有33件と主完全stepで比較する。
B32はv52chunk32/v53、B64はv52chunk32/v52chunk64/v53chunk64とDenseを同条件で測る。

v53 CPU2023 passed/5092 skipped/18既存warning(40.64s)、Ruff、両case、build PASS。
その後追加したCUDA viewパラメータ6件はCPUではskipとなり、CPU view/recipe7件を再確認してPASS。
GPUでは既存fresh分類3件+arena view CPU6/CUDA6件、その後実モデル共有33件と両primaryへ進む。

### v53 G4: B32はreserved改善、B64の上限超過は解消せず

初回job `colabjob-4020537352764c15bf516a8726132a71`はG4割当HTTP503でdriver実行前に中断。
VMなし/slot stopped/結果archiveなしを確認し、source1686filesのGit-byte一致とlifecycleを
allocation-failuresへ保存した。実行結果として数えない。pool recover後に明示的に再提出した。

source `80f326a2bb784c083cdf0a68749e6d1f44cff91d`、成功job
`colabjob-1fa4a486f14e4b3cbe632255b081b782`でraw15件PASS(4.56s、GPU9/CPU6)、
共有33件PASS(53.93s)。両primary全planの3状態full-shape独立FP64 Y/dX/全4P、
全52,428sigma更新、P/X/target hash一致PASS。全4P maxabs1.087e-4、relL2 6.291e-7。

|G4,65rounds|v52chunk32 ms|v52chunk64 ms|v53 ms|Dense ms|v53 allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|
|B32 mixed|.325980|—|.324579|.066090|33.305/94|
|B64 mixed|.496319|.404410|.404220|.074930|45.687/138|

B32はv52の33.544/98MiBからreservedが4MiB減った。速度差0.43%は強い改善と扱わない。
B64は140→138MiBに留まり、Dense106MiB以下のreserved条件は満たさない。
allocatedだけを根拠に一括処理を採用しない。通常chunk32の37.607/100MiBは両条件以内。
source archive `1233352d56c37a7d9f813ed19ff755cf96d0ad1732f095d05818a86bf9b473f3`、
result archive `57e2d9303afaf7a8d804f252f484747342b9270c2210ff5c7b850442bcbd3cca`。
1686source files/28manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4は11job、診断/失敗/割当失敗/L4は別。所有GPU停止済み。source CPU CI PASS(2m14s)。

## v54: 数値workspaceをまとめ、forward deltaをadjoint bankへ再利用（検証前）

v53のimmutable係数frameとは別に、InputBank、AdjointBank、DeltaInput、Borderを
一つのFP32 storageに256byte整列して確保する。forwardのDeltaOutputはAdjointBank領域を使う。
Yへのrestoreが終わってからbackwardでその領域へdYのbankを書き、別のDeltaInputへdXを注入する。
保存InputBankはどちらの書き込み領域とも重ならず、不変である。

storageはsave_for_backwardで保存し、backward後まで残るctx属性にTensorを置かない。
retained backwardは同じforward streamならscratchを再利用し、異なるstreamなら従来の独立scratchへ戻す。
同じstreamへの複数host threadのlaunch列はframe別lockでまとめて投入し、異なるdyのbank/queryが混ざらないようにする。
同期的GPU waitや数式変更は加えない。fused Triton zeroを必須にし、保存InputBankと同storageにある
scratchの初期化でPyTorchの共有version counterを増やさない。入力bankの値は変更しない。

full batch<=64、bank padding0、saved input bank、fused bank/zeroが数値arenaの支持条件。
基本設定とsigma cutoff.5の混在native/H、viewの非重複、同時retained全勾配、共有動的gate、
両主条件の完全stepとallocated/reservedで確認する。現段階で性能/メモリ改善は主張しない。

v54 CPU2026 passed/5134 skipped/18既存warning(40.87s)、変更対象Ruff、両case、build PASS。
raw分類/arena CPU/CUDA view19件（GPU11/CPU8）、実モデル共有+同時retained34件の後、
両primaryでv52chunk32/v53chunk64/v54chunk64/Denseを比較する。

v54初回job `colabjob-5c2e4576f7114f909bd6b22996c78e07`、source04a2cdd0でraw19件PASS、
実モデル9件（8つの動的full-gradient条件と標準retained検査）PASS。
追加cutoff optionが共有retained testの明示sigma_cutoff=.2と衝突してTypeErrorで停止した。
runtimeやgateを変えず、追加optionを既存のprepare_warps=2へ変更する。
ALL_QUERY_OPTIONS各testのmodel呼び出しの明示keywordをASTで確認し、prepare_warpsと重複しないことを確認した。
元の共有retained/Graph検査がcutoffを変更し、混在native/Hは引き続き検査する。
return1/no-timeout/source/manifest/hash/cleanupをfailed archiveへ保存し、性能には数えない。

### v54 G4: reserved大幅減、B64はDenseより4MiB多く未採用

source `f59c35968c8398e96e314489b6f02036ed95b746`、job
`colabjob-f9ee5aa8e0994f58a3ff37658cb6c09a`でraw19件PASS(4.57s、GPU11/CPU8)、
共有+同時retained34件PASS(61.14s)。全planの主3状態full-shape独立FP64 Y/dX/全4P、
全52,428sigma更新、初期hash一致PASS。v54全4P maxabs1.167e-4、relL2 6.332e-7。

|G4,65rounds|v52chunk32 ms|v53 ms|v54 ms|Dense ms|v54 allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|
|B32 mixed|.320150|.319070|.322780|.065170|33.253/82|
|B64 mixed|.485770|.396581|.401790|.073660|46.186/110|

B32はreserved94→82MiB、B64は138→110MiBと減ったが、B64はDense106MiBを超える。
allocatedはDense49.502MiB以下。速度はv53に対し1.16%/1.31%遅く、reserved制約も満たさないので
一括処理の採用根拠とはしない。partialなallocated改善を目標達成と扱わない。
source archive `ecd1f27bf5c538e5d2e484b76ecba63b2a2694aecc601fcc440129dcc509c64e`、
result archive `648cfc8df622bd82e7e95239a3246268bb6f8a1407e341417663c3fb00afd49b`。
1688source files/31manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4は12job、診断/失敗/割当失敗/L4は別。所有GPU停止済み。

## v55: immutable状態と数値workspaceを一つのforward allocationへ（検証前）

v54の入力/adjoint/delta/borderの形、値の再計算、scratch再利用、stream/host-thread保護を保ち、
同じforwardのimmutable係数frameと数値workspaceを一つのstorageに確保する候補。
Meta等のdtype別viewの末尾に256byte整列したFP32 numeric領域を置く。診断用countは引き続き別。
読み取り専用領域とmutable scratchは重ならず、元のInputBank/Meta/Coはbackwardで変更しない。

numeric領域の大きさをshapeだけから計画し、_frameは一時listへそのviewを返す。
Tensorをctx属性や長く残るroutingに追加せず、全viewのstorage寿命をsave_for_backwardに合わせる。
独立した二つの大きいallocationを一つにすることでreservedが下がるかを、同じ既存プロトコルで測る。
benchmarkのstream/warmup/capture手順、memory統計の定義、精度基準は変更しない。

v55 CPU2029 passed/5170 skipped/18既存warning(40.46s)、変更対象Ruff、両case、build PASS。
GPUではraw23件（GPU13/CPU10）、共有33件+v54/v55同時retained2件の計35件を先に検査し、
両primaryでv52chunk32/v54chunk64/v55chunk64/Denseを完全stepで比較する。

### v55 G4: 大きな単一確保でreservedが増加、不採用

初回job `colabjob-3fed8f0ba8564a2993b5da2ecade6550`はG4割当HTTP503で実行前に中断。
No VM/No結果とsource1690filesのGit-byte一致をallocation-failuresへ保存し、recover後に明示再提出。
source `70c2b8ac7b8f41e49da2c29d4a61995b8b264e26`、成功job
`colabjob-4241d37f4b274e238c19bb22e24762c4`でraw23件PASS(4.51s、GPU13/CPU10)、
共有+同時retained35件PASS(60.86s)。全planの両主3状態full-shape独立FP64 Y/dX/全4P、
全52,428sigma更新、初期hash一致もPASS。全4P maxabs1.069e-4、relL2 6.266e-7。

|G4,65rounds|v52chunk32 ms|v54 ms|v55 ms|Dense ms|v55 allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|
|B32 mixed|.326599|.329090|.326230|.066240|33.252/96|
|B64 mixed|.495790|.410560|.409540|.074780|46.186/124|

v54のreserved82/110MiBに対し、v55は96/124MiBと各14MiB増えた。
単一allocationにまとめれば必ずreservedが減るわけではないという負の結果。
B64はDense106MiB以下を満たさず採用しない。未測定のallocator内部原因は確定しない。
source archive `7b04dccda52c15225f899cf0c5330752f12d1e48535b437ea1ed52c50337b69a`、
result archive `1bc965fad64acdcec5dc96d158c94d2974a6a5ed1ffcb78b47fa5a50df4f2496`。
1690source files/31manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを厳密検証。
accepted G4は13job、診断/失敗/割当失敗/L4は別。所有GPU停止済み。source CPU CI PASS(2m18s)。

## v56: 局所位置番号のlossless signed16化（検証前）

v54の分離した係数arena+数値arenaを基準に戻し、Pos/BorderPosのみsigned16で保存する。
Aのsource ID配列はsigned64のまま。係数、Meta、bank、重み、正規化、勾配の浮動小数点精度は変えない。
支持範囲のN<=1024、最小局所block4では点配列9727要素、border511要素で、
最大有効index9726/510と無効値-1はsigned16内に正確に収まる。supportsにindex上限も宣言する。

読み込み直後にsigned32へ戻し、batch stride・shard・planeの掛け算やpoint+1は32bitで行う。
16bitのまま9726*64や510*8*64を計算してoverflowさせない。
保存payloadはPosで24A、BorderPosで8A、計32Abytes減る（A52428で約1.600MiB）。
これは予約メモリ改善の証拠ではなく、同じ既存capture/replay統計で確認する。
rawでは最大index/負sentinel/大きなstride/shardを使い、32bit版のquery/injectionとbit一致を要求する。
その後、元の全shape3状態FP64 Y/dX/全4P・動的sigma・retained・同時backward検査と完全stepで比較する。

v56 CPU2033 passed/5206 skipped/18既存warning(42.01s)、Ruff、両case、build PASS。
rawでは既存分類/view19件+compact view CPU/CUDA2件+query/injection bit一致1件の計22件（GPU13/CPU9）、
共有33件+v54/v55/v56同時retained3件の計36件、その後主2条件の完全stepを検査する。
post-run metadataのregister tileは実際のbatch chunk幅に対応させ、compilerのregister数と区別する。
この表記修正は測定済みtime/peakや実行kernelには影響しない。過去raw artifactは変更しない。


v56 source `f4530aff1e0fcab251ac1faeedb0a1a43f9d3b22`のG4初回
`colabjob-f4494cb73e1d4a2e981712c97ed12ded`と明示再試行
`colabjob-af01db36163c4325a685a430f35d19ee`は、両方とも割当Service Unavailableで
VM/driver/resultsなし。1692source filesのGit-byte一致と停止状態を保存し、性能証拠から除外する。
再試行source archiveは`ca57585a0f15d28cec0f46ce45ef8cf0b9b94d78cb20549f3dda7bb18b6fa760`。

同じsourceをL4へ明示提出した`l4job-702985faf84744eba175804442b34472`は、
raw21件PASS後、新query/injectionテストのGluonコンパイルで
`AttributeError: str object has no attribute type`となった。モデル/性能測定は未実行で除外。
呼出元raw kernelから`LAYOUT`を明示constexprで渡すテスト修正を行う。
実行runtime/sourceの浮動小数点計算は変更しない。修正後CPU対象12 passed/24 skipped、Ruff PASS。
失敗result archive `3ff2b2a5ec354c3fd0dbb0a07f73354b995fe401181e31a47502b33061104726`、
source `e2a875cba21bb6a9cf876f39fe76d9fbfa158cd6657caa6b0a4c9fb46e9aba68`、
1692source files/4manifest entries/return1/no-timeout/cleanup ackを確認し、所有session停止済み。


### v56 L4: 精度PASS、B64は短縮するがreserved超過が残る

source `4333aa1380e1d146579a4920a1b1b656738dfe86`、job
`l4job-f90348a65dca4c7ca0a1887a041e8420`。実機NVIDIA L4、driver580.82.07、
Torch2.11.0+cu130/CUDA13/Triton3.6。raw22件PASS(8.53s)、共有+同時retained36件PASS(148.89s)。
両主3状態full-shape FP64 Y/dX/全4P、全52,428sigma更新、初期P/X/target hash一致をPASS。
maxabsはY2.898e-5/dX2.471e-5/全4P1.2613e-4、relL2は1.882e-6/1.791e-6/6.330e-7以下。

|L4,65rounds|v52chunk32 ms|v54full64 ms|v56full64 ms|Dense ms|v56 allocated/reserved MiB|
|---|---:|---:|---:|---:|---:|
|B32 mixed|.708585|.708379|.743448|.082532|31.653/80|
|B64 mixed|1.297802|1.079157|1.008166|.105113|44.586/108|

v54よりpayload/allocated約1.600MiB、reserved各2MiB減る。B64のこの測定では6.58%短縮、
B32は4.95%遅延した。単独測定で普遍的な速度改善とはしない。B64はDense49.502/106MiBの
reservedを2MiB超え、速度もDense2倍を満たさない。v56 G4は未測定、L4結果で代用しない。
source archive `dd4257aabaf627999e24200568a89fee0af31f92c62ebebc67721ac6e71d8c9e`、
result archive `312ff0610a05bb8be3b0915db9dc2c6bc58d39c793c2f512fca5268c96564901`。
1692source files/31manifest entries、Git-byte一致、return0/no-timeout/cleanup ack確認。
L4 primary registry40件、G4 primary13件とは別管理。source CPU CI PASS(2m26s)。

### L4 backward consumer診断（完全stepの候補ではない）

同source job `l4job-bda08f17495e4282a46405baad9a1a8c`、v52/B32の全shape3状態独立FP64 gateを
先にPASSしてから、固定入力のbackward kernelを特殊化した。129回の中央値。

|特殊化|ms|registers|spill slots|PTXのatomic命令行数|
|---|---:|---:|---:|---:|
|dX+全4P|.230400|52|0|18|
|全4Pのみ|.161792|90|0|0|
|dXのみ|.137216|40|0|18|
|両方、境界1分割|.219136|48|0|18|

atomicを除いた全4Pにも時間が残るためatomicだけの問題ではない。一方、
削除するconsumerでregister数/生存期間/命令列が変わるので差を足し引きして内訳としない。
PTXの静的FP64算術行数は順に10/8/2/10で、実行命令数やFP64占有率の証明ではない。
全てspill0。境界1分割の出力/復元精度や完全step・メモリは検証していないので採用しない。
`norm_replay`も診断専用でありprepareとの二重加算をしない。
source archive `1984066eec6283abef74a858bb24edef8c90aa3166e01abacbece1835a9ab699`、
result archive `cc12801157e4efd3bee3abb4823edaeba82f9706e680f4c735fd9bb0be1784cd`。
1692source files/9manifest entries/return0/no-timeout/cleanup ack確認、L4診断registryへ分離保存。
両job終了後、所有GPU停止とserver上のsessionなしを確認した。

## v57: 境界4分割で数値scratchを減らす（検証前）

v56のPos/BorderPos、係数・Meta・正規化・全4Pはそのまま、境界deltaの分割数のみ8→4へ変える。
B64/N1024で3*511*(8-4)*64*4=1,569,792bytes（約1.497MiB）のscratch payloadを削る。
復元は同じ全境界和であり実数演算の式は同じだが、FP32 atomicの加算順序は変わる。
速度/両peakと誤差の実測前に改善扱いしない。
新しいv57だけがboundary_shards=4を受け付け、旧版の許容値8/16/32/64は変えない。
共有全勾配/retained/同時backward、動的sigmaに加え、v56/v57の513個の一致した広いatomを
同じseedで3状態・B3/B33・2幅にわたり照合する。判定4e-4を維持する。
CPU suite2034 passed/5240 skipped/18既存warning(40.87s)、追加scratch4分割/新GPU試験の
対象4passed/25skipped。Ruff・両case declaration PASS。GPU検証は未実施。


### v57検証中断: 新しい広いatom一致試験が既存v56でも失敗

source `fb1e647f1ca33bc825bdb65f09f7126de8d594cb`、L4 job
`l4job-aff2ad7d7a9c40f0be50ef5d094faf78`。raw24件PASS(8.59s)、
通常/retained等38件PASS後、新しい513重複・radial scale1.15・B3・step0の既存v56で
Y gateに失敗した。本来ゼロの出力領域を含む誤差があり、assert_close最大報告は
0.00498199462890625(index0,39)、11/192要素がatol/rtol4e-4を超えた。
通常のmixed full-shape条件での以前のPASSを取り消すものではないが、
その証拠だけでは重なり条件も精度保証するとはいえない。新テスト/閾値は変更しない。
v57全精度検証と両primary完全stepは未完了なので、v57速度/メモリ結果を主張しない。
source archive `4973196f83ba9ae9f58239441db9461e12de6fb85f4434bc2e064c4ad6c4321a`、
result archive `290d435b53f2c8051bd1460d31dee63843f0d01b447ba4f43f87c1e9e77e32fc`。
1694source files/5manifest entries/Git-byte一致、return1/no-timeout/cleanup ackを確認し、失敗として保存。
source CPU CIはPASS(1m52s)。所有GPUは停止済み。

次のL4診断ではv56/v57、A1/513、B3/33、radial scale1.15/1.4、3状態でY/dX/全4Pを記録する。
元のforwardから係数とprefix bankを読み、同じkernelに独立delta/borderとH保存先を与えて
Hを取得する。Hと独立FP64軸評価、Meta scaleと独立正規化を比べ、GPU H/Meta scaleから
出力profileのみをFP64で評価した射影も保存する。この射影と実Yの差は出力profile係数/
注入/復元/最終castの合成誤差であり、atomic単体の誤差とは断定しない。
数値診断の完走を精度PASSに読み替えず、違反した状態数を明示する。


最初の診断job `l4job-e35a6e4ff92b42a3ac40d61f7c0a61f6`は13状態をstdoutに記録後、
異なるdtypeのarena viewを同時torch.saveしたため保存エラーでreturn1となった。
実行済み部分でも完走/PASS扱いせず失敗registryへ保存した。1694source files/4manifest entries、
source `20639bd3e521b0171203b9ff0d845e4fe5ebca1f989914f05c783af15a4e8ede`、
result `bd48874e23ebba0f866470f6e4b005463d120f84035eb0f036c04c0b81410ff9`、cleanup ackを確認。
対象のv56/A513/B3/scale1.15/step0ではH max2.4161e-6、Meta scale max1.6203e-8、
HをFP64出力profileへ射影したY誤差max2.5501e-4に対し、元Y/射影間max4.981995e-3。
元dX max1.062012e-2、全4P max8.2341e-6も記録した。出力側の合成誤差が大きいという
部分診断だが、atomic単体/係数単体/復元単体にはまだ分離できていない。
保存用各viewを独立CPU cloneへ変え、状態ごとのpartial JSONも書く診断修正を行う。
A1の誤差*513と、A513出力とA1出力*513の差も記録し、単独誤差の一様増幅と集約誤差を区別する。


### 広いatom一致の48状態診断: 出力/入力集約で誤差が増幅

完走source `602a13be`（fb1e647fから説明のみ変更）、L4 job
`l4job-219eb24995c3454294aec156e60d01d9`。16構成*3状態の48状態を記録。
単独atomはv56/v57計24状態全て4e-4以内、A513は両版計24状態全てYまたはdXで失敗。
全4Pのmaxabsは全48状態で2.1287e-5以下だった。診断return0は状態収集の完了を示し、
数値gate PASSを意味しない。主performance registryには登録しない。

|A513の12状態最大|v56 境界8分割|v57 境界4分割|
|---|---:|---:|
|Y maxabs|.0159076|.0193256|
|dX maxabs|.0166932|.0173951|
|全4P maxabs|2.1287e-5|2.1287e-5|
|H maxabs|3.6847e-6|3.6847e-6|
|Meta scale maxabs|2.1500e-8|2.1500e-8|
|GPU H+MetaからFP64出力profile射影と参照Yとの差|.000411497|.000411497|
|実Yとその射影との差|.0158437|.0192616|
|実Y513と実Y1*513との差|.0160556|.0194735|
|実Y1の誤差*513|.00101287|.00101287|

参照Y513と参照Y1*513はmax3.013e-12以下で一致し、atomの線形和という独立参照も確認した。
単独誤差の一様増幅より、出力集約時の追加誤差が大きい。Hの精度を改善するだけでは
出力profile係数/FP32注入/復元の合成誤差は消えない。一方、H+Meta射影自体も1条件で
max4e-4をわずかに超えるため、出力側だけ直せば全条件PASSとも断定しない。
FP64演算/係数cast/atomic/scanを個別に同定するには追加の対照が必要である。
この負の結果によりv57の高速化/省メモリ化を採用せず、新たなG4性能測定も行わない。

source archive `18375f77d9eada14f920db4c044a5fba9855e384a5ba752dfd9906a2ae9a04ea`、
result archive `a57b1f8b1cab13609b91abe98c3524ded24887eb434cab3b304e34d7b528a3f0`。
1694source files/6manifest entries/Git-byte一致、return0/no-timeout/cleanup ackを確認。
13tensorのCPU clone snapshotはローカルでweights_only reload/finite検査にもPASSし、
反例Y max4.9819946e-3を再確認した。L4診断registry2件、L4 primary40件、G4 primary13件を分離。
所有GPUは全停止・serverにactive sessionなし。goalは速度/両peak/同じ精度基準でactiveかつ未達。


## 加算精度の切り分けとv58（GPU検証前）

保存済み反例1条件をCPUで再計算した`overlap-scatter-cpu.py/json`では、同じFP32係数/重みの
逐次FP32加算513回はY max約.007435、復元だけFP64へ変えても約.007443であった。
FP32積をFP64で加算/復元すると約.00023937へ下がる。
4/8/16/32個ずつ高精度で局所加算してFP32へ戻すCPU例は約.001532/.001018/.000724/.000500で、
この1条件でも4e-4を満たさない。GPU atomic順序や全条件を再現した証明ではない。
同snapshotの入力queryではGPU HをCPU FP32 FMA/和でbit再現し、参照H誤差2.4161e-6。
積/FMAはFP32のまま和のみFP64で2.1926e-6、全内積FP64で1.1460e-6となった。

L4 source `c000e0ef`、診断job `l4job-e746be15e027455e8ff812f22afe6f77`は
v56/A1/513/B3/33/2幅/3状態、二つの精度上書きで48状態を完走。
Delta/Border/復元だけFP64のA513では12状態中7失敗、Y max.000912061、dX max.001442384。
Bankの生成/queryもFP64にしても12状態中7失敗、Y max.000923288、dX max.001501149。
A1は両案全24状態PASS、全4Pは両案ともmax2.1287e-5以下。
後者のH+Meta→FP64出力profile射影はmax.000170012まで改善したが、出力profile係数の保存、
境界係数のshift、weightのcast、積にはFP32が残る。加算/入力だけの高精度化を完成扱いしない。
source `23d0625ed362aee8a560fe6530716e9c9ed6217c6dcb6341c7d7c3650451af4a`、
result `6d29f8b0be6b067d57376005ce18c8d8cd253e8bd5d6dc672373b3f8cffba621`、
1694source files/7manifest entries/Git-byte一致/return0/no-timeout/cleanup ackを検証。
L4診断registryへ分離し、速度/メモリ/採用の証拠とはしない。所有GPU停止確認。

v58は同じ実数の局所多項式・支持・sampled product L2を保ち、
係数CoとHalfInv、入力/随伴Bank、Delta/Border、重み*係数の積と復元をFP64でつなぐ。
Meta/正規化/幅stop-gradient/Polar、全4Pの縮約設定と最終FP32出力・gradientは既存契約を保つ。
旧recipeのFP32 store/load/castは同じdtypeのまま。新recipeだけFP64の型付きstoreを使う。
整数位置はsigned16のまま直ちにsigned32へ戻す。境界はFP64の4分割、batch chunk16。
各forwardの係数を全chunkで共有し、H全体は保存しない。入力Bankもbackwardへ保持せず、
そのchunkで再生成するので精度用bufferの寿命を制限する。速度改善は未証明である。

CPU2036 passed/5292 skipped/18既存warning(39.65s)、Ruff、両case declaration、build PASS。
raw係数layout検査をFP32/FP64へ拡張し、FP64係数の下位bitがFP32へ落ちていないことを要求する。
共有個別/全勾配・Graph更新・retained・同時backwardと新513重複試験も新routeへ適用する。
4e-4 gateを通った場合だけ、v52とDenseに対して両主条件の完全step/両peakを測る。


### v58 L4: 重なり精度は修正、完全stepは大幅に遅く未採用

source `c1500c698a2ce106bd11336e75b8d0920f3d8ea3`、job
`l4job-94cfdf6b08354ad88aa72b146671f6b6`。NVIDIA L4/driver580.82.07、
Torch2.11.0+cu130/CUDA13/Triton3.6。raw34件PASS(48.30s)、共有+重複+同時42件PASS(182.84s)。
同じseedの513個の広いatom一致・B3/B33・2幅・3状態もPASSし、以前の反例をこのrouteで解消した。
両主条件3状態全shape独立FP64 Y/dX/全4P、全52,428sigma更新、P/X/target初期hash一致もPASS。
v58だけの最大誤差はY1.233e-6、dX1.199e-6、全4P2.811e-5、relL2は各7.601e-8/7.889e-8/1.035e-7以下。

|L4,65rounds|v52 ms|v58 ms|Dense ms|v58 allocated/reserved MiB|
|---|---:|---:|---:|---:|
|B32 mixed|.709899|4.742494|.082511|38.255/104|
|B64 mixed|1.268093|9.172595|.103275|38.755/108|

v58のallocatedは両方Dense49.002/49.502MiB以下、reservedはB32だけDense106MiB以下。
B64 reservedは2MiB超過し、速度も大幅に退行するため目標達成/高速化候補として採用しない。
正しい高精度の計算経路と速度の両立は未完了であり、精度PASSを速度達成へ読み替えない。

独立phase診断は16行chunkあたりforward約.37888ms、backward約1.45510ms、
入力/再入力/随伴Bank約.07578/.07373/.07578ms、各restore約.08602ms。
compiler reportはforward40 registers、backward128 registers、両方spill0/shared0/local load-storeなし。
これだけでregister使用量が遅さの原因と確定はしないが、queryの係数live rangeを減らす対照を試す根拠とする。
prepare約.13107msに含まれるnorm_init約.07270msと診断専用norm_replay約.14029msを二重加算しない。
source archive `26966b6d2fa608002edb1677f9447d8cd7f378db26bce11c193e60695b6fda6f`、
result archive `19e4637ecf1925cc10f4c35849ffd03f83d7e0b02a428e46e74057d4bd39b1f8`。
1696source files/25manifest entries/Git-byte一致/return0/no-timeout/cleanup ackを確認。
L4 primary41件、L4診断3件、G4 primary13件を分離。source CPU CI PASS(2m15s)。所有GPU全停止。

### 次の係数共有式のCPU確認（GPU実装前）

半tile幅/Rをvとすると、外側の2次係数q=(1.5*v)*v、中央の係数m=(-3*v)*v。
4knotsの差分係数は[q,m-q,q-m,-q]=[q,-3q,3q,-q]で、tile境界係数はqか-2qである。
各軸について6イベントのc0/c1とq一つを保持すれば、18→13個のFP64係数にできる。
2軸・A52428では10*A*8=4,194,240bytes（約4MiB）の係数payload削減となる。
これは実測peak/reservedの削減ではない。

`compact-c2-cpu.py/json`はvを1e-8から約2まで100,003個取り、600,018係数で元のFP64演算と
bit一致を確認した。境界が中央pieceかのflagはsigned16 Posのbit14へ保持できる。
valid index0..9726へbit14を加えても最大26110で正のsigned16内、sentinel-1はそのまま。
19,456通りのflag付き位置のlossless roundtripも確認した。
この表現を使う全query/injectionで位置bitを復号し、旧native-cache等の通常layoutとの混用を防ぐ必要がある。
未実装のGPU性能や全勾配のPASSとは扱わない。

既存v28のevent-ownerも再確認した。各atom最大6イベントを整数histogram/scanで配置する方式であり、
site*atom支持表は作らないがCSR型の索引は持つ。以前のFP32版は構築込みでv27より遅かったため、
新精度routeの根拠なく採用しない。今回最も重いquery/全4Pの処理量を減らす案と比較して判断する。
