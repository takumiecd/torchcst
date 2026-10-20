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
