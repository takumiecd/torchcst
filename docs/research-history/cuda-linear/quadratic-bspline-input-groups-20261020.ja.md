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
