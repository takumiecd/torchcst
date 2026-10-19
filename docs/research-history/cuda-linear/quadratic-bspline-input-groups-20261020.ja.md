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
