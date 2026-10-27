# atom走査・位相・同期の切り分け

2026-10-26。対象sourceは`6d619931f1c775738a8086d32ae2f59f7acc2746`。
production runtimeは変更せず、CUDA sourceの診断コピーを別extensionとしてコンパイルした。
128siteでの限界を示す結果ではない。現在のY/dXには長いatom走査と重複計算が残り、
同じ数学・精度の診断版でphysical forward/backwardが約4.63→2.31msへ短縮した。
公開dispatcherへの採用、denseを上回ったという判定、完全学習stepの測定ではない。

## コード上の主因

対象は`src/torchcst/_backends/cuda/algorithms/linear/cell_mass/kernels.cu`の
`expand_tiles_kernel`。sites_per_block=16、batch4、mode8では次の構成になる。

```cuda
// 8 blocks。各blockが16siteを担当するが、全819atomを走査する。
for (chunk : all_atoms_in_chunks_of_32) {
    if (thread_id < 32) prepare_atom_and_harmonics();
    load_H_chunk();
    __syncthreads();

    if (thread_id < 68) { // 1 profile tile × batch4 × basis17
        for (atom : chunk) {
            if (tile_is_in_retained_window(atom))
                C += amplitude * H * phased_coefficient(atom);
        }
    }
    __syncthreads();
}
publish_C_and_expand_sites();
```

- line414のouter loopは、sort済みrowsのbegin/endを絞らず819atom全体を読む。
  1blockあたり26chunk、各chunkに2回のblock barrierがある。
  line419でprofileと全harmonicを作り、寄与の判定はその後のline446以降にある。
  sortは導入済みでも、今の実装には候補範囲を狭める効果がまだない。
- profile準備を行うのは主に最初の32thread。他のwarpはHをloadした後で待つ。
  Cの担当も128thread中68threadで、C ownerは32atomを順番に足す。
  block内・block間の双方で並列性が不足する。
- line456の位相はatom・harmonic・chart imageで決まり、batchには依存しない。
  それでも4batchとsin/cosそれぞれの担当で同じ位相を評価している。
- float32入力でもprofile、係数、位相、積算はdouble。
  メモリのbyte数だけから実行時間を予測できない。今回精度は変更しない。

このcaseのsource上の回数は次の通り。machine instruction数・実trafficの測定値ではない。
Yのprofile tile32、dXのprofile tile16、どちらもK=3。

| 対象 | 現状 | 重複を除く場合 |
| --- | ---: | ---: |
| Yのprofile準備 | 8×819=6,552回 | 寄与するblockのみなら4,914回 |
| dXのprofile準備 | 6,552回 | 寄与するblockのみなら2,457回 |
| 各kernelのC ownerによる所属判定 | 8×4×17×819=445,536回 | atom分割だけでは総数不変 |
| Yの非定数位相評価 | 314,496 scalar sin/cos要求 | 39,312 sincosペアで共有可能 |
| dXの非定数位相評価 | 157,248 scalar sin/cos要求 | 19,656 sincosペアで共有可能 |

位相は4batch×sin/cosの8重複をまとめられる。scalar関数とsincosの命令コストは
同じではないので、要求回数の8分の1を時間の8分の1とは解釈しない。

## 現行YのNsight Compute

診断job内でproductionのYを1launch計測した。grid8、block128、L4のSM数58。
仕事を持てるSMは最大8個で、全SMを埋めるblock数に届いていない。

| 観測 | 値 |
| --- | ---: |
| Compute (SM) Throughput | 7.50% |
| DRAM Throughput | 0.01% |
| L1/TEX hit / L2 hit | 95.55% / 83.76% |
| 平均active thread/warp | 19.65 |
| issued instruction間の平均warp cycle | 33.58 |
| barrier待ちの平均 | 約16.0 cycle、上記の47.7% |
| short scoreboard待ちの平均 | 約11.9 cycle、上記の35.5% |
| local memory spilling requests | 0 |

47.7%はkernel wall時間の割合ではなく、warpのinstruction発行間隔に関する指標。
short scoreboardはsharedの依存・特殊命令などを含み、bank conflictと断定しない。
この取得sectionにはbank-conflictの直接counterがない。
全体のDRAM帯域を使い切っているという説明は、このlaunchの観測に合わない。

profiler下のdurationは4.09msだが、非計装のGPU eventでは同じYが1.606ms。
profiler値を速度比較に混ぜない。自動表示される推定speedupも実測改善とは扱わない。

## 数学を保つ診断変更

診断用flagsはproduction APIではない。全候補でprofile tile、周期、tile窓、
正規化、Fourier mode、dtype、double内部演算を固定した。
支持境界でのセルclipや裾の追加切り捨てはしていない。

1. **early**: 既存のtile所属判定をprofile/harmonic準備の前にも置き、
   block内のどのprofile tileにも寄与しないatomの重い準備を省く。
   全819atomの走査自体は残す。
2. **phase**: 現在のchunkだけ、位相を掛けた係数をsharedへ生成する。
   1threadが別々のatomを担当し、harmonicごとにsincosを一度呼び4batchで共用する。
   1blockが複数profile tileを担当するときは元の経路へ戻す。
3. **pad_h**: Hのshared strideを32→33に変える。bank conflict仮説の単独比較。
4. **split4**: 26chunkを4つのblock担当へ分ける。
   各blockは6〜7chunk、192〜224atom相当を担当し、最後の担当だけ819atomで打ち切る。
   gridは8→32block。小さな部分出力をglobalへ書き、別kernelの一意なwriterが合計する。
   atomic、atomコピー、global係数表は追加しない。

```cuda
// 同じ出力範囲に対してatom軸を分担する。
partial[split][batch][site] = expand_assigned_atom_chunks();
// 次のkernel。partialが揃った後に、一つのthreadが一つの出力を確定する。
Y[batch][site] = sum_over_splits(partial);
```

split4は1回のYまたはdXにつきFP32で4×4×128×4=8,192byteの部分出力を使う。
Y用scratchはforward中だけ、dX用scratchはbackward中だけ生存する。
partialを入力dtypeへ保存するため加算順と丸めは変わるが、従来の許容誤差を維持する。
各blockがCを持ち、部分出力を作る方式で、Hを部分値のまま使用する変更ではない。

診断controlにも条件分岐と静的shared追加がある。
production Y/dXはREG90/thread・SHARED9,984byte/block、
診断コピーはREG104/thread・SHARED18,464byte/block。
診断版はraw harmonicとphase済み係数の両方を残して切り替えるため、shared最小実装ではない。
controlとの比較とproductionとの比較を区別する。

## 実測

NVIDIA L4、58SM、Torch2.11.0+cu130、CUDA13.0。
128×128・819atom・batch4・FP32・mode8・spacing1・sigma0.2〜12・seed120。
入力T=16/出力T=32、K=3、周期48/96、merge Z・1要素4thread、直接多項式係数。
sites_per_block=16。buildと検証は測定processから分離した。

各候補を2つのfresh processで測定し、2巡目は逆順にした。
各processはwarmup5回後11サンプル。prepare・入力検査・scratch合計を含む
physical forward/backwardの同期wall時間であり、optimizer・Graph・denseを含まない。
同じjob内のproductionと比較し、別jobの絶対wall時間を組み合わせない。

| 変更 | process1中央値 | process2中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| production | 4.638 ms | 4.631 ms | 85,504 byte | 2 MiB |
| 診断control | 4.675 ms | 4.686 ms | 同上 | 同上 |
| earlyのみ | 4.228 ms | 4.236 ms | 同上 | 同上 |
| phaseのみ | 3.827 ms | 3.818 ms | 同上 | 同上 |
| early+phase+pad_h | 3.402 ms | 3.374 ms | 同上 | 同上 |
| split4のみ | 2.719 ms | 2.696 ms | 同上 | 同上 |
| split4+early+phase+pad_h | 2.306 ms | 2.316 ms | 同上 | 同上 |

組合せはproduction比約50.0%/50.0%短縮、約2倍の速度。
controlには約0.8%/1.2%のoverheadがあるが、主要な改善幅より十分小さい。
ピーク不変は測定されたbuffer寿命を含む結果であり、scratchが0byteという意味ではない。
元のX/dY/atomを含むallocatedであり、GPU process全体のVRAMではない。

別のGPU event測定によるY/dXのみの中央値。splitには合計kernelも含む。
この表の和でphysical forward/backwardのwall時間を置き換えない。

| 変更 | Y | dX |
| --- | ---: | ---: |
| production | 1.606 ms | 1.248 ms |
| control | 1.638 ms | 1.276 ms |
| early | 1.489 ms | 0.971 ms |
| phase | 1.047 ms | 0.999 ms |
| early+phase | 0.912 ms | 0.699 ms |
| pad_hのみ | 1.638 ms | 1.277 ms |
| early+phase+pad_h | 0.911 ms | 0.699 ms |
| split2 | 0.966 ms | 0.676 ms |
| split4 | 0.540 ms | 0.377 ms |
| split8 | 0.417 ms | 0.321 ms |
| split4+early+phase+pad_h | 0.324 ms | 0.219 ms |

pad_h単独には改善がない。bank conflict解消を主因とする根拠は得られなかった。
split8は段階別ではさらに短いが、完全経路の独立process比較は未実施で、採用とはしない。

非同値のcost probeも実行した。C積算を丸ごと省くとY0.777/dX0.780ms、
非定数の位相乗算を省くとY0.987/dX0.943msだった。出力が誤るため候補には含めない。
compilerの最適化も変わるため、各差分を足してコスト割合には変換しない。

## 面積正規化で打ち消せる共通因子

`make_profile`は`D=T_M(beta)=cosh(M*acosh(beta))`とそのsigma微分を作っている。
多項式係数の分子をQ_kと書くと、raw complex Fourier係数は

\[
c_k=\frac{Q_k}{2D^2},\qquad S=P c_0.
\]

非定数の実Fourier係数とその微分は

\[
\widehat c_k=\frac{2c_k}{P c_0}=\frac{2Q_k}{P Q_0},\qquad
\partial_\sigma\widehat c_k=
\frac{2}{P}\frac{Q'_kQ_0-Q_kQ'_0}{Q_0^2}.
\]

同じ因子DとD'は消える。定数項は1/Pで微分0。
診断ではpolynomial経路だけD=1、D'=0として、既存の比と微分のコードを使用した。
DFT経路は変えない。近似精度を落とす変更ではない。

別job・別extensionの完全経路比較ではcontrol4.755/4.730msに対し、
共通因子削除のみ4.635/4.623ms（約2.5%/2.2%短縮）。
early+phase+pad_hも加えると3.379/3.330ms、split4も加えると2.339/2.362ms。
Y/dX単独はcontrol1.653/1.284ms→因子削除1.585/1.218msだった。
第一jobの組合せを上回ったとは判断せず、分担・位相共有より優先度の低い簡約として残す。

## 正しさと未確認範囲

第一jobは11候補×4shape×FP32/FP64の88case、Y/dX/全4列のatom勾配264比較が通過。
shapeはN128/A819/B4/M8、N32/A35/B5/M16、N8/A0/B2/M8、N1/A3/B5/M8。
batch末尾、空atom、小さいchart、M16を含む。
第二jobはcontrol・因子削除・組合せ2種の32case、96比較が通過した。
独立した既存Torch Fourier参照を使い、誤差基準はFP32 rtol5e-5/atol2e-6、
FP64 rtol3e-8/atol3e-10から緩和していない。

| job | dtype | Y最大絶対誤差 | dX | 全atom勾配 |
| --- | --- | ---: | ---: | ---: |
| 第一 | FP32 | 1.19e-7 | 5.96e-8 | 2.38e-7 |
| 第一 | FP64 | 3.55e-15 | 3.11e-15 | 9.88e-15 |
| 第二 | FP32 | 1.19e-7 | 5.96e-8 | 2.38e-7 |
| 第二 | FP64 | 8.22e-15 | 3.33e-15 | 1.04e-14 |

第一jobのproductionとの差は、FP32 Y/dX最大5.96e-8・勾配0、
FP64 Y6.66e-16/dX6.11e-16・勾配0。
第二jobのJSON field `max_abs_vs_original`は診断controlとの差を意味し、
productionとの差ではない。第二のcontrolと第一のproduction/controlはそれぞれ独立参照で検査した。
この検証はtile-window近似の実装同値性を対象とし、compact coreとの近似誤差を測り直していない。
DFT、全profile設定、全sites_per_block、複数GPU、公開学習経路の全面検証は未実施。

## 次に実装する順序

1. atom走査の分割と一意writerによる合計を、読みやすいproductionコードへ移す。
   数学のtileと実行上のatom分担を分離し、scratch寿命と設定を明示する。
2. chunkの位相係数をsharedで再利用し、既存tile所属判定を重い準備より前へ置く。
   診断用の二重shared表とflagsをそのままproductionへ持ち込まない。
3. 通常のCUDA全suite・独立勾配・完全経路メモリを確認し、別jobで速度を再確認する。
   その後にsortされた候補rangeの利用、Cのblock間再計算、block幅との組合せを比較する。

改善後も残る約2.3msを全てY/dXとみなさない。
[直前の段階別測定](cell-mass-block-sites.ja.md)ではprepare約0.4ms、
H/G各約0.2ms、Parameter勾配約0.62msだった。これらは今回の新規測定ではない。
`parameter_grad_kernel`は保存H/Gに加えて両軸の微分projectionを再走査する。
`project_atom<true>`には値の積算も含まれるが値は後で保存H/Gから読むため、
compilerが除去しているかを確認してから専用化を判断する。
prepareにはTorch演算とstable argsort、入力検査にはhostでのbool読み出しがあり、
Y/dX短縮後はその比重が高まる。今のsourceと測定だけで残りの限界速度を断定しない。

## 証拠と再実行

第一job: `l4job-e9a15b87b35848018370ff692a2de458`。

- source archive SHA256: `0715326f4664a4d83476cd36ddcf84c8d56368115e9884be6d8d392b922949e7`
- result archive SHA256: `137a477f3e11b0bf5facadb08ed6e06c65e7d79e3583fef31d473167e15ab7e3`
- 診断CUDA SHA256: `ed7926b9fb8e207be8cfd9cdc95102458b335c20074a35d7dcffde57f70d1acf`
- frozen driver SHA256: `e769acaa8a992e926079e48d4f1d22dc0049d0627ddc139f7adcf415764dadda`

第二job: `l4job-2d0084de38ee4c2ba20a4b7a9dfb1fbf`。

- source archive SHA256: `4f2c43b12570d8795c4c56861543c879943601135b1ae20d2a9037dc6afe29c9`
- result archive SHA256: `c1bfe742ecac56f28edd745d5ca211aac6c3ccd4ad62846ff751a434fb544f64`
- 診断CUDA SHA256: `fec4f3ca9289c8ea7503cf70e9efc7a1ba5698c29a03cb94a8c2d35bbebe014f`
- frozen driver SHA256: `a7327e276cb8c185c6c90a4e8840b9a6c913eed29a084c3d676bb43821514cdd`

両source/result archive hash、全result manifest、driver内に埋めた診断CUDAとGPU回収物を照合した。
両snapshotの321 runtime source files（Python319/CUDA1/C++1）は現行sourceとbyte一致。
併せて`src/torchcst`内の文書9件も照合した。productionの追加変更はないため、
前回のCPU全体・wheel/sdist検証は繰り返していない。

raw evidenceはignored `output/cell-mass-deep-analysis-20261026/`へ保持した。
`mechanisms/`と`normalization/`に、それぞれsource/result archive、receipt/spec、
frozen-driver.py、JSON、CUDAコピー、第一jobのNCU report/counterを保存する。
原本はhost共通poolの`jobs/JOB_ID/`にも残る。
native binary自体のhashやbinaryコピーは今回のdriverでは保存していない。
再コンパイル可能なsource/driverを保持するが、同一binaryの復元とは区別する。

再実行時はsource archiveを新しいディレクトリへ展開する。driverにはCUDAコピーが埋め込まれ、
production sourceを上書きしない。共有poolへ次の形で提出する（各jobのdriverとsourceを対応させる）。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/extracted-snapshot \
  --script /absolute/extracted-snapshot/__pool_driver__.py \
  --timeout 720 --label cell-mass-deep-analysis-repeat
```

両jobの正常終了・検証済み結果の回収と、所有pool slotの停止を確認済み。
