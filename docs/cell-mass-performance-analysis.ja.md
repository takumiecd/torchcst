# 128サイトcell-mass実装の時間分析

> 旧定義の実装・測定記録。ここでのatom中心の支持clipと境界セル経路は廃止した。
> 現在の数学・API・測定は[1周期分のタイル窓](cell-mass-tile-window.ja.md)を参照。
> 以下の旧数値を、新しい近似方式の同値なbaselineとして扱わない。

2026-10-26。[係数計算の最適化](cell-mass-coefficient-optimization.ja.md)後の追加診断。

この記録は最適化後の同じ819atom / batch4 / FP32 / mode8ケースを分析する。
目的はボトルネックの所在と再計算・並列度の関係を分けること。
計算を省いた診断は正しいY/dXを出さないので、新方式の性能結果として扱わない。

## 現在の実装が行っていること

YとdXは同じ`expand_tiles_kernel`を使う。32site×4batchを1blockが担当し、
8siteのleafごとに次を順番に繰り返す。

```cpp
for (leaf : blockの4個のleaf) {
    for (atoms : 全819atomを32個ずつ) {
        32threadでsigmaと中心からprofileを生成;
        sharedにprofileとHまたはGを置く;
        sync();
        68threadでC[4batch,17basis]を加算;
        32threadで担当leafの端セルの直接積分を加算;
        sync();
    }
    Cをsharedに渡す; sync();
    32threadで担当leafの8site×4batchを展開;
    sync();
}
outputを各threadが1回保存;
```

これはZx/Zy→H/Gの入力階層集約は使っているが、出力Cの階層マージはまだ使っていない。
Cはleafごとにゼロから作る。prepareも現在は中心の(output tile,input tile)のargsortのみで、
幅bank分類やblockのbegin/end表は未実装。ソート順は整っているが、候補範囲を絞るためには
まだ利用していない。全体の設計方針と現在の試作の実装範囲を区別する。

また、今回のbankでは支持半幅Lは入力24・出力32に固定されている。
sigmaはprofileの形を変えるが、このLを変えない。
小さいsigmaでもその裾を厳密に扱うため、狭い支持へ勝手に切り捨てることはできない。
将来の幅bank/階層分類は、対応するprofile・周期・支持の数学と合わせて実装する。

## 正確な入力を使ったCPUの処理回数集計

前回driverと同じseed120・819atomを再生成し、周期image -1/0/+1と全leafを走査した。
下記はソース上の論理呼び出し数で、GPU命令数やDRAM transaction数ではない。
コンパイラの共通式除去とcacheの効果は含まない。

| 項目 | Y | dX |
| --- | ---: | ---: |
| block数 | 4 | 4 |
| profile生成 | 13,104 | 13,104 |
| Cの支持判定 | 2,673,216 | 2,673,216 |
| 直接セル積分関数の呼び出し | 1,074,528 | 1,126,944 |
| そのうち面積ゼロで戻る呼び出し | 1,045,044 | 1,097,460 |
| 面積ありのセル積分 | 29,484 | 29,484 |
| raw係数評価（DC・C・端セルの合計） | 615,888 | 511,056 |
| blockあたりbarrier | 216 | 216 |

Yは直接積分の約97.3%、dXは約97.4%が面積ゼロ。
これらは全てmode8の積分まで実行するわけではなく、lo/hiの比較で早く戻る。
したがって「100万回×mode8の重い積分」と数えてはいけない。
それでも不要な候補走査と周期image判定は繰り返される。

同じbankでatomごとに異なるraw係数はDCを含む9個、全atomで7,371個。
ソース上の評価回数はYでその約83.6倍、dXで約69.3倍になる。
この比は達成可能な速度向上率ではない。再利用範囲がblockをまたぎ、
境界・位相・読み込みなど別の処理も残る。

## 再計算の具体例

C[b,k] += amplitude[a] * H[b,a] * coefficient(a,k)

coefficient(a,k)はbatchに依存しないが、4batchで同じ値を評価する。
同じharmonicのsin/cos成分も、raw係数部分は同じ。
端セルでもmass(a,site)はbatchに依存しないが、4回評価する。
さらに同じatomを隣接leafでもmake_profileから作り直す。
現在sharedへ保持しているのはprofileのコンテキストであり、全mode係数の値ではない。

globalへの全係数表やatomコピーを作らずとも、現在の32atom chunk内だけで
係数・セル積分を生成してshared/registerから複数batchへ使い回す余地がある。
これは採用済みの高速化ではなく、次に同値性と時間を測る候補。

## threadとblockの使われ方

128thread中、profile生成時は最大32thread、C加算は68thread、
そのleafの直接積分とC展開は32threadが担当する。
同時に全員が異なる仕事を進める割り当てではなく、各段階で担当外のthreadがある。
特に端セルの段階では各warpの8laneだけが担当し、残り24laneはその分岐を実行しない。
barrier自体の費用と、担当threadの重い計算を待つ費用は分けて測る必要がある。

## 算術・メモリ・同期の区別

入力・出力はFP32でもProfile、係数、三角関数、面積、accumulatorはdouble。
これはソースで確認できる。FP32に変えてよいか、double演算が何%を占めるかは別問題で、
境界・面積の相殺・全Parameter勾配を検証せずに精度を下げない。

Y/dXはatom3項目の論理読込157,248byte、H/Gの論理読込209,664byte。
indices、spill/stack、cache transaction、メモリ階層の実trafficはこの値に含まない。
総allocatedが小さいことは「帯域問題がない」証明にはならないし、
同じ値を読んだ回数をそのままDRAM転送量とみなすこともできない。

register108/thread・shared6176byte/blockはコンパイル時の情報。
以下のNsight Compute診断と区別し、これだけからregister/shared容量を主因と断定しない。
現在の4blockというgrid不足は、それ以前に同時に使えるSM数を制限する。

## GPUで一部の計算を省いた診断

runtime sourceを変更せず、ignored outputに診断用CUDA sourceを生成した。
Axisへruntime maskを追加し、C加算、端セルの直接積分、または両方を省く。
mask0のY/dXは製品sourceの出力とbit単位で一致した。
mask1/2/3は数学的に不完全な出力であり、採用候補ではない。

同じbinaryのmaskだけを切り替え、warmup2回の後に7サンプルのGPU event中央値を測った。
NVIDIA L4（実機58 SM）、Torch2.11.0+cu130、CUDA13.0。

| 診断 | Y | dX |
| --- | ---: | ---: |
| 元のruntime source | 31.122 ms | 29.473 ms |
| 診断用source・省略なし | 31.254 ms | 29.574 ms |
| Cへの加算を省略 | 23.341 ms | 23.874 ms |
| 端セルの直接積分を省略 | 13.682 ms | 11.491 ms |
| 両方を省略 | 5.666 ms | 5.678 ms |
| 診断用source・省略なしの再測定 | 31.261 ms | 29.578 ms |

この比較では、端セル経路を省いたときの変化がC加算の省略より大きい。
端セル経路が最初に改善すべき箇所という根拠になる。
ただし、省略により分岐や待ち方も変わるので、差を独立した関数の正確な所要時間や
得られる高速化として扱わない。両方省略した5.7msもprofile生成単独の時間ではなく、
chunkロード・判定・基底展開・同期などを含む残存経路の時間。

実機SM数58に対してY/dXのgridは各4block。通常のblock実行では同時に
最大4 SMへしか仕事を配れない。H/GとParameter勾配も各7block。
これは実際のSM busy時間・warp占有率そのものではなく、launchから決まる並列度の上界。

H/Gは同じ819atomでもprofile生成各819回、直接セル積分各9,009回
（面積あり7,371回）。一度計算したセル積分を4batchへ適用しており、
Y/dXのようにbatchごとに面積を再計算する経路ではない。
Hのraw係数評価は論理76,011回、Gは77,771回。
H/Gの所要時間差まで、この回数だけでは説明できない。
ソート順、warp間の支持位置のばらつきなどは追加の比較が必要。

## Nsight Computeで確認した実行状態

元のY kernelを13passで取得したレポートの値。dXのhardware counterは未取得。
ここでの計装時間は78.30ms、SM周波数は約807MHzで、通常計測の31.1msとは区別する。

| 指標 | 観測値 | 読み方 |
| --- | ---: | --- |
| grid / 実機SM数 | 4 / 58 | 同時に仕事を持てるSM数が少ない |
| warpあたり平均active thread | 9.26 / 32 | 分岐や役割分けによりlaneが十分使われていない |
| predicationを含む実行thread | 8.64 / 32 | 条件付き実行でも有効laneが減る |
| FP64 pipe活動率・全SM平均 | 5.505% | GPU全体では低い |
| FP64 pipe活動率・最も忙しいSM | 80.515% | 一部のSMではdouble演算を多く実行 |
| DRAM throughput / peak | 0.000914% | この診断はDRAM帯域飽和を示さない |
| L1/TEX cache throughput | 1.08% | GPU全体の表示値。cacheの全問題を否定する値ではない |

FP64のraw metricは`sm__pipe_fp64_cycles_active.{avg,max}.pct_of_peak_sustained_elapsed`。
最大値を「全SM平均」や「forward時間の80%」と読み替えない。
少数blockにdouble計算を押し込め、warp内でも有効laneが少ないというコード上の構造と
観測が整合している。各問題の寄与率を分離したという意味ではない。

命令issue間の平均約31.76cycleに対し、short scoreboard待ちは約22.25cycle（約70%）、
barrier待ちは約4.62cycle（約14.6%）。short scoreboardはshared memoryの依存に加え、
特殊数学命令や動的分岐でも起きる。今回はMemory Workload Analysisや
source単位のstall位置を採取していないので、sharedのbank conflictが原因とは断定しない。
プロファイラの自動表示する「推定speedup」も、実測改善として報告しない。

## 次に比較する順番

1. **端セル経路の不要な候補を除く。** whole_tileでない場合にも、leafと支持に重なりが
   あるかをchunk内で判定する。完全に離れたleafで8site×4batchのcell_massを呼ばない。
   周期端では正しいimageの寄与を全て残す。長期的にはソート済みbegin/endへ接続する。
2. **chunk内の再利用を入れる。** raw係数とmass(a,site)を4batch、sin/cos成分、
   隣接leafで使い回す。全atomのglobalコピー・係数表ではなく、現在chunkの
   register/sharedを候補とする。32atom×9raw係数×doubleなら2,304byteだが、
   実際には位相や追加scratch・registerへの影響も含めて測る。
3. **blockの分配を変える。** leaf/batch/Cの担当を見直し、4blockに長いatomループを
   押し込める状態を解消する。atom方向へ分割する場合は部分和の書込み・集約・
   追加メモリも数える。blockを増やしただけで速いとは決めない。
4. **算術型と特殊化を比較する。** 値だけの経路と微分経路、固定modeの特殊化、
   FP32/混合精度を独立参照のY/dX/全atom勾配の基準で検査する。
   先に誤差基準を緩めて速さを作らない。

Zx/Zyの構築は前回の段階別診断で各約0.037ms。
現時点でZのmerge方式やstream間の重なりを優先しても、約60msあるY/dXは残る。
まず演算の再利用と候補範囲を直し、その上で並列化を評価する。

## 証拠と再現

分析対象はsource commit `d25bdc4afa0461d0e05f6bc12d2bc19a554b7745`。
CUDA本体は前回の係数最適化から変更していない。
job `l4job-cc912e60ccef46f3b9ea81ba834fa5cb`、source archive SHA256
`86c642f8b57f806229fa7440c1fab04eff83496f766dd672d59a1ed9565301c0`、
result archive SHA256
`0cf965e9a383722490b98455e5cf0391ef3ab2db6a0923b83d9d229c014d4d0d`。
321 runtime source filesのsnapshotとのbyte一致と両archiveのhashを再確認した。
raw data、診断source、driver、処理回数集計scriptとJSONは
ignored `output/cell-mass-analysis-20261026/`、原本はhost共通poolのjob directoryに保持する。
処理回数はvenv Pythonで同directoryの`count_work.py`を実行して再生成できる。

測定は1job・各7サンプルであり、独立した7runではない。
GPU計測の後にNsight Computeの13passで元のY kernelを1件取得した。
計装時の時間を通常のforward+backwardの時間と比較しない。
この分析ではruntimeコードを変更しておらず、新しい高速化の採用・dense比較・
optimizer/Graphを含む完全stepの性能結果も追加していない。

取得済みNCU reportをtext/CSVに変換するため、追加job
`l4job-e6f8591896eb4aa981c17421769d6ebf`で同じreportのimportだけを実行した。
新しいCUDA測定・workloadは実行していない。
report SHA256は`9e78d764d040f45657ca40d234aea605e9328bea6ff2ed714fa9da301caf95ba`、
export result archive SHA256は
`861e07de806ad38b5ca9a6e3e130b7abc59779814e0c4c745076490425f4a67b`。
原reportとexport先の同一性、result archive hashを照合済み。
text/CSVは同じignored directoryの`ncu-export/artifacts/`に保持する。

両jobは正常終了し、使用したpool slotの停止を確認済み。
