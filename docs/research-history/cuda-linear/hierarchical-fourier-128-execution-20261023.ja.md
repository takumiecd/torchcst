# 128で確認した準備・保持・出力への受け渡し

2026-10-23。N128/B32/A128/M8/mixed/seed41をNVIDIA L4で実行した。
runtime/kernelは変更せず、checkpoint `15c90df1` の既存runnerと別の診断driverを使用。
局所31 atom、階層成分を持つ23 atom、残る74 atomは既存の直接縮約経路だった。
局所出力・最終出力・勾配・同じY領域への受け渡しを検証した。
128のprofiler記録では局所計算とZ/Hの時間的重なりは観測されなかった。
この条件で大規模GPUの並列性や高速化を主張しない。

## 最初に何があるか

- X: `[32,128]` FP32、16,384 bytes。入力値、global memory。
- P: `[128,4]` FP32、2,048 bytes。振幅・出力中心・入力中心・I logit、global memory。
- 静的table: Tensor payload合計129,504 bytes。Fourier basis、projection、lookup、
  bank/node/children等。shape・mode・recipeに依存し、stepをまたいで保持する。
- target: 診断のbackward用 `[32,128]` FP32、16,384 bytes。forwardの入力ではない。

Wは論理的に128×128だが、このCST経路では実体化しない。
以下でglobalはdevice tensor storageを指し、L2への常駐保証を意味しない。

## 1. 準備と所属の決定

Pからroutingとdemandを作り、その後全atomを一組でprepareする。
この段階ではPと固定site/tableを読み、Xの値はまだ縮約しない。
atom_chunk宣言上限は256だが、実際の配列のatom軸は128。padding確保はない。

| device配列 | shape | dtype | payload bytes | 内容 |
| --- | --- | --- | ---: | --- |
| routes | `[128,4]` | int32 | 2,048 | componentごとの共有経路の適否 |
| demand | `[2,28]` | int32 | 224 | 出力/入力で必要なnodeと子依存 |
| Meta | `[128,4,4]` | FP64 | 16,384 | 各componentの支持半長・tile幅・bank・混合重み |
| Co / Ci | 各`[128,4,9]` | FP64 | 各36,864 | 混合係数とI微分係数 |
| Norm | `[128,2,3]` | FP64 | 6,144 | 各軸のΣf²、Σf∂μf、Σf∂If |
| Beta | `[2,128,4,2,17]` | FP32 | 139,264 | 中心の位相を含む共有縮約用係数と微分 |
| Native | `[128,2,3,128]` | FP64 | 786,432 | 全profileまたは階層内部を除いた残差、および中心/I微分 |
| Bounds | `[128,2,2]` | int32 | 2,048 | 各軸の始点とcache長/残差種別 |
| bank用order / begin | `[512]` / `[13]` | int64 | 4,096 / 104 | componentをbank順に参照するリスト |

prepareの戻り値9 Tensorのpayload合計は1,028,200 bytes。
Nativeは実支持長に詰めず、atom×両軸×3成分×128 slotを確保する。
局所forwardが読むNative成分は値だけだが、この共通準備は微分も用意する。
prepareの最初でMeta/Co/Ci/Norm/Beta/Native/Boundsを全部確保するため、
値の充填前から合計1,024,000 bytesの容量を持つ。

prepare後、局所判定と所属リストを作る。これらもglobal memory。

| 配列 | shape | dtype | payload bytes |
| --- | --- | --- | ---: |
| Local | `[128]` | int32 | 512 |
| sorted keys | `[256]` | int32 | 1,024 |
| local order | `[256]` | int64 | 2,048 |
| local begin | `[5]` | int64 | 40 |

この初期状態では局所31 atomのうち23が出力1 tile、8が2 tileに所属。
所属件数は39、tile別に `[10,9,13,7]`。空の出力支持を持つ局所atomは0だった。
階層側は候補12 banks/28 nodes。tile幅は32/64/128 sites、各幅に4 banks。
出力/入力のdemandはそれぞれ14 nodes。候補容量と実際の需要は別である。
階層成分を持つ23 atomにも、境界等の直接計算が残る。

## 2. 局所block内で計算し、Yへ保存する

出力32 sites×batch4件が1つの仕事。4 tiles×8 batch群=32 jobsを8 programsで処理する。
program kはjob k,k+8,k+16,k+24を担当する。特定SMへの割当ではない。
実際の所属リストにあるatomを4個ずつ読む。

実例としてatom12は、入力sites92..99、出力sites93..100を支持する。
tile2（sites64..95）とtile3（96..127）の両方のリストに載る。
batch0..3の場合、program2がtile2、program3がtile3を担当する。

1. program2はtile2の最初のatom群 `[12,19,29,37]` をorder/beginから読む。
2. 各atomのBounds、Nativeの入力profile値、Xをglobalから読む。
   atom12ならXのbatch0..3、sites92..99を縮約する。
3. 論理的な局所縮約はatom4×batch4、出力累積はbatch4×32 sitesのFP64値。
   `h = (A/D) ΣXv` とし、`acc += h u` をblock内で計算する。
   atom12の寄与はprogram2がsites93..95、program3が96..100に加算する。
   入力縮約はこの2つの出力担当で再計算する。
4. 各担当が全atom群の累積を終えたら、同じglobal Y `[32,128]` FP32へstoreする。
   どのリストも空ならゼロをstoreし、全Y要素を初期化する。

局所kernelにはglobal Hへの出力pointer自体がない。
h/acc等はblock内の一時値で、他blockへそのまま渡さない。
profileの中心/幅微分は局所forwardでは読まない。
Xを明示的なshared配列へコピーして保持する実装ではない。
Tritonの実生成物は128 threads/program、registers/thread48、shared memory4,096 bytes/program。
これは各論理変数をregister/sharedへ一意に割り当てた証拠ではなく、
sharedを全てXのcacheと呼ぶこともできない。profilerの推定occupancyは実SM稼働率ではない。

## 3. 階層側のZとHを作る

通常のoverlap経路では局所出力をside streamへ、Z→Hを元streamへ置く。
診断では途中値を読むために全stream同期を入れており、この同期付き実行を
並列性の実測に使わない。別の通常forward profilerも後述する。

- Z: `[32,28,17]` FP32、60,928 bytes、global。
  需要のある32/64/128 siteタイルについて作る。偶数周波数は対応する子から継承し、
  残る周波数はXから計算する。28 node分の容量自体は需要14個に圧縮しない。
- H: `[32,128,3]` FP32、49,152 bytes、global。
  非局所97 atomの縮約に使う。23 atomは階層成分を使い、74 atomは既存直接経路。
  forwardは非局所の値fieldのみstoreし、局所31 atomのslotをstore/readしない。
  ただし全128 atom×3 fieldの容量を確保している。未初期化slotを数値として検査しない。

局所kernel内のhは振幅/正規化を乗じた値で、こちらのglobal Hは未スケールのΣXv。
両者を生の配列値として直接比較せず、Yへの寄与で同値性を検証する。

## 4. 合流後、同じYへ非局所寄与を加える

元streamが局所stream完了を待ち、C `[32,28,17]` FP32、60,928 bytesをゼロ初期化する。
非局所Hを出力tileのFourier係数Cへ集約し、残差をYへ加算する。
最後にCを固定basisでsiteへ展開し、同じYへ加算して完成させる。

```text
共通prepare / 所属リスト
  ├─ 局所：X + Native → block内縮約・出力累積 → Y（局所値）
  └─ 非局所：X → Z → H
                 join ↓
      H → C集約、残差を同じYへ加算
          Cを展開して同じYへ加算 → 最終Y
```

局所出力→残差加算→最終展開まで、Yのstorage addressが全て同一であることをassertした。
Y_local/Y_hierarchyという別のdevice tensorは作らない。
診断が保存した途中値はCPU上のコピーであり、実装のglobal partial-output bufferではない。

独立FP64 quadrature/KKTとの最大絶対誤差は、局所Yで5.06e-8、最終Yで2.30e-7以下。
「残差加算後のY＋Cの独立CPU展開＝最終Y」の誤差は8.28e-8以下。
serial8/overlap8両方で検査した。

## 5. 何がforward終了まで残るか

prepare一組、局所リスト、Z/H/Cはforwardのローカル変数として最後まで残る。
最後の利用直後に個別解放したり、ZとCを同じbufferで使い回す実装ではない。
forwardを抜けるとこれらはautograd保存対象ではなくなる。
saved_tensorsはX/P/demand/routesとNone2個だけで、X/Pは入力と同じstorageの参照だった。
静的tableと返したYは残る。reserved領域の解放とは別である。

同期付きoperator診断のallocated値は両方式で次の通りだった。
optimizer/Graphを含まず、trace・同期・CPUコピーがあるため、通常stepのピークとは区別する。

| 診断時点 | allocated bytes |
| --- | ---: |
| 入力・静的tableあり | 166,912 |
| routing/demand後 | 169,472 |
| prepareの7配列確保・係数計算後 | 1,193,472 |
| prepare return時（sortの一時変数もframe内に残る） | 1,202,176 |
| 局所Yのstore後 | 1,218,560 |
| Z生成後 | 1,279,488 |
| H生成後 | 1,328,640 |
| C確保・集約・最終Y生成まで | 1,389,568 |
| forwardから戻った後 | 185,856 |

backwardは現行の全atom再計算経路で、局所融合は未実装。
今回のPython逐次traceにはautograd側のbackward内部イベントは記録されず、
backward各配列のphase別寿命を実測したとは主張しない。
backward後のoperator診断peakは1,531,392 bytesであり、下記の正式step値とは別の測定。

## 通常stepの検証・計測結果

既存runnerの4 correctness worker、4 measure worker、1 Dense worker、全9 worker PASS。
独立FP64 oracleのY/dX/全P勾配、3つのlive状態、retained backward、
live変更後のGraph replayをatol=rtol=4e-4で照合した。
局所候補の最大絶対誤差はY4.24e-7、dX5.46e-7、dP4.94e-6以下。

| plan | Graph完全step ms | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| direct-samples-h | 0.746496 | 1,205,248 | 6,291,456 |
| local-control | 1.105920 | 1,522,688 | 6,291,456 |
| local-serial8 | 1.181696 | 1,522,688 | 6,291,456 |
| local-overlap8 | 1.134592 | 1,522,688 | 6,291,456 |
| Dense | 0.041984 | 34,408,960 | 48,234,496 |

準備・sort・norm・forward/backward/AdamWを含み、Graph capture/replay込みのメモリ。
一job、各planは独立worker、各時間は7 replay samplesの中央値。
7 samplesを7独立runと数えない。Denseの値にもライブラリ等のworker内保持が含まれ、
Wのpayloadだけを示す値ではない。CST速度の採用結果ではない。

directとlocalのpeak差は317,440 bytes。capture前のallocatedはそれぞれ64,512と192,000で、
常駐差127,488 bytesを実測した。残差189,952 bytesは
Zx/Zg/Cの3×60,928、routes/demandのallocator丸め後2,560、
bank用order/beginの丸め後4,608の合計と数値一致する。
これはコード・buffer容量・実測値が整合する説明であり、Graph内の全allocation traceを
採取して一件ずつ因果照合した内訳ではない。
局所なしcontrol/serial/overlapのpeakは同じで、2 streamによるY二重化の増加はない。

通常forwardの別profilerでは、serial/overlapとも局所とZ/Hの重なり0 us。
overlap記録の局所kernelは時点0..51.168 us、最初のZ kernelは136.479 us開始だった。
この一回の計装traceから、未計装GraphのSM割当や一般的な並列性は推定しない。
overlapの完全step中央値がserialより約4%短いことも、ここでは同時実行の証明にしない。

## 回収・再現・停止

- job: `l4job-6153458561a94e118b12c9d0985dfa2f`
- source commit: `15c90df1`
- source archive SHA256: `999fc4e88c5779b54ffc5282017831178edddf02e7af0112f659a92d92554a60`
- result archive SHA256: `db169848fd655a2651b75f2894a5b3dd6f8f9be5f5d1e0cffda60b42a8b7efbd`
- NVIDIA L4、GPU UUID `GPU-b052e12e-7cb6-729d-a906-619266be7fc3`、58 SM、driver580.82.07。
- Python3.13.15、Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。

returncode0、timed_out=false。result archiveをreceiptと照合、全driver artifact hashを照合、
worker source file照合3,654件。比較JSONの全record PASSを確認した。
pool全slot stopped、Colab server active sessionsなしを実行後に確認した。
新しいkernel実装、PR、push、merge、次サイズのGPU投入はしていない。

raw/driver/collectorはignored `output/hierarchical-fourier-128-20261023/`。
raw原本はhost poolのjob directoryの`results/artifacts/`にも保持する。
診断driverは既存runnerを呼び、別processでexecutorのphaseを観測するもの。
新しい性能runnerではない。storage addressを記録してもdevice Tensorを保存し続けない。
診断の途中値はCPUの`*-phase-values.pt`、shape/lifetimeは`*-phases.json`に保存した。

```bash
python3 output/hierarchical-fourier-128-20261023/collect.py \
  l4job-6153458561a94e118b12c9d0985dfa2f
```

この段階で128のforwardの保持と受け渡しを確認した。
局所のH容量削減や準備の分離は、今回の記録を基準に別の変更として検証する。
