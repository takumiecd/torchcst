# ソート付き128版：プログラムの読み方

2026-10-26。[配置と処理方針](hierarchical-fourier-atom-layout-20261025.ja.md)を、
まず入力128・出力128・atom128の有限区間のCUDAへ接続する。
既存の全atom経路は比較基準として残し、ソート版は別Algorithm/Recipeで選ぶ。
以下はコードの説明であり、性能改善を主張するものではない。

## 最初に読む3ファイル

1. [block128_executor.py](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_executor.py)：
   PyTorchの入口。何をforwardからbackwardへ保存するかを見る。
2. [block128_layout.h](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_layout.h)：
   ソートkeyとget_rangeの意味を見る。関数はCPUでも検査している。
3. [block128_kernels.cu](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_kernels.cu)：
   CUDAで並べ替え、sharedへ読み、Z → H/G → C → Y/dXと進む。

係数・支持の数式は[block128_math.h](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_math.h)、
Python/CUDAの間のshape検査は[block128_bindings.cpp](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/block128_bindings.cpp)。
各関数の先頭に役割を書き、条件・ループのbodyや複数の操作を一行へ詰めない形にした。

## forward：入口からの順番

```text
apply(x, parameters, spec, SortedBlock128FourierRecipe())
  → _SortedApply.forward
    → C++のforward_sorted：shape/device/dtypeの検査
      → block128_forward_sorted_cuda
        1. prepare_atom_layout：key作成 → sort → pack → get_range
        2. prepare_norms：packした行順で全域normを準備
        3. block128<false,true>：担当する出力を計算
```

prepareは毎forwardの現在値から作る。まだ差分更新や永続的な数値cacheを導入しない。
GPU内で完結する固定shapeの処理で、ソートのためにParameterをCPUへ読み戻さない。

### prepare_atom_layoutで作るもの

make_atom_sort_keyは、幅logitから現在のIを求める。既存component()と同じ階層境界を使い、
group=hierarchy_level(I)、T=0.2×2^groupとする。中心をTで割ってfloorした
(group, output_tile, input_tile)が主key。同じkeyのatomは元の行順に並べる。
周期・支持・混合の数学的な選択を、ソートのために変更しない。
一つのatomに複数成分があっても、そのatom自体は一行のまま。

sort_atom_keysは128 threadsの比較・交換によるbitonic sort。
比較するペア同士は重ならず、一段階ごとに全員で同期する。
keyだけを並べてから、元のParameterをsorted_parametersへコピーする。

| global配列 | shape・dtype | payload |
| --- | --- | ---: |
| sorted_parameters | [128,4] FP32 | 2,048 bytes |
| source_rows | [128] int64 | 1,024 bytes |
| atom_groups | [128] int32 | 512 bytes |
| ranges | [2,4,12,2] int32 | 768 bytes |
| 追加のlayout配列の合計 | | 4,352 bytes |

source_rows[slot]は「ソート後slotが、正本の何行目から来たか」。
これは今回の呼出し元Tensorの行対応であり、公開AtomStateの不変IDを別途導入する変更ではない。
key、groupの区切り、最大支持半幅はprepare kernelのshared内だけで使い終える。
原本のParameter・勾配・optimizer momentは並べ替えない。
norm等の既存配列は別であり、このpayload合計はGPUピークメモリではない。

### get_rangeが返す範囲

ranges[axis][tile][group]が[begin,end)を持つ。axis=0はforward、axis=1はdX。
tileは担当する32 sitesの番号0..3、groupは階層番号0..11。

- forward：group内は出力key順。担当site範囲をgroupの最大支持半幅だけ広げ、
  対応する中心keyの最初と最後を二分探索する。
- dX：入力keyはgroup全体で昇順ではない。最初の実装は入力候補を探し、
  その最初から最後を含む広めの一つの範囲を取る。途中に混ざるatomも一括計算する。

例えば入力候補がslot 2と5なら、dXの範囲は[2,6)。slot 3と4も含む。
この方式のdX探索効率は未確定だが、必要な寄与は落とさず、追加compactionもしない。
候補を受け取った後の支持外ゼロ・完全なタイル・境界残差の扱いは、元の計算式を保つ。

現行Specは有限な中心支持なので、このget_rangeに周期wrapは入れない。
FlatTorus等へ対応する場合は別の数学契約・範囲生成として追加する。

## block128の中：同じthreadsで役割を切り替える

1. **init**：入力XまたはdYをsharedへ読む。ソート版ではParameter 2 KiBもsharedへ置き、
   H/G・C・直接出力で再利用する。元のglobal表は保持する。
2. **Z**：build_z_tileで、入力タイルの(batch,basis)要素を分担する。
   子Zが必要な親を作る前に同期する。
3. **H/G**：各threadが別atomを担当。自分のslotがbegin/end内なら、全入力支持からH、
   または全出力支持からGを作る。範囲外のthreadも値0を書き、同期へ到達する。
4. **C**：H/Gをsharedへ公開して同期する。build_cの各threadは一つのC要素を担当し、
   各groupのbegin/end内のatomから寄与を足す。
5. **出力**：store_outputでCを展開し、直接・境界部分も足して担当Y/dXを保存する。

H/G計算中はZを読む。全threadがZを使い終えた同期の後、同じarenaをCへ再利用する。
Cができた後にも同期し、出力担当threadが完成したCを読めるようにする。
各H/G、各C要素、各出力要素の書き手は一意で、atomicAddを追加しない。

| blockのdynamic shared予約 | 全atom版 | ソート版 |
| --- | ---: | ---: |
| forward | 21,376 bytes | 23,424 bytes |
| backward | 38,656 bytes | 40,704 bytes |

増分はParameter表の2,048 bytes。H/Gをglobalへ保存する変更ではない。
現在は全28 nodeのZと全128 atomの枠を使う。atom数のchunk化や需要Zだけの生成は次の設計対象。

## backward：H/Gを全域で計算する担当と、勾配の戻し先

```text
_SortedApply.backward
  → 元のX/Pのversionを検査
  → block128_backward_sorted_cuda
    1. block128<true,true>：G → 入力側C → dX
       各小バッチ群のtile 0だけは、全atomのH/GからdP部分和も保存
    2. reduce_parameters：小バッチ群の部分和を合算して元のParameter行へ保存
```

dP担当blockはdXの範囲外のatomも計算する。atom勾配には全支持のH/Gが必要だからである。
他のdX担当blockは、その入力範囲に必要なGだけを作る。

\[
dP_{\mathrm{original}}[source\_rows[slot],f]
 =\sum_{\text{batch group}}dP_{\mathrm{partial}}[\text{batch group},slot,f].
\]

source_rowsは順列なので、各元行に一度だけ保存できる。
元X/P、sorted_P、norm、slot対応とrangesは呼出し固有のautograd保存値とする。
次のforwardのworkspaceで上書きせず、retained backwardでも同じ対応を使う。

## 選び方と確認範囲

新しい研究Algorithmはresearch_cuda_hierarchical_fourier_block128_sorted。
[Plan](../../../benchmarks/cuda/linear/plans-hierarchical-fourier-block128-sorted.json)と
[Case](../../../benchmarks/cuda/linear/cases/block128-sorted-fourier-m8-mixed-128-a128.json)に登録した。
公開CSTLinearの既定選択には接続していない。

CPU検査は独立したTorchの成分/支持定義と比較し、両軸・全4担当タイルで必要なatomを
範囲から落とさないこと、ソート順、元行対応、更新後の再準備を確認する。
GPUでは同じ範囲の一致に加え、Y/dX/全atom勾配、retained backward、元Tensor変更検出と
既存runnerの独立oracle・可変値のGraph replayを確認した。結果は以下に記す。

## 検証結果：128サイトの単一ケース

GPUで実行したsourceは`4d0ba1ce5286e978389af7238fc987df56fdfc89`。
jobは`l4job-c2c0e66412d8416c9d24f2f090c3a867`。NVIDIA L4、driver 580.82.07、
Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0、Python 3.13.16。
GPU UUIDは`GPU-bd94f626-e729-188a-7792-7b8ca790c595`。

- CPU全体：2,271 passed / 5,758 skipped、既存のTorch JIT警告18件。
- 対象CPU検査：10 passed / 12 CUDA-dependent skipped。host C++で共有headerを実際にコンパイル。
- L4対象検査：22 passed。CUDAのsort結果・rangesがCPU版と一致し、更新後も再準備される。
  batch1/4/5でY/dX/全atom勾配、retained backward、元X/Pの変更検出を検査。
- 既存runner：独立したquadrature/KKT FP64 oracleに対し、全sites・全atomを検査してPASS。
  live値を更新した3状態と、更新後のGraph replayを含む。
- ソート版の独立oracleに対する最大絶対誤差：Y 2.9321e-8、dX 2.9571e-8、dP 1.3888e-6。
- Ruff、Plan/Case宣言往復、文書リンク、wheel/sdist buildを確認。
  両packageに新しいblock128_layout.hが同じbytesで含まれる。
  build backendは隔離環境に用意し、共有venvへ追加インストールしていない。

同じjobのm8-mixed / atom128 / batch4 / FP32 / AdamW、warmup2・7回のreplayで、
未計装の完全stepを測った。prepare・sort・norm・forward・backward・optimizerを含む。
7 replayは一つの独立run内のサンプルであり、7回の独立実験ではない。

| 方式 | Graph完全step中央値 ms | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| 既存direct-samples-h | 0.572416 | 1,061,888 | 6,291,456 |
| 全atom版cuda-block128 | 46.424065 | 33,792 | 6,291,456 |
| ソート版cuda-block128-sorted | 41.463806 | 38,400 | 6,291,456 |
| dense対照 | 0.032768 | 34,351,616 | 48,234,496 |

この一例ではソート版は全atom版より短時間だが、既存direct/denseより遅い。
汎用の速度採用・他の幅分布・他GPU・多次元へ一般化しない。
allocatedとreservedは別であり、GPUプロセス全体やshared/register込みの総容量を示していない。

### 実データのget_rangeの例

このjobのソート後のgroup別atom数は、group0..7で8,16,23,21,20,17,19,4。
例えばgroup2はslot [24,47)を占める。
forwardの出力担当tile1、つまりsites32..63に対し、get_rangeはgroup2の[27,39)を返した。
この12 atomを追加選別せずH → C/直接出力へ渡す。
全source_rowsと両軸のrangesは下記layout-report.jsonに保存してある。

### 証拠と再現

ignored証拠はrepo内の
`output/block128-sorted-20261026/l4job-c2c0e66412d8416c9d24f2f090c3a867/`へ保全した。
source/result archive、spec、receipt、raw stdout/stderr、pytest.log、runner.log、
benchmark.json、layout-report.json、verification.jsonを保持する。

- source archive SHA256：`780c1072f3ffaf9c9b478bee150dd9f6468369a122e46a7c721efbd9ef4f2118`
- result archive SHA256：`f14c5e15bef2d43211212f8b0e6e1944017684889c80a6dfad6a784703a8f530`
- workerの全412 source hashを凍結archiveと照合。全7 recordのsource commit/hashが一致。
  layout-reportのnative側5ファイルもarchive・現在の作業sourceと照合した。
- 結果回収成功と、共有poolの全slot停止・未完了jobなしを確認した。

```bash
python -m pytest -q tests/test_hierarchical_fourier_block128.py \
  tests/test_hierarchical_fourier_block128_layout.py
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-block128-sorted.json \
  --case benchmarks/cuda/linear/cases/block128-sorted-fourier-m8-mixed-128-a128.json
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-block128-sorted.json \
  --case benchmarks/cuda/linear/cases/block128-sorted-fourier-m8-mixed-128-a128.json \
  --source-commit 4d0ba1ce5286e978389af7238fc987df56fdfc89 \
  --output output/block128-sorted-reproduction.json
```

GPUコマンドは上のsourceをcheckoutした対応環境で実行する。
最初の開発範囲は引き続き128サイト・各側1次元の有限区間で、周期格子や一般次元対応は未実装。
