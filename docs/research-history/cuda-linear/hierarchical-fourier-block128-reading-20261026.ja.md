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
既存runnerの独立oracle・可変値のGraph replayを確認する。実行結果は検証後に追記する。
