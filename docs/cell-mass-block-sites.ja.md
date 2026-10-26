# profileを固定したblock担当site数の比較

2026-10-26。[C共有](cell-mass-c-reuse.ja.md)後の数学・profile・周期を固定し、
Y/dXの実行上の担当範囲だけを変える。初期の比較はsites_per_block=32/16/8。

続く[atom走査・位相・同期の診断](cell-mass-deep-analysis.ja.md)では、
site担当だけを小さくしても各blockの全atom走査が残ることを確認し、
atom軸の分担と位相共有をproductionとは別の診断コピーで比較した。

## 数学のタイルと実行の担当範囲

`CellMassAxisSpec.tile_sites`はprofileの周期・選択窓を決める宣言であり、
今回のチューニングでは変更しない。新しい`physical_linear(..., sites_per_block=...)`は
1blockに渡す最大site数だけを指定する。Axis configへ混ぜず、forwardごとのctxへ保存し、
backwardのdXでも同じ実行設定を使う。H/G・Z・Parameter勾配の割り当ては同じ。
thread数128、batch幅4、atom chunk32、shared上限、係数計算、dtypeも固定。

```python
# 入力profileのtile_sites=16 / 出力profileのtile_sites=32、周期48/96は固定。
y = physical_linear(spec, x, atoms, z_method="merge", sites_per_block=16)
```

受け付ける設定は8/16/32。既定は比較元と同じ32。
小さい入力ではsite数と従来の4leaf分の上限も使うため、実際の担当幅は
min(site_count, sites_per_block, 4*leaf_sites)。
これは限られた実行方式の比較であり、数学のprofileを作り直す処理ではない。

## 並列度と再計算の交換

128site・819atom・batch4・mode8、入力profile tile16/出力tile32の場合。
基底は17個。各出力要素には一つのblockだけが書き、atomicやglobal部分和は追加しない。

| 担当site数 | Y/dX各grid | 各kernelのprofile準備 | YのCスカラー数 | dXのCスカラー数 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 4block | 3,276回 | 272 | 544 |
| 16 | 8block | 6,552回 | 544 | 544 |
| 8 | 16block | 13,104回 | 1,088 | 1,088 |

profileタイルよりblockを小さくすると、そのタイルのCをblockごとに作り直す。
block数を増やして同時に仕事を持つSMを増やせても、各blockのatom loopは短くならない。
担当するsiteの基底展開だけが短くなる場合もあるため、grid数だけで採用しない。

同じextension binaryで3設定を切り替え、Y/dX/全atom勾配を独立参照と照合する。
速度とメモリは検証から切り離したfresh processで測る。
初回32→16→8、次回8→16→32の順にし、各設定2processで比較する。
1process内の複数時間サンプルを独立runとは数えない。

## 初回のGPU比較

source commit `793d1764`。CPU全体は2006 passed / 2852 skipped / 既存JIT警告18件。
NVIDIA L4（58 SM）、Torch2.11.0+cu130、CUDA13.0で対象104件
（CUDA76件・CPU/math28件）がpassed、failed/skippedは0。
819atomのFP32/FP64両方で、全設定のY・dX・全atom勾配は32site設定との差が全て0。
独立Torch参照への最大絶対誤差も全設定で同じだった。
FP32はY5.96e-8、dX3.73e-8、全Parameter勾配5.96e-8、
FP64はY1.80e-15、dX2.33e-15、全Parameter勾配1.50e-15。
近似方式・精度・勾配の定義を変えていない。

時間は同じextension binaryを使い、各設定2つのfresh processで測定した。
128×128・819atom・batch4・FP32・mode8、spacing1、sigma0.2〜12、seed120。
入力T=16/出力T=32、K=3、周期48/96、merge Z・1要素4thread、直接多項式係数。
各processはwarmup5回後15サンプル。prepareと入力検査を含むphysical forward+backwardで、
optimizer・Graph・公開CSTLinearは含まない。JIT/build時間は除外。

| 担当site数 | Y/dX各grid | process1中央値 | process2中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 4block | 5.170 ms | 5.211 ms | 85,504 byte | 2 MiB |
| 16 | 8block | 4.633 ms | 4.626 ms | 85,504 byte | 2 MiB |
| 8 | 16block | 4.664 ms | 4.626 ms | 85,504 byte | 2 MiB |

16siteは32siteより約10.4%/11.2%短縮。8siteを16siteより速いとは判断しない。
allocatedは全て83.5KiBで、元のX/dY/atomの17KiBを含む。GPU process全体のVRAMではない。
shared上限9,984byte/block、register90/thread、stack40、表示上LOCAL0も共通。
コンパイル時リソースと、SM busy/warp占有率/実trafficの計測は区別する。

別の段階別GPU event中央値は次の通り。和で全体時間を置き換えない。

| phase | 32site | 16site | 8site |
| --- | ---: | ---: | ---: |
| prepare | 0.388 ms | 0.391 ms | 0.400 ms |
| Zx | 0.0369 ms | 0.0399 ms | 0.0369 ms |
| Zy | 0.0358 ms | 0.0358 ms | 0.0369 ms |
| H | 0.196 ms | 0.196 ms | 0.210 ms |
| Y | 1.618 ms | 1.608 ms | 1.611 ms |
| G | 0.215 ms | 0.215 ms | 0.215 ms |
| dX | 1.748 ms | 1.246 ms | 1.251 ms |
| Parameter勾配 | 0.621 ms | 0.622 ms | 0.624 ms |

主な改善はdX。32site blockが二つの入力profileタイルのCを作っていたのに対し、
16site blockは一つを担当する。C総数は同じでもblock内の仕事が減る。
Yは32siteでも一つのprofileタイルのCを担当していたため、16/8へ分けても
各blockのatom loopとC生成は短くならず、実測時間はほぼ同じだった。
8siteは16siteよりprofile準備やCの重複を増やすが、追加の改善は観測していない。
このパターンは担当範囲と測定の整合を示すもので、新しいhardware counterを取得した結果ではない。

## 初回の証拠

job `l4job-dd206c68dd9547208ada17537bbddc6e`。
source archive SHA256
`80336765bfcfe6e21a466da5cc378014a1974c66824a7d8dff481db729d877b3`、
result archive SHA256
`ae9068af5fcc16e9c157f408f05d93011f880dac53cf3dd73f5ec04805f0077e`。
全6測定processのnative binary SHA256は共通で
`ee9c0924e3f448da0dd79e6c23c78ca0ac3e6a10baf7fad5b0ec1578356d6131`。
両archive hashと321 runtime source filesのsnapshotとのbyte一致を照合した。
wheel/sdist buildと両artifactの321 runtime source filesのbyte照合も通過。

raw evidence・driver・CPU/build log・package照合は
ignored `output/cell-mass-block-sites-20261026/`、原本はhost共通poolのjob directoryへ保持する。
15サンプルを15独立runと数えない。dense・optimizer/Graphを含む完全学習step比較は未実施。
再実行: `python -m tools.kernel_dev test --suite cpu`。
CUDA/nvcc/Ninja環境では
`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。

## 独立jobでの32/16site再確認と選択

改善幅が約10%だったため、別のL4 jobで32/16siteを再測定した。
同じruntime source、同じ入力・精度・profile、同じ測定手順を使い、
32→16と16→32の順で各2つのfresh process、warmup5回後15サンプルずつを測った。
初回buildだけを独立した準備processへ分離し、実行時間へ含めない。
このjobではruntimeが不変なので104件のcorrectness suiteは重ねて実行していない。

| 担当site数 | process1中央値 | process2中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 32 | 5.341 ms | 5.339 ms | 85,504 byte | 2 MiB |
| 16 | 4.772 ms | 4.769 ms | 85,504 byte | 2 MiB |

約10.66%/10.68%短縮し、初回と同じ傾向を確認した。
段階別ではYが1.618→1.608ms、dXが1.746→1.249ms。
絶対wall時間はjob間で異なるため、異なるjobの32siteと16siteを組み合わせて比率を作らない。
初回と確認jobでそれぞれ同条件の32siteを基準にする。

この128site・819atom・batch4・FP32・L4のcaseでは16siteを推奨する。

```python
y = physical_linear(spec, x, atoms, z_method="merge", sites_per_block=16)
```

省略時は互換性のため32のままにする。他のshape/deviceへ16を自動適用するdispatchは追加しない。
8siteには16siteを上回る改善の証拠がなく、今回は選ばない。

Yを大きく改善するには、担当siteを細分化するだけでなく、各blockが走査するatom数や
Cの担当自体も分割する必要がありそうだ。これは次の仮説であり、今回実装・採用した結果ではない。
追加の部分和buffer・集約が必要になる案は、その時間とメモリを含めて比較する。

確認job `l4job-593c6d654272429492975e5be2db8426`。
source archive SHA256
`6909be197c6ea3060013b5383eaecd3281ff27d4ac30e04d04a19fad931d0081`、
result archive SHA256
`3f382e53856da5324cc5cedd8f0b1e02835b59aa2b87f3c257a1110e22e44834`。
4測定processのnative binary SHA256は共通で
`cadb2d9145d765640edb2f684c3af9b81e24d080fa1e8936c1091ab3ebc40af8`。
両archive hashと321 runtime source filesの現行sourceとのbyte一致を照合した。
証拠は同じignored directoryの`confirmation/`とpool job directoryへ保持する。

先の確認job `l4job-111e7f85039644118b89b66a43cbe808`は、初回JIT/buildを
測定用childの90秒timeoutへ含めたdriverの不備で失敗した。
測定artifactは出ておらず、全ての速度比較から除外する。
source archive SHA256は`245f8e50f458607c1dc61f28f7edb15d57c81a6026e45c1fc6d2cfa9c969d978`、
result archive SHA256は`b089a8e0ed65bfbce08bfc3f80a4d52ea369311bf457898e287b127096501e80`。
両hashを照合し、stderr・driver・source snapshotを保持した。カーネルのcorrectness失敗ではない。
修正版はbuildを別processの240秒枠へ分離し、上記の確認jobで正常終了した。

初回比較・除外job・確認jobの終了と、使用したpool slotの停止を確認済み。
