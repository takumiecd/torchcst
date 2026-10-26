# profileを固定したblock担当site数の比較

2026-10-26。[C共有](cell-mass-c-reuse.ja.md)後の数学・profile・周期を固定し、
Y/dXの実行上の担当範囲だけを変える。初期の比較はsites_per_block=32/16/8。

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
