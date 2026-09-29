# Strip所属探索の全station走査を除去

> この文書の `prototypes` 実行例は commit `190a1bb3fa6259fde493a30d901d5cab7c2ebb82` 時点の
> 歴史的な再現手順です。削除したコードの参照方法は
> [旧実験コード](../../../../../experiments/cuda/linear/legacy-prototypes.ja.md)を参照してください。

2026-09-26。[目標atom密度の再測定](../../../../../experiments/cuda/linear/notes/target-atom-density.ja.md)で見つかった所属探索を改善した。

続く[I/B分類と一時メモリの改善](batched-support-preparation.ja.md)では、分類のまとめ処理と
整数配列の再利用・32bit化を行った。以下はその変更前の測定である。

## 変更

`CircleRouting`に固定pitchと局所候補検索の適用条件を保存し、GPU準備で条件を満たす場合は
`owners_local`を使う。CUDA Graph内やwarm forwardで新しいhost側tensor読み出しはしない。
固定chart bufferの変更時は既存のplan version検査で再構築する。

各atomについて、次の処理を行う。

1. decode済み座標のatan2から円周上のarcを求める。
2. 最初のstationからの周期内距離をpitchで割り、候補番号を求める。
3. 候補番号の前後2station、および両端のstation、最大7候補を調べる。
4. 旧実装と同じFP32距離式で比較し、同距離なら小さいstation番号を選ぶ。

Stripの区間は順序付きで互いに重ならず、全体が円周1周未満に収まる。
厳密演算なら位置を含む/直前の区間とその次、および円周の端を調べれば最近傍が含まれる。
候補推定のFP32丸めを考慮して前後2stationまで広げ、候補内の判定式は変更しない。
1 atomあたりの走査数がGから最大7になり、該当する形状ではO(A*G)をO(A)へ置き換える。

適用条件はFP32・3station以上・有限な幾何・区間の非重複と1周未満、さらに
`256 * eps_float32 * (abs(start) + period) < pitch`。
座標の丸め幅をpitchに対して小さく保てない形状は既存の全走査に戻す。
すべての将来サイズにこの高速経路が適用されるとは限らない。
今回の4096²/8192²（4096/16384station）は高速経路を使う。

atomの移動量には新しい制限を設けず、I/B区分・stable sort・pack・計算本体は維持した。
全Wを保存する処理はCST forwardに追加していない。

## 正確さ

A100で次の4ファイル、計133テストが通過した。

```bash
PYTHONPATH=src:. python -m pytest -q --color=no \
  tests/test_local_strip_routing.py tests/test_large_strip_preparation.py \
  tests/test_triton_linear.py tests/test_block_strip_linear.py
```

- 3/4/31/1025/4096/16384stationの全stationの開始・終端・中間境界とその前後1ULP。
- 正負の開始位置、零spacing、大きい円周上の空き区間、端数の最終station、非連続center配列。
- signed zeroとatan2の継ぎ目、円周を何度も超える位置、ランダム位置。
- owner/order/offsetを旧GPU全走査と整数完全一致で比較。抽出点は独立Torch参照とも比較。
- 極端な座標桁でのfallback、CUDA Graph再生後の大きな位置更新。
- 既存の準備・forward・勾配・I/B継ぎ目・更新・host同期なしの検証。

最初の追加テストはLinePatternの生成引数を誤り8件が失敗したため修正した。
既存125テストは初回から通過し、修正後は追加8件も通過した。

大型ベンチマークでは同じ入力で旧/newのprepared全配列を完全一致で検証し、
全atom参加のcanonical評価1,152点を使ってdense対照を点検した。
両方式の全出力比較は従来のatol=rtol=3e-5を維持する。

## 測定方法

A100 80GB PCIe MIG 3g.40gb、42 SM、Torch2.6.0+cu126/Triton3.2.0。
FP32・TF32無効、CST tile64×64、入力128行、seed21。
atom数は `round(N*K*0.05)`。4096²が838,861、8192²が3,355,443 atoms。
初期配置はすべてI区分で、学習後の偏りやB多数の性能を代表しない。

同じプロセス・同じ固定geometry・同じatom・同じ入力で新旧を交互に計測。
benchmark専用にplanの局所候補検索flagを切り替え、旧経路を再現する。
新旧どちらもdecode、I/B分類、stable sort、packを含む。
CUDA Graph3rounds、rep=20msの中央値。dense W生成とJIT compileは計測外。

計算本体は既存の局所合算を使用。atom直接方式は全出力の正確さを確認し、
今回その速度の反復測定は行わない。計算本体のコードは変更していないため、
測定対象を所属探索の変更による差に絞った。

## 結果

単位ms、中央値［最小–最大］。

| 経路 | 4096² | 8192² |
| --- | ---: | ---: |
| dense | 1.380 [1.379–1.405] | 6.413 [6.410–6.419] |
| 準備・旧全走査 | 83.585 [83.413–83.708] | 1123.968 [1121.966–1125.198] |
| 準備・局所候補 | 18.818 [18.818–18.827] | 75.238 [74.863–75.248] |
| 局所合算forward・旧全走査 | 198.998 [198.722–200.224] | 1565.777 [1564.429–1566.744] |
| 局所合算forward・局所候補 | 134.516 [133.691–135.278] | 521.495 [521.173–521.721] |
| 局所合算forward・準備済み | 113.950 [113.646–113.961] | 443.339 [443.248–443.429] |

改善率：

- 4096²：準備が4.44倍高速、局所合算forward全体が1.48倍高速（時間32.4%減）。denseの97.4倍の時間が残る。
- 8192²：準備が14.94倍高速、局所合算forward全体が3.00倍高速（時間66.7%減）。denseの81.3倍の時間が残る。

## プロファイル

CUPTIの通常実行3回平均。CUDA Graphとは別測定であり、hardware stallやcache hit率は未測定。

| 形状 | 旧所属探索ms | 新所属探索ms | 新forwardのI/B分類ms | 新forwardの計算本体ms |
| --- | ---: | ---: | ---: | ---: |
| 4096² | 65.4465 | 0.0723 | 15.2725 | 114.4360 |
| 8192² | 1032.6389 | 0.2824 | 67.1109 | 445.2200 |

全station走査は主要な費用から外れた。残る準備の中心はI/B分類であり、
8192²では約67ms。局所合算の計算本体は約445msで、現在の全体時間の大半を占める。
所属探索の削減だけではdenseとの差は解消しない。

## メモリと次の作業

追加allocationピークは変わらず、4096²が65.894MiB、8192²が259.921MiB。
既存input/model/prepared/比較用dense Wを除くPyTorch allocatorの追加ピークで、
出力を含む。総学習メモリやCUDA contextを含む値ではない。
今回の変更はkernel内部の走査削減で、準備の外部配列の数は減らしていない。

次はI/B分類と準備の一時配列を整理し、その後に局所合算の計算本体を改善する。
backendの標準選択は今回変えていない。既存のGPU準備は適用条件を満たせば自動で新探索を使う。

## 再現と保存

- 実装commit：`bc7cab580d0798c202dcc9264a9ecc6f96b974b9`。
- テスト修正・測定source：`ac26fae0db4c13853b33224f9ae8ac8c4aafb26e`。
- Source archive SHA256：`db5dacb37ae68bd027720d5d3e94365f2969743e687a7962c72170118de92088`。
- リモート：`srv11/cst-lab/torchcst-local-routing-ac26fae`。
- 結果bundle SHA256：`51be6c2a019e7291c223d7307c8548cd9693a668afbd7e7bbfae06deac7f57b0`。
- ローカル：`output/triton-a100-20260926/local-routing/`。

```bash
PYTHONPATH=src:. python -u -m prototypes.benchmark_local_routing \
  --output-dir study --source-commit ac26fae0db4c13853b33224f9ae8ac8c4aafb26e
```

source/結果のリモート・ローカルhash一致、133テスト通過、両ケース完了、A100/42 SM、
prepared全配列の完全一致、10件の出力比較（dense自己比較2件を含む）、36 timing samples、
6 profilesを確認した。Ruffとdiff whitespace検査も通過した。
