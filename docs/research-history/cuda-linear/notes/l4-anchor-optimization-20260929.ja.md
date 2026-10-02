# L4・1024² CSTのatom×site削減と次の算法（2026-09-29）

## 実験条件

Colab ProのNVIDIA L4、PyTorch 2.11.0+cu128、Triton 3.6.0。1024×1024、52,429 trainable Triweight atoms（5%）、FP32、TF32無効。M=16/128は線形層へ入る総行数。Graph capture後24 replayの同期wall中央値を採った。全方式は同じ初期atomとランダムMSE targetで27回AdamW更新し、baseline・anchor・必要時denseを同一プロセスで交互測定した。校正されたPOD基底の訓練seedは21・37、未知seedは53・71。全Wは校正または照合時だけ生成し、アンカー学習ステップには置かない。

## 何が速くなったか

固定補間の16×32アンカーは64×64ブロックの1/8 siteで真のCST atomを毎回評価する。入力と出力の固定基底を共有し、atom勾配はアンカー行列の随伴から求める。学習対象は引き続きCST atomの振幅・中心であり、アンカー行列を自由パラメータとして学習していない。

L4のM=16、16×32では、以下の逐次変更を測った。実験間にGPU状態の揺れがあるため、各行はその実験内の対応比較を見る。

| 変更 | anchor完全ステップ | 効果・判断 |
| --- | ---: | --- |
| 初期固定アンカー | 0.622 ms | 既存CSTの約1.74倍速 |
| ブロック局所basis | 0.618 ms | dense block-diagonal basisより約12 µs速い比較もあり、以後採用 |
| atom勾配を8候補並列 | 0.512 ms | 1候補版0.616 ms。24×48ではレジスタspillのため4候補に留める |
| forwardのatom並列化 | 0.519～0.531 ms | 1候補版が最速または同程度なので不採用 |
| fused AdamW | 0.475 ms | foreach 0.520 ms。baselineにも同じfused設定を適用 |
| Torus decodeのforward/VJPを各1 kernelへ融合 | 0.368 ms | PyTorch decode版0.477 ms。独立PyTorch VJPとの相対L2は約1.0×10⁻⁷ |
| 実アンカーsiteのAABBで候補絞り | 0.365 ms | 元16×64 tile AABB版0.369 ms。値・勾配は独立全atom参照と約2×10⁻⁷／1.4×10⁻⁷ |

最終16×32設定の同一L4 runでは以下。denseはCSTから生成した全Wを自由パラメータとし、同じfused AdamWを使う。従って速度の比較対象だが、学習可能なパラメータ数とモデルは異なる。

| M | 既存CST | trainable 16×32 anchor CST | dense | anchor / 既存CST | anchor / dense |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.056 ms | **0.365 ms** | 0.083 ms | 0.347 | 4.42 |
| 128 | 1.153 ms | **0.382 ms** | 0.173 ms | 0.331 | 2.25 |

16×32、M=16のprofile上のGPU kernel時間は約358.5 µs。そのうちfused AdamW約75.8 µs、atom勾配約56.4 µs、アンカー重み生成約51.6 µs、支持判定約41.7 µs、bucket histogram/scatter計約20.5 µs。Nsight Computeの対象kernelではL2 hit率が約85～88%で、DRAM帯域は飽和していない。したがって次の大幅改善には候補atomやsiteとの評価回数自体を消す方策が要る。Nsightの単体kernel時間はGraph完全ステップ時間と直接比較しない。

## アンカー数と精度

等間隔・局所三次補間では、M=16初期出力誤差は8×16で11.33%、8×32で6.36%、12×32で1.51%、16×32で0.75%、24×48で0.16%。16×32より減らすだけでは速度改善も小さく、誤差が大きくなる。

次に、校正seed21・37の各64×64 CST blockから共有行POD基底と4系列ごとの列POD基底を学び、DEIMでサンプル点を選んだ。学習時に全Wを生成せず、固定基底と選択siteのみを使う。未知seed53・71の初期状態では次の通り。射影誤差は共有部分空間そのものの限界、補間誤差は実際に選択siteだけから復元した値。

| PODアンカー | site/block | seed53/71 射影W誤差 | seed53/71 補間W誤差 | seed53/71 M16出力誤差 |
| --- | ---: | ---: | ---: | ---: |
| 8×16 | 128 | 1.56/1.63% | 2.52/2.64% | 2.40/2.82% |
| 12×24 | 288 | 0.39/0.40% | 0.88/0.91% | 0.84/0.97% |
| 16×24 | 384 | 0.16/0.16% | 0.43/0.43% | 0.43/0.47% |

8×16 PODは同じ128 siteの等間隔三次補間の約11%出力誤差を大きく減らしたが、現16×32の水準には届かない。12×24は現16×32よりsiteを44%、16×24は25%減らす。POD基底で実際にatomを学習した結果は以下。seed53/71は基底校正に使っていない。

| 設定 | M | seed | 完全ステップ | 初期出力誤差 | 初期atom勾配誤差 | 27更新後の出力差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12×24 POD | 16 | 53 | **0.351 ms** | 0.914% | 0.909% | 2.314% |
| 12×24 POD | 128 | 53 | **0.361 ms** | 0.894% | 0.841% | 4.038% |
| 16×24 POD | 16 | 53 | 0.420 ms | 0.442% | 0.448% | 1.503% |
| 16×24 POD | 128 | 53 | 0.425 ms | 0.437% | 0.424% | 3.004% |
| 16×24 POD | 16 | 71 | 0.421 ms | 0.413% | 0.418% | 1.543% |

PODのアンカー値とそのatom VJPは、128²で全atomを直接評価したPyTorch参照と相対L2約2.0×10⁻⁷／1.5～1.6×10⁻⁷で一致。これは**近似モデル自身の勾配実装**の検証であり、真のCSTとの0.4～0.9%差を消すものではない。固定ランダムMSEで27回更新した結果に過ぎず、実タスクの精度・長期学習の品質はまだ測っていない。

## 複数モデルとの再相談と次の実験

Grok 4.7、Gemini 3.1 Pro、Claude Opus 5.5へ現実測を渡して再相談した。各回答は未検証の提案として扱い、現行候補リストと計算量を突き合わせた。Geminiの空間hashは既存の局所候補リストと重複し、乱択atomは現行候補より多くなる。Claudeの一般4D多項式案は項数を過小評価しており、そのままでは使えない。Grokの近接atomクラスタを二次momentでまとめる案は検証余地があるが、中心・precisionの5次元moments、振幅重み、支持境界での直接評価が必要。まずクラスタ内部に入る実アンカーpairの割合を測らないと速度改善は主張できない。元回答には未検証の推論や誤りがあるため、[原文](l4-anchor-optimization-consultation)と採否を分けて保存した。

特にTriweightの`max(q,0)^3`は支持境界で二階微分まで連続であり、「境界で一次勾配が不連続」という回答は誤り。dense完全ステップ時間も別のパラメータ化・学習則の実測値であって、CST算法の厳密な下界ではない。

より直接的な数学案は、Torusのstation内でTriweightの支持内関数を**局所10項基底**に展開し、各atom×列の支持行区間へ係数をrange-addするもの。forwardは行prefixでアンカー値を作り、backwardはアンカー勾配×各基底のprefix区間和からatomパラメータ勾配へ戻す。理想的な単位円幾何なら支持内で正確に表せるため、atom×行×列の評価をatom×列×10項へ変えられる。ただし現行FP32のsite座標丸め、支持端の判定、10係数のscatterとprefixの費用が未検証。最初にFP64 oracleと現行CSTのW・atom勾配を比べ、16行と64行の局所展開で係数範囲・支持端の誤分類を測る。通った場合だけL4でrange-add込み完全ステップを試す。

固定POD基底とsite位置はGPUに保持して再利用できる。atomパラメータが毎step動くのでアンカー値は再計算する。候補リストのstep間再利用には、前回除外した全atomが更新後も支持域外に留まる保証が必要であり、現時点では毎step再構築する。

実装は [`anchor_atom_training.py`](../retired-code.ja.md)、[`torus_decode_trainable.py`](../retired-code.ja.md)、[`sweep_pod_anchor_basis.py`](../retired-code.ja.md)。完全ステップ再現は [`profile_anchor_atom_training.py`](../retired-code.ja.md)。生JSON・Nsight CSV・POD校正基底は [`benchmarks/cuda/linear/evidence/l4-anchor-optimization-20260929/`](../../../../benchmarks/cuda/linear/evidence/l4-anchor-optimization-20260929) に保存。代表実測ソースはcommit `d53086c`、Git archive SHA256 `4572c63e792ecb1afbe5841b8f9cf0965667f08255dc1cb0f4de350a76086f16`。Colab L4 sessionは停止し、active sessionなしを確認した。
