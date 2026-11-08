# 物理半径のLM FFN実行検査

[English（正本）](physical-lm-radius-20261108.md) | 日本語

非公開で明示登録する `LM_PHYSICAL_LEVELS` Planは、FP32の512→2048と
2048→512、8192Atom、1〜256行、M8/base1/merge2/origin0/spacing1、
物理座標 `(a,r,mu_out,mu_in)` のcell-mass FFNを対象にする。
default registryには追加せず、既存Polar Planの範囲も変えない。
実験用adapterであり、default変更や速度向上の主張ではない。

`Y = sum_j a_j (X v_j(r_j,mu_in_j)) u_j(r_j,mu_out_j)^T` では、
共有する物理半径の微分に**両側**のfactorの微分が必要になる。
`dY/dr_j = a_j [(X dv_j/dr_j) u_j^T + (X v_j) (du_j/dr_j)^T]`。
Polarのpacked reductionは意図的にこの成分を省いているため、物理版は既存の
完全なbatched physical parameter kernelに分岐する。FP32 momentsと
level-gridのdX/Y経路を使い、forwardの半径clampは追加しない。
Adam後の射影は研究consumerが別途扱う。

検査は独立したmetadataの採否とserialization、Polar/default registryの拒否、
両FFN方向の7/65/256非連続行、全4成分の勾配、同じgraphでの再backwardを含む。
consumerでは学習前に独立求積、CPU参照VJP、初期関数の一致、完全更新・optimizer・
checkpoint復元も確認する。CPU検査をGPU検証として扱わない。

protocolと結果はcst-researchのstudy `ffn-independent-radius-20261108` に記録する。
演算子source凍結後にsource/runtime/result hashを残す。公開physical宣言は既存の
ものであり、この予備比較は公開APIやselectorへの採用を行わない。

その後、Ada比較を演算子 `88c2e8a7c4aa312f4f096c5169d712ace49ac979`、
consumer `40df1a94349fccb5134413a5659eed69c83f108c` で完了した。
CPU/CUDA回帰231件、3条件の新規モデル検査、学習後checkpoint監査3件に合格。
各976更新・999424tokenで、DEV CEはPolar3.42441、物理半径学習3.42438、
物理半径固定3.42865。38ファイルのcompact archiveを独立にhash照合した。
SHA256は `dcae47ec745b72c548b35c4cccdf849f6de103d48b018eadba2994263d6b8c3f`。
[日英の結果・半径と勾配の曲線](https://github.com/takumiecd/cst-research/blob/feat/lm-attention-output/docs/ffn-independent-radius-20261108-results.ja.md)
を参照。短期1seedの開発実験であり、収束・非劣性・採用は主張しない。
今回の実装の物理更新時間は、観測されたPolar更新の1.71倍だった。
