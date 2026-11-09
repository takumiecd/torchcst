# Connected M16と包絡のLM用明示選択

[English](connected-m16-envelope-lm-20261109.md) | **日本語**

別のAlgorithmとして既存の微分可能なradius-envelope写像と、検証済みの行連続配置
physical M16 Algorithmを合成する。revision10のraw行は
`[amplitude, radius_fraction, output_center, input_center]`のままで、optimizerは
復号した物理半径ではなくこのraw行を更新する。振幅から半径への微分と、割合から
半径への微分を両方維持する。不変な写像定数とrevision9のphysical宣言だけをprepareし、
学習値は毎回復号する。小規模な既存FP64包絡写像は変更せず、M16本体と保存する
geometry/momentsはFP32を維持する。

採用判定はmetadataのみ。既定のconnected M16、半径[.5,50]内、振幅cap<=32と、
physical Algorithmのeager FP32・行数1～256・一次微分の条件を要求する。
既存Algorithmや本番dispatchの対応範囲は広げない。consumerは
`ConnectedEnvelopeAlgorithm`を明示登録し`CONNECTED_ENVELOPE`を選択する。
connected宣言のrevision10 checkpoint guardを維持する。

検証では、raw写像の完全なchain ruleを含む長方形の独立積分、実FFN寸法・Atom数、
Adam moment/counterを含むタスクモデルの直列化・再開、次の更新を変えない観測hookを
要求する。短期LM品質と幾何診断は別のcst-research研究で扱い、推奨recipeや公開既定値は
ここでは採用しない。

## direct階層の順序固定経路

最初の包絡タスク検証では全Aの数値oracleを通過したが、次更新のbit一致に失敗した。
最初に報告された埋め込みParameterの最大差は7.451e-9。physical実装には、異なる
小さいtile階層から同じsiteへ浮動小数atomic加算する部分があった。
別の`ConnectedOrderedAlgorithm`と`ConnectedOrderedEnvelopeAlgorithm`で、
level/row/siteごとのscratchを持ち、最終siteの担当が4つのdirect階層を固定順で加算する。
profile、包絡、optimizer、既存Planの識別子は維持する。
新経路はexpansionごとに`4*B*N`個のFP32 scratchを追加し、実タスクの時間・allocator peakは
別途測定する。公開deterministic/Graph modeは引き続き保証しない。
新しい明示選択経路の数値・bit一致・再開検証は未完了であり、以前の速度測定は直前の実装の結果。
