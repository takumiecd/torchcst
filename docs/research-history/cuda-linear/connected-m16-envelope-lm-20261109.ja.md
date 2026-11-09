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
