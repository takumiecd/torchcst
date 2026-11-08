# 振幅範囲を広げたenvelopeコホート

[English](radius-envelope-amplitude-range-20261109.md) | **日本語**

初期radius_fraction=1では半径が全て50、.5ではL=.5,U=50時の最小半径が5になる。
8192Atom・振幅上限1では、既存の有界初期化が.5/1のFFN-up目標分散.0004に届かない。
上限1/fraction1ではdown目標.00005にも届かない。これは初期化の実現性の問題で、
学習失敗ではない。.25は実現できる。

ユーザーはDenseの初期二次モーメント目標を保ち、全条件共通の振幅範囲を広げる方針を
承認した。明示的な別候補`RADIUS_ENVELOPE_WIDE_AMPLITUDE_LM` /
`RadiusEnvelopeWideAmplitudeLMAlgorithm`を用意し、amplitude_max<=32を対象にする。
従来のreference/cached Planは上限1を維持。写像・physical CUDA演算・端点・全4勾配・
optimizerフィールドの単位・公開デフォルトは変えない。定数準備は不変scalarだけを保存し、
Parameterとautograd保存は呼出しごとに保持する。自動dispatcher採用・速度改善の主張ではない。

公開初期化のseed123では、上限32でfraction0,.25,.5,1のup/down目標をすべて実現できた。
up振幅RMSは約.264,.395,1.156,10.029、downは.094,.140,.409,3.562。
相対moment誤差は.005以内。これは期待operator energyの目標であり、各サンプルの
実出力RMSの厳密一致ではない。モデルの実activation RMSも記録する。
上限1を保ったままfractionだけ変えると、初期スケールの不足と半径の学習を混同する。

GPUでは|a|<=32について、両方向・7/256行・半径上限4.7/50で、FP32物理行を保つ
FP64参照によるY/dX/全フィールド、retained backward、保存復元後のoptimizer継続を検査。
既存atol1e-4/rtol4e-3を維持する。native対象は幅512/expansion4、A8192、R1..256、
FP32/TF32無効、eager、M8。別コホート・Graph・公開デフォルトには別途証拠が必要。
