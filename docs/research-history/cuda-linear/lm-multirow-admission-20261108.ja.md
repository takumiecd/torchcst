# FP32多トークンLM射影の対応範囲

[English](lm-multirow-admission-20261108.md) | 日本語

merge済みPR154と、別途検証済みのA100 MIG/RTX4090正方Linear観測を引き継ぐ。
公開登録しない明示plan `LM_LEVELS` / `LM_LEVELS_STREAM64`を、実LM射影の
正方N256/A2048、256↔1024/A4096、正方N512/A4096、512↔2048/A8192へ追加する。
R1..256、FP32、origin0/spacing1、Polar M8/base1/merge2、半径[.5,50]のみ。
既存packed FP32・geometry再利用・階層・窓再計算kernelの算術は変更しない。
既存Algorithm、既定selector、大規模・語彙・lookupの対応範囲は変更しない。

学習前に4090で新scopeと全勾配testを実行し、6shape・両保持方式を既存nativeと比較。
非連続入力、周期境界、振幅0、端数行、retained backwardを含む。
続いてresearch repoの既存qualification/learnerを使い、実256token行の全CPU FP64
Fourier Y/dX/全Atom VJP、独立Gauss積分、既存の同一勾配Polar optimizer比較、
CST Oも含む手計算masked Attention、未来token勾配の厳密0、全モデル更新、
保存と再開後の次更新を確認する。数値基準は変更せず、失敗/OOM/非有限/timeoutで
停止し、その試行を保全する。

最初の固定タスク族は幅512、pre-LN4block、8head、FFN倍率4、context128、batch2、
accumulation4、c8、seed17、FP32/TF32無効。fresh Dense、FFN単独、QKV単独、
QKV+FFN、QKVO、QKVO+FFNを比較する。CSTはlevelsを使用。stream64も別途検証するが、
学習recipeを黙って切り替えない。research protocolで実行前に要求1Mtoken
（976更新/実999424token）、20Mtokenのdata/LR horizon、minimum-001/q0初期化、
基準LR3e-4・Atom群×16を固定する。モデル精度と完全updateのメモリ・時間を
単体kernel結果と区別する。既存learnerのzero/LR設定と更新後診断は計時外であり、
その時計定義を維持する。

限定研究opt-inであり公開dispatcher採用ではない。runtimeは本repo、タスク・
qualification/configはcst-researchで管理する。source、入力、protocol、binary、
結果hashと原証拠を保全する。語彙embedding/出力・学習位置embeddingは後続段階とし、
直接lookup・重み共有・専用operatorの検証が必要である。
