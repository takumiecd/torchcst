# 全学習stepのCUDA Graph検証

基準はmain/#125の `7aa7c89e42a4dd515630f8358e4ade880ddea482`。
#126/#127は閉鎖済みのまま保全し、本branch `kernel/full-step-graph` とは混ぜない。
目的は真B4の全モデルforward/loss/backward/実optimizer/指標集計を1 replayにまとめ、
外側のguard・入力供給・mutation通知込みの利益を検証すること。

## 限定した試作

最初に同一凍結初期値、N1024/A52428/FP32公開・FP64内部、TF32 off、native CSTAdamW、
同じowner-window-recompute算術と既存stable sortでcapture成功とbyte状態を確認する。
専用native入口はこの研究branchだけに追加し、通常のcapture拒否やdispatcherを変更しない。
専用入口もshape/dtype/軸/launch等の検査を維持する。既存sort/optimizerの大規模な再実装が
必要なら、この最小実験では止めて記録する。

速度比較へ進む条件は、通常eager・準備済みeager・Graph間でのlogits/loss/全grad/全model・
Adam状態のbyte一致、毎replayの正しいstep進行、最新Parameterからの分類である。
単なるcapture成功や最終P一致だけを対応完了とはしない。hooks、live設定、storage、
retained reader、version通知、有限値検査、入力・状態の寿命も対応範囲と検査対象にする。
対応外はそのstep全体をeagerへ戻し、guard費用を計時へ含める。

成立した場合はCST/dense各3実行方式、初期8更新後・基準学習後、256 minibatch×6 paired反復。
同じallocation内で条件順を釣り合わせ、各条件は別process。区間前後だけ同期する。
メモリはoptimizer準備・warmup・capture前からreplayまでのallocated/reserved実peakを測る。
準備時間とbreak-evenも別記する。短区間を45,000 stepの実学習結果とは呼ばない。

10%短縮は継続判断の目安で、予測値やユーザー指定ではない。
メモリ増加ゼロもユーザーの確定条件ではないため、増減を隠さず速度と併記する。
raw driver・結果・checkpoint・source hashはignored `output/full-step-graph/` とdefault poolへ保全する。
現時点ではcapture・正しさ・速度は未測定で、採用・mergeは行わない。
