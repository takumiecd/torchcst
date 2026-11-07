# 入力行数だけを変える比較

[English](multirow-batch-20261108.md) | **日本語**

残る性能差でbatchが支配的かというユーザーの問いを確認する。Linearで効くRは
FFNの場合batch×tokenの平坦化した行数である。R256固定の幅比較だけではRの影響は特定できない。
N=16384・A=131072を固定し、R=8/64/256で初期Parameterと凍結x/target/dYの同じ先頭行を使う。
minimum-001/q0/Xavier、lr1e-4/decay0、FP32 IEEE/TF32 off、A100 MIG3g.40gbは同じ。

数値sourceとcompiled extensionは幅比較のimmutable3dd8115dを維持する。
input_prefix orchestrationはpayload SHA256、再生成したP・分散合わせreport、id/rows以外の
全Case項目を検証し、既存Linear workerを呼ぶ。変えるのはデータの選択で、学習・計時loopではない。
driverのGit commit・byte hashは数値sourceと分けて残す。

CST FP32並列とDenseをfresh processで比較する。A順は行数8→64→256、各CST→Dense。
B順は行数256→64→8、各Dense→CST。各観測はwarmup10＋primary計時51で、61更新全部が
allocator peak対象。データ常駐、zero_grad・dX消去・F/MSE/B・optimizerを含む。
その後、forward/loss・backward・optimizerを個別同期する診断を5更新追加する。
別の計装結果であり、medianを足し引きしてprimary時間を求めない。
primary state hash・peakは診断前（clock61）で、実際のprocessは66更新を実行する。

CST workerでは既存の初期・更新後の全Atom FP64 Fourier VJP検証を維持する。
source/GPU資格は幅比較を再利用し、そのR1/9/256境界・端数と全R256/A8N検証に続く比較である。
初期データ相違、数学検証不合格、nonfinite、OOM、時間上限で止め、失敗を保全する。
全12観測、primary51sampleと診断5sample、allocator peak、初期化・radius分布を比較する。
根拠なくRの影響を単一kernelへ帰属したり、この更新から精度・収束を判断したりしない。
公開default・Graph・別GPU・Atom/LR/カーネル則のsweepは含めない。

本書は計時前の固定protocolで、完了結果ではない。
