# owner-windowの同一状態・同期方式・checkpoint交差診断

PR #126の単層13.4〜13.6%短縮と、MNIST全体の改善なしを説明する短い診断。
新しいkernel最適化や既定dispatcherの変更は行わない。
元の結果は [followup記録](owner-window-followup-20261103.ja.md) を参照する。

## 条件と順序

`benchmarks.training.owner_window_diagnostics` は既存MNISTのmodel、optimizer、data loader、
正規化を再利用する。初期値と両学習後checkpointをそのまま使い、長期学習をやり直さない。

1. 同じP/x/dYでbaseline/C/raw/DFT/combinedを比較。同一Plan再実行を対照とし、
   Y/dX/dP、nativeの入力window、全モデルのlogits/loss/grad、更新後の全状態を検査する。
   同じgradientを両Planへ与えた更新も比較。共有状態から短区間を逐次比較し、最初の更新差を
   発見したら、その直前のcheckpointとminibatchを保存して一回の再実行で照合する。
2. 同じモデル・全optimizer状態から、同じ256 minibatchを両Planで実行。
   毎step同期／区間同期の2方式、4反復、順序を反転して比較。主指標は区間wall timeの平均。
   初期performance状態だけはbaselineの共通8更新後。学習後の2状態は既存checkpointを使う。
   事前準備をwarmupし、Parameter/moments/stepを元のstorageへcopyして正確に復元する。
3. 両学習後checkpointをそれぞれ両Planで動かし、状態差と実装差を分ける。
   別passのCPU/CUDA profileでnative window kernelを確認し、timelineを保存する。

性能計測はeager/TF32 off、CPU math threads1、FP32 Parameter、内部FP64、真B4。
全train/test Tensorと全epoch用orderをGPUへ常駐させ、gather、cross entropy、全層backward、
実optimizer、GPU loss/accuracy集計を含む。評価・checkpoint復元・計装はwall timeから除く。
profileの時間を未計装の性能値と混ぜない。CUDA Event間隔にもCPU供給による空き時間が
入り得るため、それを純粋なGPU演算時間とは解釈しない。

native保存windowの未使用bankは未初期化なので比較せず、実際のnative keysが要求する
有効bankのFP64領域だけを照合する。CPU診断とnative owner keysの差も別に記録する。
profileではTorchDispatchMode等によるfallbackを避け、実際のkernel名を確認する。

## 準備時点の検証

CPU上で初期値・両checkpointをbaseline/combinedの各modelへ復元し、model/optimizerの
全Tensorが一致することを確認した。Parameter・exp_avgを意図的に変化させ、同一storageへ
復元する検査も通過。lint/formatを確認済み。これはGPU結果ではない。

入力checkpointとdatasetはignored outputへ保全し、pool driverへhash付きpayloadとして
含める。通常のsource snapshotへ非ignoredの生データを混ぜず、pool以外の直接uploadはしない。
driverとraw resultsは `output/owner-window-diagnostics/` とdefault pool job原本へ保持する。
結果・job/source/result hash・判断は実行後に追記する。
