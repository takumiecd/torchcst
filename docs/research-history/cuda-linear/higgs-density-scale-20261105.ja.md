# HIGGS中間層のatom密度と幅の探索

## 問いと比較条件（実行前）

`28 → N → N → 2`、ReLUを2個置く。入力・出力層の**種類**はdenseで、
重みを凍結しない。全3層を学習し、中間層だけをCSTまたはdenseにする。
中間層のbiasは両方式で無し。Nは1024/2048/4096/8192。

atom数は `A=floor(N*N/divisor)`、divisorは200/100/50/20/10、
従ってαは0.005/0.01/0.02/0.05/0.1。1atom4変数で、層のParameter比率は約4α。
既存のpublic radiusセル積分、M8、物理spacing1、radius範囲[0.5,50]を使う。
N増加時に相対支持域は狭くなる。radius分布も記録する。
旧HIGGS実験の球面Triweight・B128・中間bias付きとは別protocolであり、旧結果再現とは呼ばない。

基準denseは通常の`nn.Linear`初期化。CSTはpublic初期化を保ち、gain校正しない。
同じseedで入力層・出力層の初期値とデータ順は全密度・方式で共通。
dense初期値はαに依存しない。CSTの初期作用素はdenseより小さいため、
初期各層activation/logit RMSを記録し、初期化の影響をαの表現力と混同しない。

## データと探索段階

公式HIGGS CSV.gz（11,000,000行）をCPUで全走査する。
training[0,10400000)、validation[10400000,10500000)、
test[10500000,11000000)を維持し、標準化は全trainingのみから計算する。
初回探索の学習は先頭100,000行、validationは固定先頭8,192行。
testは別ファイルに保存し、探索jobへ転送せず、選択に使用しない。
α最小値の確定には全validation・複数seed・十分な学習時間での追試が必要。

真batch4、FP32、TF32無効、CPU math1thread、seed17。
全方式でAdamW proposal＋既存CST Polar更新、普通層/dense WD0.01、
atom raw WD0・振幅WD0.01。学習率だけを同じ候補集合から選択する。

初回の有限探索:

1. 正しさを先に確認。全5密度のprivate Graph Y/dX/全dPをeagerと比較。
   N1024の最小・最大密度で、全モデルeager/Graphの2epoch×16更新を比較。
   最小密度の全atom public CPU VJPと、独立quadratureを確認する。
2. N1024/α0.05でlr {0.0001,0.001,0.003}を両方式各5,000更新。
   validation lossで各方式のlrを独立選択する。全候補と選択を残す。
3. N1024/2048/4096で全5密度、N8192でα0.01/0.05。
   各幅に通常denseを1個置き、全て同じ100,000例・1epoch=25,000更新。
   accuracy、cross-entropy、ROC-AUC、時間、allocator allocated/reserved peakを記録。

これは1seed/1epochの探索であり、非劣性やα最小値を確定する実験ではない。
途中で数値・Graph資格の失敗があれば該当探索を停止し、失敗を保存する。
学習の進みと実行時間から次の学習延長・seed追加を決める。

## 実行・証拠

L4共有poolの1台を使う。A100は既存データ再利用のためstatusを確認したがHTTP403。
GPU種の比較ではない。HIGGSデータを公式配布元から取得し、source/array/file SHA256を記録する。

HIGGS専用private Graph入口だけに上記密度を許可し、旧MNISTの5%固定入口と
public入口のcapture拒否は維持する。タグ・shape・全層trainabilityもreplay前に確認する。

時間は同期したgather/forward/loss/backward/optimizer/metricsの完全step。
preparation/capture/復元/retirement込みも別に報告する。
optimizer作成前にpeakを一度resetし、以後capture・過去epoch評価を含む累積peakを報告。
GPUプロセス全使用量ではない。CST実行にdense W/dWを作らない。
学習中のvalidation評価もnative B4で行い、その時間は学習時間から分ける。

raw証拠はignored `output/higgs-density-20261105/` とpool job directoryに保存する。
最終test結果・複数seed結果・通常batchでの実用性は、この初回探索に含まれない。
