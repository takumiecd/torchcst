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


## 初回探索の結果（検証済み）

L4で全35試行（資格8・lr探索6・本探索21）を完了した。
データ・ソース・結果archive、228 manifest files、全35試行の実checkpointを照合し、
入力層・出力層のweight/biasも全試行で実際に更新されたことを確認した。
実際に計測したソースは `640fbd25aa34b373c1aa38ce631a6bd6b6e39928`。
後続commitのreport/verifierは学習算術を変更しない。

**α=0.05を必要最小とは見なせない。今回の短期条件では0.005〜0.01が有望だった。**
8192・α0.01は同幅denseより約2.10倍速く、allocatedは約9.62%、reservedは約14.20%。
accuracyは70.300%対69.238%、ROC-AUCは0.779910対0.757109だった。
α0.05ではCST25.458msに対しdense11.680msであり、5%のままでは速度優位はない。
8192で5%から1%に下げると、CST自体の学習時間は約4.57分の1になった。

一方、幅1024の低密度CSTはROC-AUC約0.786〜0.787、学習約19〜20秒で、
8192のCST（AUC0.780、139秒）より品質・時間・メモリで良い。
**このHIGGS条件で幅を大きくする品質上の利点は確認できていない。**
またdenseの品質が大きいNで低下しており、N1024/α0.05だけで選んだlr0.001の
流用・学習不足・batch4等を切り分けていない。CSTの一般的な精度優位とは主張しない。

### 全21条件

validationは固定8,192件、seed17、1epoch/25,000更新。accuracyの差とAUCの差は
別の指標であり、非劣性や最小αの判定ではない。時間は準備・評価・checkpoint出力を除いた
同期完全学習step平均。最後の列はPyTorch allocator allocated / reserved MiB。

| N | 方式 | α | validation accuracy | ROC-AUC | ms/step | allocated / reserved MiB |
|---:|:---|---:|---:|---:|---:|---:|
| 1024 | dense | — | 70.105% | 0.773803 | 0.379 | 50.90 / 80 |
| 1024 | cst | 0.005 | 70.764% | 0.785942 | 0.779 | 32.61 / 40 |
| 1024 | cst | 0.01 | 70.972% | 0.787075 | 0.805 | 33.25 / 42 |
| 1024 | cst | 0.02 | 71.130% | 0.787305 | 0.793 | 34.54 / 44 |
| 1024 | cst | 0.05 | 71.069% | 0.783365 | 0.768 | 38.38 / 50 |
| 1024 | cst | 0.1 | 70.349% | 0.779653 | 0.897 | 45.87 / 86 |
| 2048 | dense | — | 70.544% | 0.775771 | 0.642 | 111.40 / 120 |
| 2048 | cst | 0.005 | 70.459% | 0.783252 | 0.770 | 36.10 / 44 |
| 2048 | cst | 0.01 | 70.569% | 0.784130 | 0.793 | 38.66 / 50 |
| 2048 | cst | 0.02 | 70.654% | 0.784386 | 0.940 | 43.81 / 86 |
| 2048 | cst | 0.05 | 70.764% | 0.782191 | 1.599 | 61.41 / 102 |
| 2048 | cst | 0.1 | 70.850% | 0.782457 | 2.736 | 90.27 / 142 |
| 4096 | dense | — | 68.787% | 0.758406 | 2.981 | 352.40 / 360 |
| 4096 | cst | 0.005 | 70.898% | 0.784140 | 1.073 | 46.97 / 86 |
| 4096 | cst | 0.01 | 71.350% | 0.786282 | 1.612 | 58.63 / 100 |
| 4096 | cst | 0.02 | 69.995% | 0.781123 | 2.658 | 79.48 / 140 |
| 4096 | cst | 0.05 | 70.374% | 0.779585 | 5.944 | 148.22 / 202 |
| 4096 | cst | 0.1 | 70.203% | 0.777693 | 11.502 | 260.35 / 352 |
| 8192 | dense | — | 69.238% | 0.757109 | 11.680 | 1314.40 / 1324 |
| 8192 | cst | 0.01 | 70.300% | 0.779910 | 5.573 | 126.50 / 188 |
| 8192 | cst | 0.05 | 70.276% | 0.773966 | 25.458 | 492.71 / 658 |

数値・lr探索・hardware・archive SHAは
[higgs-density-scale-results-20261105.json](higgs-density-scale-results-20261105.json) に保存した。
準備込み時間と比率もJSONの各observationにある。

### 数式と解釈範囲

中間層のatom密度とモデル全体の学習変数数は次の関係を持つ。

$$
A=\lfloor\alpha N^2\rfloor,\qquad
P_{\mathrm{dense}}=N^2+31N+2,\qquad
P_{\mathrm{CST}}=4A+31N+2.
$$

従ってα0.005は中間層のscalar Parameter比で約2%、α0.01は約4%。
GPUメモリ比はこの比率と同一ではなく、実測allocated/reservedを別に使う。
最小αを確定する前に、十分な学習予算・各幅のdense tuning・複数seed・全validationを
用いて許容差を事前固定する必要がある。今回の結果を下限や普遍的なスケール則としない。

次の有望な範囲は、まずN1024/2048、αを0.001〜0.01に下げた探索と学習延長。
8192は実装の大規模検証・メモリ評価として有用だったが、同じ課題でさらに幅を増やすより
密度境界と収束を確認する方を優先する。新しい追加GPU探索はこの初回結果には含めない。

### 再実行

計測ソースcommitへcheckoutし、記録したTorch/CUDA環境と依存を用意する。
自前L4上での直接実行例は次。所有者の共有L4では同じcampaignコマンドを
共有poolへ提出するdriver内から呼び、直接CLI実行でleaseを迂回しない。
各outputは新しいパスを使う。

```bash
PYTHONPATH=src python -m benchmarks.training.higgs_data \
  --csv-path /path/to/HIGGS.csv.gz --output-directory output/higgs-data-replay
PYTHONPATH=src python -m benchmarks.training.higgs_campaign \
  --data output/higgs-data-replay/screen.npz --output output/higgs-campaign-replay
```

公式圧縮CSVのSHA256は結果JSONに記録したものと照合する。
学習側はnative B4を使い、testファイルはcampaignに渡さない。

### 正しさと証拠

- L4実機テスト161 passed / 2 skipped。
- 最小密度の全5,242 atomでY/dX/全20,968 scalar dPをpublic CPU FP64 Fourierと照合。
  最大絶対差はY1.3733e-9、dX1.5913e-9、dP1.1825e-6で既存gateを通過。
  別に31atomの独立48点quadratureも通過。
- α0.005/0.1・CST/denseのeager/Graphを2epoch×16更新で比較。
  計8回・288 Tensor field比較がbyte一致し、clock16/32も一致した。
- 全35試行の最終model/optimizer/gradientの実TensorをログのSHAと照合。
  source1,170 fileとresult228 manifest fileを検証。大規模をhashだけで済ませずraw checkpointを保持。
- データは全1,100万行を別block幅・Welfordでも再読し、全splitのraw/normalized SHAと
  統計を検証。旧HIGGS統計のFP32値ともbyte一致。testは品質評価に使っていない。
- CUDA binaryそのものは保存していないため、記録したbinary SHAの再hashはできない。
  固定ソース・runtime・実checkpoint・結果archiveの検証と区別する。

raw証拠: `output/higgs-density-20261105/`、および共有poolの
`l4job-99a6978940574057a7607ad1c34a9bba/results/`。
再集計は `python -m benchmarks.training.higgs_report --results <artifacts> --output <fresh-directory>`、
実Tensor照合は `python -m benchmarks.training.higgs_verify --job-directory <job-directory>
--source-commit 640fbd25aa34b373c1aa38ce631a6bd6b6e39928 --output <fresh-json>`。

### 先行失敗と停止

`l4job-d08ef957deb04a85849d22aba272a9e3` はColabのensurepip欠如でsetup前に失敗。
ジョブ専用pip targetに切り替え、システム環境は変更しなかった。
`l4job-bedac2f368b04687b31099128421a21b` はGPUテスト159件・独立oracle通過後、
初期state_dictのdict metadataをTensor扱いした保存処理で停止。
CPU上でextra-stateごと保存し、save/load round-trip試験を追加して修正した。
両失敗のarchive・manifestも検証して保持している。

最終ジョブは成功し、結果回収後にCLI session cacheが失われて通常stopが割当を解放できず、
pool recoverも同じ理由で失敗した。所有記録・結果回収・endpoint一致を確認し、
supervisor lock下で**当該所有endpointだけ**をunassignした。その後通常pool recoverを完了し、
server assignmentsが空、全pool slot stoppedを確認した。DBやslot記録を手編集していない。
