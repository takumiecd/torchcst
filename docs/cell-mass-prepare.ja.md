# キー生成と入力値検査をまとめる

2026-10-26。[denseの約2倍のstep時間という目標](cell-mass-dense-step.ja.md)に向け、
prepareで繰り返していたTorch演算とhostへの値の読み戻しを減らす。
profile・面積正規化・double内部演算・誤差基準は変えない。

## 処理と保存先

従来の入口はXのfinite検査、atomのfinite検査、sigma範囲検査を別々のTensor演算で行い、
それぞれ結果をhostのboolとして読んでいた。続いて座標ごとのremainder、加算、除算、
floor、整数変換などでソートキーを作っていた。

新しい入口では、メタデータと値を次のように扱う。

```cpp
// Python: Tensor値を読まない形状・dtype・device・stride・設定の検査。
validate_input_structure();
validate_cuda_support_and_execution_options();

// native.prepare_inputs: error_wordを0へ初期化し、同じstreamで投入。
prepare_inputs_kernel<<<...>>>();
// 各threadはXの1要素とatomの1行を、範囲に入る場合だけ調べる。
// 有効atomにはoutput/inputのtileキーを一つ書く。

// Python: 値を読むのはエラー値1個だけ。
errors = error_word.item();
raise_on_invalid_values(errors);
rows = stable_argsort(keys);

// ここから従来と同じZ/H/Y、backward。
```

`prepare_inputs_and_layout`が値の確認からstable sortまでを呼び出す。
Z/H/Yは検査に成功した場合だけ起動する。
standaloneの`prepare_layout`は従来のTorchソート補助として保持し、通常のforwardでは使わない。
Torch参照は`validate_inputs`でメタデータと値の両方を従来通り検査する。

| データ | 場所 | 寿命 |
| --- | --- | --- |
| 元atom・X | 元のglobal配列 | 従来通り |
| keys[A] | temporary global、int64 | prepareとsortの間 |
| error_word | temporary global、int32 1個 | hostで読んで検査するまで |
| rows[A] | global、int64 | backwardの最後まで |

atomコピー・global係数表は作らない。keysとerror_wordはctxに保存しない。
error_wordは有効入力では0のままで、atomicは実行しない。
無効値を発見したthreadだけatomicOrでbitを立てる。

- bit1: Xに非finite値
- bit2: atomに非finite値
- bit4: sigmaが範囲外

hostはfiniteエラーをsigmaエラーより優先して、従来と同じValueErrorを返す。
ゼロ振幅でも幅・中心の検査とキー生成を行い、atomを除外しない。
空のX/atomにも対応する。Graphはerror_word.item()があるため引き続き未対応。

## キーと丸め

各軸でchart長Q、tile幅T、site間隔dとして

\[
q(\mu)=\left\lfloor\frac{(\mu-origin)\bmod Q+d/2}{T}\right\rfloor\bmod (N/n_T),
\qquad key=q_{out}(N_{in}/n_{T,in})+q_{in}.
\]

nativeのキーはprofileのtile ownerと同じdouble演算で計算する。
FP32 TensorにTorch演算を重ねていた旧キーとは、tile境界の丸めで順序が異なる場合がある。
atomのprofile・窓・値を変更する処理ではなく、順序の変更に伴う加算丸めは独立参照で検査する。
同じキーの行順はstable argsortで元の順に保つ。

sigmaの上限判定は別に、従来のTorchのscalar/Tensor比較と同じ規則を保つ。
上限min(Pin,Pout)/3をatomのdtypeへcastしてから比較する。
FP32に丸めた上限がdoubleの上限よりわずかに大きい場合も、従来と同じ受理・拒否になる。

## 検証と比較

CPU全体は2006 passed / 2959 skipped、既存JIT警告18件（42.57秒）。
Ruff・clang-format・差分検査を通過。wheel/sdistをbuildした。
追加GPUテストはキーとstable行順、全atom fieldのNaN/Inf、XのNaN/Inf、
sigma0/負値、FP32上限の上下丸め、上限直後、複合エラーの優先順位を検査する。
既存のY/dX/全atom勾配・保存状態・空入力のテストも維持する。

比較元runtimeは`8fe67646`。同じN128/A819/B4・FP32・初期写像と入力、
同じGPU fused AdamW、TF32=False、Graphなしで、dense・変更前CST・新CSTを比較する。
検査とprepareを時間から除外しない。CPU参照勾配による更新delta・momentsの検証も行う。
GPUの検証・時間・メモリ・source/result hashは、回収後に以下へ記録する。

## L4での結果

source commit `9f60fc7b`。NVIDIA L4、Torch2.11.0+cu130、CUDA13.0。
対象suiteは211 passed / 0 failed / 0 skipped。
Y/dX/全atom勾配の従来基準を維持し、CPU参照勾配を同じGPU fused AdamWへ渡す
Parameter更新delta・moments・stepの検査も通過した。
CSTのCPU参照への最大絶対誤差はY5.96e-8、dX4.47e-8、全Parameter勾配5.96e-8。
更新deltaとstepは差0、exp_avg差7.45e-9、exp_avg_sq差5.82e-11で変更前と同じ。

以下は同じjobのfresh process各2つ、2巡目は逆順、warmup10回後30サンプル。
検査・prepare・部分出力合計・optimizer更新を含む同期wall時間である。

| 方式 | forward/backward process1 / process2 | AdamW込みstep process1 / process2 | AdamW込みpeak allocated / reserved |
| --- | ---: | ---: | ---: |
| dense | 0.243 / 0.234ms | 0.455 / 0.437ms | 17,310,208byte / 22MiB |
| 変更前CST | 1.867 / 1.876ms | 1.885 / 1.888ms | 115,712byte / 2MiB |
| 検査・キー生成を融合したCST | 1.364 / 1.353ms | **1.379 / 1.368ms** | **115,712byte / 2MiB** |

変更前CSTからstepが約26.9%/27.5%短縮した。CST/dense比は約3.03/3.13。
目標の2程度には未到達で、このjobのdense基準では0.911/0.874msが目安になる。
forward/backwardのみのallocated peakも88,576byteで変更前と同じだった。
denseの大きい常駐割当の内訳は[先の比較](cell-mass-dense-step.ja.md)と同じく未分解で、
このメモリ差を重み行列の圧縮率とは解釈しない。

検査・キー構築・stable sort全体を別のGPU event envelopeで測ると、
変更前0.615ms→変更後0.113ms。hostの読み戻しと起動の隙間も含む値で、
GPUでのソート単体の時間ではない。この差を全stepから差し引いて採用判断することもしない。
shape/configのhostメタデータ処理はこのstage診断の対象外。

## 証拠と再実行

job `l4job-aea4f391e9e24cbab4fc4ec8a7613656`。

- source archive SHA256: `d3b46bcdf5ecaa83b911cc5e0656f1c729b174ec7729cd414ff09244cc646031`
- result archive SHA256: `5288278468cedbfa3a2897739cb6611693756309d3bc0b3553a12043103aa919`
- driver SHA256: `da95d3cfabefa853d5fc7c0e8cbb94968fdf94299f875c820a5a18cab00b768b`
- 本体binary SHA256: `c2fcf1d3ef97cd6d8e6cae0316a31b68f5b6c9e30472dbe48a20b37912ee014f`

両archive hash、25result manifest、321 runtime source filesと現行・wheel/sdist、
driverに埋めた変更前executor/CUDA/validatorと回収物、回収binaryのhashを照合した。
source/result・binary・JSON・CPU/build log・receipt/specは
ignored `output/cell-mass-prepare-20261026/`に保持した。原本はhost共通poolのjob directory。
結果回収と所有slot停止を確認済み。
再実行はsource archiveを新しいdirectoryへ展開し、含まれるdriverを共有poolへ提出する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/extracted-snapshot \
  --script /absolute/extracted-snapshot/__pool_driver__.py \
  --timeout 720 --label cell-mass-prepare-repeat
```
