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
