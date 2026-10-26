# denseの約2倍のstep時間を目標にする

2026-10-26。ユーザーの目標は速度がdenseの2倍になることではなく、
**1stepの所要時間をdenseの約2倍程度に収めること**。
比較指標は次で固定する。
最新の[prepare融合](cell-mass-prepare.ja.md)では、同じ条件のCST/dense比が約3.1へ改善した。
以下の約4.3は融合前の記録として残す。

\[
r_{step}=\frac{t_{CST,step}}{t_{dense,step}},\qquad
\text{目標は }r_{step}\text{ を2程度にすること。}
\]

[atom分担の最適化](cell-mass-atom-splits.ja.md)で得た約2.46倍は変更前CSTとの速度比。
denseへの到達度とは区別する。以後、同じshape・batch・dtype・精度設定・optimizer・
Graph条件で測ったdenseを基準にする。メモリも完全stepのallocated/reservedを比較する。

最新の条件別結果は[投影微分の再利用](cell-mass-projection-cache.ja.md)と
[メモリ制約を含む比較](cell-mass-memory-frontier.ja.md)を参照。
128・batch4では約1.9倍の時間でdense以下のメモリ、batch16では両立未達。

## 今回の1stepの境界

biasなしの1層、N128×128、batch4、FP32、CSTは819atom（N²の約5%）。
入力Xと上流勾配dYはseed120で固定する。CSTはmode8、sigma初期値0.2〜12、
入力profile tile16/出力tile32、周期48/96、16site/8split/16thread、merge Z。

CPUのセル積分参照から初期のdense重みを構成する。

\[
W_{oi}=\sum_a A_a U_{ao}V_{ai}.
\]

Wの構成は初回の測定外で行い、以後dense側はWを学習Parameterとして更新する。
CST側は元のatom配列を学習Parameterとする。
初期の線形写像・Y・dXは同じだが、パラメータ空間と更新後の学習軌跡は異なる。

各stepは次を含む。

1. Xと学習ParameterのgradをNoneへ戻す。
2. forward、固定dYによるbackward（dXと各モデルのParameter勾配）。
3. 更新ありの比較では同じGPU fused AdamWを実行する。

AdamWはlr1e-4、betas=(0.9,0.999)、eps1e-8、weight_decay0.01、
foreach=False、fused=True。warmup後もParameterとmomentsを実際に更新する。
CSTのprepare・入力検査・partial合計は各stepの時間に含む。
TF32=False。CSTのdouble内部演算を維持し、denseは通常のFP32 Linearを使う。
Graphは両者とも使わない。JIT/build・初期profile/W構成・loss層・他のnetwork層は測定外。
公開CSTLinear/CSTOptimizerの統合を完了したとは扱わない。

## 検証方法

各モデルのY・dX・Parameter勾配を独立したCPU参照と照合する。
更新については、そのCPU参照勾配を同じGPU fused AdamWへ渡し、
Parameterの更新delta・exp_avg・exp_avg_sq・stepを実行側と比較する。
これは勾配実装の差とoptimizer実装の差を混ぜないためで、
勾配・更新の誤差基準はrtol5e-5/atol2e-6を維持する。

初回job `l4job-5bd11e629a554315a7a0a5ffa9255c94`は、CPU非融合AdamWと
GPU融合AdamWの更新deltaを直接比較した検査で停止した。
CSTの大きな中心座標を含む更新で最大3.05e-5の差があり、
異なる更新の演算順・丸めを混ぜた比較だった。速度測定は始まっておらず、全比較から除外する。
修正版は上記の同じGPU optimizerへCPU参照勾配を渡す方式を使う。
kernelや精度、誤差基準は変更しない。

失敗jobのsource SHA256は
`d88e990c236d9862d72be8fb1b97ea056a38277dae94b63debf0de23a4425284`、
result SHA256は
`543aeb4d7ed05354c0c60c4b10d7860b005e5ce627de75268a0f273f7c486244`。
両hash・manifest・321 runtime source filesを照合し、source/result・stderrを
ignored `output/cell-mass-dense-step-20261026/failed-v1/`へ保持した。

## 実測

検証成功後、forward/backwardのみとAdamWを含むstepを分けて記録する。
各設定2fresh process、2巡目は逆順、warmup10回後30サンプル。
同じjobのdenseとCSTから比を作り、異なるGPU jobの絶対時間を組み合わせない。

runtimeは`e9adbebc`（実コードは`ee131b7d`と同じ）。
NVIDIA L4、Torch2.11.0+cu130、CUDA13.0、TF32=False。
各stepを同期して測ったwall時間の中央値は次の通り。

| 境界 | dense process1 / process2 | CST process1 / process2 | CST/dense |
| --- | ---: | ---: | ---: |
| forward+backward | 0.234 / 0.232ms | 1.875 / 1.889ms | 8.00 / 8.16 |
| forward+backward+fused AdamW | **0.438 / 0.452ms** | **1.886 / 1.910ms** | **4.31 / 4.23** |

目標は後者の比を2程度にすること。今回のdenseでは目安が0.876/0.903ms/stepとなる。
現状は目標未達で、CSTにはさらに約2.15/2.11倍の短縮が必要。
forward/backwardのみの比を、optimizerを含む目標への到達度として扱わない。
optimizerあり/なしの差も、単独のoptimizer kernel時間とは解釈しない。
CPUの起動処理とGPUの演算が重なり、両modelで隠れる時間が異なるためである。

| 境界 | dense peak allocated | CST peak allocated | dense / CST peak reserved |
| --- | ---: | ---: | ---: |
| forward+backward | 17,178,624byte | 88,576byte | 22 / 2MiB |
| AdamWを含むstep | 17,310,208byte | 115,712byte | 22 / 2MiB |

CSTのstep peak allocatedは113KiB。Tensor以外のGPU process全VRAMは未測定。
denseの割当にはwarmup後も残る大きな常駐割当があり、
forward/backwardのみのbaseline allocatedは17,108,992byte。
W・X・dYの合計69,632byteだけでは説明できず、今回その内訳を分解していない。
この差をdense行列そのもののbyte数や圧縮率に換算しない。
学習Parameter数はdense16,384、CST3,276で、約5分の1。

更新検証は同じGPU融合optimizerへCPU参照勾配を渡す修正後に通過した。
両modelともParameter更新deltaとoptimizer stepの差は0。
CSTのexp_avg差最大7.45e-9、exp_avg_sq差最大5.82e-11、denseの両moments差は0。
CSTのY/dX/Parameter勾配のCPU参照への最大絶対誤差は5.96e-8/4.47e-8/5.96e-8。
初期のdenseとCSTのY差は8.94e-8、dX差は1.19e-7で、従来の基準内だった。
全40更新後もCSTのsigma範囲は約0.19997〜11.99550で、有効範囲内。

## 次の優先箇所

現在のstep時間はdenseの約4.3倍であり、変更前CST比の約2.46倍改善だけでは目標に届かない。
まず次を同じ誤差・step時間・メモリ基準で検討する。

- prepareのキー構築と入力検査のGPU処理をまとめ、複数のTorch起動とhost同期を減らす。
  既存のshape/finite/幅の検査を黙って省略しない。
- H/G・Parameter勾配に残るprofile/係数・微分計算の重複と、
  FP32入力でもdoubleになっている中間演算の必要範囲を調べる。
  精度変更の候補はFP32/FP64のY・dX・全atom勾配で検証し、基準を緩和しない。
- 変更後もforward/backwardだけで採用せず、このAdamW込みのstepでdenseとの比を測る。

Graphや公開CSTLinear/CSTOptimizerの比較は別に必要で、今回の数字をそのまま一般化しない。

## 証拠と再実行

成功job `l4job-04602472bd8742e9842fbbecfc77914f`。

- source archive SHA256: `3167e348c0a88e9cc7779144443758ab005d7ce14beb9ed773fc2adc1f798ba5`
- result archive SHA256: `5c1fcade216a2b495a9442c4963528d78be3b7f77f2e1b1ab6d6bede1ccb27fb`
- driver SHA256: `f73b1b624d574f994893db99c5bf6e705f9c3cd16975b0ee2d8d995aa6e29ee8`
- CST binary SHA256: `8234e5e227031eae609d3510725e7159829f6f9e8c7167466e9d8ee8fa6e1fff`

両archive hash、14result manifest、321 runtime source filesと現行source、
driver、回収したbinaryを照合した。raw evidenceとreceipt/specは
ignored `output/cell-mass-dense-step-20261026/validated/`へ保持した。
原本はhost共通poolのjob directoryへ保持する。失敗jobも前述の別directoryへ残す。
両jobの回収と所有pool slot停止を確認済み。
runtime不変のためCPU全体/wheel/sdistの再実行はしていない。

source archiveを新しいdirectoryへ展開し、そこに含まれるdriverで再実行する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/extracted-snapshot \
  --script /absolute/extracted-snapshot/__pool_driver__.py \
  --timeout 480 --label cell-mass-dense-step-repeat
```
