# Operator の宣言と live state の接続

2026-10-01、`codex/cuda-dispatch-registry-v1`。単一 Chart を基本にし、現行の
入出力 Chart の組を小さな layout 型として保持する。型・field・利用例は
[Operator](../../../operators/README.md)を参照。

`5881935` は共通 OperatorSpec / SingleChartSpec / ChartPairSpec、既存 Module の
Chart / Kernel / Atoms を参照する Operator、Torch materialized / factored の
共通実行入口。checkpoint のキーと Parameter の所有は変えない。Tensor の更新、
Parameter の置き換え、Module の dtype/device 変換、deepcopy を検証した。
trainable な Chart の実状態を参照し、snapshot を毎回実行に持ち込まない。
custom Kernel の既存実行には宣言を要求せず、宣言を利用するときだけ明示実装する。

次の checkpoint は normalized Strip を同じ OperatorSpec で宣言し、専用の
NormalizedStripSpec に適合確認して既存 CUDA registry へ渡す接続。LogWidthSpec は
signed amplitude / exp(clamp(log_sigma)) / center の意味を宣言する。operator 全体の
L2 正規化・norm floor・幅の clamp・通常の勾配を維持する。Direct / Polar activity
や Chart ごとの正規化をこの専用 Algorithm に流さない。CUDA 内の旧 OperatorSpec
名は互換 alias。連続した Strip の点は同じ座標を表す Product として canonicalize
できるが、非連続 pitch、Profile・幅・正規化の異なる宣言は拒否する。

## ローカル検証

Python 3.11.15 / Torch 2.13.0。第一 checkpoint は 549 passed / 189 skipped。
CUDA 接続後は 552 passed / 193 skipped、既存 warning 18 件。追加した 22 ケース
のうち4件はCUDA実機専用で、このローカル実行ではskip。独立 Gaussian oracle で
materialized / factored の値、dX、全 atom gradient、入力・出力 Chart の座標勾配を
照合した。既存の単一 Chart の独立 radial oracle と optimizer/checkpoint テストも
そのまま通る。shape / layout / mathematical contract の誤対応も確認した。

```bash
python -m pytest -q
python -m pytest -q tests/test_operators.py tests/test_cuda_dispatch.py \
  tests/test_normalized_strip_public.py
```

CUDA実機ではlive bindingのCPU照合と、pairのGraph capture/replay中に宣言や
parameter_dimの設定確認を行わないことを追加で検証する。既存のnormalized Strip
benchmark driverで配布wheel全テスト、独立FP64 oracleとGraph/AdamW、N=1024 / batch=128
のfull/window/dense完全ステップを測る。普通のsigma-threeとsharpは別のfixture。
allocated peakはcapture/replayを含め、reservedと区別する。実機結果は回収後に記載する。
