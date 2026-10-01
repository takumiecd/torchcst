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
Operatorを再bindingしないことを追加で検証する。既存のnormalized Strip
benchmark driverで配布wheel全テスト、独立FP64 oracleとGraph/AdamW、N=1024 / batch=128
のfull/window/dense完全ステップを測る。普通のsigma-threeとsharpは別のfixture。
allocated peakはcapture/replayを含め、reservedと区別する。実機結果は回収後に記載する。

## 移動前の実行との直接比較と配布

移動前 `2684f52` の src を ignored evidence に保存し、同じ seed=21 / FP64 の
19 完全ステップを比較した。6 種類の pair Kernel（Separable、Amplitude、Direct、
Polar、AmpWidth の interpolating / inverse）で materialized / factored、5 種類の
Profile の単一 Chart、normalized Strip の full / window。y、dX、全 Parameter の
勾配、AdamW 更新後の Parameter と y は `rtol=0, atol=0` で一致し、全値が有限。
checkpoint のキーも同じ。原始スクリプト・before Tensor・baseline の src は
`benchmarks/cuda/linear/evidence/operator-integration/` に保存した。

```bash
PYTHONPATH=benchmarks/cuda/linear/evidence/operator-integration/baseline/src \
  python benchmarks/cuda/linear/evidence/operator-integration/compare.py \
  --save benchmarks/cuda/linear/evidence/operator-integration/before.pt
PYTHONPATH=src python benchmarks/cuda/linear/evidence/operator-integration/compare.py \
  --compare benchmarks/cuda/linear/evidence/operator-integration/before.pt
uv build --wheel --out-dir benchmarks/cuda/linear/evidence/operator-integration/wheels
```

ローカル wheel に package 全体の 143 Python module が source と同じ bytes で
含まれることを確認した。wheel SHA256 は
`eb1335b7ca6d8256ea6155b201fbdc322a7c620479c7357a369a93cf5e90451d`。
提出後の code の差分は root の `__all__` の export 順を lint 規約に合わせる変更だけ。

## L4 提出

source commit `b9e88ae809e9dd310dcf32b3d04ebb1569098b5b`、
job `l4job-477845f9741548a5ad0a36b6002fd447`。source archive SHA256 は
`5a061998fafc10857635b2c2e9482073504b87e06ef2161dcd7285ffd0b0fb9f`。
提出時は clean、新しい raw evidence が archive に含まれないことも確認した。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_dispatch_registry.py" \
  --label operator-integration-wheel --timeout 1200 \
  -- b9e88ae809e9dd310dcf32b3d04ebb1569098b5b --full-suite
```

### 全テストの回収と Graph テストの条件修正

`l4job-477845f9741548a5ad0a36b6002fd447` は driver exit 1、timeout false。
配布wheelの全テストは **809 passed / 1 failed / 2 skipped**。追加した Graph テストが、
Amplitude._split の既存の parameter_dim 呼び出しを拒否した。これは Python の
Chart 次元 metadata の確認で、snapshot や GPU scalar の読み戻しではない。
テストの条件が過剰だったため、宣言再生成の禁止を維持し、parameter_dim の禁止を
Operator の再構成（__post_init__）の禁止に修正した。Graph テストの成功や性能測定の
成功とは扱わない。benchmark は全テスト failure により実行されていない。

GPU は NVIDIA L4、Python 3.13.15 / Torch 2.11.0+cu128 / CUDA 12.8 / Triton 3.6.0。
receipt と結果archive SHA256 を照合して raw output を保存し、所有runtimeの停止を
確認した。この条件修正はテストだけで、Operator / Kernel の計算は変えていない。
他の全テストは既に通っているため、修正した Operator suite と既存のdispatch/profile
契約を配布wheelから再実行し、独立oracle・Graph/AdamW・完全ステップ測定へ進む。

### 修正後の回収済み結果

測定 source commit `9432900e18724bf93e328f7e8f510fe5d5596039`。
job `l4job-61b810060e264ca7ae83bf324926aff1`、driver returncode 0 / timeout false。
source archive SHA256 は
`56c8b775d10489871dc90f2994cef412cdfaf7fd7b8f4ecf73b5c762ae06d66a`。
結果 archive SHA256 は
`3909a5c3facf7c02657738317453484d6026162f1bec144c1752115b5f0fa8af`。
receipt / archive、結果の source hash、配布 wheel の全 Python module と提出ソースの
hash を照合した。結果を ignored evidence に保存し、pool の全 slot が stopped
であることを確認した。

Operator / dispatch / normalized Strip / compact Profile の配布 wheel テストは
**119 passed / 0 skipped / 0 failed**（25.95 秒）。修正した pair Graph capture/replay
テストも通り、宣言を作り直したり Operator を再構成したりせず、出力・dX・全 atom
gradient が eager と一致した。単一 Chart と pair の CUDA 値・勾配の CPU 照合も通る。
初回全テストの他809件は通っており、修正後に全812件を一括再実行したという
主張はしない。計算 module 142件は初回全テストと同じ bytes、root の export 一覧のみ
lint による並び順の変更。ローカル wheel からも Operator 18 passed / 4 skipped。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_dispatch_registry.py" \
  --label operator-graph-contracts --timeout 600 \
  -- 9432900e18724bf93e328f7e8f510fe5d5596039 --operator-suite
```

実機 NVIDIA L4 / driver 580.82.07 / 23034 MiB、Python 3.13.15、
Torch 2.11.0+cu128、CUDA 12.8、Triton 3.6.0。小さい混合 fixture の独立 FP64 oracle で
full/window の y・dX・全 atom gradient・weight・更新後の値/勾配を確認し、
Graph/AdamW の検証も通る。最大 absolute error は y 2.88e-8、dX 3.46e-8、
atom gradient 1.98e-4（relative L2 6.92e-5）、既存の許容範囲を通る。

完全ステップは N=1024（1024×1024 weight）、batch=128、atoms=52429（約5%）。
FP32 / TF32 off / seed=21、AdamW lr=1e-4 / weight_decay=0.01 / fused / capturable。
broad は初期 sigma=3、sharp は初期 sigma=0.199 で支持点に近い center の別 fixture。

| fixture | implementation | eager median ms | Graph median ms | allocated peak MiB | reserved peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| broad | normalized_window | 3.154 | 1.517 | 46.16 | 86.00 |
| broad | normalized_full | 1.531 | 0.617 | 51.85 | 106.00 |
| broad | dense_linear | 0.689 | 0.128 | 50.50 | 106.00 |
| sharp | normalized_window | 2.736 | 0.308 | 46.16 | 86.00 |
| sharp | normalized_full | 1.618 | 0.540 | 51.85 | 106.00 |
| sharp | dense_linear | 0.733 | 0.127 | 50.50 | 106.00 |

時間は warmup 後の forward/backward/AdamW の完全ステップ。初回 compile / 準備
時間は含めない。allocated peak は warmup 済み model・勾配・optimizer 状態を含め、
capture/replay を reset 後に測定する。process 全体の GPU 使用量は未測定。
初回宣言分離の測定に対して、この fixture の allocated peak は同じ値だった。
各方式は独立 process の順次測定で、paired alternating timing ではない。
1024×1024 の全 atom gradient を独立 oracle で網羅したとは主張しない。dense は
別の parameterization の参考値。この変更による速度改善、方式の承認や既定
dispatch への昇格を認定するものではない。
