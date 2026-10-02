# CSTOptimizer 移行の記録

変更前の checkpoint は `33504c6`。ユーザーは旧 optimizer 全体の削除と
後方互換の破棄を明示的に承認した。

## 変更

- 公開 optimizer は `CSTOptimizer` と `OptimizerStateAdapter` の二つにした。
- 旧 AdamR / ParameterAdam / Normalized / Quadratic family、config、functional
  dense optimizer、N/D moment system・solver、専用 AtomGrad 経路を削除した。
- Linear / Conv の backward は通常の backend autograd のみを使う。
- 旧 optimizer 専用のテストと利用ガイドを削除し、Kernel / Geometry / execution
  の既存テスト中にある学習 step は新しい wrapper に移行した。
- historical GPU / MNIST research notes は旧 revision の記録として残す。
  新しい wrapper にその結果を転用しない。

## 検証

ローカル環境は Python 3.11.15、Torch 2.13.0、CUDA なし。
CPU 全 suite は 482 passed / 196 skipped。旧 API の削除により旧 optimizer 専用
テストも削除しており、以前の suite 件数と単純比較しない。
新規 optimizer テストは 39 passed / CUDA 専用 3 skipped。
Ruff と wheel build が通った。

SGD、Adam、AdamW、RMSprop、Adagrad、Adadelta、Adamax、Rprop の Euclidean
complete step を同一初期値・入力・勾配の base optimizer と照合した。
Sphere center と vector state の接空間制約、Polar / Direct のゼロ勾配時の活動
減衰、欠損勾配・lr=0 の扱い、group ごとの lr、scheduler、closure、checkpoint
復元後の次の step、所有と parameter order・state shape の拒否も確認した。

最初の新規テストでは Kernel constructor の必須設定の不足で失敗した。
fixture に amplitude / width 設定を与えて修正した。計算式を変えて解決していない。

CPU GradScaler でも有限 step と overflow 時の step / state のスキップを確認した。
CUDA AMP の包括的な検証や学習精度比較を行ったという主張はしない。

## Installed-wheel L4 検証

- Source: `1df491e`。
- Job: `l4job-f5b1cab9acb942a588a0872996937f59`。
- Snapshot SHA256: `e5411338277598aa41c56c15c286fd64d1febcab9d75633eaa8567968ae89505`。
- Result archive SHA256: `0def36c37e50ddd0c5bd20ba5f8dc7902a60a1b92b380fb7cd633678e7ec7bf0`。
- Wheel SHA256: `94b7d5501ab9334d60af8c58285365a99a15c751d7a62d49c6427aa13d16ec9d`。
- NVIDIA L4 23034 MiB、driver 580.82.07、Python 3.13.15、Torch 2.11.0+cu128、
  CUDA 12.8、Triton 3.6.0。

全 suite は **742 passed / 2 skipped / 0 failure / 0 error**、496.56 秒。
新規 optimizer suite は **42 passed / 0 skipped**。CUDA 上の SGD / Adam /
AdamW complete step と base の照合が通った。既存の Triton DirectAmpWidth
training test も新 wrapper に移行して通った。配布 wheel の全 125 Python module
を snapshot manifest と照合し、現在の計算 source と byte 一致を確認した。
ローカル wheel も同じ SHA256 であり、39 CPU optimizer tests が通った。

既存 normalized Strip の independent FP64 oracle、support / normalization、
dX / 全 atom gradient、更新後の値、plain AdamW の Graph 検証も通った。
window の小 fixture の atom gradient は max error `1.9776835e-4`、relative L2
`6.9198e-5` で、既存 tolerance 内。ここを zero-error と表現しない。

### 既存 CUDA 経路の回帰測定

以下は **plain AdamW を使う既存 normalized Strip 経路** の complete step であり、
CSTOptimizer wrapper の性能測定ではない。N=1024、batch=128、52429 atoms、
FP32、TF32 off、seed=21、fused/capturable AdamW、lr=1e-4、weight_decay=.01。
sharp fixture は sigma-three の ordinary performance fixture と区別する。

| Fixture | Implementation | Graph median ms | Peak allocated MiB | Peak reserved MiB |
| --- | --- | ---: | ---: | ---: |
| broad | window | 1.516 | 46.16 | 86 |
| broad | full | 0.632 | 51.85 | 106 |
| broad | dense | 0.127 | 50.50 | 106 |
| sharp | window | 0.305 | 46.16 | 86 |
| sharp | full | 0.545 | 51.85 | 106 |
| sharp | dense | 0.126 | 50.50 | 106 |

各 case は独立 subprocess。allocated / reserved peak は warmed model、gradient、
AdamW state と capture / replay を含む。total GPU process usage は未測定。
compile / preparation time を step time に含めていない。dense は異なる学習
parameterization なので、その時間から algorithmic speedup を主張しない。

wrapper 自体の Graph capture、旧 optimizer に対する速度・GPU peak memory の
改善は主張しない。有限性検査は同期し、snapshot は atom table サイズである。

### 再現と回収

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/checkout-of-1df491e \
  --script /absolute/checkout-of-1df491e/benchmarks/cuda/linear/validate_dispatch_registry.py \
  --timeout 600 --label cst-optimizer-installed-wheel -- 1df491e --full-suite
```

検証済みログ・wheel・source snapshot・receipt は ignored な
`benchmarks/cuda/linear/evidence/cst-optimizer/l4job-f5b1cab9acb942a588a0872996937f59/`
に保存した。host queue の原本も保持している。supervisor は正常終了し、slot 1
の owned L4 runtime が `stopped` になったことを確認した。
