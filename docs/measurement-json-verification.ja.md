# JSON measurement path verification (2026-10-02)

測定 JSON を唯一の設定入力にし、Colab の複数 GPU を同じ経路で測定する。
Actions は repository 内の JSON パスを受け取る。GPU / Case / 回数の独立した選択欄は置かない。
依頼を commit、source hash、Case / Plan snapshot とともに固定し、各 GPU の観測を中央で照合する。

Colab client は `tools/colab/` に置いた公開ツール。T4 / L4 / A100 / H100 / G4 を明示的に指定でき、
利用者の端末で設定した Colab アカウントを検証する。個人の Skills の導入は利用条件にしない。
機種の指定を受け付けることと kernel の実機検証は区別する。今回の実機確認は L4 と G4 のみ。

最初の測定 (source `5386b4c`) は両 GPU の正しさ・完全 step・回収・PostgreSQL への保存を通過した。
ただし同じ seed でも CUDA 乱数による input / target hash は GPU 間で異なった。
[最初の記録](../benchmarks/automation/results/colab-20261002.json)を維持するが、同一入力の比較として扱わない。

修正した source `a725bc2` は初期値を CPU で生成する。公式 driver の子プロセスを
`ATEN_CPU_CAPABILITY=default` で起動し、実際の capability と生成方式を記録する。
DB adapter revision 2 は旧 CUDA 生成と新 CPU 生成を異なる protocol に分ける。
Raw artifact / 旧 projection は上書きしない。

CPU と実 PostgreSQL の suite は654 passed / 117 skipped。ruff / actionlint も PASS。
小さい独立 FP64 oracle と1024²・M128・52,429 atoms・sigma3・FP32・TF32 off・fused capturable AdamW の完全 step を測る。
完全 shape の全勾配の独立証明や kernel の認証・既定採用は行わない。
Allocated / reserved peak は Graph capture/replay を含む実測。GPU process usage は未測定。

```bash
PYTHONPATH=src python -m benchmarks.automation run \
  --request benchmarks/requests/colab-linear-l4-g4.json \
  --output output/measurement-cpu-rng-20261002
PYTHONPATH=src python -m benchmarks.automation validate \
  --request output/measurement-cpu-rng-20261002/request.json \
  --results output/measurement-cpu-rng-20261002/measurement/results
```

Neon への実接続と GitHub 上からの workflow 実行は未確認。
`NEON_DATABASE_URL`、protected environment、ログイン済み CPU bridge runner を設定して接続する。
[利用と初回設定](../benchmarks/automation/README.md)を参照。

## 修正後の実機結果

[機械記録](../benchmarks/automation/results/colab-cpu-rng-20261002.json)に、job ID・request / archive hash・誤差・全時間サンプル・実機 metadata を保存した。
L4 と G4 の input / target / 初期 Parameter hash は一致。CPU capability は両方 `DEFAULT`。
request ID は `50a21876917a3d75a8f6fa97a023fe3984fefcf74d3d53e0a3b0c4a4bbb6d277`。GPU は1台ずつ起動し、回収後に全 owned slot の停止を確認した。

| GPU | 候補 | Graph median ms | Peak allocated bytes | Peak reserved bytes |
| --- | --- | ---: | ---: | ---: |
| L4 | full | 0.614642 | 54366208 | 111149056 |
| L4 | window512 | 1.505537 | 48404480 | 90177536 |
| L4 | dense | 0.127916 | 52955648 | 111149056 |
| G4 (RTX PRO 6000 Blackwell Server Edition) | full | 0.244590 | 54366208 | 111149056 |
| G4 (RTX PRO 6000 Blackwell Server Edition) | window512 | 0.713940 | 48404480 | 90177536 |
| G4 (RTX PRO 6000 Blackwell Server Edition) | dense | 0.091290 | 52955648 | 111149056 |

この測定は各 GPU で1回・各候補7 sample。速度順位の一般化・アルゴリズム改善の主張は行わない。
dense は Parameter の構成が異なる性能参照。GPU が速いことを L4 のアルゴリズム改善とは扱わない。

実 PostgreSQL に2 runs / 10 worker records / 62 metricsを保存し、同じ原本の再送で増えないこと、byte-exact export、GPU / protocol / provenance の取得を確認した。
元の request、source / result archive、全 logs は ignored `benchmarks/automation/evidence/colab-cpu-rng-20261002/` に保存済み。ローカル検証用 DB はデータを保持してサーバーを停止した。
