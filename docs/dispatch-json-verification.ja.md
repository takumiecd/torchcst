# Dispatch JSON の実機検証

2026-10-02、JSON を実ファイルに書き、読み戻した ExactSelector を公開 CSTLinear に
渡して NVIDIA L4 で実行した。各 Plan の選択は完全一致 entry を使い、既定の
bootstrap へ偶然流れた成功ではないことを primitive trace で確認した。
JSON は `demonstration-only` と表示する動作確認 fixture。DB による順位付けや承認済み
policy の出力ではない。

## 確認結果

- CPU suite: 602 passed / 133 skipped。CUDA と PostgreSQL の未接続テストは skip。
- L4 の対象4ファイル: 108 passed / 0 failed / 0 skipped。
- 二つの登録済み Algorithm を JSON の Plan ID から選択して実行。
- 独立 FP64 oracle で W / y / dX / 全5 atom 勾配を照合。混合 fixture は sharp / one-hot、
  clip / floor、512行境界を含む。許容誤差は既存の atol=rtol=3e-4、W は2e-5。
- AdamW を含む Graph replay 後の Parameter・勾配・optimizer 状態を eager と比較。
  capture 後に中心と幅を変更し、更新後の oracle でも照合。
- 未観測の入力行数7では JSON に明示した fallback Plan を使用し、y / dX / 全 atom 勾配を照合。
- 不明な Plan ID と異なる PyTorch version の実ファイルを拒否。
- ローカルでは、読み込み後に JSON ファイルを削除しても選択と演算が動くことを確認。

追加した補助 CPU テストの最初の全体実行では、ランダムな FP32 値の打ち消しで
参照との丸め誤差が出た。経路を検証する fixture を二進数で厳密に表せる入力に変更し、
全体を再実行した。修正は `7de4994`。GPU の測定ソースはその直前の `3c8901d` であり、
現在の runtime / benchmark の167 Python file と SHA256 が一致することを確認した。

## 完全 step の測定

1024² / M128 / 52,429 atoms（約5%）/ 通常 sigma3 / FP32 / TF32 無効。
fused capturable AdamW、lr1e-4、weight_decay .01。各ケースは別 process。
入力・target は同じ seed、二つの CST candidate の初期 atom SHA256 は一致。
各 CST ケースも、実ファイルを読み込んだ選択器で eager / Graph の完全一致を確認した。
小さい混合 fixture の独立全 atom oracle と、大きい完全 step の測定は区別する。

| Plan / 参照 | Graph 中央値 ms | peak allocated MiB | peak reserved MiB |
| --- | ---: | ---: | ---: |
| normalized_full | 0.625282 | 51.8511 | 106 |
| normalized_window | 1.513336 | 46.1655 | 86 |
| dense | 0.127243 | 50.5024 | 106 |

forward / backward / optimizer を含む同期 wall time 7 samples の中央値。
peak は Graph capture 直前に reset し、capture と replay を含める。
全 GPU process usage は未測定。測定 peak を workspace 上界として扱わない。
今回の結果から新しい性能 policy を自動採用していない。

## 環境・証拠

実際の検証環境は PyTorch 2.11.0+cu128 / CUDA12.8 / Triton3.6.0 / Python3.13.15、
NVIDIA L4、compute capability8.9、SM58、driver580.82.07。
pool の system probe は CUDA13.0。実験は job 内の独立 venv で CUDA12.8 を使い、
system package は変更していない。

source commit: `3c8901d9493e6912007275ed383f58e9bd54bf32`。
job: `l4job-132c93c6daf643529ee6e8b1a4418323`。
source archive SHA256: `c7d75ea932a8dbac9f70199a9010c69f019cd9753b1b274df9a26bc9d77659dc`。
result archive SHA256: `3e2fe01d9a283aa5e11af84b87aba3322d89819393aeec41057966765cf89d64`。
16 result file の manifest hash と source / result archive の hash を再照合した。
結果を回収し、pool-owned VM の stopped を確認した。

[機械記録](../benchmarks/cuda/linear/results/dispatch-json-20261002.json)。
実 JSON、trace、環境、stdout / stderr、source / result archives は
`benchmarks/cuda/linear/evidence/dispatch-json-20261002/l4job-132c93c6daf643529ee6e8b1a4418323/`
に保存し、ignore している。

## 再現

CUDA 環境で、存在しない出力ディレクトリを指定する。

```bash
python -m benchmarks.cuda.linear.check_dispatch --output-dir output/dispatch-check --bench
```

L4 pool の driver は evidence 内に保存した。system 環境を変えずに
検証した CUDA12.8 venv を再構築する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/evidence/dispatch-json-20261002/driver.py" --label dispatch-json --timeout 1200 -- 3c8901d
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

元のソースを再現する場合は source archive を使う。現行 checkout で同じ手順を
実行する場合は、末尾の source commit 引数を実際の checkout に合わせる。
