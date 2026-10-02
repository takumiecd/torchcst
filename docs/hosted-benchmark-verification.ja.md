# GitHub-hosted Colab 測定の確認（2026-10-02）

[run 37000213462](https://github.com/takumiecd/torchcst/actions/runs/37000213462) で
GitHub-hosted Ubuntu の `prepare → measure` が成功した。
source は `49adf035ae002cbc4091e19f819ef026c4b96731`、入力は
`benchmarks/requests/colab-linear.json`。自分の Mac runner は起動していない。
feature branch の確認なので、この実行の Neon job は意図どおり skip された。

- 実機は NVIDIA L4 / CC8.9 / 58SM、Torch2.11.0+cu128 / CUDA12.8 / Triton3.6.0。
- full / window512 は独立 FP64 oracle の小さな mixed fixture で全 atom 勾配まで PASS。
  1024² の通常 sigma-three 条件では、完全な training step の時間と capture/replay の peak を別に測定した。
  full-shape の独立全 atom oracle を確認したという意味ではない。
- source inventory、source / result archive の SHA256、result binding を回収後に照合した。
- owned session `cst-pool-569d60b0a0e7-1` を停止した。実際の session ID は
  evidence の `owned-sessions.json` を正として参照する。
- pool job は `l4job-1b71cdc6682c4fc1a355ddd8416dc3c2`。

| 測定 | eager median ms | graph median ms | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: | ---: |
| full | 1.667753 | 0.634520 | 54,366,208 | 111,149,056 |
| window512 | 2.894124 | 1.529089 | 48,404,480 | 90,177,536 |
| dense | 0.710066 | 0.127574 | 52,955,648 | 111,149,056 |

peak は CUDA Graph capture/replay を含む PyTorch allocator 計測。
GPU process 全体の使用量は未計測。新しい algorithm の速度改善として扱わない。
[機械記録](../benchmarks/automation/results/hosted-colab-20261002.json)に
request ID、source・archive hash、全 timing sample、oracle 誤差と検証範囲を残した。
raw archive / JSON / sanitized logs は ignored `benchmarks/automation/evidence/hosted-actions-20261002/` に保存した。

## 失敗と修正

- run36998784995: Python3.11 では CLI0.6.0 を install できず、GPU の確保前に停止。
  測定 runner を3.13にした。
- run36998925522 / 36999441747: CLI0.6.0 の無制限 dependency が
  `jupyter-kernel-client==1.0.2` に解決され、`KernelClient` API がなく失敗した。
  確保した GPU は停止済み。動作している0.14.0に固定し、確保前に API の存在を検査した。
- CLI cache が stop で消える前にも credential 値を取得し、diagnostic log を保存時に redact する。
  live OAuth / session config を artifact に含めない。

## 中央取り込みと検証範囲

新しい fetch 経路で既存 run36995360838 の GitHub metadata と artifact digest を読み直し、
upstream main から request を再構築した。専用 role で Neon に再取り込みし、
元 JSON の byte-exact export と観測数が増えないことを確認した。以前の bridge の出所は保持する。

CPU suite は655 passed / 135 skipped、既存 TorchScript warnings18件。
関連75件・Ruff・actionlint も確認した。GPU / local PostgreSQL tests は CPU suite では skip。
本番 Neon の上記確認は別に実接続で行っている。

別参加者の Colab アカウント、追加の GPU preset の実機測定はこの確認の対象外。
参加手順は [運用ガイド](benchmark-contributions.ja.md)、入力は [JSON 一覧](../benchmarks/requests/README.md)を参照。

## Main / Neon の通し確認

[run 37004242555](https://github.com/takumiecd/torchcst/actions/runs/37004242555) は
merge commit `19d203cedf8395752d18c5eefa7defbd27cc2d6f` を既存の L4 JSON で測り、
GitHub-hosted Ubuntu の `prepare → measure → ingest` が全て成功した。
Neon に新しい5 records / 31 metrics を保存し、専用 role で元 JSON の byte-exact export、
request / source SHA / workflow run ID / origin を照合した。

[中央取り込み run 37005008609](https://github.com/takumiecd/torchcst/actions/runs/37005008609) でも
同じ観測を GitHub metadata / artifact digest / upstream request と再照合して取り込んだ。
DB は4 runs / 20 records / 124 metrics のままで、再送は観測を増やしていない。
[機械記録](../benchmarks/automation/results/hosted-neon-20261002.json)に今回の job / request / archive hash、
測定値、Neon の照合と再送結果を保存した。raw evidence は同じ ignored evidence directory に保持した。

owned GPU の停止を確認した。self-hosted runner の登録・起動は0台。
検証専用 Environment と `BENCHMARK_VALIDATION_REF` を削除し、
`benchmark-colab` / `benchmark-database` の main-only policy を維持している。

参加者の fork と別人の Colab アカウントは未実行。
Fork での測定には参加者自身の Colab 初回認証設定、中央での fork artifact 読み取りには
別途 read credential の設定が必要。今回の中央取り込み確認は upstream 自身の実行に対して行った。
