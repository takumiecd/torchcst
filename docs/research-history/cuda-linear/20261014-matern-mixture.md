# Periodic Matérn mixture: definition and first shared executor

2026-10-14。source checkpoint `a7f23ce655d10525bb3ae49d0a2cc106dc33a7d6`。
数学的定義・幅の連続混合・sampled L2・計測値は
[定義と実装の文書](../../matern-mixture.ja.md)を正本とする。

## 結論と統合範囲

任意のlive sigmaを隣接2つの共通固定幅profileへ連続混合する新しい定義は成立した。
Torch参照と研究Triton shared/directを統合する。公開CUDA defaultの変更は行わない。
共有scanの再利用は直接実装に対して大きく有効だったが、N1024/A52428/B32ではdenseに負けた。
実装の正しさと工学的な速度採用を分ける。数学的family全体の不可能性を示す結果ではない。

8/16幅各2回、12 isolated workersの全初期/post24 oracle gateと完全step測定がPASS。
CPU1,900件、L4上82件（66 CUDA / 16 CPU）の回帰検査もPASS。
code reviewで確認した原点tiny guardとchart座標精度の制約は修正・回帰検査済み。
wheel/sdist、isolated wheel import、Ruff、GitHub CPU validationはPASS。

## 実験provenance

GPUは全jobで明示的にL4を選択し、hardwareを検査した。
Torch2.11.0+cu130、CUDA13.0、Triton3.6.0、driver580.82.07。
未計装primaryの完全stepにはprepare/L2、forward/MSE、backward、fusedAdamW、
Polar/periodic更新が含まれる。CPU thread poolsは1に固定。
時刻・samples・source全file hash・更新clock・幅hash・memory scopeはraw JSONに保存。

| scope | pool job | source archive SHA256 | verified result archive SHA256 |
| --- | --- | --- | --- |
| final qualification | l4job-9cc1b9903ac940e2b605963356c7a897 | 8bcfea307af2742986ed8f270a26fa19ad79586f0775ba41fa1777757fd042a2 | cb271bc0631149e07594ac2236be3d528c5f39a4d7d3f0efabb200d27f3391dd |
| separate phase diagnostics | l4job-0f23230c8a5a44009865f4220e9e593c | 84fcd1a1281285427c72ec7d9c9c8ade7d5b9f6900f2b6914e00674bc9057b08 | 61a8672a28081a8a435acbe0222bfab9b76f96b501dd525cf3e8376b0326eb41 |

診断jobは別copy・別Graphで、主時間の加算・減算や追加独立primary runとして扱わない。
初回のRecipeはblock64を測定。block128の大shape性能は未測定。
process全体のVRAMは未測定。DB submission/dispatch adapterの対象化は未実施。

## 初期失敗と回復

- l4job-0e2393ee105b4fca9671f83044ab886f：48/56 regression PASS。
  Gram反転anchorがFP32へ丸められall-dP gateを超え、Graphではhost knots copyが失敗。
  anchorのFP64保持とimmutable GPU knots cacheで修正した。
- l4job-8d1654a3302549f38b007e197552d2a7：56 regression PASS、full初期all-dP FAIL。
  FP32 Polar幅解釈の約2e-8差が狭い山の中心VJPへ増幅されることをCPUで再現し、
  幅のFP64解釈とFP64 batch VJP縮約で修正した。gate4e-4は維持した。
- l4job-0cf8392fe3c94c44a03f2422df6036fd：Colab execution/output通信timeout。
  検証済み結果なし。数学や速度のFAILには数えない。owned endpointの消失をpool recoverで
  確認し、queued final jobを新しいL4で実行した。他のT4 assignmentは操作していない。
- queuedだけでcancelした2jobsもspec/sourceを保全した。GPU結果として数えない。

原点のFP64昇格でsource dtypeのtinyを変えてしまう回帰は独立reviewで発見した。
source tinyを保持し、oracleのradius³ underflowも避け、零/subtiny/floor両枝の検査を追加。
公開FP32 chartがsiteごとに座標を丸める契約は維持し、研究CUDAは座標buffer FP64を要求する。

## 再現と保全

```bash
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/matern-mixture-1024-a52428-b32-k8.json \
  --plans matern-shared matern-direct dense --output output/matern/cohort-k8.json
python -m benchmarks.cuda.linear.periodic_comparison \
  --case benchmarks/cuda/linear/cases/matern-mixture-1024-a52428-b32-k16.json \
  --plans dense matern-direct matern-shared --output output/matern/cohort-k16.json
```

所有者L4では共有poolのdriverとして上記runnerを使う。direct CLIでpool VMを操作しない。
primary source/driver/result/receipt、失敗と診断、CPU logs、wheel、Git bundleはignored
`output/matern-mixture-20261014/` に保存し、同じ相対pathで所有者primary checkoutへcopyする。
`preservation-manifest.json`のSHA256でコピーを照合する。
一次pool原本は `~/.local/state/colab-l4-pool/jobs/<job>/`。
branch `kernel/matern-mixture`、PR106、Git bundleも保全する。
owned L4 slotsは全実験後に停止を確認した。
