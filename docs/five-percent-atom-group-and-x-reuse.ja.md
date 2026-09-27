# 5% atom 密度での群判定と入力 X の再利用

2026-09-27。Strip + Torus、Triweight、tile 64×64、入力 128 行の A100 前向き計算を調べた。目的は atom の並び替え・寄与判定と入力行列 X の再利用を、準備時間・実行時間・作業メモリを含めて判断すること。dispatch は変更していない。

## 条件と比較方法

A100 80GB PCIe MIG 3g.40gb、42 SM、FP32、PyTorch matmul の TF32 無効、seed 21。形状は 4096² / 8192²、atom 数はそれぞれ 838,861 / 3,355,443（5%）。準備を毎回行う `--full` と、準備済み tensor を渡す計測を分けた。同じ atom・X・dense 重みで候補を交互に実行し、CUDA Graph 3 round の中央値を使った。dense は**生成済み W を使う**参考値で、重み生成とピークメモリは含まない。正しさは canonical W の 1,152 点と全出力を確認し、出力許容差は atol=rtol=3e-5。

## atom 群の寄与判定

8 station/形状を抽出し、FP64 の atom×site 全組み合わせを真値とした。4D 埋め込み座標の保守的な球境界を用い、支持域外と判断できる割合を測った。初期配置では各 station 約 205 atom。近接 atom を並べた場合の理想化した群も含む。

| 省略単位 | 4096² | 8192² |
| --- | ---: | ---: |
| 個々の atom×site が支持域外 | 55.11% | 55.09% |
| 1 atom × 16×32 site 群、境界で省略可能 | 21.83% | 21.93% |
| 8 atom 群 × 16×32 site 群、境界で省略可能 | 12.02% | 12.39% |
| 8 atom 群 × 4×8 site 群、境界で省略可能 | 25.23% | 25.36% |

これは**省略可能な組の割合**であって速度ではない。小さい site 群ほど判定は強いが、群数・分岐・索引の費用が増える。初期配置 8 station だけなので学習後分布への外挿はできない。近接群への並び替えが安いという仮定も置いていない。

実際に atom ごとの 16×32 site 境界判定を既存の融合 kernel に入れると、4096² の full forward は 14.69→22.85 ms に悪化した。全出力は許容差内。境界の縮約と動的分岐の費用が、距離評価の節約を上回る。この方式は既定 kernel から取り除いた。ソース履歴は `a229a97`、同一 run 結果は `cull-4096.json`。4×8 の群判定は kernel 化していないため、性能向上を主張しない。

現在の `prepare` は毎回 atom の owner/I・B を分類して stable sort し、pack する。群内でさらに空間ソートするには追加費用がある。古い群索引をそのまま再利用すると、学習で移動した atom を取りこぼし得る。更新後の分類・境界を毎回検証し、失効時には漏れのない再構築が必要。群の二重計上にも注意する。

## X の再利用：採用した変更

既存の融合 kernel は 1 CTA の atom ループの外で X[128,32] を読み、16 出力行に再利用していた。出力行 tile を 32 にすると、同じ X を 32 行へ再利用し、station 当たりの CTA 数を半減する。作業 tensor の形や split-K=8 は変わらない。64 行 tile は 4096²・8192² ともに 32 行より遅かった。列幅 16/64 も 32 より遅かった。

| 計測 | 4096² 旧16行 → 新32行 | 改善 | 8192² 旧16行 → 新32行 | 改善 |
| --- | ---: | ---: | ---: | ---: |
| full forward（準備を含む） | 14.66 → 14.20 ms | 3.1% | 58.83 → 56.13 ms | 4.6% |
| 準備済み forward | 11.98 → 11.48 ms | 4.2% | 47.46 → 44.74 ms | 5.7% |
| 生成済み W の dense 参考値 | 0.662 ms | — | 3.029 ms | — |

4096²/8192² の full 比較は commit `7206124`、準備済み比較と既定設定の検証は `4e2caa0`。全候補が全出力検査に合格。新設定は `default_fused_config` の A100・正確な 5%・対象形状・M=128 に限定した。A100 で境界/継ぎ目/CUDA Graph 更新テスト 2 件合格、ローカルで CPU 対象 5 件合格（CUDA 対象は skip）。

split-K を 2/4/8/16 にした 4096² full の列方向距離再利用版は 16.02/15.10/14.69/14.68 ms。8 と 16 はほぼ同等で、16 は作業 tensor を倍にするため 8 を維持した。split-K=8 の partial tensor は 4096² で 16 MiB、8192² で 32 MiB。32 行化によってこの allocation は増えない。全学習ピーク、allocator 予約量、optimizer state を含む総メモリは今回測定していない。

## 棄却した距離式

通常の `||site||²+||center||²−2 site·center` は巨大な torus 半径の二乗同士の相殺を起こすため、kernel 化前に棄却した。小片の基準半径からの差を使う式も 4096² で全出力の 33 箇所が許容差を超え、最大差 3.97e-5 だったため採用しない。実験ソースは git 履歴 `9bbc9b7` に残し、現行 kernel から削除した。

Claude Sonnet 4.6、Gemini 3.1 Pro、Grok 4.7 に読み取り専用で相談した。群の厳密境界と X の複数出力タイル共有は採用・測定の対象にした。提案に含まれた学習中の索引更新猶予、旧群へのフォールバック、chart 上の周期距離は、現行の 4D 埋め込み距離に対する厳密性が確認できないため採用していない。

## 再現と記録

ソース archive SHA256: 群診断 `065855c8505946a26e2d324ffaa1734debfedd8be3838f04d2274c1752b88288`、tile 比較 `bd60d7237d9ec1f9fe0bb137d93c1835f937745423f0581233565c41d38b5a4e`、既定設定確認 `80c25ae78d3691a9c1ac6c3de7c027b3f59a1de3da8794b90ee8cc33b3abc6b7`。いずれもリモートで一致を確認した。

ローカル結果は `output/triton-a100-20260927/`（gitignored）。SHA256 は次の通り。

| JSON | SHA256 |
| --- | --- |
| `group-4096.json` | `775b23a05b25a7016e9831c1d2d586e25b84f771a05531a69a1812763133c6ba` |
| `group-8192.json` | `4bb99d02b2740ca30e8f0576b61785b79e8b653f1e70fac1e776ce61a9945c85` |
| `cull-4096.json` | `51bdd89d4dc983560b7c75e1084812f135a207dd2a686f3cfa7c8fef9809ce28` |
| `sweep-4096.json` | `19f943de271b8d2802560ccd3d478ad1c4c9a370920493c7be6a483c4f765f78` |
| `tile-4096.json` | `a1279f8b094c3ec6dd3aa8be693278ee19288769197fb5fbc36cdb93aec99c2e` |
| `tile-8192.json` | `840b6580fee942c5691f78fd2a92b9fd550b9105e30df50d45dfb42fbc0b5386` |
| `default-prepared-4096.json` | `0fb13cd0f22db57dd1e49e67bbf2a7287bbadc102f61ca35b89811b2c7a69ee8` |
| `default-prepared-8192.json` | `7d1f4fc4b3de8a6c91a5195c73bebbdbb88cc13c446e8228f65696c47c558038` |

再現コマンドの例（隔離 checkout で実行）：

```bash
PYTHONPATH=src:. python -m prototypes.diagnose_atom_site_groups \
  --size 4096 --stations 8 --source-commit 2b69412 \
  --output output/group-4096.json
PYTHONPATH=src:. python -m prototypes.benchmark_split_k_five_percent \
  --size 4096 --batch 128 --full --splits 8 --column-factored \
  --tile-rows 32 64 --tile-columns 16 64 \
  --source-commit 7206124 --output output/tile-4096.json
PYTHONPATH=src:.:tests python -m pytest tests/test_fused_compute_configs.py \
  -q -k 'config15 or default_five_percent'
```

次は、学習後の atom 分布でも小群の省略率とソート費を測り、索引の追加メモリが妥当な場合だけ群内ソートと安全な更新経路を kernel 化する。X の読み込み量については L2/DRAM traffic を測り、さらなる共有の必要性を判断する。
