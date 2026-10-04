# CUDA Linear benchmarks

現在の backend の正しさ・速度・メモリを確認する入口。
新しい研究候補の追加からPR統合までの手順は
[カーネル開発手順](../../../docs/kernel-development.ja.md)を参照。
試作実装をここへ置かず、測定対象は `src/torchcst/_backends/` にある実装とする。
公開 API の統合確認と、registry の plan を強制する比較を分ける。

| ファイル | 役割 | 測定範囲 |
| --- | --- | --- |
| `run.py` | Plan catalog と Case を読み、選択 Plan を強制して比較 | 独立 oracle、forward・backward・AdamW、eager / Graph、capture を含む peak |
| `check_dispatch.py` | 実機条件の JSON を明示的に書き、読み戻して公開 CSTLinear を実行 | 完全一致・未観測 fallback・不正ファイルの拒否、独立 oracle と Graph。任意で完全 step を別 process で測定 |
| `check_normalized.py` | 公開 CSTLinear の統合確認 | 独立 oracle、全5 atom 勾配、更新後の支持、Graph replay と AdamW 状態。任意で完全 step を測定 |
| `validate_wheel.py` | shared L4 pool の検証 driver | 配布 wheel の全維持テスト、上記の統合確認、通常幅・鋭い支持・dense の完全 step |
| `strip_torus.py` | 既存 Strip/Torus の fused / split と参照経路の診断 | 準備、forward、forward/backward。optimizer は含まない |
| `strip_torus_dense.py` | Strip/Torus と保存済み dense の比較 | forward / backward、FP32 と別条件 BF16。メモリは warmed baseline からの追加割当 |
| `strip_torus_split.py` | Strip/Torus の split reductions の比較 | forward / backward、forward Graph。optimizer は含まない |
| `manifest.py` | Plan catalog / Case / 固定 snapshot の読み込み | GPU 不要の schema・recipe・比較条件検証 |
| `fixtures.py` | Chart・Kernel・model の共通構築 | 純粋な Spec と設定境界の State 構築。テストからも利用 |
| `reference.py` | 正規化 Triweight の独立 dense oracle | backend の距離・重み生成を呼ばず、サンプル点から直接計算 |

`__init__.py` を含めて11 Python file。benchmark から `tests/` を読み込まない。
旧 `experiments/`、探索用 CLI、移行段階ごとの validation driver は削除した。
旧 module 名や `--tests-file` / `--chart-suite` の互換入口は用意しない。

## 正規化 kernel の確認

```bash
python -m pip install -e '.[dev,cuda]'
python -m benchmarks.cuda.linear.run --list-plans
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/normalized-1024-broad.json --validate-only
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/normalized-1024-broad.json --correctness-only --output output/normalized-check.json
python -m benchmarks.cuda.linear.run --case benchmarks/cuda/linear/cases/normalized-1024-broad.json --source-commit COMMIT --output output/normalized-step.json
python -m benchmarks.cuda.linear.check_normalized --output output/public-check.json
python -m benchmarks.cuda.linear.check_dispatch --output-dir output/dispatch-check --bench
```

[plans.json](plans.json) は名前付き Plan の一覧。
各 Plan に Algorithm ID、revision、schema version、全 recipe フィールドを書く。
registry の `load_plan` が登録済み recipe 型へ変換・検証する。
export は `dump_plan`、単一 Plan の JSON text の往復は `dumps_plan` / `loads_plan` を使う。
[Plan の JSON API](../../../src/torchcst/_backends/README.md#plan-の-json-export--import)を参照。省略による既定値の補完や未知の設定は受け付けない。
現在 full / window512 の検証済み recipe 値のみを受け付け、自由な tuning sweep は用意しない。

[cases/](cases/) の各 JSON は測定条件（形状、atoms、幅、dtype、seed、warmup、rounds、optimizer）と、
比較 Plan 名の一覧、明示的な baseline、dense の有無を持つ。
`--plans PATH` で catalog を変更できる。`--device` は実行 GPU の index。
`--list-plans` と `--validate-only` は GPU 不要。実機での適合性は実行時に registry が検査する。

実行開始時に両ファイルを一度だけ読み、選択された Plan と Case の固定 snapshot を
output の隣の固有ディレクトリへ保存する。worker はその SHA256 を検査してから読み込む。
各 Plan の独立 oracle 確認を新しい process で行い、全て通った後に各完全 step を計測する。
実行時に dispatch の自動選択や別 Plan への fallback は行わない。
候補間の初期 Parameter / input / target の hash 一致も確認する。
結果には元の JSON の hash、固定 snapshot、実際の Plan、source hash、実機環境、誤差、時間分布、メモリを残す。
実行 UUID と UTC 開始時刻も記録する。同じ数値を得た別実行と結果の再送を区別する。
完成した `run.py` の JSON は [PostgreSQL 保存入口](../../database/README.md)で追記・検索・復元できる。
固定された測定依頼を Colab で実行し、中央で照合して保存する入口は
[official measurement tooling](../../automation/README.md)を参照。
`--snapshot PATH --snapshot-sha256 HASH` は固定した run を coordinator に直接渡す。
元の Case / catalog を読み直さず、同じ候補・初期条件を実行する。
失敗した worker の結果と途中までの記録も保存し、計測を中断する。

`dense: true` は通常の dense Linear を追加する。dense は性能の参照で、CST と同じ Parameter
空間の演算ではない。`normalized-1024-sharp.json` は鋭い支持の別条件。通常の sigma3 と混ぜない。
fixture は現在 normalized Euclidean Strip / FP32 / 1024²・8192²に限定する。
独立 oracle は境界を含む小さい混合 fixture、完全 step は Case の大きい fixture を使う。

`check_normalized --algorithm normalized_full normalized_window` は Algorithm ID から
通常の Plan と FixedSelector を構築して公開入口を確認する。専用の memory 指定はない。
`check_normalized` と `validate_wheel` は従来の NVIDIA L4 検証条件を維持する。
`run` と Strip/Torus の CLI は他の CUDA GPU でも使える。
全 CLI の設定は `--help` を参照。wheel gate は1024²に限定する。
`check_normalized --bench` は既定で1024²と8192²を測るため、通常の確認では
`--sizes 1024` を明示する。

## 配布 wheel の L4 gate

[`colab-l4-pool`](../../../tools/colab-l4-pool/SKILL.md) を使う。
この driver は system package を変更せず、job-local の Torch 2.11.0+cu128 / CUDA 12.8
環境でテストする。初期 system probe の CUDA 13.0 と実際の検証 runtime を区別する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_wheel.py" --label wheel-gate --timeout 600 -- SOURCE_COMMIT
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

N1024²、M128、約5% atoms、FP32、TF32無効、fused capturable AdamW の契約を維持する。
各時間・メモリ測定を別 process にし、peak allocated / reserved は Graph capture と replay
を含める。GPU process usage は別の指標で、この driver では未測定。
大きい fixture の完全 step 測定と、小さい fixture の独立全勾配 oracle を区別する。

## Strip/Torus の診断

```bash
python -m benchmarks.cuda.linear.strip_torus --output output/strip-torus.json
python -m benchmarks.cuda.linear.strip_torus_dense --source-commit COMMIT --output output/strip-torus-dense.json
python -m benchmarks.cuda.linear.strip_torus_split --output output/strip-torus-split.json
```

これらは現在の fused backend の診断であり、完全学習 step の結果とは比較しない。
BF16 は別精度の参照。Parameter bytes や追加 tensor 割当を GPU 全体の peak と呼ばない。

## 記録

作業出力は ignored `output/`、raw log / trace / tensor / snapshot は ignored
[evidence/](evidence/README.md)、小さい検証記録は [results/](results/) に保存する。
GPU・runtime・source commit/hash・演算条件・時間分布・誤差・メモリの範囲を記録する。
benchmark の成功は kernel / dispatch の認証や既定採用を自動で行わない。

過去の数値や不採用理由は [研究履歴](../../../docs/research-history/cuda-linear/README.md)、
削除した実行コードの取得方法は [削除台帳](../../../docs/research-history/cuda-linear/retired-code.ja.md)
に保存している。履歴中の旧コマンドは記録当時の source commit で再現する。
[今回の整理と検証](../../../docs/benchmark-cleanup.ja.md)を参照。

## Plan 一覧の入口の検証（2026-10-02）

source `8f2ec40` の CPU は515 passed / 117 skipped。
同じ source の配布 wheel を NVIDIA L4、Torch 2.11.0+cu128 / CUDA 12.8 / Triton 3.6.0
で確認し、631 passed / 1 skipped / 0 failed。skip は Linux で使えない macOS socket binding。
full / window512 の独立 oracle、公開 API の更新後 Graph / AdamW、両 Case の完全 step が通った。
各 Plan の初期 Parameter hash と、dense を含む input / target hash は一致。
source / wheel / installed の152 Python file を照合し、local と L4 の wheel hash も一致した。

[検証記録](results/plan-catalog-20261002.json)に固定 Plan・Case、誤差、全時間サンプル、
peak allocated / reserved、source と archive の hash を保存している。
raw artifact は ignored evidence に保存済み。共有 pool の owned GPU は停止済み。
この確認は小さい独立 oracle と完全 step 測定の範囲であり、認証や既定採用は行わない。

Plan の専用 JSON API は source / 配布 wheel の CPU suite で各533 passed / 117 skipped。
full / window の dict・JSON text・ファイル往復、recipe 型の復元、GPU 実装を import しないこと、
不正・重複・非有限な設定の拒否と benchmark の固定 snapshot を確認した。

`check_dispatch` は catalog の二つの Plan を明示的に使う動作確認。生成する JSON は
`demonstration-only` と表示し、性能順位や承認済み policy として配布しない。
`dispatch-*.json`・選択 trace・oracle 誤差・不正 JSON の拒否理由を結果ディレクトリに保存する。
`--bench` は1024²・M128・52,429 atoms・sigma3・FP32・AdamW の完全 step と
dense 参照を別 process で測定する。小さい混合 fixture の独立全 atom oracle と区別する。

[L4 の JSON 実機検証記録](../../../docs/dispatch-json-verification.ja.md)と
[機械記録](results/dispatch-json-20261002.json)を参照。

## Small PolarAmpWidth product research

The existing runner also accepts the research-only `local_polar_product` fixture.
Use the separate catalog with the same runner, fresh-process isolation, frozen
run declarations, FP32/TF32-off settings, correctness gates and capture-inclusive
memory measurements:

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-product.json \
  --case benchmarks/cuda/linear/cases/local-64-broad.json \
  --output benchmarks/cuda/linear/evidence/local-64-broad.json
```

Sizes are 16/32/64, batches 16/16/32 by default, and atoms are rounded from 5% of
N*K. Profiles `sharp`, `few`, `broad`, `wide`, and `mixed` initialize rho=1, 2, 3,
16, or mixed widths through the **production polar activity map**. Minimum sigma
and spacing are both 1 in these fixtures. `sharp` also aligns centers to grid
sites; one-hot eligibility is checked from actual support and the norm floor.
These specify **only the initial state**. Sigma is recomputed from the current
PolarAmpWidth angle/activity radius every forward, including graph replay.
`sigma_updates` records initial/final per-atom widths, the count that changed,
and maximum change. The dynamic fixture fails if no widths change. Width changes
follow the existing CST update policy, including radial regularization and dormant
expansion; no fixed sigma tensor is injected into the forward or backward.
Initial and final diagnostics are outside timing and outside the measured peak;
peak statistics are saved immediately after capture/replay.

Plans compare fused local H, saved H, fused without ordering, and the same
normalized CST computed with Torch factors. An ordinary dense Linear is a
separate performance reference. Candidate plans share atoms, inputs, loss and
AdamW proposal settings. CST candidates then apply the production Euclidean
polar finite-chord/activity/radial update policy. Dense uses ordinary AdamW.
The graph-safe fixture specialization is checked against public CSTOptimizer,
including AdamW moment state. It does not add public CSTOptimizer graph support.

`initial_support` reports exact full/local support counts, local intervals,
normalization-floor cases and touched tiles, **outside timing**. The diagnostic
uses full factors and is not the future sparse runtime preparation path. Step
measurements do include polar decode, ordering, full-domain norm preparation,
forward, dX, all atom gradients, AdamW proposal, and polar policy update.
The current paths select fused/saved for the whole call; these reports do not
implement a mixed per-atom routing policy or guarantee any L1/L2 residency.

For the matched width/reuse experiment use `plans-local-rho.json` and
`cases/local-{64,128}-rho{1,1_5,2,3,4,8,16}.json`. These labels select initial
sigma/spacing; sigma is updated during every training step. Across widths only
initial polar radius changes, keeping centers, amplitudes and inputs matched.
`polar_support` and `polar_support_saved` both read the actual input support for
H; the latter materializes H for reuse. See the
[rho experiment record](../../../docs/research-history/local-product/20261004-rho-sweep.md).

`--phase-diagnostics` adds a separate graph with external CUDA events after
primary timing/memory measurement. It reports forward+loss, backward and
optimizer phases with instrumentation overhead; training continues, with no
sigma freezing. Primary uninstrumented complete-step timing and capture/replay
memory peaks remain the comparison metrics. Reported final sigma is the end of
the primary measurement, before phase diagnostics run.

## 小さい積カーネルのサイズ比較（研究）

`local-size-{32,64,128}-{early,middle,late,narrow-only}.json` はbatch32、
約5% atoms、同じ初期rho分布・PolarAmpWidth更新を使う研究用Case。
`plans-local-persistent.json` のsaved-H、毎回pack、配置を保持するhybridを
既存runnerで比較する。各Caseに同じshapeのdense完全step参照を含む。

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-persistent.json \
  --case benchmarks/cuda/linear/cases/local-size-64-middle.json \
  --phase-diagnostics --core-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-size-64-middle.json
```

`--core-diagnostics` は測定後、初期状態を別途再構築し、正規化と配置の準備を
除いたforward（H生成とY）をGPUイベントで測る。100回のforwardを一つのGraphに
入れ、Pythonからのreplay間隔の影響を避ける。入力・係数を反復利用する診断であり、
backward/loss/optimizerは含まない。dense診断は同じCST演算子のWを参照用に準備する。
候補はWを生成しない。FP64 Torch factor参照との一致を確認するが、独立したscalar
全勾配oracleと完全step測定は別のgate。完全stepのpeakは診断の前に保存する。
診断時間を完全stepから差し引いて残りの時間を推定しない。


## Local contraction tuning (2026-10-05)

The consolidated `plans-local-tuned.json` includes warp/block tuning, bounded
middle-band unrolling and the measured negative vector-reduction alternatives.
`local-tuned-{64-middle,64-narrow-only,64-sigma3,128-early}.json` compares the
control and atom32/4-warp variants in the existing runner:

```bash
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-local-tuned.json \
  --case benchmarks/cuda/linear/cases/local-tuned-128-early.json \
  --polar-update fused --phase-diagnostics --kernel-diagnostics \
  --output benchmarks/cuda/linear/evidence/local-tuned-128-early.json
```

Use the catalog to explicitly select `persistent-supportprep-band-unroll` when
comparing ordinary initial-sigma3. These remain research recipes; shared sigma
is refreshed on every step. No public default selection is changed.
See the [measured record](../../../docs/research-history/local-product/20261005-contraction-launch.md).
