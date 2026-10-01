# CUDA registry と plan の初期実装

2026-10-01。`codex/cuda-dispatch-registry-v1` の研究変更。
現在の NormalizedStripLinear の full/window を registry 経由で実行する。
学習済みの性能木、kernel の承認ポイント、GPU 別の認証は未実装。
既存の CUDA kernel と autograd の計算式・保存状態は維持する。

## オブジェクトと所有

| 型 | フィールド／操作 | 所有するもの |
| --- | --- | --- |
| OperatorSpec | operation_id、semantics_id、sizes、origin、spacing | 固定された演算・幾何の契約。sigma .03–3.25、norm floor 1e-6、五つの parameter の意味は semantics に固定。 |
| DeviceInfo | type、index、name、compute_capability、sm_count | 選択された実機の facts。CPU テストには架空 facts を注入できる。実行時は実デバイスと照合。 |
| DispatchContext | operator、input_shape/strides、dtype、atom_count、device、required_grads、execution_mode、deterministic、precision、workspace_limit_bytes | metadata のみ。M/N/K は導出し、Tensor 値は読み戻さない。 |
| FullRecipe | id、atom_num_warps、sorted_forward/backward、support、enable_fp_fusion、saved_support_flags、tuple_grads | full の完全設定。初期版は現行既定の組のみを許可。 |
| WindowRecipe | id、window_rows、enable_fp_fusion | window の完全設定。初期版は rows512 の組のみを許可。N が小さい場合の実行窓は min(512,N)。 |
| ExecutionPlan | schema_version、algorithm_id、algorithm_revision、recipe | 解決済みの不変な実行設定。algorithm_revision=v1 は実装契約の版であり、Git SHA の代替ではない。 |
| DispatchDecision | plan、tree_revision、matched_path、reason、evidence_ids、workspace_upper_bound_bytes | 選択理由。互換ポリシーなので evidence_ids は空。 |
| AlgorithmEntry | id/revision、operation_id/semantics_id、recipe_type、validate_recipe、supports、workspace_bound、load_executor | 実装への登録。入力や backward の保存状態は保持しない。 |

登録表は `(algorithm_id, revision)` で参照する。重複登録、未知版、recipe の
型違い、非対応条件を拒否する。実行 adapter は Tensor と OperatorSpec を
明示的に受け取り、既存の autograd Function を呼ぶ。backward は forward の
設定・保存 Tensor をそのまま使用し、selector を再実行しない。

静的な幾何とデバイスで確定する offset だけを adapter にキャッシュする。
norm、support、row bucket、作業領域、勾配は各呼び出しの所有である。
CUDA モジュールは executor を実行するときだけ import する。

## 配置

- `../cuda/schema.py`: 不変の contract。
- `../cuda/context.py`: 実 Tensor から metadata を集める。
- `../cuda/registry.py`: 登録と直接 plan 実行。
- `../cuda/algorithms/normalized_strip/`: 登録、共有 guard、既存 full/window への adapter。
- `../cuda/dispatch/select.py`: 現行 memory 指定と同じ互換選択。
- `../../../../../benchmarks/cuda/linear/run.py`: 同じ registry/plan を使う検証と完全step。

既存の低レベル kernel は `../normalized_strip/` に残す。今後の algorithm も
登録契約と recipe を揃え、独立した oracle と adapter を追加する。

## 選択と実行

1. module が元の入力を flatten/contiguous にし、実行 context を作る。
2. memory=full は full を選ぶ。memory=window は window の適格性を確認し、
   不適格なら full を検証して戻る。両方非対応なら実行前に失敗する。
3. registry は plan、実 Tensor の metadata、精度・device の一致を検証する。
4. executor が既存 autograd 接続を呼ぶ。
5. backward は呼び出しごとの保存状態から dX と全 atom 勾配を返す。

benchmark が window を直接指定した場合は fallback しない。非対応は失敗と
して記録する。実行後の OOM を捕まえて別 algorithm に再実行しない。
CPU と空入力/空atom は、既存 module の経路を維持する。

scratch 全体の証明済み上界は未確定なので `workspace_bound=None` とする。
window の weight/halo の式は routing/state 等を含まず、全体上界には使わない。
workspace_limit_bytes を指定した場合は、上界がない plan を拒否する。
これは測定 peak allocated、reserved、GPU process usage とは別の情報である。

## 共通 benchmark の入口

```bash
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run \
  --algorithm normalized_window --size 1024 --rows 128 --profile broad \
  --dense --source-commit "$(git rev-parse HEAD)" --output output/registry-broad.json
```

最初の対応は normalized full/window、FP32、N1024/8192、5% atom。
`--profile sharp` は構成した鋭い支持 fixture として別記する。
候補と normalized full baseline は同一初期parameter/input/optimizer契約で
実行する。dense Linear は直接weightを更新する別の性能参照であり、CST の
parameter 更新や独立勾配 oracle の代替ではない。

正確性は小さい混合 fixture の FP64 sampled-site oracle と W、Y、dX、全五勾配を
照合する。大きい shape は完全stepの計時・メモリ測定で、全atomの独立勾配
照合を実施したという意味ではない。各ケースの計時は新しい subprocess で行い、
同期 wall time の全sampleと中央値、capture/replay peak allocated/reserved を保存。
GPU process usage は未測定として None。測定は同じfixtureの独立process比較で、
交互の対応計時ではないため、微小差による勝者の認証には使わない。

ソースファイルhash、source commit、GPU実機、CUDA/PyTorch/Triton、recipe、seed、
測定条件を JSON に残す。承認用の全形状suiteやケースファイルschemaは次の段階。
出力が PASS でも承認状態は not assessed であり、自動採用しない。

## 検証

CPU: contract、fallback、直接指定の拒否、複数forwardのautograd接続、CPU公開API。
GPU: installed wheel の同じテスト、独立oracleの必要勾配の組合せ、複数live forward、
Graph capture後の中心/幅変更、AdamW state、通常sigma3/sharp別の完全stepとpeak。
GPU driver は `benchmarks/cuda/linear/validate_dispatch_registry.py`。

### 検証済み checkpoint

- source commit: `ee4effd70822fd3287e8e6775e211e2ada7937dc`。
- ローカル CPU 全体: 495 passed、189 skipped（CUDA 等）、PyTorch 2.13.0。
- pool job: `l4job-1000bf47ff0c40c8ae73420ea4d8b158`、succeeded。
- frozen source archive SHA256: `ab71a235e9bfd39ac43fdbaed975f559ebe03b901917213f3ce2a0cd75393544`。
- result archive SHA256: `a921f181e8584e406d7b4e953a1adf638767efa399a7e7d1ba6f3200bc64cccc`。
- NVIDIA L4、GPU-70468812-5019-6f2d-1e66-ca0bb8c6a823、driver 580.82.07、
  PyTorch 2.11.0+cu128、CUDA 12.8、Triton 3.6.0、Python 3.13.15。
- installed wheel の関連テスト: 94 passed。直接planの必要勾配の組合せと
  複数live forwardを含む。独立oracleとGraph/AdamW状態検証もPASS。
- full/windowの通常sigma3とsharp fixtureで、初期parameter hashの一致を確認。
- 結果archiveはpoolでSHA照合済み。owned L4 slotのstoppedを確認。

N1024、M128、atom 52429、FP32、TF32無効、fused/capturable AdamW。
時間は同期wallの7sample中央値。peakはmodel/input/gradient/optimizerを含み、
Graph capture/replayを含む。GPU process全体の使用量は未測定。

| fixture | plan / reference | eager ms | Graph ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 通常 sigma3 | normalized_full | 1.56768 | 0.63450 | 51.85 | 106 |
| 通常 sigma3 | normalized_window | 3.30984 | 1.52948 | 46.16 | 86 |
| 通常 sigma3 の性能参照 | dense Linear | 0.72651 | 0.12715 | 50.50 | 106 |
| sharp-only fixture | normalized_full | 1.64341 | 0.54902 | 51.85 | 106 |
| sharp-only fixture | normalized_window | 2.81897 | 0.30727 | 46.16 | 86 |
| sharp fixture の性能参照 | dense Linear | 0.67416 | 0.12788 | 50.50 | 106 |

通常sigma3ではwindowはfullより遅く、peak allocatedは小さい。
今回の変更は選択・実行・測定の接続であり、L4のアルゴリズム改善の主張ではない。
N8192はrunner対応だが今回未測定。別GPUの認証・性能、交互対応計時、
全形状suite、kernel承認ポイント、学習済みdispatch木は未実施。

生データ、wheel、展開ソース、ログ、frozen sourceはignoreされた
`benchmarks/cuda/linear/evidence/registry-v1/l4job-1000bf47ff0c40c8ae73420ea4d8b158/`
に保存した。元のpool job directoryにも保持している。raw evidenceはGitへ追加しない。

再現（現在のcheckoutを固定して一台のL4 queueへ投入）:

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_dispatch_registry.py" \
  --label cuda-dispatch-registry-v1 --timeout 900 -- "$(git rev-parse HEAD)"
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

すでにsupervisorが稼働している場合は二重起動せず、submit後にwaitする。
