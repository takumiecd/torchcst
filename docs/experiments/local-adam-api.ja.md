# CSTLocalAdam 公開API化（2026-09-09）

実装コミット: `afbbb4f`。既存の全modelを受け取る方式に合わせて
`CSTLocalAdam(model, cst=LocalAdamConfig(...), dense=AdamWConfig(...))` を追加した。
`model.parameters()` は受け取らない。CSTLinearと通常のLinearを同一modelに
置き、既存のcoordinatorで更新提案・検証・一括commitする。

## 数式と責務

実験 `local_both` + `no_trust=True` + `update_damping=0.01` を移植。
alphaの輸送・再圧縮とCの輸送は同じatom内のみ。Cはwhitenした累積parameter
勾配の外積EMA。CST更新は `(M + update_damping I) delta = -lr b_hat`。
再圧縮と更新にbatched Choleskyを用い、全体Gramと反復線形solverは使わない。
whiteningとCの平方根には小さい固有値分解を使う。trust regionは設けない。
詳細な数式・引数・使用例はREADMEの「Atom-local Adam without a trust region」。

幾何はAtomLocalTangentGeometry、二次モーメントはTransportedParameterRMS、
optimizerは既存の混在所有権・AtomGrad capture・checkpointを利用する。
公開実装にexperimentsへの依存や動的なmethod差し替えはない。
Cはparameter dtype、基底はFP64。checkpointのload時に基底をparameter dtypeへ
落とさず、初期stateもFP64としてdeferred commitと保存・再開を揃えた。

## 検証

- ローカル全体: **424 passed / 73 skipped**。ruffとgit diff --checkは成功。
- A100（既存MIG環境）: 対象4ファイル **24 passed / 0 failed**。
  コマンド: `PYTHONPATH=src:. python -m pytest -q tests/test_local_optimizer.py tests/test_local_first_moment.py tests/test_unconstrained_update.py tests/test_transported_parameter_rms.py`
- CSTLinear×2 + nn.Linear混在モデルで、FP32/FP64、通常/deferred execution、
  CPU/CUDAの実験版との複数step更新の完全一致（atol=rtol=0）。
- checkpoint復元後の複数step更新も完全一致。基底FP64/Cのparameter dtypeを検証。
- 通常Linear部分はtorch.optim.AdamW oracleと一致。
- 不正なCST更新が混在modelへ部分適用されないことを通常/deferred両経路で確認。
- 全体Gramを要求したら失敗するテスト、直接solveのdense oracle、
  同じatomから見えなくなった方向のC履歴除去、基底変換時の共分散保存を確認。
- trust radiusのconfig引数は受け付けず、正則化更新oracleでは旧radiusを超える
  displacementが切られないことを確認。

GPU snapshot: `/home/jovyan/work/srv11/cst-lab/torchcst-local-public-20260909`。
GPUログ: 同ディレクトリの `validation.log`。

今回は公開API移植の検証であり、新しいMNIST精度・速度sweepは実施していない。
以前のno-trust結果（3 seed、精度中央値80.55%、warm step中央値18.93ms）は
数式選定の根拠であり、公開API版の再測定として扱わない。
デフォルトlrは既存流儀の1e-3で、実験設定0.05を一般推奨にはしていない。
CUDAでもeigh等の内部同期まで排除したとは主張しない。
