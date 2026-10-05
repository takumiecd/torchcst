# AtomStateとAlgorithmStateの所有境界

ModuleはAtomStateを所有し、固定atom IDと物理行を分離する。配置変更はParameter、
既存gradient、明示したatom軸を持つoptimizer状態を一緒に移動し、完了後に世代を進める。
既存`atoms.py`と数学的kernel、CUDA kernel、recipeには変更を加えていない。
公開契約とcheckpoint形式は[状態所有の説明](../atom-state.ja.md)に記載する。

## 検証したsource

- baseline: `b49b982fe6cf51119b6d4d2bdfe4b3ee0f22f450`
- candidate: `a9c07feed8140c01746ea4e2cf79244e3829a0d3`
- L4 job: `l4job-97ce356c8092495c8b97ecdaf0438dfe`
- source archive SHA256: `3336d9756da353a96a24ef5fbd2d5ee549b1e33b644303025e893ae745589bca`
- baseline archive SHA256: `c51ae42710a0938c5baf0f187de252843908080e9abde7976dbb908cbac9c026`
- retrieved result archive SHA256: `b2df6f236e65bcb0a18f4f68a1f88c657e771c2054ad4300a95fd19babe9d07d`

ローカルCPU suiteは820 passed / 326 skipped。Ruff、Plan/Case宣言検査、
wheel/sdist buildも通過した。PostgreSQL専用検証と他GPUは実施していない。

L4はNVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0。
CUDA関連suiteは123 passed。FULL/WINDOWで実際のatom配置変更後のY/dX/全5座標勾配を
独立FP64 oracleで検証した。既存public gateは境界、幅・支持の更新、Graph replayも通過。
小さいfixtureの初期dP相対L2誤差は両方式で約6.92e-5、更新後は1.79e-7以下。
大きい性能fixtureの全atom独立oracleは実施していない。

## 完全stepの観測

N=1024、B=128、A=52429、broad、seed=21、float32、TF32無効。
公開Module経路と通常のfused/capturable Torch AdamWを使う既存
`benchmarks.cuda.linear.check_normalized.benchmark`を実行した。
CSTOptimizerのcoordinate policyのGraph対応を意味しない。

| 方式 | eager ms baseline → candidate | Graph ms baseline → candidate | capture/replay peak allocated bytes | peak reserved bytes |
| --- | --- | --- | --- | --- |
| FULL | 1.785 → 2.068 | 0.6093 → 0.6106 | 54369792 → 55209472 | 111149056 → 111149056 |
| WINDOW | 3.310 → 3.425 | 1.5169 → 1.5112 | 48408064 → 49247744 | 90177536 → 92274688 |
| dense | 0.728 → 0.683 | 0.1278 → 0.1266 | 52955648 → 52955648 | 111149056 → 111149056 |

各source/方式を独立processで1 run、各runは同期付きwall clockの7サンプル中央値。
Graph前のwarmup/captureも既存プロトコルに従う。初期Parameter hashは全CST方式で一致。
atom IDの双方向int64対応によりCSTのallocated peakは839680 bytes（約0.80 MiB）増えた。
FULL eagerでは約16%、WINDOW eagerでは約3.5%の増加を観測した。Graph時間の差は小さいが、
単一runから統計的同等性や高速化は主張しない。GPU process全体の使用量は未測定。

## 再現と保存先

```sh
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:.:tests python -m pytest -q tests/test_atom_state.py tests/test_algorithm_state.py tests/test_cst_optimizer.py tests/test_linear_parameters.py tests/test_normalized_strip_public.py tests/test_cst_conv.py tests/test_tiled_linear.py
PYTHONPATH=src:.:tests python -m benchmarks.cuda.linear.check_normalized --output output/public-gates.json
```

比較driverはignored `output/atom-state-validation/driver.py`に保存した。
raw logs、source/result archives、receipt、driver specは同ディレクトリの`retrieved/`と
host poolの該当jobディレクトリに保持する。取得archiveのhashはreceiptと照合済み。
jobはexit 0 / timeoutなし、supervisorは終了し、使用したpool slotはstoppedと確認した。
新worktreeは`~/.codex/worktrees/atom-state/torchcst`、branchは`codex/atom-state`。
旧execution-layout worktreeの回収先はprimary checkoutの
`output/worktree-recovery/execution-layout-20261005T105552/RECOVERY.md`。

次の共通registry整理では、現行RegistryのLinear用入力検証と共通のID/Plan管理を分離する。
この記録のGPU数値は上記candidateに対するものであり、将来の構造変更を検証したものではない。
