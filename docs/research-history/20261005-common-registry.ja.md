# backend・演算共通のAlgorithm Registry

共通Algorithmにimmutable ID/revision/operation/semantics、recipe検証、対応条件、
workspace上界、入力検証と処理/state lifecycleを統合した。Registryは任意の名前付き入力を
Algorithmへ渡し、Linear用のx・parameters・OperatorSpecを要求しない。
Fixed/Ordered/Exact Selectorも独立したContextを扱える。

CSTLinearの全経路を同じRegistry/Planへ接続し、Torch materialized/factored/tiled/
normalized referenceとCUDA FULL/WINDOW/Strip + Torus fusedを登録した。selector自身の
Registryを使うためcustom Registryも動作する。`_backends/linear.py`、`normalized.py`と
CUDA専用のcontrol-plane型は削除した。既存backend名はPlanへのaliasとして残す。

OperatorはChart/Geometry/Kernelへの共通入口を保つ。normalized専用Operatorは追加せず、
regular site配置の抽出はAlgorithm内に置く。既存Torch normalized executorと配置抽出は
移動前とbyte単位で一致する。`atoms.py`とCUDA compute kernel/recipeは変更していない。
具体的なproduction Polar/optimizer Algorithmの登録や既定dispatchの性能による昇格は含めない。

## source・検証

- baseline: `a9c07feed8140c01746ea4e2cf79244e3829a0d3`（AtomState導入後）
- candidate: `62eae3ae8c5d5f286011e57de69b9402bce885ca`
- runtime変更のCPU checkpoint: `3aa35a84f246d0047ddeed2810c7e38d6a804485`
- CPU suite: 824 passed / 326 skipped。candidateの差分はgeneric fixtureの対応条件修正だけで、focused 4 testsも通過。
- GitHub CPU validation、Ruff/format/whitespace、normalized/local-contraction宣言、wheel/sdist buildも通過。
- L4 job: `l4job-8d89ced2368d4493a762a424579a1d0d`
- source archive SHA256: `3086c0c61c63a1bea4fcf4fdede2190dc8a6afc6ab1f5000fe3ad4d14aa88617`
- baseline archive SHA256: `ed6f9bddc5e031fd8b2a963edc3cbe6fdbf56348e6bc82c53655e5e48da6e15c`
- retrieved result archive SHA256: `b95ef7adfb7be898f7ba49202f99ccf14a82ea6caa460e358d99c1e323973ead`

実機はNVIDIA L4、driver 580.82.07、Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0。
L4上のexpanded suiteは252 passed。FULL/WINDOWの配置変更後Y/dX/全5座標勾配、
必要勾配の組合せ、二つのlive forward、Strip/Torus fusedを含む既存の公開経路を検証した。
小さい独立FP64 oracle、境界、幅・支持更新、Graph replayのpublic gateも通過した。
driverはexit 0、timeoutなし、result archive hashはreceiptと照合した。
PostgreSQL専用検証と他GPUは実施していない。

## 完全stepの比較

N=1024、B=128、A=52429、broad、seed=21、float32、TF32無効。
通常のfused/capturable Torch AdamWを使う既存public Module benchmarkで測定した。
CSTOptimizer coordinate policyのGraph対応を追加したものではない。

| 方式 | eager ms baseline → candidate | Graph ms baseline → candidate | capture/replay peak allocated bytes | peak reserved bytes |
| --- | --- | --- | --- | --- |
| FULL | 2.004 → 2.013 | 0.6107 → 0.6149 | 55209472 → 55209472 | 111149056 → 111149056 |
| WINDOW | 3.341 → 3.235 | 1.5053 → 1.5048 | 49247744 → 49247744 | 92274688 → 92274688 |
| dense | 0.734 → 0.701 | 0.1244 → 0.1261 | 52955648 → 52955648 | 111149056 → 111149056 |

各source/方式を独立processで1 run、run内7サンプルの同期付きwall clock中央値。
初期Parameter hashはCSTの全比較で一致し、optimizer stepは全方式20回。
この観測で共通化による追加GPUメモリはなく、時間差は小さい。統計的な同等性や高速化は
主張しない。大きい性能fixtureの全atom独立oracleとGPU process全体使用量は未測定。
mainからAtomState導入までの約0.80 MiBのID map増加とeager overheadは
[先行記録](20261005-atom-state.ja.md)を参照する。この比較はその費用をbaselineに含む。

## 接続失敗と回収

先行job `l4job-53d819e927384bdcae35d095ace3db78`はJupyter output待ちのtimeoutで
完了確認・result回収ができず、停止要求も一度Not Foundで失敗した。コードの正否は判定できない。
共有poolの`recover`を実行し、停止確認後に上記jobを明示的に再提出した。
失敗jobのsource/spec/transportとslot lifecycleをignored
`output/common-registry-validation/interrupted/`へ保持した。成功jobのarchives/receipt/logsは
同ディレクトリの`retrieved/`とhost poolのjob directoryに保持する。
成功jobはresult回収・hash検証後のremote source削除で接続を失い、cleanup要求がtimeoutした。
supervisorは所有VMを停止し、pool statusで全slotの停止を確認した。remote source削除の
完了は未確認として記録し、最終transportとslot lifecycleもローカルへ保持した。

## 再現

```sh
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:.:tests python -m pytest -q tests/test_backend_registry.py tests/test_atom_state.py tests/test_algorithm_state.py tests/test_cst_optimizer.py tests/test_linear_parameters.py tests/test_normalized_strip_public.py tests/test_cst_conv.py tests/test_tiled_linear.py tests/test_cuda_dispatch.py tests/test_dispatch_selectors.py tests/test_geometry_declarations.py tests/test_kernel_declarations.py
PYTHONPATH=src:.:tests python -m benchmarks.cuda.linear.check_normalized --output output/public-gates.json
```

比較driverはignored `output/common-registry-validation/driver.py`に保存する。
benchmarkには既存`benchmarks.cuda.linear.check_normalized.benchmark`を使い、baseline archiveから
分離したprocessのPYTHONPATH/cwdで再現する。workspaceは`~/.codex/worktrees/atom-state/torchcst`。
