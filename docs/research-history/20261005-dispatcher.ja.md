# Typed InputsとDispatcherへの移行

## 変更と契約

Registryから実行を分離し、登録・取得・Plan宣言/JSON検証に限定した。
Context.validate_inputsとRegistry.validate/executeは削除する。共通Dispatcherが
typed Inputsの検証、bindingからのContext構築、Plan選択/対応判定、State取得、準備と実行を接続する。
強制Planもこの入口を使う。Selector.selectはmetadataだけの便宜APIとして同じ検証処理を使う。
未選択/非対応だけをfallbackの対象にし、不正Plan・入力不足や準備/実行失敗を隠さない。

Algorithmはinput_typeを宣言し、create_state(binding, recipe=...)とexecute(state, inputs)を使う。
Stateはbinding/Recipeと準備依存Contextを保持する。Atomsを持たない演算も空のStateで実行できる。
CSTLinearはLinearInputs(x)とbinding interfaceを提供し、引き続き学習状態と実行Stateを所有する。
直接LinearBindingは既存Operator、または宣言/Parameterを結び付ける。既存AtomStateがある場合、
Parameter置換後も現在の所有状態を参照し、古いTensorを保持しない。deepcopyからcacheを除外する。

Plan/selector JSONの形式、既存の数学・数値契約、compute kernel/recipe、atoms.pyは変更しない。
Chart/Geometry/Kernelは既存Operatorから取得し、normalized専用Operatorは追加しない。
本番Polar/optimizer Algorithmsの登録、optimizerやrelayoutのGraph対応は含まない。
境界は[実行の契約](../algorithm-dispatch-boundaries.ja.md)を参照。

## 検証ソースと環境

- baseline: `9a86f3887ab1a084cfa50da3fc422c2da64c96d3`。共通Registry導入済み、Dispatcher移行前。
- candidate: `a398fc43f5aecc962a0fc75dda3c247bbba70be9`。
- L4 job: `l4job-54dfcdcd8abb4d009d94745d7425f678`。
- NVIDIA L4、driver 580.82.07、23034 MiB。
- Python 3.13.15 / Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0。
- baseline tar SHA256: `9d19ba1f213b8ef4f3ae618e41d990e119b4d9f3da726932e6a0471a9bbf8f73`。
- submitted source archive SHA256: `4589b1a4d1dab7f3ab6ff8d717c542bd78e1f07e3c2a3c8381b777ed0431121b`。
- verified result archive SHA256: `53fd7dabb835a84ab5527fb82e80d9fdfbad4bb2e8af6e95dd3409be866415e3`。
- driver SHA256: `668a4f7735c2ac764addb86ff7c40ce17b972069a69ae4d527d235e15bc6ab18`。

## 結果

最終ローカルCPUは835 passed / 326 skipped、既存TorchScript deprecation warning 18件。
Ruff/check/format、whitespace、normalizedとlocal-contraction 3 Caseの宣言検査はPASS。
wheel/sdistの隔離buildはHatchling 1.32.4で成功し、GitHub CPU validationもPASS。

L4上で14ファイルの拡張suiteは281 passed。atom配置/全optimizer状態、Algorithmの状態、
typed入力、空State、fallback失敗分類、Module/直接実行、共通Selector/Registry、Plan codec、
geometry/kernel宣言を検証した。全テストがGPU演算をするという意味ではない。
小さい独立FP64 oracleのY/dX/全座標dP、更新後支持とGraph検証はPASS。
FULL/WINDOWの初期dP相対L2は約6.92e-5、更新後はそれぞれ約1.79e-7 / 8.55e-8。
直接Plan runnerのlocal-contraction-64-middle Caseもcorrectness-onlyでPASS。

既存check_normalized.benchmarkによるN=1024 / B=128 / A=52429 / broad、FP32、TF32無効、
seed=21の完全forward/backward/optimizer stepを比較する。optimizerは普通のfused/capturable
Torch AdamWで、CSTOptimizerのGraph対応を示すものではない。
各source/方式は独立processで1 run、run内7同期wall-clockサンプルの中央値、20 optimizer steps。
CST比較の初期Parameter hashは全て一致する。

| 方式 | eager ms baseline → candidate | Graph ms baseline → candidate | peak allocated bytes | peak reserved bytes |
| --- | --- | --- | --- | --- |
| FULL | 2.0123 → 2.0899 | 0.6152 → 0.6140 | 55209472 → 55209472 | 111149056 → 111149056 |
| WINDOW | 3.4190 → 3.3853 | 1.5020 → 1.5026 | 49247744 → 49247744 | 92274688 → 92274688 |
| dense | 0.7630 → 0.7433 | 0.1262 → 0.1258 | 52955648 → 52955648 | 111149056 → 111149056 |

capture/replay込みpeakは同じ。FULL eagerはこの観測で約3.9%増えた。
速度改善・統計的な同等性は主張しない。大きい性能fixtureの全atom独立oracleと
GPU process全体のメモリ使用量は未測定。元mainに対するID mapの約0.80 MiB増加は
[AtomState記録](20261005-atom-state.ja.md)にあり、今回のbaselineには既に含む。

## 先行検証と回収

先行candidate `b96348896ea044d0e136d3d7d758666459ace29f`はjob
`l4job-02fbac5da28a4dd0aa316d01a4f43749`で280 passed、同じoracle/Graph/direct runnerを通過した。
その後、直接bindingのParameter置換追従とdeepcopy cache除外を追加し、最終ソースで再検証した。
先行result SHA256は`384d3ed683b4368215328dfccf36bf08052d38a72eca41f6ff89db0420432171`。
先行FULL eagerは2.2554 → 2.0936 msであり、最終runとはbaselineの時間も変動している。
異なるcandidateの2 jobを同一ソースの2独立runとして数えない。

初期CPU検査ではserialization fixtureの新input_type未宣言と、変換テストのParameter置換の
前提に失敗した。fixtureを更新し、Torchの明示的なoverwrite変換設定で置換を検証した。
負のログもignored outputに保持する。両GPU jobはhash検証後にremote source削除が完了し、
supervisorが所有VMを停止した。pool statusで全slot停止を確認した。
raw source/result/receipt/transport/lifecycleはignored `output/dispatcher-validation/first/`、
`final/`とhost poolのjob directoriesへ保存する。

## 再現

```sh
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:.:tests python -m pytest -q tests/test_dispatcher.py tests/test_plan_serialization.py tests/test_atom_state.py tests/test_algorithm_state.py tests/test_cst_optimizer.py tests/test_linear_parameters.py tests/test_normalized_strip_public.py tests/test_cst_conv.py tests/test_tiled_linear.py tests/test_cuda_dispatch.py tests/test_dispatch_selectors.py tests/test_backend_registry.py tests/test_geometry_declarations.py tests/test_kernel_declarations.py
PYTHONPATH=src:.:tests python -m benchmarks.cuda.linear.check_normalized --output output/public-gates.json
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run --plans benchmarks/cuda/linear/plans-local-contraction.json --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json --correctness-only --output output/local-correctness.json
```

比較driverはignored `output/dispatcher-validation/driver-final.py`に保存する。
既存runnerをsourceごとに分離したprocessで実行する。PostgreSQL専用検査と他GPUは未実施。
