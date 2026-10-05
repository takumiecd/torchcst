# LinearとAtom更新の分離

## 契約と変更

Linear Algorithmsを各backendの`algorithms/linear/`へ整理し、Atomの座標更新を
共通の`atom_update`演算へ分離した。CSTOptimizerはKernelの種類で分岐せず、
`AtomUpdateInputs(previous, step_size)`とlive Operatorを参照するbindingを共通Dispatcherへ渡す。
base optimizerは提案・moment・clockを担当し、更新Algorithmは提案へ座標法則を適用する。
LinearのPlanと更新Planは独立する。

既定は`torch_atom_update@v1`。KernelのTorch参照を呼び、Sphere/TorusのGeometry retractionを
維持する。既存の勾配射影・OptimizerStateAdapterのベクトル状態輸送も保つ。
`update_selector=`は共通Selectorを受け取り、PolarのTorch/CUDA更新も同じ演算の候補になる。
CUDA候補は連続CUDA FP32と二つのEuclidean 1D中心に限定し、対応条件をAlgorithmへ置く。
一般のfallbackはTorchの座標法則であり、非Euclideanの提案をそのまま採用しない。

PolarのTorch参照・Graph対応executorとCUDA融合executor/kernelを独立ディレクトリへ移した。
CUDAのPolar update kernelとStrip/Torus preparation kernelは整理前からバイト単位で不変。
`atoms.py`、optimizer checkpoint manifest 3、配置の所有・世代管理は変更しない。
CSTOptimizer全体のGraph captureは未対応のまま。研究runnerのGraph検証は個々の更新候補を対象とする。

更新snapshotは呼び出し元が所有し、shape/device/dtype、非alias、detach、正のstepを選択前に検査する。
強制した非対応Planはbase optimizerがParameter/momentを進める前に拒否する。
ExactSelectorの共通codecはTorch dtypeのmetadataを宣言として保存できるようにした。
Plan/Case/測定JSON、DB schema/adapterは変更せず、実DBへの書き込みは行わない。

## ローカル検証

runtime候補`08d14fd80882f4a1bc59f6e87b507563ddea2fed`で、CPUは855 passed / 330 skipped、
既存TorchScript deprecation warning 18件、32.25秒。選択器のJSON往復、Sphere/Torus fallback、
複数stepのParameter/m/v/clock、失敗前の非更新、配置変更後のState再準備を含む。
変更ファイルのRuff F/I・format・whitespace、local-contraction宣言検査はPASS。
wheel/sdist隔離buildはHatchling 1.32.4で成功。
wheel SHA256は`ac8851ef95065d0eb457a675139e402e83ed718f3218a369d3752de8db631fd9`。
同じcommitのGitHub CPU validationはrun `37271316049`でSUCCESS。

## 先行GPU検証と通信回復

job `l4job-19276a5c40244a81985f26dbdab9465b`はallocation後の通信切断でdriver未実行。
allocation前にserver assignmentが空、後に所有allocationから一つだけ生成されたendpointを確認し、
共有pool supervisor lock下でそのendpointだけを解除した。server残存0を確認してpool recoverを行った。
他sessionやqueue DBを書き換えず、回復の証拠はignored outputのorphan-recovery.jsonに残す。

明示再提出job `l4job-318a74ffbcc740958ea7ea414fb0d019`、候補`10e9c25c`は409 passed / 2 failed。
新しいAtom更新とGraph replayは通過した。非対応dtypeの古い例外型期待とStrip/Torusの
境界tile ownership一致テストが失敗し、計測は開始しなかった。dtypeテストは共通Dispatcherの
UnsupportedPlan契約へ修正した。失敗ログとsource/resultは保持する。
source archive SHA256: `53b6392068141b51c805fccf3da8833256901aff3a95dff0adeb50c2f75aef7e`。
verified result archive SHA256: `e126623f03ec1590e4d4091a31a21df437d9ccab5c26eea1f5974f9c80e883c8`。
このjobのremote source cleanupと所有VM停止は完了した。

## 整理前との回帰比較

job `l4job-e08a60d990b44aebadcd390083e6a69e`は、同じL4上でbaseline `1f0b42e0`と
候補`08d14fd8`を別source/processで実行した。baselineでも上記2件の失敗を再現した。
候補のdtype拒否は修正後に通過。境界ownershipの不一致は両sourceで同じindex 176、
Torch arc=62.5、station 11/12への距離=2.375の同距離点で、Torch owner=11 / CUDA owner=12。
数値kernelは変更していない。この既存の厳密一致の不具合は本整理では修正していない。
テスト本体は弱めず、別診断で失敗を記録した。残りのsuiteは390 passed / 1 deselected、
profiler warning 1件。全GPUテストが通ったという意味ではない。

normalized FULL/WINDOWの独立FP64 oracle、Y/dX/全座標勾配、更新後支持・Graph検証はPASS。
local-productの既存runnerもbaseline/candidate × Torch/fused更新の4 runでPASS。
小さい独立oracleの全atom dP相対L2は約1.12e-6、Polar update誤差は0。
候補の完成JSONは提出前検査も通過した。N=64 / B=32 / A=204、seed=41、FP32、TF32無効、
mixed幅、rho=[0.25,0.75,2,8]、fused/capturable AdamW、lr=1e-4、weight_decay=.01。
各source/更新候補は1独立run、warmup 5、同期wall-clock sample 21。21独立runではない。

この候補では、Torch更新のeager完全stepは2.6806→3.2231 ms、融合更新は1.7162→2.3184 ms。
Graphはそれぞれ0.148643→0.149058 ms、0.072091→0.072900 ms。
allocated peakは152064→154112 bytes、reservedは6291456 bytesで同じ。
新しいlive binding用Chart bufferを含む。eager増加を受け、宣言整合性確認の重複した
recursive buffer走査を減らした。数値法則・live buffer更新の検査は維持する。

このjobのsource archive SHA256は`9b438b5b1f4f4522cf8458bde8229d3878e61a011f895dc1d730f0e42ebc637f`、
verified result archive SHA256は`5087e942ff72fca6b83245ce45609f89b55eb8bdb12bff16660de22b449896cf`。
remote source cleanup・所有VM停止を確認し、raw logs/source/result/receiptを
ignored `output/atom-update-validation/final/`へ保存した。

## 最終runtimeの検証と計測

runtime候補`b93964f5a2eb7b61fc9a2d5254a5cae1e529158e`、job
`l4job-6c40cb596af849018020e27b290abc60`で再確認した。
NVIDIA L4 / GPU-ddc13250-725c-48c0-91d1-255a0dc1ec8c / driver 580.82.07 / 23034 MiB、
Python 3.13.15 / Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0。
CPU再検証は855 passed / 330 skipped / 18 warnings、30.57秒。
wheel/sdist build、Ruff F/I・format・whitespaceはPASS。
wheel SHA256: `8a111ad377cbb70c85b203e757c9d12a1ec6cc7a7ad90f94066f6688716db618`。
GitHub CPU validationは同じruntime commitのrun `37272271990`でSUCCESS。

GPUの対象10ファイルは259 passed、247.45秒。Atom更新、optimizer/moments、Polar融合、
local persistent layout、AlgorithmState、Selectors、CUDA dispatch、Registry、normalized Stripを再検証した。
先行390件と重複するので加算しない。既存の境界routing不一致は上記診断のまま残る。
normalized FULL/WINDOWの独立oracle・更新後支持とGraph、4本のlocal-product完成runと候補の
artifact検査は全てPASS。baselineは同じ`1f0b42e0`、比較条件は前節と同じ。
全CST runの初期Parameter SHA256は`9313fd297dc0f314528d6e6b288358a3404e0551a7695328abfe41ec25103907`。

| 方式 | eager ms baseline → candidate | Graph ms baseline → candidate | peak allocated bytes | peak reserved bytes |
| --- | --- | --- | --- | --- |
| CST / Torch更新 | 2.693062 → 3.128923 | 0.151126 → 0.151064 | 152064 → 154112 | 6291456 → 6291456 |
| CST / 融合更新 | 1.737954 → 2.222503 | 0.072733 → 0.073343 | 152064 → 154112 | 6291456 → 6291456 |
| dense / Torch更新run内 | 0.672917 → 0.690439 | 0.039234 → 0.039616 | 34179584 → 34179584 | 48234496 → 48234496 |
| dense / 融合更新run内 | 0.665990 → 0.681295 | 0.039579 → 0.039237 | 34179584 → 34179584 | 48234496 → 48234496 |

denseは普通のAdamWであり、Polar更新は行わない。allocated/reservedはcapture/replayを含む。
各方式1独立runの21同期sample中央値で、統計的同等性・速度改善は主張しない。
共通化後もeagerは約0.44/0.48 ms増えた。CPU側の入力・対応・宣言/State確認のコストを含む。
このfixtureでは融合更新のGraph完全stepはTorch更新より短いが、CUDA更新を既定採用しない。
N=1024/8192性能と他GPU、実PostgreSQL、GPU process全体のメモリ、性能shapeの全atom独立oracleは
本phaseでは未実施。CSTOptimizer自体のcapture対応を証明する測定でもない。

source archive SHA256: `11b5eaaa3c9d8bf51408713a189012b562e5f2bdb22fba45eb9dd1630d7c0343`。
verified result archive SHA256: `bca0fde6ab32086833d951f2576c36380647faa805a36030f7679003c6fe17be`。
baseline archive SHA256: `647c8af35dcc588ee066df8d436d4f6876f02fab115cabd2d472b2c4d648a52f`。
検証済みresultをignored `output/atom-update-validation/metadata-final/`へ保存した。
POOL_CLEANEDのackと全slotのstopped状態を確認した。各sourceの独立processを使うdriverは
ignored `output/atom-update-validation/driver-metadata.py`に保存する。

## 再現

```sh
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:. python -m tools.kernel_dev test --suite atom-update
PYTHONPATH=src:.:tests python -m pytest -q tests/test_atom_update_dispatch.py tests/test_cst_optimizer.py tests/test_local_polar_update.py tests/test_local_product_research.py tests/test_local_persistent_layout.py tests/test_algorithm_state.py tests/test_dispatch_selectors.py tests/test_cuda_dispatch.py tests/test_backend_registry.py tests/test_normalized_strip_public.py
PYTHONPATH=src:. python -m benchmarks.cuda.linear.check_normalized --output output/public-gates.json
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run --plans benchmarks/cuda/linear/plans-local-contraction.json --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json --polar-update torch --output output/local-torch.json
PYTHONPATH=src:. python -m benchmarks.cuda.linear.run --plans benchmarks/cuda/linear/plans-local-contraction.json --case benchmarks/cuda/linear/cases/local-contraction-64-middle.json --polar-update fused --output output/local-fused.json
```

計測driverはcatalogをbaseline Planだけに絞った複製で実行する。上の直接runnerコマンドは
元catalogの全candidateを含む。source_commitとfrozen source/result hashを合わせて確認する。

後続のCUDA境界routing修正と検証は
[Torus境界の記録](20261005-torus-routing-boundary.ja.md)を参照。
