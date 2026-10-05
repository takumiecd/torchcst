# Torus境界でのCUDA tile ownership修正

## 原因と契約

PR #29をGitHubでmergeし、main `a062af5cb7d9a2a9cb0e663ab01d3c8b8e21ecb1`をbaselineとする。
以前の`[257-64-4]`境界テストでは、中心`[-0.7730104923248291, 0.6343932747840881]`に対して
Torch atan2=2.454369306564331 / arc=62.5、Triton/libdevice atan2=2.45436954498291 /
arc=62.50000762939453だった。Torchはstation 11/12の距離を同じ2.375と計算し、11を選ぶ。
CUDAは12を選ぶ。FP fusionの有効/無効を変えても角度の差は残った。

診断job `l4job-2ef39af95004468189f4d18f1ac57393`で元テストの失敗とこれらの値を確認した。
source SHA256: `8d7bda6a3baa80d12b0b38c29ee40e93410f32a77747a3a2ffaeb2789612665b`。
verified result SHA256: `9237bf7e9898b73dcb675ce30385a79abb8f5c18ffb122c9090c90c9f5b14a74`。

CircleRouting.arcsでTorch参照のsigned arcを計算し、Torch ownersとCUDA ownersが共有する。
local / exhaustive / chunkedのCUDA kernelはそのarcを入力にして距離比較と振り分けを続ける。
数学的なowner契約と小さいstation IDへのtie breakを変えず、epsilonや許容誤差を導入しない。
追加scratchはatan2出力とarcのO(A)。全atom×stationのTensorは作らない。
arcは毎回計算し、Graph replayでも更新された中心を読む。幅・勾配・atom所有・moments、
Algorithm/Dispatcher/AtomState、recipe/semanticsは変更しない。別revisionの数学契約は導入しない。

元の境界テストを維持し、観測された中心とnextafter近傍、非連続入力、3種類のrouting、
中心を変えたGraph replayの比較を追加した。chunkedでは複数scanにまたがる同距離の候補も比較する。
期待ownerは実行環境のTorch参照から取得し、GPU/runtimeによるatan2の丸めの違いに対して
特定のtile IDを固定しない。

## 検証ソースと結果

- runtime candidate: `6fbd2cd68a212e8d9343969b6cabed1fb301a5eb`。
- validation job: `l4job-95ecac296213416693bd71c4f47290ed`。
- NVIDIA L4 / GPU-2a0c0f60-761d-f82e-5183-710488c4e2c7 / driver 580.82.07 / 23034 MiB。
- Python 3.13.15 / Torch 2.11.0+cu130 / CUDA 13.0 / Triton 3.6.0。
- baseline archive SHA256: `44fa17e2a391e146c63fc5db8a2392833177cf39ff486a9691e55847bbf4f69a`。
- submitted source archive SHA256: `3b437383d2d36fcc53bb5da409f079a296ec38e2e9276ccfa13cd561f10720f3`。
- verified result archive SHA256: `f757ecd36ee80e2869c9b3ddba4d1cdf1dae2f4f4b2181b71866d1073bc809bd`。
- driver SHA256: `040c08075ed50c3da15d9a81b79e380b0d00a742d544b98e489dcbff1e56c56e`。

CPUは855 passed / 333 skipped、31.35秒、既存TorchScript deprecation warning 18件。
変更ファイルのRuff F/I・format・whitespace、宣言検査、Hatchling 1.32.4のwheel/sdist buildはPASS。
wheel SHA256: `7ba2fd395b017172a6b0ef66df119632a71ec9019eb63a58c37043d482f00b84`。
同じruntime commitのGitHub CPU validationはrun `37275402931`でSUCCESS。

GPUのrouting gateは15 passed / 62 deselected、10.57秒。
続くtriton_linear / triton_schedule / tiled_linear / cst_optimizer / atom_update_dispatchの
5ファイルは167 passed、65.13秒、cuBLAS contextとprofilerのwarning 2件。
15件は167件に含むので加算しない。以前の失敗ケースも除外せず通過した。
Y/dX/全atom勾配、intrinsic/ambient、profiles・幅・空atom・seams、schedule、更新後支持、
Graph replay、host値の読み戻しがないwarm preparationを検証した。
最終commitでは研究ノートと、特定Torch/CUDA版に依存するliteral owner期待値の削除だけを追加する。
Torch参照とのowners/order/offsets完全一致のassertionは維持し、CUDA runtimeは同じ。

## 完全eager stepとメモリ

同じL4上でbaseline / candidate × CST / denseを別processで測定した。
N_out=64 / N_in=128 / B=32 / A=410（dense要素数の約5%）、station_rows=16、
Strip/Torus、Direct activity + radial Triweight、可変幅minimum=.2 / birth=.5 / maximum=.8、
FP32、TF32無効、seed=21、AdamW lr=1e-4 / weight_decay=.01 / fused=True / capturable=False。
CSTは公開CSTOptimizerを通して勾配射影・座標更新・moment輸送を含む。
各processで5 warmup、21同期wall-clock sampleの中央値。各方式1独立runであり、
21独立runや統計的同等性・速度改善の証拠とはしない。

| 完全eager step | ms baseline → candidate | peak allocated bytes | peak reserved bytes |
| --- | --- | --- | --- |
| CST | 9.011533 → 8.982372 | 163840 → 163840 | 2097152 → 2097152 |
| dense | 0.732280 → 0.740333 | 17229312 → 17229312 | 25165824 → 25165824 |

初期atom / input / targetのhashは全processで同じ。dense初期weightも同じ。
atom SHA256: `ef747dd07cdd8d496210c3736a9c8d2c083e08cd2e674fd0138cbdf16ce7c8b4`。
dense weight SHA256: `f58b86ced063e4f231ffd50c067a072ba72a3270209406e65e993cf0b7d561a9`。
denseは同じ初期CST weightを持つ普通のLinear + AdamWで、以降の最適化軌跡はCSTとは異なる。
peakはwarmup後の完全eager stepを対象とし、reservedはallocator予約量。
公開CSTOptimizerのGraph captureは未対応なので、完全Graph step時間/peakは測らない。
Graph対応routing/forward/backwardの正しさは上述のsuiteで確認した。
GPU process全体のメモリ、他GPU、実PostgreSQL、大きいN=1024/8192の性能は未測定。
このprobeは診断用raw JSONであり、既存Linear schema-v1提出やDB取り込み用結果ではない。

## 証拠と再現

共有poolで結果SHA検証・remote source cleanup・所有VM停止を確認した。全slotはstopped。
source/result/receipt/transportとdriverはignored `output/torus-routing-validation/diagnosis/`、
`final/`、`diagnose.py`、`validate.py`、`step-probe.py`へ保存した。
初期driverのtest import path修正とdense比較のprocess分離を行うため、queue上の2 jobを
実行前にcancelし、新しいfrozen driverとして明示再提出した。cancelしたjobはGPU未実行。

```sh
PYTHONPATH=src:. python -m pytest -q
PYTHONPATH=src:.:tests python -m pytest -q tests/test_triton_linear.py -k 'routing_boundary or fused_routing_matches'
PYTHONPATH=src:.:tests python -m pytest -q tests/test_triton_linear.py tests/test_triton_schedule.py tests/test_tiled_linear.py tests/test_cst_optimizer.py tests/test_atom_update_dispatch.py
PYTHONPATH=src:. python output/torus-routing-validation/step-probe.py --mode cst --output output/cst-step.json
PYTHONPATH=src:. python output/torus-routing-validation/step-probe.py --mode dense --output output/dense-step.json
```

sourceごとに独立processを使う。元のAtom更新整理の測定とはfixture・optimizerが異なるので
時間やメモリをそのまま横断比較しない。
