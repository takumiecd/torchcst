# Tangent作用・PCG・学習stepの転送と同期の実測

2026-09-08。対象実装は `8d51ecc`。A100 80GB PCIe **MIG 3g.40gb**、
PyTorch 2.6.0+cu126。今回は計測・発生元の確認であり、optimizerの更新式は変更していない。

**準備済みのGram/cross-Gram/weighted作用にH2D・D2Hは見つからなかった。
読み戻しと同期は、主にPCGのPython制御、前処理のライブラリ呼び出し、
更新solverのCPU処理にある。** また、転送がなくても多数のkernel起動が残る。

## 方法と範囲

`experiments/tangent_transfer_profile.py`で、PyTorch profiler/KinetoのCPU/CUDA
イベントを取得した。Nsight Systemsの実行ファイルは今回の環境にはなかった。
CUPTIによるCUDA kernel/memcpy/runtimeイベントは実際に取得できている。

- 前回と同じ784入力・10出力、両chartは1次元linspace、1 atomは3 parameter。
  Amplitude + Gaussian Separable、float32、tile=32、TF32無効、host 1 thread。
- 固定作用のPCGはlambda=0.01、rtol=1e-5、最大128反復。全学習stepは同じlambdaと
  rtolで最大256反復。前処理は既存のatom-local Cholesky。
- 初期化、GPUへの入力配置、warmup、checkpoint/JSON出力は計測scope外。
  学習入力・targetはGPUに常駐する32サンプル。データローダーの転送は対象外。
- `training_step`はzero_grad、forward、backward、optimizer.stepを含む。
  他のscopeは準備済み状態からの作用・再圧縮単体。
- 計測終了時の明示的な`cuda.synchronize()`は別scopeに置き、表から除外。
  GPUイベントはCPUのlaunch/copy correlationで帰属させるため、CPU scope終了後に
  完了したGPU処理も取りこぼさない。
- Python発生箇所は別実行のTorchDispatchModeで監査。これはGPUトレースや時間計測には
  入れていない。学習stepは実行ごとに進むため、profile側133反復・監査側136反復であり、
  表の回数はすべて**profile側**。固定PCGは両側80反復で一致。
- profilerは時間を増やすため、通常実行の3回中央値も別途記録した。トレース中のAPI時間、
  GPU busy時間、通常実行時間を混ぜて性能寄与率を算出しない。

## 実測回数

bytesはCUPTIのmemcpyイベントに記録された合計。D2Dは別集計であり、この表には含めない。
「stream同期」はscope内の`cudaStreamSynchronize`呼び出し回数。

| K | 処理 | H2D 回数 / bytes | D2H 回数 / bytes | stream同期 | GPU kernel数 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 85 | prepare | 0 / 0 | 1 / 1 | 1 | 66 |
| 85 | 輸送 cross-Gram作用 | 0 / 0 | 0 / 0 | 0 | 145 |
| 85 | weighted Gram作用 | 0 / 0 | 0 / 0 | 0 | 299 |
| 85 | direct再圧縮 | 0 / 0 | 19 / 67 | 19 | 825 |
| 85 | PCG再圧縮・80反復 | 0 / 0 | 245 / 252 | 325 | 14,593 |
| 85 | 学習step全体 | 1 / 2,040 | 421 / 4,520 | 555 | 24,781 |
| 256 | 輸送 cross-Gram作用 | 0 / 0 | 0 / 0 | 0 | 947 |
| 256 | PCG再圧縮・117反復 | 0 / 0 | 356 / 363 | 473 | 115,889 |

prepareではsnapshotのGPU内コピーも5回・1,080,860 bytes発生している。
weighted作用にはmetricのrow/columnのGPU内コピーが2回・3,176 bytesある。
これらはH2D/D2Hではない。

## 発生元

### 1. PCGの毎反復のPython判定

[`tangent_solve.py`](../../src/torchcst/_derivatives/tangent_solve.py)の
`if not torch.isfinite(curvature) or curvature <= 0`は正常系でもGPU scalarを2回読み戻し、
残差の`if ... <= rtol * norm`がさらに1回読み戻す。

K=85では毎反復3回×80＝240回。それに入口の有限値/ゼロ/Cholesky検査4回と、
最後の真の残差の`float(...)`1回を加えて245回。内訳は1-byte boolが244回、
8-byte floatが1回。すべて`aten::_local_scalar_dense`へ帰属した。
K=256でも3×117+5＝356回で一致する。

転送量は小さいが、各読み戻しに依存したhost分岐がある。`double()`などの
GPU上でのdtype変換をH2D/D2Hと数えてはいない。

### 2. 前処理のcholesky_solve内部

同じファイルの`precondition()`が呼ぶ`torch.cholesky_solve`から、
`aten::_cholesky_solve_helper`に帰属する追加の同期・割当が確認できた。

| K | 前処理のstream同期 | cudaMalloc | cudaFree |
| ---: | ---: | ---: | ---: |
| 85 | 80 | 160 | 160 |
| 256 | 117 | 234 | 234 |

したがってK=85の325同期は、scalar読み戻し245回＋前処理80回。
K=256の473同期も356＋117で説明できる。Pythonの`.item()`相当だけを
取り除いても、前処理内部の同期が残る。この帰属はCUDA runtimeイベントの
External idから得ており、前処理そのものが遅さの主因と断定するものではない。

### 3. 更新方向solverのCPU往復

[`optim/quadratic.py`](../../src/torchcst/optim/quadratic.py)の
`torch.stack((values.flatten(), spectral.flatten())).cpu().tolist()`が、
固有値と係数をCPUへ移す。255 parameter×2系列×float64＝**4,080 bytesのD2H**。
CPUのPythonリストで半径制約のscalar探索を行った後、
`values.new_tensor(interior)`で解の係数255個をGPUへ戻す。
これは**2,040 bytesのH2D**として記録された。

既存の全Gramを使う更新solverの経路であり、今回作ったGram作用とは別の発生元。
これに加えて固有分解内部の8-byte D2H、有限値・stationarity等のscalar検査がある。

### 4. prepare、cache照合、direct参照解法

prepareの`torch.isfinite(p).all()`のPython判定で1-byte D2H。
geometryの`torch.equal`によるcache照合、momentと更新の検査も学習stepで読み戻す。
shape/device/dtypeなどCPU側metadataの検査とは区別する。

direct再圧縮にも19回のD2Hがあり、そのうち15回の4-byte転送は
`aten::_linalg_svd`に帰属した。参照解法も転送・同期ゼロではない。

## 次に直す対象

H2D/D2Hを計算ループから除くには、次の処理境界を変更する必要がある。

1. **PCGの制御をGPUに置く。** curvature、残差、有限性・breakdownの状態をdevice上で保持し、
   検査結果を毎反復Python boolへ変換しない。最終的な真の残差検証と、失敗時に不正な
   parameter/momentをcommitしない契約を残す。結果確認をどのhost境界で行うかも明示する。
2. **atom-local前処理をまとめる。** 準備したfactorを使い回し、各反復のライブラリ内部の
   同期・一時割当を避ける実行を検証する。今回のq=3では小さな独立求解が対象。
3. **更新solverのscalar探索をGPUに残す。** 固有値・係数を`.cpu().tolist()`にせず、
   spectral solutionまでdevice上で作る。これはGramのmatrix-free化とは別に扱える。
4. **cacheと検査をdevice常駐の実行に合わせる。** 既知のstep/current/previousの関係を
   使い、tensor値比較に頼ったcache判定を減らす。検証を単に削除しない。

速度についてはこれらに加え、**Gram作用の融合が必要**。K=256の輸送はH2D/D2Hが
0回でも947 kernelあり、PCG全体では115,889 kernelが記録された。
転送だけを除いて高速化が完了するという結果ではない。
K=85 PCGのトレースではscalar読み戻しに帰属するstream同期のAPI時間は約1.59 ms、
前処理側は約0.99 msだった。これだけで通常実行約207 msの遅さを説明することはできない。
プロファイラのオーバーヘッドとAPI/GPU時間の重なりもあるため、削減後の通常実行で
効果を再測定する。

## 成果物・再現

- [計測スクリプト](../../experiments/tangent_transfer_profile.py)
- [集計と発生元・source fingerprint・trace SHA-256](tangent-transfers-results.json)
- [元のChrome trace群のアーカイブ](../../output/transfers-traces.tar)（約19.5 MB、ローカル保存、git管理外）
- 個別traceは`output/transfers-traces/transfers-k85/*.json.gz`と
  `output/transfers-traces/transfers-k256/*.json.gz`。gzip展開後にtimeline viewerで読める。

元のtraceは変更していない。capture後、同じtraceを再解析してruntimeのCPU-op帰属を追加した。
JSONにはcapture時のsource fingerprintと再解析時のscript SHA-256を別々に保存している。
元アーカイブと全個別traceのSHA-256、K=85/256間のcapture fingerprint一致を検証済み。

```sh
PYTHONPATH=src python -m experiments.tangent_transfer_profile --atoms 85 --output output/transfers-k85
PYTHONPATH=src python -m experiments.tangent_transfer_profile --atoms 256 --operations transport pcg --output output/transfers-k256
pytest tests/test_tangent_transfer_profile.py -q
```

集計器のテスト3件通過。計測境界の同期の除外、遅れて完了するGPUイベントの帰属、
CUDA traceが取れなかった場合に「転送0」と誤判定しないこと、busy時間の重複除去を検証。
Ruff通過。
