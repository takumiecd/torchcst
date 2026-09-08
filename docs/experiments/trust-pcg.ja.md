# 一次optimizerのmatrix-free更新solver

`CSTAdam(update_solver="pcg")` を追加した。既定値は `"spectral"` のまま。
再圧縮も `recompression="pcg"` にすると、separable metricの学習経路で
全parameter-space Gramと固有分解を使わない。

## 解く問題と合格条件

目的関数は従来どおり、`bᵀd + 1/2 dᵀHd`、`||d|| ≤ r`、
`H = Jᵀ D J / lr`。再圧縮の `(JᵀJ + damping I)α = b_raw` とは別の求解である。

更新は `(H + σI)d = -b` を解く。σは半径制約の乗数で、第一momentのdampingとは独立。
全atom間の項を保持し、atom内blockは前処理だけに使う。

- σ=0はゼロ初期値の非前処理CG。特異・整合系で、厳密演算ではEuclidean最小norm解を
  保つ。振幅ゼロの方向にも対応し、半径を超えた場合は正のσの探索へ進む。
- σ>0はatom-block前処理付きPCG。半径探索は `[0, ||b||/r]` から始める。
- 真の線形残差eから `||d - d_exact|| ≤ ||e||/σ` が得られる。このnorm区間が
  半径の内側・外側を確定できる場合にだけ、探索区間を狭める。
- 同じσで予算を使い切った場合は、CGの残差と探索方向を引き継ぐ。
  線形求解が収束済みでも半径を判定できない場合は、内側の許容誤差を厳しくして再開する。
- 内部のfactor・求解はFP64。返却するparameter dtypeへの変換後に、改めてHを作用させて
  検査する。丸め後の候補が不合格なら内側の精度を上げて再試行する。
  CGの更新式から得た残差だけでは合格にしない。

合格には次の両方と半径条件を要求する。b=0はd=0として扱う。

```
|| (H + σI)d + b || / ||b|| ≤ update_rtol
σ |r - ||d|| | / ||b|| ≤ update_rtol
```

予算不足・数値異常は、eagerでは例外、device executionではエラーラッチと更新抑制になる。
全Gramのsolverへの暗黙のfallbackはしない。`optimizer.check_errors()` が明示的な同期境界。

悪条件・ほぼ零空間の方向では、目的関数と最適性残差が近くてもparameter変位が大きく
異なり得る。`update_rtol` は変位の相対誤差の保証ではない。長期学習精度の保証でもない。

## GPU実行とメモリ

最初の実装では、既存のatom対を縮約する重み付きGram作用を反復ごとに使った。
85 atomsの学習stepで約9.5秒になったため、更新solver用の作用を次の流れに変更した。

1. kernel所有のfirst factor derivativesから、方向に沿ったδu、δvを作る。
2. 最大16出力行ずつ `Jd` を作り、同じkernelでDと`1/lr`を作用させる。
3. その行ブロックをVJPでparameter空間へ戻し、加算する。

全Gram、全Jacobian、固有ベクトル、Krylov基底は保存しない。
一時的なvisible空間の値は最大16行。factorと前処理のblockを含むworkspaceは、
chartサイズ・1 atomの座標数q・反復予算が固定ならKに対して線形である。
作用の演算量は `O(K q inputs outputs)`。CPU参照経路は既存のatom対の縮約を使う。

zero/positive shiftで再利用可能なCUDA Graphをそれぞれ持つ。changing inputとして
parameter、factor、Dのrow/column、右辺、σ、内側の許容誤差、CG継続状態を渡す。
収束・区間判定はGPUに残る。ただし固定予算のgraph nodeは終了後も起動し、
host側は外側の固定回数をscheduleする。この起動コストと常駐bufferは残る。

## API

```python
optimizer = CSTAdam(
    model,
    device_execution=True,
    recompression="pcg",
    first_moment_damping=1e-3,
    update_solver="pcg",
    update_max_iter=512,
    update_shift_steps=32,
    update_rtol=1e-5,
)
```

`update_max_iter` は探索1 roundあたりのCG予算。全CG反復数はこれより多くなる。
`update_shift_steps` は同じσでの継続・精度調整も含む外側の予算。
両者を増やすとworkとgraphの起動コストが増える。単に「PCG 512反復」とは解釈しない。

`last_step.site_results` にσ、真の相対残差、相補性の誤差、実際のCG反復数、
外側round数を返す。`evaluations` はdevice条件でskipされたものを含むH作用のschedule数。
診断値の `.item()` や表示は同期になるので、logging境界で行う。

## A100での実測

2026-09-08。A100 80GB PCIe MIG 3g.40gb、PyTorch 2.6.0+cu126、Triton 3.2.0。
784入力・10出力・85 atoms（255 parameter）、Amplitude Gaussian Separable、float32、
TF32無効、host 1 thread。再圧縮は両方式ともPCG、damping=0.01、最大256反復。
更新PCGは512反復/round、外側32 round、rtol=1e-5。

同じ初期seed・入力生成手順から、それぞれ独立して学習stepを進めた。
zero_grad、forward、backward、optimizer.stepを含む。時間はprofilerなし3回の中央値。
各方式で変位が少し異なるため、後続stepのparameter値は完全には同じではない。

| 項目 | spectral更新 | PCG更新 |
| --- | ---: | ---: |
| 学習step | 38.88 ms | 787.42 ms |
| H2D | 0回 | 0回 |
| D2H | 1回・8 bytes | 0回 |
| stream同期 | 1回 | 0回 |
| GPU kernel数 | 2,916 | 244,340 |
| warmup後 allocated | 27.62 MiB | 48.06 MiB |
| 通常実行 peak allocated | 36.47 MiB | 56.91 MiB |
| process内の初回 | 3.48 s | 9.63 s |

**この小規模条件では、PCG版は約20倍遅く、実測メモリも多い。既定値にはしない。**
全Gramを消した効果より、追加のFP64 factor・2つのCG graph・workspaceの常駐コストが大きい。
parameter数に対して二乗の配列をなくすことと、小規模問題のpeakを減らすことは別である。
大きいKでの実用的なメモリ・速度の優位性は、この測定からは主張しない。

新方式のprofile対象stepは、更新CG 10,171反復・外側25 round。
真の相対残差2.43e-6、相補性誤差3.33e-6で合格した。再圧縮もrtol=1e-5を満たし、
学習step後の `check_errors()` が通った。残るGPU kernel数は大きく、shiftごとの求解と
固定予算のinactive nodeの起動が、次に減らす対象になる。

warmup、初回compile/capture、計測終了時の明示的な同期、診断値の読み戻しはsteady測定外。
初回時間はdisk cacheの状態を含むため、完全なcold buildの測定ではない。
メモリはPyTorch allocatorのallocatedであり、process全体・reserved量ではない。
D2DコピーはH2D/D2Hと区別してJSONに残した。新方式でもD2Dは878回・約75.9 MBある。
GPU timelineの全kernel・全memcpyと、計測scopeに帰属したイベント数の一致を確認した。

## 独立した精度監査

別の入力系列で3step進め、各stepの同じparameter・momentから、明示的なFP64 Jacobianと
全Gramを別計算で作って固有分解solverと比較した。このdense処理は監査スクリプトにだけ
存在し、速度・メモリ測定には含めない。監査では外側予算64 round、rtol=1e-5を指定した。

| step | 参照との目的関数の相対差 | dense作用での相対残差 | 変位の差のnorm |
| ---: | ---: | ---: | ---: |
| 1 | 1.54106e-09% | 7.18e-06 | 7.08e-06 |
| 2 | 0.0532645% | 2.77e-06 | 0.0017 |
| 3 | 0.116274% | 2.44e-06 | 0.0067 |

全stepが設定したKKT許容誤差を満たした。目的関数の相対差は最大約0.116%、変位の差の
normは最大約0.0067（半径0.25）。「変位が参照と完全一致する」とはしていない。
この短い検証だけで長期学習精度が同じとは言えない。

監査中には、FP64候補が合格してもFP32へ丸めると残差が許容値を超えるケースがあった。
その場合に精度を上げて再試行し、実際に返すnative dtypeの値を検査する処理を追加した。
また、半径付近で内側の許容誤差が粗く、区間が判定できず止まるケースも回帰テスト化した。
許容誤差を緩めたり、不合格の更新をcommitしたりする解決にはしていない。

## 検証・再現

- A100全体: **401 passed**。
- ローカルCPU: **355 passed, 46 skipped**。Ruffとdiffの空白検査も通過。
- 零空間・振幅ゼロ・半径内外・ゼロ右辺・FP64参照・GPU replayのparameter/D/右辺変更、
  出力行/featureの端数、NaNと予算不足での更新抑制、checkpoint復元を検証した。
- 全Gram経路を呼ぶと失敗するテストの下で、optimizerの2stepを実行した。

```sh
PYTHONPATH=src python experiments/tangent_transfer_profile.py --device-execution --update-solver spectral --atoms 85 --operations training_step --output output/spectral-k85
PYTHONPATH=src python experiments/tangent_transfer_profile.py --device-execution --update-solver pcg --atoms 85 --operations training_step --output output/final-stream-k85
PYTHONPATH=src python experiments/trust_pcg_oracle.py --atoms 85 --steps 3 --update-shift-steps 64
pytest -q
```

- [結果JSON・source fingerprint・trace SHA-256](trust-pcg-results.json)
- [raw traceと検証ログ](../../output/trust-pcg-final-traces.tar)（ローカル保存、git管理外）
- [転送計測](../../experiments/tangent_transfer_profile.py)
- [独立したdense監査](../../experiments/trust_pcg_oracle.py)

両方式のcapture fingerprintとローカルの実装・計測スクリプトが一致することを確認済み。
監査スクリプトのSHA-256も結果JSONに記録した。
