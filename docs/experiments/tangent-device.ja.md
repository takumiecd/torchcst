# 一次optimizerのGPU実行

2026-09-08。`CSTAdam` に明示的な `device_execution=True` を追加した。
再圧縮をGPU制御のPCGにし、factorからGramを作用させる部分をTritonで融合した。
`direct` の既定値と更新式は変えていない。

## 数式と実行

再圧縮は引き続き `(JᵀJ + λI)α = b` を解く。λ > 0 を必要とし、Dはこの式に
入れない。atom内Gramだけを小さなCholesky前処理に使い、作用そのものでは全atom間の
項を計算する。対角近似による高速化ではない。

各atomの表現を `v uᵀ` とすると、方向xからまず `δu = du x`、`δv = dv x` を作る。
次にtarget座標ごとに、source atomにわたって次の4項を加算する。

```
(u_t · u_s)(dv_t · δv_s) + (u_t · δu_s)(dv_t · v_s)
+ (du_t · u_s)(v_t · δv_s) + (du_t · δu_s)(v_t · v_s)
```

この2段階を2カーネルで実行する。重み付き作用では入力・出力の内積にseparableな重みを
入れ、epsilonの無重み項も足す。parameterの解釈はkernel側のprepared factorが所有する。
optimizerはatomの座標レイアウトを解釈しない。

PCGのcurvature、残差、終了状態はGPU scalarで保持する。小さな前処理の逆作用は
FP64 Choleskyから準備し、反復ごとの `cholesky_solve` をなくした。
収束候補が出た時と返却直前に真の残差を検証する。CUDA Graphは固定の最大反復数を
replayし、収束後は重いGram計算をdevice条件でskipする。終了後の小さなkernel起動は残る。
Graphの入力にはparameter/factor/右辺を毎回渡すため、過去の値を固定して再利用しない。

更新方向の半径制約付き二次問題は、全weighted Gramの固有分解と、GPU上のspectral探索で
解く。cuSOLVERのinfoはGPUに残し、有限性、PSD性、固有分解の残差・直交性もGPUで検査する。
固有値と係数の `.cpu().tolist()`、解のGPUへの再アップロードは除去した。

`optimizer.check_errors()` が明示的な同期境界になる。異常時はparameterとmomentの
Tensor更新を抑制し、以後も失敗をラッチする。Python側のstep情報は進むため、正常な
checkpointを新しいoptimizerに復元して再開する。通常のeager実行は即時検査を維持する。

## 測定条件

前回の [転送調査](tangent-transfers.ja.md) と同じA100 MIG 3g.40gb、
PyTorch 2.6.0+cu126、Triton 3.2.0。784入力・10出力、Amplitude Gaussian Separable、
float32、1 atomあたり3 parameter、TF32無効、host 1 thread。
固定PCGはλ=0.01、rtol=1e-5、最大128反復。学習stepは最大256反復。
時間はprofilerなしの3回中央値であり、精密な統計評価ではない。
初期配置・コンパイル・warmup・計測終了時の同期・JSON用scalar読み戻しはsteady測定に含めない。
学習step後のラッチ確認も測定外で行い、正常な更新であることを確認する。


## 実測結果

| K | 処理 | 変更前 ms | GPU実行 ms | 比率 | kernel数 前→後 | D2H 前→後 | stream同期 前→後 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 85 | prepare | 0.881 | 0.921 | 0.96× | 66→68 | 1→0 | 1→0 |
| 85 | 輸送 | 2.052 | 0.268 | 7.65× | 145→2 | 0→0 | 0→0 |
| 85 | 重み付き作用 | 4.102 | 0.488 | 8.41× | 299→6 | 0→0 | 0→0 |
| 85 | PCG | 207.200 | 15.525 | 13.35× | 14,593→1,226 | 245→0 | 325→0 |
| 85 | 学習step | 334.970 | 38.408 | 8.72× | 24,781→2,914 | 421→1 | 555→1 |
| 256 | 輸送 | 11.640 | 1.308 | 8.90× | 947→2 | 0→0 | 0→0 |
| 256 | PCG | 1466.842 | 147.598 | 9.94× | 115,889→1,226 | 356→0 | 473→0 |

新実行のH2Dは全測定区間で0回。学習stepに残るD2Hは8 bytesで、
`torchcst::cusolver_eigh` に帰属するcuSOLVER内部の転送だった。Python scalar監査は
空で、PCGではライブラリ起因の反復ごとのmalloc/free・同期もなくなった。
cuSOLVER内部の1回は未除去であり、学習step全体が転送・同期ゼロになったとは言わない。

固定PCGの真の相対残差はK=85で8.78e-6（78反復）、K=256で8.16e-6（118反復）。
以前の80/117反復と異なるのは加算順序・前処理の数値実行が変わるためで、
同じrtol=1e-5を満たす。学習stepは132反復、相対残差9.70e-6で、
`check_errors()` も通過した。これは短い計算・更新の検証であり、長期学習精度の比較ではない。

## 初回コストとメモリ

| K | 処理 | このprocessでの初回 ms | warmup前 allocated MiB | warmup後 allocated MiB | 通常実行 peak allocated MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 85 | PCG | 2462.8 | 2.20 | 11.36 | 11.36 |
| 85 | 学習step | 15377.9 | 11.36 | 36.78 | 45.63 |
| 256 | PCG | 7114.9 | 6.36 | 17.60 | 17.61 |

数値はPyTorch allocatorのallocatedであり、process全体のGPU使用量・reserved量ではない。
同一processで順に測るためK=85学習stepには前に作った単体PCGのcacheも含む。
初回時間にはdisk compile cacheの状態が影響し、完全なcold build時間とは限らない。
Graphはparameter/factor/右辺の静的bufferと反復workspaceを保持する。
これを無視して「一時メモリがほぼゼロ」とは数えない。

輸送・PCGは全Gramを保持しないが、更新方向のsolverは依然として
全weighted Gramと固有分解を使う。全optimizerの二乗メモリ問題は残っている。
次の性能課題は、この更新solverと、Gram作用のGPU内での読み込み・演算の効率化。
Kが増えてもkernel起動数は増えなくなったが、atom対の演算量自体は二乗で増える。

## 検証と再現

- A100上の全テスト: **371 passed**。
- ローカルCPU: **329 passed, 42 skipped**。Ruff、diffの空白検査とも通過。
- 追加テストは、独立したdense autograd Jacobianとの作用・PCG解の一致、
  FP32/FP64、parameterと右辺の変更後のGraph replay、ゼロ右辺、null方向を含む
  半径制約、eagerとの2 step一致、NaN後の更新抑制・エラーラッチを検証する。
- wheel buildでnative wrapperの `.cpp` 同梱を確認した。

```sh
pip install -e '.[cuda]'
PYTHONPATH=src python experiments/tangent_transfer_profile.py --device-execution --atoms 85 --operations prepare transport weighted pcg training_step --output output/final-k85
PYTHONPATH=src python experiments/tangent_transfer_profile.py --device-execution --atoms 256 --operations transport pcg --output output/final-k256
pytest -q
```

初回にCUDA開発環境を使ってC++拡張をbuildする。上記の実行ではCUDAとC++の開発環境を
備えたLinuxを使った。ユーザー側のloggingやcheckpoint保存、データローダーの転送は
今回の測定範囲に含まない。

- [集計JSON・source fingerprint・trace SHA-256](tangent-device-results.json)
- [Chrome traceアーカイブ](../../output/tangent-device-traces.tar)（ローカル保存、git管理外）
- [測定スクリプト](../../experiments/tangent_transfer_profile.py)

K=85/256のsource fingerprintが一致し、ローカルの実装・計測スクリプトとも一致することを
確認した。アーカイブと個別gzip traceのSHA-256をJSONに保存した。
