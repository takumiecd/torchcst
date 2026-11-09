# 順序固定M16包絡の検証と実行コスト

[English](connected-m16-envelope-lm-results-20261109.md) | **日本語**

別の順序固定physical/包絡経路がGPU83テストを通過した。raw写像のchainを含む長方形の
独立積分、混合半径で3反復のbit一致、従来方式の回帰検証を含む。
実幅512/2048・A8192の全写像oracleは初期とタスク976更新後に成功。
全4タスク条件で、元の厳密なmodel/optimizer直列化後の次更新とhook非干渉ゲートを通過。
許容差の緩和、公開dispatcher/既定値、deterministic/Graph対応の追加はない。
直前の非順序固定経路は別IDで保全している。

[短期LM研究](https://github.com/takumiecd/cst-research/pull/36)では半径の予測一致度が大幅改善し、
1seed・1M-tokenの精度差は残った。読み取り専用の信号分析と解釈はcst-researchに記録する。
この演算記録は精度面での採用の主張ではない。

## 同条件のLinear単体完全step

RTX4090 CC8.9、Torch2.9.1+cu128/CUDA12.8/Triton3.5.1、IEEE FP32、B256・N8192・A65536、
warmup5回・標本15回、独立1実行。既存Linear runner/physical fixtureによりforward、MSE、
dXを含むbackward、fused AdamW、半径/中心の復元を含む。
転置・geometry・再計算も時間内、コンパイル・証明は時間外。Denseは標準初期化の速度・メモリ対照。
eagerのみでGraphは使わない。

| Plan | 中央値 ms | allocated / reserved MiB |
|---|---:|---:|
| 直前の行連続配置 | 12.810783 | 525.502930 / 560 |
| direct階層の順序固定 | 12.922911 | 557.502930 / 592 |
| Dense | 4.908139 | 1072.251953 / 1342 |

この実行では順序固定版が約0.9%遅く、scratchが32MiB増えた（4*B*N個のFP32と一致）。
Denseより2.63倍遅く、allocated peakはDenseの52.0%。model・勾配・Adam状態・一時領域を含む
allocator peakで、プロセス全体使用量ではない。反復したspeedupの主張や他GPU/形状の検証ではない。

実規模A65536/B256で行連続配置とVJP比較し、Y/dX最大絶対誤差1.193e-7・相対L2<=5.377e-8、
物理dPの全4列はこの条件でbit一致。既存runnerの独立20Atoms境界/広い半径の積分と、
4step AdamW状態ゲートも成功。別のタスクcheckpoint監査では全8中間ファイルのhash、
4最終modelのParameter digest、有限FP32 moments、時計976（DST973）を照合。
checkpointはNFSに残し、生コピー/logはignoredに保全する。

## ソースと再現

タスク実行はoperator00aae6afaf3615da6bccd986c75c14cd7b2ca64b、benchmark登録は
752240cd94ab8f9cc8b95ff3d4c9cebc389cd8c8を使用。
両者のruntime treeは0fe7b7214529217e257df8c0369f31a2905dbe77で一致する。
Jobはconnected-m16-ordered-4090-20261109-01。
ソースarchive `da4490ecd41b2526241314c141d3198dabdfab7ae8ab4cf1e2191fa43873a5fd`、
結果 `6ec0c5e958cb6c1b9163cf8b49ae064ea1a9010320d807bb2d1f08bd827d97c1`。
記録されたsource hashの1370比較が固定archiveと一致。ローカル取得物とNFS証拠を両方保全した。
終了後GPUは0%・40MiB、Hubは起動維持。無関係なプロセス・データは変更しない。

```bash
python -m benchmarks.cuda.linear.run \
 --plans benchmarks/cuda/linear/plans-connected-radius-ordered.json \
 --case benchmarks/cuda/linear/cases/connected-radius-8192-r256-c8-ordered.json \
 --polar-update radius --source-commit 752240cd94ab8f9cc8b95ff3d4c9cebc389cd8c8 \
 --output output/ordered-step.json
```

[集計結果](connected-m16-envelope-lm-results-20261109.json)。
ローカルCPU runtime一式は2255成功・2654skip・警告19、その後のcatalog roundtripと宣言確認も別途成功。
CUDA skipはGPU証拠に数えない。Graph・高階微分・他GPU・公開既定値採用は未検証。
