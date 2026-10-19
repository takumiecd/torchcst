# 係数とHの保持量を減らす比較

目的はB32/B64混合σ条件の完全学習step時間をDenseの2倍以内、capture/replay込みのallocated/reserved両peakをDense以下にすること。未達時は目標を達成扱いにしない。

v17は同じforward状態のMeta（FP64）から `_jump` を呼び、FP64の係数を利用直前に生成する。係数全体とCutsのsnapshotは保持しない。support・正規化・Polar・task幅stop-gradient・全atom勾配契約は変更しない。H/Hμ/G/GμのFP32丸め境界も維持する。

Hは非保存・Hのみ・H/Hμ保存を比較し、係数保存かつH非保存も対照にする。非保存のbackward H再計算にも明示的FP64 fmaを使う。v14/v16等の既存Planの意味は保つ。配置は各forwardで検証し、古いHや古いσをstep間で使い回さない。

独立FP64 Y/dX/全atom勾配のmaxabs/relative L2閾値はともに4e-4。まず回帰と混合B32/B64の既存runnerを実行する。現時点で速度・メモリ改善は未検証。

正規化の追加候補 `norm_fma` は、区間内のquartic sampled normとその中心微分の積和を明示的FP64 fmaへ書き換える。積分近似やdtype低下は行わない。通常版は従来の演算順序を維持し、既存recipeにこの設定を暗黙適用しない。診断のnorm replayも選択した同じ算術を使う。

## 係数再生成とH寿命の初回結果

job `l4job-2fdf8239d48b4b38b35569b1b663ceab`、source `3060bb20`。L4 / Torch 2.11.0+cu130 / Triton 3.6.0、TF32 off。同一初期P/X/target、N1024/A52428、σ=.1/.4/1.6/6.4/12、atom/stepで可変。完全stepの未計装Graph replay中央値、21サンプル、独立runは1件。構築・sampled L2・全勾配・fused Polar更新を含む。

| 方法 | B32 ms | B32 allocated / reserved MiB | B64 ms | B64 allocated / reserved MiB |
|---|---:|---:|---:|---:|
| 係数/H/Hμ保存 fma | 1.617621 | 81.288 / 172 | 2.495226 | 96.791 / 188 |
| 3-plane係数/H/Hμ保存 fma | 1.654250 | 75.688 / 158 | 2.540011 | 91.388 / 196 |
| 係数再生成、H非保存 | 3.254185 | 51.688 / 118 | 5.603754 | 55.696 / 114 |
| 係数再生成、Hのみ保存 | 3.087970 | 58.088 / 142 | 4.625459 | 68.496 / 140 |
| 係数再生成、H/Hμ保存 | 2.910342 | 64.488 / 140 | 4.400782 | 79.794 / 168 |
| 係数保存、H非保存 | 1.819975 | 68.488 / 134 | 2.913453 | 72.693 / 132 |
| Dense | .082954 | 49.002 / 106 | .106239 | 49.502 / 106 |

全再生成はallocatedを大きく削減する一方、最速対照より大幅に遅く、Denseのメモリ以下にも到達していない。正の採用結果とはしない。係数保存/H非保存もメモリとの交換に留まる。

別の診断Graphでは、B64のstream forward/backwardは対照 .814/.957ms → 全非保存1.727/3.196ms。register数は対照96/96 →128/255（いずれもstatic spill slots 0）。B32でも .453/.507ms →.958/1.692msに増加した。係数snapshotを廃止して省いたprepareは約.06msであり、利用側の費用増の方が大きい。register数から実occupancyや物理DRAM転送を確定しない。診断時間を足して完全stepの代用にしない。

独立full-shape FP64 gateの最大誤差Y 1.217888e-6 / dX 1.562121e-6 / 全atom勾配3.255087e-5、relative L2 ≤1.404e-7。全6 CST候補で52,428 atomのσ更新を確認。query経路と追加lean候補のGPU回帰132 passed /647 deselected（130 CUDA +2 metadata）。この最大誤差はrunnerの3状態の集計であり、pytest全fixtureの最大値ではない。

source snapshot SHA256 `d36de4936d30c24ce72472356868c15454d94f8c992a4c37bb8a5d257b63634f`、result archive SHA256 `7f1957f39374453e0ef2afd8dc09ddd8153c10dcbb46eb77c35228b75eaa712e`。archive/manifest48件、runtime/test/benchmark1491ファイルのcommit一致と明示的remote cleanup acknowledgementを検証した。原本と検証scriptは `output/bspline-lean-20261020/` に保持する。

## 正規化FMAの結果

`l4job-c305d49f027b4d6cb9a11b6b69253c4b`、source `57f67574`。通常norm →明示的FP64 fma normで、B32 prepare診断は約.162→.152ms、B64も約.162→.153ms。完全stepは全非保存B32 3.250416→3.241170ms、B64 5.608794→5.596981ms、H/Hμ保存B32 2.907646→2.900254ms、B64 4.403427→4.395914ms。peakは不変。完全step差は約.2–.3%で独立runは1件のため、有意な改善として採用判断しない。

全recordがPASSし、回帰142 passed /647 deselected（140 CUDA +2 metadata）。今回の2job全体のrunner最大誤差はY 1.217888e-6 / dX 1.562121e-6 / 全atom勾配3.660886e-5、relative L2 ≤1.404e-7。各CST候補の全52,428 atomでσ更新を確認。CPUは最終runtime sourceに対して1977 passed /3555 skipped /18既存warning。Ruff・宣言検査・wheel/sdist内348 runtimeファイルのbyte一致も確認した。

source SHA256 `477528c40cb2d4d3ceeb5ff8d4162fe2d0ce5cac01f0a968efc0b82a82a7efca`、result SHA256 `6c81762e233dfbc6c928a2dc01b481483142299e74e3e59e851523ddf01d4904`。manifest42件とsource1493ファイルのcommit一致、returncode0 / timeoutなし / cleanup acknowledgementを確認した。

## 割当・寿命の診断

`l4job-ce81e30c40e44111bc5bc086ac14727f` は既存runnerの `measure` を独立childごとに呼び、B64でmemory historyを有効にした。計装された時間は採用判断から除外する。primaryのpeak取得直後にallocator snapshotを採り、通常の計測と同じallocated/reserved peak値であることも確認した。snapshotはその時点の現在値であり、peakそのものではない。

全CST候補で常駐8.125MiBのblockが4つ、計32.5MiB。Python stackは2つが `saved_executor.forward` の `x @ w.T`、2つがbackwardの `dy @ w` を指す。小σ側のW/GEMM経路を残す費用として次の実装で評価する。これは割当元の計測であり、実DRAM転送量やL2 hit率の計測ではない。

| B64 snapshotのpool | fma full係数 | fma compact係数 |
|---|---:|---:|
| default pool reserved / allocated MiB | 50 / 36.885 | 90 / 36.885 |
| Graph pool reserved / allocated MiB | 68 / 1.050 | 78 / 1.050 |
| 現在の総reserved MiB | 118 | 168 |
| 計測したreserved peak MiB | 188 | 196 |

同じactive payloadでも空きblockの配置・保持するsegmentが異なることを確認した。fullでは8.125MiB blockを2つずつ20MiB segmentへ配置する一方、compactでは各blockが別の20MiB segmentへ配置され、他の空き領域も残る。係数payload減少だけではreservedが下がらない。現在snapshotの差を、履歴上のpeak差8MiBと一対一には対応させない。

source SHA256 `db3eab2ac50c2639f0ccbb351fba5d4867e35c3f6ec3488e70e2c2132a7cd6ca`、result SHA256 `cf30aa2b0c191e5d8f029662b09c4191c7f784034f47207a627e8ac80bed59c1`。manifest22件、source1493ファイルのcommit一致、正常終了とcleanupを検証した。全owned L4の停止も確認する。

今回のfrozen runnerには、no-cacheでも従来の係数snapshot再利用という説明文が残っていた。bool `coefficient_cache=false` と実ソース・payloadが実際の処理を示す。後続runnerではこの説明を修正するが、過去のJSONは改変しない。

## 次の実装

目標は引き続きDenseの2倍以内、allocated/reservedともDense以下。現候補は全て未達で、公開dispatcherへ採用しない。次は、小σ用のW構築・GEMM・dWを、軸別直接X→HとH→Y、および対応するG/dX/atom pullbackへ置き換える。大σ側の最速FMA queryは残す。小σも含めたCST全体からW/GEMMを除くことで、上記32.5MiBの常駐割当とW/dW保持を不要にできるか、完全stepと独立gateで確認する。H保存の量もB32/B64別に比較する。これは次の未実装候補で、まだ性能・精度の結果ではない。

再現は共有L4 poolから、`python -m benchmarks.cuda.linear.run --plans benchmarks/cuda/linear/plans-quadratic-bspline.json --case benchmarks/cuda/linear/cases/bspline-1024-a52428-mixed-lean-normfma.json --polar-update fused --phase-diagnostics --kernel-diagnostics --source-commit 57f67574 --output output/lean.json`。B64は対応する `b64-mixed` Caseを使う。GPU回帰は `pytest tests/test_quadratic_bspline_cuda.py -k 'query_ or lean_recipe' -q`。
