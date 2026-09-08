# 更新用Gramの対角・atom内block近似

2026-09-08。`CSTAdam`に`update_approximation="diagonal"`と`"atom_block"`を追加した。
既定値は`"full"`。**更新solver単体は速くなったが、未調整のMNIST精度は低下した。**
近似を選べるAPIとして残し、全Gramの既定経路は維持する。

## 変更する数式と変更しないmoment

同じseparable Dと線形項bに対し、更新だけを次の問題にする。

```
min_d bᵀd + 1/2 dᵀH_approx d,  ||d|| <= r
H = Jᵀ D J / lr
diagonal:   H_approx = diag(diag(H))
atom_block: H_approx = block_diag(H_aa)
```

対角版は`d_i = -b_i / (H_ii + σ)`、block版は各atom内の小さな固有系を使う。
σは**全atomで共通**。atomごとに別の半径でclipする実装ではない。
σ=0で特異なら、零固有値とその方向の右辺も判定する。零方向に力があれば正のσで
半径境界を解く。固定80回のscalar探索は残るが、更新用PCGは呼ばない。

再圧縮は従来の`(JᵀJ + λI)α = b_raw`のまま。旧frameからの輸送にも全atom間の項を残す。
対角化するのは更新用Hだけで、DのEMA・αの表現・再圧縮の連立方程式は変えない。
ただし更新後のparameterが変われば、次stepのJ、moment、再圧縮の反復数も変わり得る。

既存の`second_moment="atom_diag" / "atom_block"`とは異なる。
今回は`second_moment="separable"`を固定し、同じ観測量とEMAを使って比較する。

## APIと実装

```python
optimizer = CSTAdam(
    model,
    update_approximation="diagonal",  # "atom_block" / "full"
    update_solver="spectral",
    second_moment="separable",
    recompression="pcg",
    first_moment_damping=0.01,
    device_execution=True,
)
```

local近似はfactorized tangentとseparable momentが必要。`update_solver="pcg"`との
同時指定は拒否する。`update_max_iter`と`update_shift_steps`はPCG用なので影響しない。
`update_rtol`は近似後のHについて真の相対KKT残差と相補性を検査する。
全Hの最適解に近いことを保証する値ではない。

- 対角成分はkernel所有のfactor微分から直接縮約し、全Gramもatom内q×q行列も作らない。
- block版は各atomのq×qだけをFP64で計算する。CUDAはbatched Jacobiの固定sweepを
  CUDA Graphで再利用し、eigen残差と直交性をdevice上で検査する。奇数qも内部padで扱う。
- parameter dtypeへ丸めた後の変位で残差と半径を再検査する。失敗はeagerで例外、
  deferredでエラーラッチと更新抑制になる。近似を強めて黙って再試行する処理はない。
- atom配置の解釈はkernel側にある。solverはqやfactorのshapeを使用し、振幅の列番号を
  仮定しない。CPU/GPUの数値テストには振幅ゼロのatomも含む。
- `site_results.objective`は近似目的関数。`iterations=80`はscalar探索の固定予算を示す。

## MNISTの比較

A100 80GB PCIe **MIG 3g.40gb**、PyTorch 2.6.0+cu126、host 1 thread、TF32無効。
現行APIのAmplitudeBandwidthSeparable、K64・4 coordinates/atom、28×28入力chart、
10出力、factored backend。train 8,192 / test 2,000、batch128、128更新。
lr0.05、betas(0.9,0.99)、半径0.25、separable D、damping0.01。
全方式で再圧縮PCG最大1,024反復・rtol1e-5。local更新のrtolも1e-5。

seed17/29/43ごとに同じ初期parameter・batch列のhashを照合した。
各method/seedは独立process、GPU試験は直列。過去の別実験の数値をbaselineに流用していない。
**全9runが128stepとも数値検査を通った。** lrや半径を近似別に調整した試験ではない。

| 更新近似 | seed17 | seed29 | seed43 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| full | 80.60% | 75.00% | 78.50% | **78.03%** |
| diagonal | 68.75% | 68.80% | 72.45% | **70.00%** |
| atom_block | 70.80% | 68.95% | 69.65% | **69.80%** |

この条件では全atomの結合を捨てると約8 percentage points低下した。
block版が対角版より必ず良いという結果でもない。3 seedの未調整pilotであり、
近似版を調整した場合の上限精度や他タスクでの性能を断定するものではない。

時間は最初の2stepを除いた各runの中央値を取り、表ではその3 seed平均を示す。
学習stepはzero_grad/forward/backward/stepと完了待ちを含む。評価・logging・エラー確認は外。
更新区間はCUDA eventsで測り、Hの構築を含む。異なる学習軌道上の時間である。

| 更新近似 | 更新区間 | 学習step全体 |
| --- | ---: | ---: |
| full | 3.96 ms | 58.71 ms |
| diagonal | 1.65 ms | 77.58 ms |
| atom_block | 2.22 ms | 78.54 ms |

更新単体は短縮したが、**この学習条件ではstep全体は遅くなった**。
再圧縮の実反復数の中央値も、fullの194〜260程度に対し、対角274〜521、block288〜489へ
増えている。更新の近似によって次のframeが変わるため、再圧縮まで一定時間とは限らない。
再圧縮単体の時間はこのrunnerで独立には計測していない。

## K256での速度比較

別の固定乱数MSE workload。784入力・10出力、Amplitude Gaussian Separable、K256・768 parameters。
lr0.001、半径0.25、再圧縮PCG1,024反復・damping0.01。各方式5stepを進め、最初の2stepを
除いた3stepの中央値。全方式とも5step合格。MNISTとはモデル・データが異なる。

| 更新近似 | 更新区間 | 学習step全体 | peak allocated |
| --- | ---: | ---: | ---: |
| full | 12.44 ms | 273.30 ms | 68.42 MiB |
| diagonal | 0.75 ms | 223.23 ms | 68.42 MiB |
| atom_block | 1.36 ms | 230.75 ms | 68.49 MiB |

更新単体では対角が約17倍、blockが約9倍速い。このworkloadではstep全体も約16〜18%短縮した。
各方式で後続parameterが異なり、再圧縮反復数も異なるので、全体の差を更新solverだけの
効果とはしない。メモリはPyTorch allocatorのpeakで、process全体のGPU使用量ではない。
このKではforward/backward・再圧縮のpeakが残り、学習step全体のpeak削減は見えていない。

## 検証と再現

- 独立したautograd Jacobianから全Gramを作り、同じ対角・blockだけを残したdense参照と比較。
  これは近似問題を正しく解く検証であり、全Gramの解との一致を要求するテストではない。
- 全体で一つの半径、特異blockと零方向の右辺、ゼロ右辺、振幅ゼロ、変化するparameter/D/右辺、
  座標幅1/3/4/6、native FP32残差、checkpoint、非有限値での更新抑制を検証。
- 全Gram構築を禁止したまま2step実行し、αの再圧縮・輸送が全dense Gramの参照と一致することを確認。
- A100全suite **429 passed**。ローカル **373 passed, 56 skipped**。Ruff・構文・diff検査も通過。

K85の別のMSE workloadでCUPTI/Kinetoの転送監査も行った。warmupと計測終了のdrainを
除いた学習scopeでは、対角・blockとも **H2D 0、D2H 0、stream同期0**。
別passのPython scalar抽出監査も空で、最後の`check_errors()`が通過した。
GPU kernel数は対角2,822 / block2,929、D2Dは49 / 53回だった。
全traceのGPU kernel・memcpy数とscopeへの帰属件数の一致も確認した。
これはwarm実行の結果であり、初回compile/captureやloggingの転送をゼロとする主張ではない。

全実験のproduction codeのhashがローカルと一致し、初期parameterとbatch列のhashも
同一seedの3方式で一致することを確認済み。API・実験runner・結果をコミットして残した。

```sh
PYTHONPATH=src:. python -m experiments.local_tangent_learning --data /path/to/MNIST/raw --seed 17 --approximation diagonal --output output/mnist-17-diagonal.json
PYTHONPATH=src:. python -m experiments.trust_pcg_scaling --atoms 256 --solver spectral --approximation diagonal --compression-max-iter 1024 --output output/scaling-k256-diagonal.json
PYTHONPATH=src:. python -m experiments.tangent_transfer_profile --device-execution --update-approximation diagonal --atoms 85 --operations training_step --output output/profile-diagonal
```

- [集約結果・source fingerprint](local-tangent-results.json)
- [MNIST runner](../../experiments/local_tangent_learning.py)
- [生データ・trace・ログ](../../output/local-tangent-final.tgz)（ローカル保存、git管理外）
