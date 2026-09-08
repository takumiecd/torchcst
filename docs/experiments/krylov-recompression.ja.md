# 全項を残すKrylov更新とJVP→VJP再圧縮

2026-09-08。一次optimizerの数式とモーメントを保ち、求解方法を変更した。
二次optimizerや対角近似の既定値は変更していない。

## 数式とAPI

更新は従来と同じ問題を解く。

\[
\min_{\|d\|\le R}\ell^T d+\tfrac12d^T H d,
\qquad H=J^T D J/\eta.
\]

`update_solver="krylov"` は右辺から対称Krylov基底を作り、FP64で2回再直交化する。
列基底を \(Q_m\)、\(T_m=Q_m^T H Q_m\) として、
\((T_m+\sigma I)y=-Q_m^T\ell\)、\(d=Q_m y\) を解く。
半径・shiftの探索は小行列を使い回し、全Hに対するCGをshiftごとに再開しない。
直交基底なので元のEuclidean半径を保つ。GLTR型の部分空間法だが、有限精度の
3項漸化式だけに依存する三重対角実装ではなく、明示的な小さい射影行列を使う。
対象は現在のPSD一次問題であり、一般の不定値trust-region solverではない。
部分空間法の参考は[TRLIB](https://arxiv.org/abs/1611.04718)。

FP32へ丸めた実際の変位に全Hを新たに作用させ、残差・半径・相補性を検査する。
基底や小行列の品質検査も通らなければcommitを抑制し、診断を返す。
予算不足で密行列に切り替えたり、rtolを緩めたりはしない。

```python
optimizer = CSTAdam(
    model,
    device_execution=True,
    update_solver="krylov",
    update_basis_size=128,
    update_basis_memory_mb=64.0,
    update_check_interval=32,
    update_rtol=1e-5,
    recompression="pcg",
    recompression_action="jvp_vjp",
    recompression_max_iter=1024,
    first_moment_damping=0.01,
)
```

`update_basis_memory_mb` はQとHQの2配列に対する上限で、FP64なら16Nm bytes。
実際の基底容量はN、指定本数、メモリ制限から決める。factor、射影行列、直交化の一時領域、
CUDA Graph cache、forward/backwardは別途必要であり、process全体のメモリ上限ではない。
K1,024/N3,072では128本のQ/HQに6 MiB、K2,048/N6,144では12 MiB。
CUDAは固定スケジュールで、収束後の作用をmaskする。診断のiterations/evaluationsは
有効な作用数であり、起動したkernel数ではない。後続の小行列処理・起動コストは残る。

再圧縮は \((J_t^T J_t+\delta I)\alpha=b\)、履歴の輸送は
\(J_t^T J_{t-1}\alpha_{t-1}\) のまま。新しいactionは可視空間を16行ずつ通し、
source側のJVPの後、destination側のVJPを計算する。全atom間の結合を残す。
可視空間のscratchは最大16I、factor方向にはO(K(I+O))が必要。
従来のatom対の縮約と新経路は `recompression_action="pair"|"jvp_vjp"` で選択できる。
既定値は従来どおりspectral更新とpair作用。入出力幅によって有利な縮約は変わる。

## 規模試験

A100 80GB PCIe MIG 3g.40gb、PyTorch 2.6.0+cu126、TF32無効、host 1 thread。
Amplitude Gaussian Separable、784入力/10出力、FP32 parameters、seed17、batch32、MSE。
各方式は独立processでGPUを直列利用。同shapeで初期値・データをそろえた。
5stepの最初の2stepを除いた中央値。全体時間にはforward/backward/optimizerと完了待ちを含む。
更新時間にはfactor準備も含む。全stepで検査し、失敗後の抑制されたstepを時間に数えない。
メモリはPyTorch peak allocatedで、native libraryやdriverの全使用量ではない。

| atoms | 更新 / 再圧縮action | 全体 ms | 更新 ms | warm peak MiB | 検査合格 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1,024 | 旧PCG / pair | 16,370 | 10,980 | 246.8 | 5/5 |
| 1,024 | Krylov / pair | 5,579.53 | 86.62 | 181.25 | 5/5 |
| 1,024 | Krylov / JVP→VJP | 221.12 | 87.75 | 181.25 | 5/5 |
| 2,048 | Krylov / JVP→VJP | 414.31 | 145.58 | 297.18 | 5/5 |
| 85 | Krylov / JVP→VJP | 125.66 | 81.30 | 35.32 | 5/5 |

旧PCGは[以前の同条件試験](trust-pcg-scaling.ja.md)の値。
このK1,024の例では全体が約74倍速くなった。まずshift探索のやり直しを減らし、
次に再圧縮の二乗のatom対計算を除くことで短縮した。
K1,024/2,048では更新の有効基底は96本で足りた。
この短いMSE試験は大規模モデル全体の学習精度を示すものではない。
小規模にはspectralの方が速く、既定値を一律Krylovに切り替える根拠はない。

K1,024の初回更新は別processでFP64の全J/Hを組み立て、固有分解参照解と照合した。
Krylov+pairでは独立残差2.33343e-6、変位差norm6.2522e-9。
目的関数の絶対差は2.77e-13。丸め後の変位で要求1e-5を満たす。
JVP→VJP再圧縮を選んだ独立照合でも同じ初回更新の結果になった。

## 長めの学習と基底予算

前回のMNIST試験と同じ64 atoms/256 parameters、8,192 train/2,000 test、batch128、
lr0.05、betas0.9/0.99、radius0.25、damping0.01、128更新、seed17/29/43。
初期parameterとbatch順のhashを比較する。separable Dと全項の式を維持する。

基底128本ではseed17/29/43がそれぞれ101/110/100更新目に停止した。
再圧縮は全て合格し、更新側が128本を使い切って不合格になった。
不合格時には変位0を返しcommitを抑制するため、診断に出る残差1は返却変位の残差であって、
最後に試した射影候補の残差ではない。途中のaccuracyを最終精度として扱わない。


基底予算256本なら3seedとも128更新を通過し、実際に必要だった基底は最大160本だった。
同じ初期値・batch順のhashが一致することを確認し、更新と再圧縮の一方ずつを変える
追加試験も行った。すべて全項で、対角近似は使っていない。

| 更新 / 再圧縮 | seed17 | seed29 | seed43 | 平均accuracy | 全体ms* |
| --- | ---: | ---: | ---: | ---: | ---: |
| spectral / pair（従来） | 80.60% | 75.00% | 78.50% | 78.03% | 58.71 |
| spectral / JVP→VJP | 76.95% | 79.25% | 79.60% | 78.60% | 53.48 |
| Krylov / pair、基底256 | 78.80% | 77.50% | 76.05% | 77.45% | 224.28 |
| Krylov / JVP→VJP、基底256 | 79.80% | 69.85% | 78.10% | 75.92% | 222.90 |

*各runのwarm中央値を3seedで平均。新規9runはすべて128/128更新・再圧縮検査合格。
従来値は前回のfull試験。今回のKrylovはこの小規模では遅く、accuracyの同等性も
確認できなかった。特に両方変更したseed29の低下は無視できない。
一方だけ変えた場合にもseedごとの増減があり、この3seedだけで特定の演算経路が
恒常的に精度を上げる・下げるとは断定しない。初期値の違いではなく、演算順序、
FP32/FP64での作用、許容誤差による軌道差が候補だが、原因の特定まではしていない。

小規模の主経路はspectralのままとし、JVP→VJP再圧縮とKrylov更新は独立に選択する。
大きいatom数での速度改善と、長期学習でのaccuracyの確認を分ける必要がある。
Krylovの基底容量は実験に合わせた値であり、128本でいつでも解けるという契約ではない。

## 低rank前処理の実験

内部実験として、全Gの作用で \(Y=G\Omega\) を作るNyström前処理を追加した。
直交sketchは独立したseedで作り、学習の乱数列を変えない。FP64の安定化shift、
小さいCholeskyと固有分解から \(\widehat G=U\Lambda U^T\) を構成する。
前処理の逆作用は

\[
P^{-1}=I+U\left[(\lambda_{\min}+\delta)(\Lambda+\delta I)^{-1}-I\right]U^T.
\]

これは[Frangella–Tropp–UdellのNyström前処理](https://arxiv.org/abs/2110.02820)に基づく。
近似は前処理に限定し、PCGの作用と真の残差は全Gを使う。
optimizerの公開option・既定値には追加していない。
rank0は既存atom-block前処理、rank32/64はNyström。G作用、前処理の構築、PCGを同じ
CUDA Graph内で毎回実行し、構築込みの時間・peak memoryを測る。
初回compile/captureと再利用する直交sketchの生成はwarm測定外。
この比較はK1,024の初回再圧縮の同じ右辺を3回測るもので、学習全体の速度ではない。


| 前処理 | 有効PCG反復 | 構築込み中央値ms | peak MiB | 独立FP64残差 |
| --- | ---: | ---: | ---: | ---: |
| atom block | 254 | 116.25 | 49.62 | 7.76e-6 |
| Nyström rank32 | 40 | 94.77 | 51.08 | 3.47e-6 |
| Nyström rank64 | 6 | 106.63 | 52.60 | 5.75e-7 |

全試行でnative残差・独立残差が1e-5以下。rank32は構築込みで約18%短縮した。
rank64は反復数が少なくても全体はrank32より遅い。1,024反復分の固定スケジュールを
maskする起動コスト、小行列処理とsketch構築のコストが残る。
この結果だけでは学習全stepでの利益を確認していないため、内部の実験実装として保持する。
公開optimizerへの前処理選択APIや、自動rank選択は追加していない。
実験runnerのsketch予算は四つのFP64 N×r配列相当をチェックするが、factorのFP64化、
一時領域、graph cacheは別に必要で、上表のpeakも合わせて評価する。

## GPU転送と検証

K85のwarm transport、再圧縮、training stepをKinetoと別のscalar抽出監査で確認した。
全区間でH2D/D2H転送とPython scalar抽出は0。D2Dは再圧縮11回、training step334回。
training stepのkernel起動は19,262回あり、転送をなくすだけでは起動コストは消えない。
初期compile/captureや診断の取り出しは測定外。profiled時間を通常の速度比較に使わない。
CUDA runtimeにもhostのstream/device synchronizeはなく、GPUのstream間event待ちは残る。

最終検証: A100 **471 passed**、CPU **398 passed / 73 skipped**、`ruff check .`合格。
小行列の多shapeコンパイル上限はdynamic compileで解決し、Nyström構築の入れ子graphの
event依存は、親graph内に小行列の演算を取り込むことで解決した。
パラメータを変えたgraph replayと独立dense oracleを回帰テストに含める。

## 再現

```sh
PYTHONPATH=src:. python -m experiments.trust_pcg_scaling --atoms 1024 --solver krylov --recompression-action jvp_vjp --compression-max-iter 1024 --output output/krylov-k1024.json
PYTHONPATH=src:. python -m experiments.local_tangent_learning --data DATA/MNIST/raw --output output/mnist-krylov.json --approximation full --solver krylov --recompression-action jvp_vjp --basis-size 256 --seed 17
PYTHONPATH=src:. python -m experiments.recompression_nystrom --atoms 1024 --rank 32 --output output/nystrom-r32.json
PYTHONPATH=src:. python -m experiments.tangent_transfer_profile --atoms 85 --device-execution --update-solver krylov --recompression-action jvp_vjp --operations transport pcg training_step --output output/krylov-profile
```

測定値・各runのsource fingerprint・失敗条件は[集約JSON](krylov-recompression-results.json)に保存する。
