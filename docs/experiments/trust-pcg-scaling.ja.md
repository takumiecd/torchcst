# 一次optimizerの規模拡大試験

2026-09-08。85 atomsでの結果を拡張し、256 / 1,024 / 2,048 atomsと、
1,024入力・256出力の単一層をA100で試した。**大規模モデル全体の学習精度試験ではない。**
atom数とvisibleな重みサイズを別々に変え、求解・時間・メモリの限界を調べた。

全Gramをなくすメモリ上の効果は大きいatom数で現れる。しかし、現実装がそのまま
大規模学習に使えるという結果にはならなかった。再圧縮の反復予算不足、更新solverの
反復予算不足、forward/backwardのmaterializationを分けて扱う必要がある。

## 条件と測定方法

- A100 80GB PCIe **MIG 3g.40gb**、可視メモリ40,192 MiB。
  PyTorch 2.6.0+cu126、host 1 thread、TF32無効、FP32 parameters。
- Amplitude + Gaussian Separable、1次元のinput/output chart、1 atomあたり3 parameters。
  atomの配置・振幅は現行APIの初期化に従う。seed17、batch32、固定の乱数入力・MSE目標。
  同じshapeのsolver比較は同じ初期parameterとデータ。更新後の軌道は各solverで異なる。
- 共通: `device_execution=True`、`recompression="pcg"`、damping=0.01、
  lr=0.001、radius=0.25、再圧縮・更新PCGのrtol=1e-5。
  基本予算は再圧縮256、更新512反復/round・外側32 round。
  追加試験は再圧縮1,024、更新外側64 round。許容誤差は緩めない。
- shape/solverごとに独立process、GPU試験は直列。通常は5step、最初の2stepを
  初期captureと初回history transportのwarmupとして除外し、残り3stepの中央値。
  各step直後、計測外で `check_errors()` とsolverの`converged`を検査し、失敗したら停止。
  失敗後の抑制された更新を高速なstepとして数えない。
- 学習stepはzero_grad/forward/backward/stepと完了待ちを含む。
  更新区間はCUDA eventsで`_solve`全体（Gramまたはfactorの準備も含む）を測る。
  profilerは使わず、この試験ではH2D/D2Hの回数を再監査していない。
- peak allocated / reservedはPyTorch allocatorの値。CUDA Graphの常駐bufferを含むが、
  driverやnative libraryなどprocess全体のGPUメモリを表す値ではない。
  cold試行にはcompile/captureが含まれるため、通常stepの時間と比較しない。

## atom数を増やした結果

入出力は784 × 10に固定。`backend="auto"`は全条件でmaterializedを選ぶ。

| atoms / parameters | 基本予算・spectral更新 | 基本予算・PCG更新 |
| --- | --- | --- |
| 256 / 768 | 5step検査合格 | 5step目の更新が外側32 roundで不合格 |
| 1,024 / 3,072 | 初回の再圧縮が不合格 | 初回の再圧縮が不合格。更新求解自体は合格 |
| 2,048 / 6,144 | 初回の再圧縮と更新診断が不合格 | 初回の再圧縮が不合格。更新求解自体は合格 |

初回の再圧縮の真の相対残差はK1,024で4.79e-5、K2,048で4.12e-4。
どちらも256反復を使い切り、要求した1e-5を満たさない。
初回の更新の右辺は観測したgradientから作られるため、後で失敗する再圧縮の近似解を
使った右辺ではない。PCG更新の合格と、optimizer全体の不合格は両立する。

K2,048の初回試行のピークはspectral **1,919.9 MiB**、PCG **456.6 MiB**。
PCG側は約76%少ない。peak reservedはそれぞれ3,024 / 852 MiB。
これは**両方とも学習stepとしては不合格だった初回試行のメモリ**であり、
定常学習の速度・メモリ優位性や、長期学習の成功を意味しない。

予算を増やすとK256は両方式とも5step合格した。学習stepの中央値は
spectral 273 ms、PCG 3,067 ms、peak allocatedは68.4 / 98.7 MiB。
この規模ではPCGの常駐factor・graphのコストがまだ大きく、速度も約11倍遅い。
PCGは5step目に外側37 roundを使ったため、基本予算32での失敗と整合する。

K1,024でも追加予算のPCGは5stepとも合格した。再圧縮は263〜305反復、更新は
外側23〜40 roundを使った。通常3stepの学習時間中央値は**16.37秒**、更新区間は
**10.98秒**、peak allocated **246.8 MiB**、peak reserved 406 MiB。
計測stepごとに全体から更新区間を引くと残りは約5.39〜5.60秒であり、更新solver以外にも
大きな時間がかかっている。これは再圧縮だけを独立に測った時間ではない。

同じ追加予算のspectralは再圧縮を通過したが、4step目の更新診断が不合格だった
（projected gradient norm 1.55e-5）。したがって両方式の成功した5stepの速度比は出さない。
K2,048での追加予算試験は実施していない。

## 入出力を広げた結果とbackend

K256、1,024入力・256出力。visibleな重みは262,144要素だが、trainable parametersは768。
autoのbackend判定は`K * (I + O) <= I * O`ならfactored、それ以外はmaterialized。
この条件では327,680 > 262,144なのでmaterializedになる。

materialized経路は各atomの`[K, O, I]`を作ってから足す。
この配列1つだけでFP32の256 MiBとなり、backwardの中間値も増える。
「dense weightの要素数がfactorより少ない」という判定だけでは、このpeakを捉えられない。

auto + spectralは5step合格、中央値314 ms、peak allocated 844.5 MiB。
auto + PCGは4step目に外側予算不足で停止した。合格した3step目の1回だけでも
学習stepは39.65秒、更新区間は39.37秒、peak allocated 881.4 MiBだった。
これは不合格になる軌道の途中の単発測定であり、成功した5stepの中央値ではない。

同じ初期値で`backend="factored"`を明示し、3stepだけ再実行した。両方式とも3step合格。
比較可能な3step目の値は以下。**warm測定1回で、長期安定性や中央値の統計的な比較ではない。**

| solver | backend | 3step目の学習時間 | 3step目のpeak allocated |
| --- | --- | ---: | ---: |
| spectral | auto → materialized | 0.310 s | 844.5 MiB |
| spectral | factored | 0.312 s | 71.1 MiB |
| PCG | auto → materialized | 39.648 s | 881.4 MiB |
| PCG | factored | 38.985 s | 99.9 MiB |

atomのdense展開を避ける効果は大きい。一方、更新solverの遅さはほぼ残る。
backend間ではFP32の演算順序が変わり、後続のCG反復数やparameter値に小さな違いがある。

## 実装上の意味

1. **更新用の全Gramを消す方針にはメモリ上の意味がある。**
   固定q/I/Oでfactor・vectorの保存量はKに線形。一方、現状のstreamed更新作用は
   `O(K q I O)`で、出力幅を増やすと1作用あたりのworkも増える。
2. **再圧縮は別に改善が必要。** 現在のCUDA再圧縮はatom対の縮約で、全Gramを
   保存しないものの作用の演算量はKに二乗で増える。ここにもJVP→VJPとcross-frameの
   JVP→VJPを使う候補を比較したい。速度と反復数、FP32での真の残差を別々に検証する。
3. **更新solverは反復数・前処理・graphの固定起動数が課題。** 外側予算を増やすと
   解ける条件はあるが、速くはならない。atom内blockは全atomの結合を近似せずに
   前処理として使う現設計を保ち、より強い前処理とshift探索のwork削減を検討する。
4. **forward/backwardのbackendも全体のメモリに効く。** solverだけの省メモリ化では
   atomごとのdense materializationを消せない。auto選択のcost modelも改善候補になる。

追加予算のspectralでは`converged=False`をrunner側で不合格として止めたケースもある。
現行optimizerのspectral経路は、この診断値だけではエラーラッチを立てず、有限性・半径等を
満たせばcommitし得る。したがって表の「検査合格step数」は実際のcommit数とは限らない。
再圧縮またはPCG更新のdeferred検査に失敗した場合の更新抑制とは区別する。
両solverの収束判定の定義も同一ではない。全stepを独立dense oracleで照合したわけではない。

## K1,024での独立dense照合

速度・メモリ試験とは別processで、追加予算のPCGの初回更新を検証した。
同じparameterとmomentからFP64の全Jacobian、`Jᵀ D J / lr`を別計算で作り、
固有分解参照解と比較した。ここでの参照は通常のspectral経路のFP32 Gramとは異なる。

- dense作用で測った真の相対残差: **2.71171e-6**（要求1e-5以下）。
- 目的関数: PCG -0.00104949660、参照 -0.00105015145。
- 目的関数の相対差: **0.06236%**。相補性の相対誤差: 3.22729e-6。
- 変位差のnorm: 0.0009209。PCG変位のnorm: 0.2491989、半径0.25。
- 再圧縮を含むstepの`check_errors()`も合格。

この1stepの照合は更新solverの大きいatom数での数値確認であり、長期の学習精度を保証しない。

## 再現

```sh
PYTHONPATH=src:. python -m experiments.trust_pcg_scaling --atoms 1024 --solver pcg --output output/k1024-pcg.json
PYTHONPATH=src:. python -m experiments.trust_pcg_scaling --atoms 1024 --solver pcg --compression-max-iter 1024 --update-shift-steps 64 --output output/extended-k1024-pcg.json
PYTHONPATH=src:. python -m experiments.trust_pcg_scaling --atoms 256 --input 1024 --output-width 256 --solver pcg --backend factored --steps 3 --output output/factored-wide-k256-pcg.json
PYTHONPATH=src:. python -m experiments.trust_pcg_oracle --atoms 1024 --steps 1 --compression-max-iter 1024 --update-shift-steps 64
```

- [測定runner](../../experiments/trust_pcg_scaling.py)
- [集約JSON・source fingerprint](trust-pcg-scaling-results.json)
- [先行する85 atomsの実験と数式](trust-pcg.ja.md)
- [raw JSON・ログ・実行スクリプト](../../output/scaling-final.tgz)（ローカル保存、git管理外）

今回の変更は実験runnerと記録のみ。optimizerの既定値・数式・実装は変更していない。

14条件の最終結果と独立dense監査を保存し、全測定の実装fingerprintがローカルと一致することを確認した。
計測runnerはbackend選択引数追加前後の2版を使い、各結果にそのhashを記録した。
Ruff、Python構文検査、diffの空白検査が通過。production codeの変更はないため、
この実験追加で全ライブラリテストの再実行はしていない。
