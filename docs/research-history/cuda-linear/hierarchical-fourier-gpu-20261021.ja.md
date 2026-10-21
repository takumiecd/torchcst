# 中心支持FourierのZ→H→C実装 2026-10-21

branch: `kernel/hierarchical-fourier-tiles`。
shapeと階層の基礎は[CPU確認](hierarchical-fourier-range10-20261021.ja.md)。
この実装は研究用Algorithmであり、公開CSTLinear dispatcherの採用ではない。

## 数学と今回の実行契約

FourierPeakSpecの形は追加・変更しない。M4はq下限0.70、T基底0.05、L_min=0.0125、
M8はq下限0.30、T基底0.2、L_min=0.05を使う。
周期・階層のquintic混合を含め、各成分は中心から±Lの外でゼロ。
whole_tilesによる別関数への置換はしない。

生の入力profileをv、出力profileをuとすると

\[
 D_a=\max(\|u_a\|_2\|v_a\|_2,10^{-6}),\quad
 W_{ji}=\sum_a A_a u_a(j)v_a(i)/D_a,
\]
\[
 Z_{b,t,r}=\sum_{i\in t}X_{bi}\phi_{t,r}(i),\quad
 H_{ba}=\sum_iX_{bi}v_a(i),\quad
 C_{b,s,r}=\sum_a(A_a/D_a)H_{ba}c^u_{a,s,r},\quad
 Y_{bj}=\sum_r C_{b,s,r}\psi_{s,r}(j)+Y^{direct}_{bj}.
\]

完全に支持内にあるタイルだけをZ/Cで共有する。両端の部分区間と小Iは同じFを直接評価する。
境界の直接処理を省いた方式ではない。複数周期は別bankで保持し、出力site担当が合計する。
Cは各bank/tileを一つの担当が合計し、残るatomicは直接評価のY/dX書き込みである。

Pの4座標は`(amplitude, output_center, input_center, width_logit)`。
`I=.1+9.9*sigmoid(width_logit)`をatomごとに独立に学習する。
これは既存Polarのtask幅stop-gradientやactivity更新とは別の研究契約であり、
同じ更新則の速度改善だとは扱わない。両比較候補は同じ直接座標AdamWを使う。
Iを旧sigmaへ換算した実効幅の保証は導入しない。

L2は全成分を合計したFの実site二乗和をFP64で計算し、全productで一度だけfloorを適用する。
連続積分への近似置換は使わない。中心・I・ノルム・混合重みの微分を含む。
幅・中心・振幅は毎forward/replayの現在値から再計算する。

## 保存と寿命

- 固定: 有限候補のperiod bank、site基底、bank/node metadata。X依存値やlive係数を含まない。
- 呼び出し全体: 入力Zと出力C。backwardで入力Zと出力勾配Zを再生成する。
- atom chunk内: 係数・係数I微分・phase係数・ノルム・H/Gとそれぞれの微分。
- autograd保存: XとP。Tensor versionを検査し、更新後Pで古いforwardを再計算しない。

既定chunk=4096、比較用1024/8192も宣言できる。
全atom×全siteのfactorやWを作らず、chunkの最後の使用後に作業領域を再利用する。
Hをbackwardまで全保存する候補はまだ実装していない。

この初版は候補bankのZを毎回構築する。CPUの容量admission・需要bankのみの生成・
階層間carryをすべて移植したものではない。小Iだけの共有Planにも固定bank費用が残る。
最小保存32/64/128点は実行設定で、形状選択のT基底とは独立。
入力/出力profileは軸別に縮約し、2次元タイルのjoint 289成分を保存しない。

## 検証と測定の入口

```bash
python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier.json \
  --case benchmarks/cuda/linear/cases/hierarchical-fourier-m8-mixed-256-a512.json
python -m tools.kernel_dev test --suite hierarchical-fourier
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier.json \
  --case benchmarks/cuda/linear/cases/hierarchical-fourier-m8-mixed-256-a512.json \
  --source-commit <実際のcommit> --output output/hierarchical-fourier/result.json
```

既存Linear runnerへ専用fixtureを接続した。
CPU oracleは独立した192点Gauss積分、全KKT solve、独立した階層選択、全site・全atomを使う。
Y/dX/振幅/両中心/幅logitの全勾配、3動的状態、retained backward、
capture後の入力・幅・中心変更を検査する。許容値は従来の`atol=rtol=4e-4`。
完全stepは係数・並べ替え・Z/C構築・L2・forward/backward・AdamWを含み、
同じ入力・batch・更新率のDenseを別processで測る。allocatedとreservedを区別する。
この新契約のDB adapterは未追加なので、既存adapterへの混入を明示的に拒否する。

## 実行履歴

- L4初回`l4job-c8da9c73e97540be8bfcddae13b12583`: 接続切れで結果未回収。
  owned VMの停止を確認し、性能・正しさの証拠から除外した。
- Kyutech: 稼働済みRTX PRO 6000でコンパイルのみ試し、Triton3.5.1のlayout passエラーで終了。
  ユーザーの2026-10-21の指示で以後使用しない。サーバー停止・設定変更は行っていない。
- Colab接続確認`l4job-1ffdc670561e4f9da75f28a9526c1023`: PASS、回収検証済み。
- L4 smoke `l4job-25fb571d31e84dee9ab934034ba3c4f5`: M4/8×direct/shared×2状態、8照合PASS。
  Torch FP64参照に対する最大誤差Y 2.4e-7、dX 2.3e-7、全P 3.9e-6。
  これは16 atomのsmokeであり、独立oracle・大規模完全stepの代替ではない。
- 初回runner `l4job-70ff21f28505436c879e02d6fc6cafd5`: retained eager graphのstream所有が
  captureへ残りFAIL。保存graphを解放してからcapture streamを準備するよう修正。

L4: Torch2.11.0+cu130、CUDA13.0、Triton3.6.0。
raw source/results/receiptsはhost-wide poolの各job directory、ローカル整理先は
`output/hierarchical-fourier-gpu-20261021/`。未回収・未完了の結果から速度を主張しない。

## 最初の完全stepと切り分け（改良前）

`l4job-c7604cacbae24596b309940f7d9b417c`、source `1cf37f71`。
N=256、A=512、B=8、IEEE FP32、独立process、7 replay標本の中央値。
M4/8×small/mixed/large×direct/sharedについて、全atomの独立oracleを3状態と
動的Graph replayの4回ずつ、計48照合PASS。最大誤差は全Pを含め約5.3e-6。
GPU専用pytestも5件PASS。初期fixtureの「空／floor」候補は常にその状態になる保証が
不足していたため、後続sourceでは専用atomを幾何的に保証しCPUでも検査している。

| M | I分布 | 直接ms | 共有ms | Dense ms |
| --- | --- | ---: | ---: | ---: |
| 4 | small | 3.637 | 3.742 | 0.0522 |
| 4 | mixed | 3.893 | 4.041 | 0.0522 |
| 4 | large | 4.994 | 5.287 | 0.0522 |
| 8 | small | 5.659 | 5.792 | 0.0522 |
| 8 | mixed | 6.312 | 6.162 | 0.0522 |
| 8 | large | 8.284 | 6.272 | 0.0522 |

M8 largeでは共有で約24%短縮したが、Denseの約120倍であり速度目標を満たさない。
全ての条件で512 atomの幅logitが更新された。時間は準備・sort・L2・AdamW込み。
測定は同じbatchのreplayであり、Xの変更は別のcorrectness gateで確認した。

capture/replay込みの総peak allocated/reserved:

| M / route | allocated B | reserved B |
| --- | ---: | ---: |
| M4 direct | 724,992 | 6,291,456 |
| M4 shared32 | 906,240 | 6,291,456 |
| M8 direct | 1,118,720 | 6,291,456 |
| M8 shared64 | 1,384,448 | 6,291,456 |
| Dense | 35,162,624 | 48,234,496 |

Denseにはライブラリ作業領域も含む。256²のWだけの容量ではない。
この条件での総peakの比較であり、5% atom密度・N1024以上への一般化ではない。

別job `l4job-56a2baa67c1847858aac51f0c31e907a`で未計装の主測定とは分けてprofileした。
M8 small/directのkernel時間合計約5.60msに対し、contract 2.777ms、
prepare 1.800ms、norms 0.570ms、deposit 0.342ms、pack 0.070ms。
M8 large/sharedではcontract 2.578ms、prepare 1.804ms、norms 1.072ms、
aggregate 0.312ms、deposit 0.289ms。合計を完全step時間の代わりに使わない。
prepareは固定projectionと解析的係数の評価であり、学習中のKKT solveではない。

## 小支持の評価再利用候補

既存B-spline `native_cache_kernels.py`の、少数siteのprofileを呼び出し内だけ保持する
考え方を利用した。L2でFP64評価したF・中心微分・I微分をFP32で保存し、
H/GとY/dXへ使う。混合成分を合計済みのFを一度だけ注入する。
M4で最大16点、M8で最大32点、共有bankがない軸だけに適用し、条件外は従来の正確な経路へ戻す。
余分な容量は最大`chunk*2axes*3values*capacity*4` byte、全atom・全siteの表ではない。
ノルムとその微分は引き続きFP64、全productにfloor一度。
同時に、重みがゼロのperiod成分の解析的係数評価を省く。
この変更は座標の固定・幅bankへの丸め・別profileへの置換を行わない。

`l4job-9505059e5d444813b6dbce49c178674c`、source `d24555ad`。
GPU pytest9件と、M4/8×small/mixed×3 Plan×4状態の48独立oracle照合がPASS。
最大誤差は全Pを含め7.4e-6以下。保証した空／非空floor atomも含む。
同一run内のN256/A512/B8完全step:

| M | I分布 | 直接・非cache ms | 直接・cache ms | 共有＋cache ms |
| --- | --- | ---: | ---: | ---: |
| 4 | small | 3.828 | **1.570** | 1.683 |
| 4 | mixed | 4.067 | **3.072** | 3.231 |
| 8 | small | 5.059 | **1.705** | 1.855 |
| 8 | mixed | 5.682 | 4.218 | **4.070** |

Denseは同じ0.0522ms。smallの直接cacheでもDense比M4約30倍、M8約33倍であり、
速度目標を満たしたとはしない。係数のactive成分化だけで全条件が速くなったわけでもない。
小Iでは共有bankを作る費用がまだ残るため、直接cacheが最速だった。
M8 mixedは共有の効果が残る。現時点で単一Planの普遍的な勝ちを主張しない。

直接cacheのallocated peakはM4 929,792B、M8 1,520,128B。
共有cacheはM4 1,111,040B、M8 1,785,856B、reservedはいずれも6,291,456B。
比較Denseのallocated 35,162,624B / reserved48,234,496Bを下回るが、
FP32 native cacheを追加した分は非cacheより多い。速度と容量の交換を明記する。

CPU suiteはこの時点で2,233 passed /5,660 skipped /18 warnings。
GPU skipをGPU検証には数えない。リポジトリ全体のGPU suiteを完了したとの意味ではない。

## KKTを使う場所と連続係数の評価

生のFourier係数をc(q)、q=I/Lとし、中心値1・端点値0と所定の偶数階端点微分0を
線形制約Ab=hとする。連続L2の係数metricをGとすると、制約付き射影は

\[
 \min_b\tfrac12(b-c)^T G(b-c),\qquad Ab=h,
\]
\[
 b(q)=P c(q)+d,\quad
 P=\mathrm{Id}-G^{-1}A^T(AG^{-1}A^T)^{-1}A,\quad
 d=G^{-1}A^T(AG^{-1}A^T)^{-1}h.
\]

ここでKKTはこの固定射影を導出するために使う。A/G/hはMだけで決まり、P/dは初回に
作ってcacheする。学習中のatom/stepごとにはKKTを解かず、現在のqに対する係数を評価する。
独立oracleでは別実装のKKT solveで照合するが、これは測定workerとは別processであり
完全step時間には入らない。L2正規化はこの係数射影とは別の処理である。

最後の候補`table_coefficients=True`はb(q)をChebyshev基底で表し、Clenshawで
値とq微分を連続評価する。空間profile自体はFourierのままで、qの丸めや固定中心を導入しない。
M4は24項（最高次数23）、q範囲[.69,1]、M8は32項（最高次数31）、[.29,1]。
実使用のq下限.70/.30を覆う。10001点CPU照合では値の差約2–4e-15、
微分の差約2–3e-12。これは係数評価の数値近似であり、数学的な厳密等価とは呼ばない。
GPUでの許容誤差は変更しない。既定値はFalseのまま、明示的な研究Planとして残す。

## 5% atom密度での拡大試験と最終判定

N=1024、A=52,428、B=32、spacing=.1、全atomのIがstepごとに更新される条件。
`l4job-791bc55c9f2e4036b5607a4118e4ce7d`では、M4 small/M8 mixedの
直接cache・共有cacheについて全atom独立oracleの16状態がPASS。
直接cacheのGraph完全stepは193.054/1220.084ms、共有cacheは198.252/1265.757ms。
Denseは0.079872ms。小さいNの結果から良好な拡大性能を推定することはできなかった。

最終比較`l4job-a80d412aa53a4825aaa3640081e19949`、数値source `a493e68a`。
各Planの3動的状態＋Graph変更の全atom照合、計24状態がPASS。
Y/dX/全Pをまとめた最大絶対誤差は3.20e-5以下、最大相対L2誤差1.55e-6以下。
同じrunの7標本中央値（準備・sort・L2・更新込み）:

| 条件 / Plan | Graph ms | eager ms | allocated peak B | reserved peak B |
| --- | ---: | ---: | ---: | ---: |
| M4 small / direct-cached | 193.106 | 193.864 | 19,929,600 | 35,651,584 |
| M4 small / direct-table | **158.534** | 159.268 | 19,929,600 | 35,651,584 |
| M4 small / shared32-table | 163.599 | 164.799 | 21,816,320 | 37,748,736 |
| M8 mixed / direct-cached | 1220.343 | 1221.210 | 31,597,056 | 56,623,104 |
| M8 mixed / direct-table | **1186.344** | 1187.233 | 31,597,056 | 56,623,104 |
| M8 mixed / shared64-table | 1231.828 | 1233.147 | 33,806,848 | 58,720,256 |
| Dense | **0.079872** | 0.695–0.716 | 51,382,784 | 111,149,056 |

係数評価の変更は直接cacheに対しM4約17.9%、M8約2.8%短縮したが、
最良値でもDense比約1,985倍／14,853倍。速度目標「Denseの2倍以内」は明確に未達。
allocated/reservedはこの単層独立processでDense以下だが、Denseのライブラリ作業領域を
複数層で共有するネットワーク全体の省メモリを保証しない。process全体のdevice使用量は未測定。
単一独立run内の比較であり、複数runの安定性や、全I分布・全サイズの普遍的優位ではない。

速度面は研究試作としても不十分で、公開CSTLinearへの採用・最速dispatcher登録は行わない。
この測定だけでFourier/H方式そのものの限界が証明されたわけでもない。
現実装にはatom×batchごとのprofile評価、候補bankの一括準備、chunkごとの再構築が残る。
上記profileは診断用で、DRAM帯域・register spillをhardware counterで特定した結果ではない。
特にCPUで検討した需要に基づく階層carryを全部実装した結果とは区別する。

最終source `defab975`では数値runtimeを変えず、片軸のみfloor近傍のGPU fixtureを追加。
`l4job-baf39f7f1fe940a5a15bc9783afdcab7`でGPU専用13件PASS（76.20秒）。
CPU suiteは2,235 passed /5,664 skipped /18 warnings、wheel/sdist buildもPASS。
Colabの全owned slotは停止確認済み。Kyutechにはユーザー指示以降アクセスしていない。

## 回収と同一性

ローカルの`output/hierarchical-fourier-gpu-20261021/pool/`にsource/result archive、
spec、receipt、stdout/stderr、個別JSONを保存し、sourceとresultのarchive SHA256を照合した。
`final-result-summary.json`に全Planの値、`artifact-sha256.json`に回収ファイルのhashを保存。
最終性能jobと現在のruntime/benchmark/Planのbyte一致、最終QA jobと現在のGPU testの一致も確認。
raw evidenceはignored研究出力で、Gitに大量の生成物を取り込まない。

| job | source archive SHA256 | result archive SHA256 |
| --- | --- | --- |
| density / 791bc55c | af0b48707ef11207aaa4af6b8e4fa27ddbd260e5be9cfccf0a1655356aab5a5a | 3dade074a2b049636f92406b5ac7aabb71b2d70e2d3d9d283df8a38b29b0fdac |
| coefficient / a80d412a | 6a199a514fd199c5ab2c926e65a8b559a14926c9e8d41dca362da9d95a1489cd | 51846664e4dd283811cb452ab0199e0080f04a450be38e9e645b5660bd4dc53b |
| final QA / baf39f7f | f0a8fca0d1f77e33fa264b1deded362d905bb682babe6630248054168d064c12 | a209ce86ab72e0dc026866f430f6a28685b949b2d32090816aed12add7bed032 |
