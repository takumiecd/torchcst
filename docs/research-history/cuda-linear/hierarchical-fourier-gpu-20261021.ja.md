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
