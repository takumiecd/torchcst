# 階層Fourierの幅・共有数による実行経路の比較 2026-10-21

研究branch: `kernel/hierarchical-fourier-tiles`。
開始source `d1add4eb`。[前回の測定](hierarchical-fourier-reuse-20261021.ja.md)を引き継ぐ。
公開dispatcherの採用ではない。ユーザーの依頼で探索を一旦終了し、階層タイルを次sessionの軸として保持する。
再開入口は [引継ぎ](hierarchical-fourier-handoff-20261021.ja.md)。数値実装checkpointは `ecb6987c`。

## 数式と実行経路

学習幅は従来どおり `I=.1+9.9 sigmoid(s)`、中心・幅・振幅はatomごとに独立な連続Parameter。
ここでのIはGaussianの標準偏差ではない。M4/M8それぞれ既存のprofile、周期/階層の
quintic blend、中心±Lの支持、離散site上の積L2 floorを一度適用する契約を維持する。

atom a の混合Fourier成分を F_ac、各成分の支持に完全に含まれるタイルの和集合を E_ac と置く。

\[
R_a(x)=\sum_c {\bf1}_{x\notin E_{ac}} F_{ac}(x),\qquad
F_a(x)=R_a(x)+\sum_c {\bf1}_{x\in E_{ac}}\beta_{ac}^{T}\phi_c(x).
\]

\[
Z_{bct}=\sum_{x\in t}X_{bx}\phi_c(x),\qquad
H_{ba}=\sum_x X_{bx}R_a(x)+\sum_{c,t\subset E_{ac}}\beta_{ac}^{T}Z_{bct}.
\]

中心・Iの微分も同じ分解を使う。正規化は常に
`sum_x (sum_c F_ac(x))**2`を評価し、成分間の交差項を残す。
L2評価時に生じたR・R_mu・R_Iをchunk内の既存FP64 sample cacheに保存し、H/GとY/dXへ再利用する。
新しい窓、支持の拡張・切捨て、成分ごとの正規化はない。
完全タイル内部のRがゼロのblockではXを再読込しない。部分区間は同じ有限Fourier級数の評価値を使う。
cache容量超過では既存の正確な経路に戻る。

## 離散L2の有限和

大きい幅のタイル経路では、連続積分へ置換せず同じ離散和を係数から求める候補も比較する。
成分ペアの支持の共通部分に入るsite番号を lo,...,hi-1、N_s=hi-lo として、

\[
C(\omega)+iS(\omega)=
\frac{\sin(N_s\omega\Delta/2)}{\sin(\omega\Delta/2)}
\exp\{i\omega[\Delta(lo+(N_s-1)/2)-\mu]\},\quad \Delta=.1.
\]

ゼロ周波数は極限N_sを使い、極近傍はFP64の安定な展開を使う。
ω_cn=nπ/L_c、a_cn=weighted coefficient とすれば、

\[
\|F\|_2^2=\tfrac12\sum_{c,d,n,m}a_{cn}a_{dm}
[C(\omega_{cn}-\omega_{dm})+C(\omega_{cn}+\omega_{dm})].
\]

各ペアのC/Sはそのペアの共通支持で計算し、混合成分の交差項も含める。
`sum F F_mu`は各ペアの積に
`[(omega_n-omega_m) S_minus+(omega_n+omega_m) S_plus]/4`を掛ける。
`sum F F_I`は片側係数を既存のI微分係数へ置換し、C和へ1/2を掛ける。
CPUでは元のsite評価とautogradに対して値・中心/I微分を照合した。

浮動小数点での微小normの相殺を避けるため、観測領域内に中心があり、共有成分があり、
cacheに収まる幅だけに適用する。端だけが残る支持・非空floor・空支持・容量超過は従来のFP64 site和を使う。
いずれも同じ離散正規化を計算する実行経路の違いである。

## 毎stepの共有数による振り分け

GPUで `(周期, 階層, 軸, タイル)` ごとの候補atom数を数える。同じbankには各atomから高々一成分が入る。
ある成分が両軸で必要とする完全タイルの候補共有数がすべて閾値以上なら、その成分をタイル経路にする。
それ以外も同じFourier関数の直接GPU評価で残す。選択結果はbackwardまで保持し、次のforwardで再構築する。
CUDA Graph capture中もCPUへ判断を戻さない。

これは候補共有数に基づく費用heuristicであり、選択後の共有数が閾値以上に保たれる固定点保証ではない。
候補数と選択後の数を別に記録する。中心の一致やσの量子化・固定アンカーは要求しない。

内側の一タイルの値の縮約だけなら、S site、R=2M+1項、A共有atomについて
直接のASに対して集約RS+ARという演算数の目安になる。
`A>RS/(S-R)` はS32でM4約12.52、M8約36.27、S64でM4約10.47、M8約23.15。
これはL2・境界・振り分け・勾配・起動費用を除いた理想化であり、実測の損益分岐ではない。

## 現在の階層規則が許す範囲

幾何的な最小幅は次のとおり（閾値ちょうどではcoarse blend weight=0、直後から有効）。

| 最小タイル点数 | M4の最小I | M8の最小I |
| --- | ---: | ---: |
| 32 | 6.0667 | 1.1143 |
| 64 | 12.1333（I上限10の外） | 2.2286 |
| 128 | 24.2667（範囲外） | 4.4571 |

これは現在のq floor=.7/.3とT/L_minによる範囲であり、Fourier一般の制約ではない。
この比較は形を維持するため、その規則を変えない。M4中幅や小幅でのタイル経路は縮退した直接評価になる。

## 比較条件

N1024、B32、spacing .1、IEEE FP32/TF32なし、FP64 geometry/profile/norm。
小I .101〜.351、中I1.2〜3.2、大I5〜9.9、atom数64/1024/8192。
各fixtureにゼロ振幅、端、非空floor、空支持の特殊4atomを含む。
大I/1024atomでは4領域へ中心を集中させる初期配置も比較する。
中心は各atomで異なり、学習中の拘束はしない。
通常密度5%のA52428 mixedは別の最終比較とする。

既存Linear runnerを使い、独立workerの完全forward/backward/fused capturable AdamWを測る。
振り分け・sort・係数・L2・cache構築・H保存を含む。
正しさは独立FP64 quadrature/KKT oracleの全site・全atomでY/dX/全4Parameter勾配、
3状態・retained backward・変更後Graph replayを同じatol=rtol=4e-4で確認する。
メモリはcapture/replay込みのallocated/reservedを分ける。GPU process使用量は未測定。

## 回収済みの結果と停止状態

新しい実験は停止。待機中の所有jobは取消済みで、poolのslot 1/2/3はすべてstopped、
serveプロセスも終了している。Kyutechは使用していない。
通常密度A52428の新候補比較、大Iの残り、cluster比較、全82 GPU regressionは
**待機中に取消**。実行済み・検証済みと扱わない。現在の共有数別dispatcherは未確定。

全てColab L4、N1024/B32。各行は一つの独立job、完全step中央値。
MBは10^6 bytes、capture/replay込みのallocatedピーク。候補ごとに独立FP64のY/dX/全atom勾配gateを通過。
今回の短いjobでの数%の差は独立再測定していない。

| M | 初期I | atom数 | direct H ms | 幅sort + direct H ms | Dense ms | 最速CST allocated MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | .101–.351 | 64 | 0.2468 | 0.2755 | 0.0798 | 2.226 |
| 8 | .101–.351 | 64 | 0.2806 | 0.2990 | 0.0799 | 2.277 |
| 4 | .101–.351 | 8192 | 21.8368 | 21.2111 | 0.0788 | 17.765 |
| 8 | .101–.351 | 8192 | 23.9053 | 23.2356 | 0.0809 | 18.161 |
| 4 | 1.2–3.2 | 64 | 0.3543 | 0.3953 | 0.0809 | 2.226 |
| 8 | 1.2–3.2 | 64 | 0.4844 | 0.5130 | 0.0799 | 2.277 |
| 4 | 1.2–3.2 | 8192 | 30.8511 | 27.8057 | 0.0799 | 17.765 |
| 8 | 1.2–3.2 | 8192 | 44.6638 | 39.2817 | 0.0788 | 18.161 |
| 4 | 5–9.9 | 64 | 0.6052 | 0.6308 | 0.0799 | 2.226 |

Denseのallocatedは51.383MB、reservedは111.149MB。CST reservedはこの表で27.263〜33.554MB。
ここで小I/中Iの直接経路が速いことは、大I・高共有数に対する階層タイルの否定ではない。

初期の境界cache pilot（source `a70325a3`、A1024、大I）では、従来tile32から
M4 14.5869→10.2257ms、M8 28.2778→17.4080msへ短縮した。
同jobのdirect Hは6.6540/12.5092msでなお速い。境界cache allocatedは15.110/16.697MB、
chunk4096の小容量cache候補は3.768/6.141MBまで下がるが13.4523/23.7896msになる。
このpilotは後続のゼロ残差blockのX読込省略・係数正規化導入前であり、現HEADの測定と混同しない。

現HEADの一般有限和による係数L2候補は精度を満たしたが、M8中I/A8192で
32site count0=170.7069ms、count8=171.7248ms、64site count8=131.0843msとなり、
sort directの39.2817msより遅い。allocatedも20.219/20.350/19.512MBで、
sort directの18.161MBを超える。この実行方法を高速化成功とは扱わない。
原因をFP64演算・レジスタ・spillと断定するhardware counterは未取得。

## 検証範囲と証拠

- 修正後source `ecb6987c` のcritical GPU gate: **30 passed, 52 deselected**。
  coefficient norm、cache128/512、sharing0/8、warps1/4、特殊atom、sortの全勾配、
  retained backward、順序が変わるGraph replayとParameter version guardを含む。
- CPU全体はsource `227a6c3a` で2258 passed / 5723 skipped。wheel/sdist buildも同sourceで成功。
  後から追加したsortを含む全体82 GPU testは未完了。30 gateと完全caseごとの照合を超える主張はしない。
- `56d24205` の2 jobは共有数kernelのloop変数型衝突でcompile failure。
  `6fb6a39b…` は係数L2 kernelのstatic/dynamic変数衝突で18 fail/12 pass。いずれも修正し、失敗runを速度表から除外。
- 長いmatrix job `aceb9bdf…` は通信timeoutで結果archive未回収。pool経由で復旧しVM停止。
  結果は全て除外する。後続の短いjobのみ以下に記録した。
- 論理的な読込省略と実測DRAM転送量は区別する。物理DRAM/spill counter未取得。
  demandで未使用計算を抑制してもatlasの確保容量は最大形状で固定。
  親の偶数周波数のみ子から継承でき、奇数周波数は新しい読込が必要。

回収済みsource/result archiveのSHA256（collectorでartifact hashも照合済み）:

| cohort | job | source SHA256 | result SHA256 |
| --- | --- | --- | --- |
| boundary-pilot | `l4job-1c0115e620324826ba2fb60ea0e9bab7` | `333a2ed7d3f89a44acaa1947f29ece9ebdd9b36ac2ce198c9f1377539ed265df` | `76d9ee4560fac65d1ee75284962241e549dd947ab319ac4525bf6586227ed7a3` |
| m4-small-a64 | `l4job-a932dfddbac741e5acff2973e55476b7` | `30f98940d1c1a0a983f2b842d482592937a017567b402d7d858d07d6504e7b11` | `6b8904598b3d5e1f7980d972a890fdd9958cc96c747c72ad4ae4f6b5c1f422ad` |
| m8-small-a64 | `l4job-90825bb32c67487ab58b226deaf2e6c0` | `955335ee6cbb7f2b83a8ff9f0e316d772ae6c57cc34951930514a344e1454857` | `35b8d30957bc61c8d38bf779d63556b79162a08a001c128e4531e99056fcaea2` |
| m4-small-a8192 | `l4job-72f90dcaaf804f168824e9ffa5ec6f17` | `b54cf6b676856764236ffd30b1735231661c512e2381721ca75ec17cb3ee3dc2` | `0fdfa5c952d2e98da99b1b960b582115139a6483a36377a26046ca5c8e8742a8` |
| m8-small-a8192 | `l4job-a9240f9c93f147448a66c218d02e56af` | `60774c934f869d52f15b2832f5d15a108b192a0618af5750633d10cc0473a40a` | `3df03ef3761d0d32c530e8cfb6023acdb460a50bd1c9f7423caee50efbc6d997` |
| m4-medium-a64 | `l4job-dca74a85f362454685d4b1258d943c60` | `193dfede267b25edcbf1c258794ca65aa62d3e0b6a0e4d1cf891a4d12f7e0917` | `15a7bbc96f792869b317647e07362782a490b03117333b2f34362429e6dc9bd6` |
| m4-medium-a8192 | `l4job-d321030f74e54cf0840bf611810f567f` | `0aeb10e5815cfd6497d9acfd65a0f66c97706130c4fb214b6fba44b4ad82ca24` | `be56014408fa6d0f9d738872cdf9af3764e2063e82bfdc5543688e5eceb33c00` |
| critical-gates-fixed | `l4job-cadef25c931744b5a1b53bc9964a4ae5` | `926d9ef579bfde8d0d5c5872913dadd9f67ba4d0e253cc7f23c4bc787a16e87b` | `19d1388e30cff3f26f2b14870651e91a53446bc7945dd2f6a14860ded9cb680b` |
| m4-large-a64 | `l4job-1b66db4dc98c4acfa59918c8a89397a3` | `fe00ad3a8bc5907e8ed4e05b8f228370a133f3fab77b640844c42297ff33c6c6` | `ffd85ae029885fe07b15fecff3faf9fed39cc63c79b2e689834ac5aa08df3b73` |
| m8-medium-a64 | `l4job-25bce23fe4514fbbb8d461839003c5b7` | `2c78210629fbb906672c855b324d74706e1805a91e870b62cec103feea0976d9` | `a579e1681a2d22c6340bfff6e595eb261a2cdd4bd49b27537541c9d92168d888` |
| m8-medium-a8192 | `l4job-947dc6dfde77471ca9b1b4866cd7a78a` | `6ecdde351a1bc42a6541e04338a5b21ee9cc90680bfd6ac005cc88ce947f8fdb` | `8dbb30c347c420c079638fdcac45b5a424c7094e14ebd274f0b04a6349c27b58` |

raw evidenceはignored `output/hierarchical-fourier-dispatch-20261021/` の
`results-summary.json`（51測定行、11成功job）、`measurements.csv`、driver、CPU/build logと、
`~/.local/state/colab-l4-pool/jobs/<job>/` のsource/results/receiptに保存。
通常密度の前回比較は[前回記録](hierarchical-fourier-reuse-20261021.ja.md)を参照。
今回取消した通常密度jobで置き換えない。Dense2倍以内の目標達成は確認できていない。
