# Owner-window FP32 body: 数値成立性の研究

## 測定前の固定条件

#128 (`ac4c8980cb65c7a64aefe37cf20cccabd8d65c3d`) に積んだ別候補。
main/既定dispatcherは変更しない。まだ採用・高速化を主張しない。
まず内部native入口だけで成立性を検査し、合格した場合だけPlan/Caseと
全モデル測定へ進む。内部入口は公開hooks/live設定の代替APIではない。

N1024/M8/spacing1、Polar、radius bounds [.5,50] のみ。
Parameter/公開入出力はFP32。decode・wrap・分類、local displacementの減算、
最後のPolar pullbackは既存FP64を保つ。その他の係数、phase、cell積分、
raw/window、H/Gと中心微分、C縮約、da/dmu集約をFP32にする。
atom/chunk/level順、kernel分解、正規化、学習する中心・半径、支持、全dP、
dX、radius task-gradient detach、optimizer則は維持する。
fixedDFT/C配置変更/tight mask/cooperative raw/fast-math/TF32は追加しない。

## 事前ゲート

- 独立CPU FP64 Fourierおよび小集合の独立quadratureに対し、Y/dX/全dPの
  **各要素**で `abs(error) <= 3e-5 + 3e-4*abs(reference)`、全値finite。
  平均/RMSが良くても最大要素の不合格を隠さない。
- 参照勾配をFP32へ一度丸めたものと候補勾配を、同一Adam初期状態に入力。
  `u=mhat/(sqrt(vhat)+eps)` のmax差 <= .01、各Parameter列RMS <= .001。
  post-Polar更新差はs/tで `1e-6 + 2 ULP(FP32)`、中心は周期距離で比較。
  同一勾配を入れるoptimizer対照はbyte一致。
- constant/one-hot/alternating/random、owner集中、tiny amplitude、極端なs/t、
  zero/clamp、周期・分類境界の隣接FP32値、wide radius、小勾配(.1eps..10eps)
  を含める。通常checkpointの全52,428 atomも検査する。
- eager/Graphの同じ精度同士はbyte一致。異精度には上記数値基準を使う。
- 不合格は、係数c/q/dq、active window、H/G・中心微分・pullback前で診断。
  未初期化のinactive raw/window領域は比較しない。

数値ゲートに不合格なら、実装不具合か丸めかを切り分けて保全し、
この候補の速度測定・長期学習は行わない。結果を見て許容差を緩めない。

合格した場合のみ同じL4上で初期/学習後checkpoint、256 minibatches×6反復の
fresh processでFP64/FP32のeager/Graphと同等条件のdenseを比較する。
未計装完全step、guard/入力供給込み、capture込みpeak allocated/reservedを記録。
継続目安は両状態10%以上短縮・paired 95%下限5%、eagerの明瞭な回帰なし。
これは研究の継続目安であり、ユーザー指定の性能・メモリ上限ではない。
長期学習は局所数値・性能ゲートを通った場合だけ行う。

#128 harnessのrestoreは符号付きzeroも含むbyte比較に修正し、capture後の
実際のlive開始状態を計時直前に再取得して元の開始状態と照合する。
過去#128の原本・測定結果は変更しない。

## なぜ出力だけでは判定しないか

正規化された各atomの入力因子を $v_a$、出力因子を $u_a$ とすると、
$H_a=\sum_i X_i v_{ia}$、$G_a=\sum_j dY_j u_{ja}$。
一定入力・一定dYでは、各因子のcell mass総和が1なので $H_a=G_a=1$、
$\partial_{\mu_i}H_a=\partial_{\mu_o}G_a=0$ が数学上の基準となる。
このzero近傍ではFP32の三角関数・加算の残差が、Yの許容差内でも中心勾配を変えうる。

cold Adamの最初の更新方向は $u(g)=g/(|g|+\epsilon)$。
例えば $\epsilon=10^{-8}$、参照 $g=0$ に対して残差 $g=10^{-6}$ があれば、
勾配の絶対差は小さくても方向は0から約0.9901へ変わる。
これは感度の説明例であり、この候補で観測した値ではない。
実検査では参照勾配をFP32へ一度丸め、同じcold状態と実CSTAdamWにも入力して比較する。

CPUの事前確認: 独立48点Gauss-Legendre積分とpublic CPU FP64 Fourierの
小集合Y/dX/dP一致、eps近傍の符号差を更新方向gateが検出すること、
restoreが負のzeroを復元し同一byteの定数Tensorを再変更しないことを確認した。
標準CPU suiteは `d6b46df0` で **1,707 passed / 2,133 skipped**。
skipはCUDA/外部環境等の未実行であり、GPUでの合格を意味しない。

## 中間試行と測定器の修正

CUDA算術実装は `d6b46df023b6bb52e8df2cb55abf387e22350d24` 以後変更していない。
以下は同一候補の測定器修正であり、新しい最適化候補を追加したものではない。
失敗jobと部分結果も削除せず保全した。

1. `l4job-72ee46ac693e4da79145b9ba043f6c92` / `d6b46df0`:
   compileと小集合8条件は完了。全モデル構築でexplicit execution Planが不足し停止。
   この時点の実optimizer検査はlr=1e-3、amplitude decay=0で、凍結学習条件と異なった。
   これらのpost-P数値を凍結条件の結果として扱わない。勾配とcold Adam方向差は保全。
2. `l4job-33af50fbf262403796258f84c88912cd` / `43ca36f7`:
   追加境界fixtureと全atom初期状態まで完了。学習後のoptimizer検査で、モデルから
   deepcopyした既存optimizerのCPU stepがCUDA controlに混入して停止した。
   optimizer初期状態を測定対象と混ぜず、現在のPとKernel宣言だけから新しいsiteを作るよう修正。
3. `l4job-1402338e7e204e02b023917372a0138e` / `134dcb1e`:
   保存済み勾配8条件を、凍結cohortのlr=1e-4、amplitude decay=.01、
   raw decay=0、betas=(.9,.999)、eps=1e-8、native CSTAdamWで再評価。
   通常範囲のatomだけでもconstant/alternatingは不合格で、同一勾配対照のP/moments/stepはbyte一致。
4. `9efb89bd16ed7f7082c27453389eefc1944d1580`:
   fresh cold optimizer・正しい係数・FP64での周期距離を使う最終検査source。
   高価な全atom VJPは、後段optimizer等が失敗しても残るよう先に保存する。

極端な `s=t=FP32(1e20)` の行では、候補と参照に同じ勾配を入れても既存のFP32
Polar update後のs/tが非有限になる。これはFP32 body固有の失敗と区別する。
既存updateはFP32の `s*s+t*t` を使っており、この値ではoverflowする構造にある。
基準側が非有限の行はpost-Pの通常誤差に混ぜず、行番号と比較対象数を明記する。
全fixtureのadmissionはそのまま不合格とし、正常な部分集合の成功に読み替えない。

一定入力の診断では、元のFP32 windowからのHmu最大値は約2.714e-7、FP64基準は
約4.118e-16。FP64で生成した同じwindowを一度だけFP32に丸めて再計算する対照では
約6.135e-12に減った。測定したpullback前値をCPUのFP64 Polar式へ入れた全dPは、
候補の実際のdPとこのfixtureでbyte一致した。
この対照はwindow生成の丸めが主要な入口である証拠だが、
「windowだけFP64へ戻した完成版」の全数値・速度合格を示すものではない。

## Dense対照の追加（ユーザーの問いを受けて、結果取得前に固定）

ユーザーから「denseもFP32なのに、CSTだけ誤差が大きいのか」と指摘された。
上記gateは研究用の厳しい互換性基準であり、FP32の実用性そのものの合否ではない。
これまでdenseを同じ精度基準で調べておらず、CSTだけが悪いという判断はできなかった。

凍結初期effective Wを使う通常Linearで、同じB4のX/dY6条件をCPU FP64参照と比較する。
TF32 off、PyTorch FP32 GEMM、同じepsと学習率のfused AdamW。
CSTのdPとdenseのdWは座標・スケール・要素数が異なるため、max値だけで学習品質を順位づけない。
加えて `(1, FP32(1e-8), -1, 0)` の全24順序をK4縮約に入れる、意図的な打ち消し対照を行う。
これは通常データと分け、都合の良い順序だけを選ばず全結果を残す。
今回のgateがdenseにも厳しすぎる可能性を調べる対照で、過去の不合格を合格へ変更するものではない。
学習精度悪化・不安定化やFP32使用不能は、この局所gateだけでは立証されない。
