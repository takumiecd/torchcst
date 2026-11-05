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

## 最終CST数値検査

`l4job-b61d61c266a74d7494026eb346c089d2` は完了。測定sourceは `9efb89bd16ed7f7082c27453389eefc1944d1580`。
L4 `GPU-3c84e6bf-2edd-ff9f-cd44-5688ed0c7540`、driver580.82.07、
Torch2.11.0+cu130、CUDA13、Triton3.6、Xeon2.20GHz/12 logical cores、CPU math threads1。

13条件すべてで、候補・FP64基準ともY/dX/全dPの各要素が事前の混合許容差内、finiteだった。
小集合11条件は独立48点quadrature、全52,428 atomの2状態はpublic CPU FP64 Fourier参照。
学習後の状態は、既存FP64経路で学習したcheckpointのPを評価したもので、
FP32候補を45,000 step学習したという意味ではない。

| 全atom状態 | Y max abs / relative L2 | dX max abs / relative L2 | dP max abs / relative L2 | cold Adam max delta-u | .01を超えた成分 |
| --- | --- | --- | --- | ---: | ---: |
| 初期 | 2.138e-9 / 4.132e-7 | 1.373e-9 / 4.154e-7 | 1.426e-6 / 3.424e-7 | .020336628 | 1 / 209,712 |
| 学習後 | 2.056e-7 / 3.762e-7 | 2.592e-7 / 3.897e-7 | 1.484e-6 / 3.524e-7 | .010854959 | 1 / 209,712 |

列別direction RMS最大は初期9.504e-5、学習後5.475e-5で、RMS基準.001は満たした。
最大値gateだけが通常2状態で超過した。初期の最悪はatom16327の入力中心勾配で、
参照 -2.4097635e-9、候補 -2.1042841e-9、差約3.055e-10。
学習後の最悪はatom4603のt勾配で、参照 -9.4994974e-8、候補 -8.4252790e-8。

constant/alternatingではmax delta-u=.944368124/.958893418。
一定入力の最悪中心勾配は参照1.388e-16、候補1.698e-7。
正しい学習率でのpost-Pの中心周期距離maxはconstantで1.221e-4、alternatingで9.584e-5。
通常norm部分集合だけでもこの失敗を再確認した。same-gradient対照はPおよび全3 optimizer
Tensor（step/exp_avg/exp_avg_sq）がbyte一致した。

これらは厳しい局所互換性gateに不合格という結果であり、実用の学習精度低下や
「FP32では使えない」の証明ではない。全勾配の通常の相対L2誤差は約3.4～3.5e-7であり、
微小勾配へのAdamの感度と分けて扱う。学習品質の判断には別途事前に定めた学習比較が必要。
今回の候補は既定へ採用せず、速度・capture memory測定・長期学習には進めていない。

GPU環境の回帰検査は **191 passed**（既存window/whole-step検査188＋新CPU参照検査3）。
FP32候補の公開Plan登録や、FP32全stepのhooks/live setting/capture契約を認証したという意味ではない。
最終code sourceのCPU標準suiteも再実行し **1,707 passed / 2,133 skipped**。
既存Plan/Case宣言検査、対象lint、diff checkもPASS。
GitHub CPU validation run687は `9efb89bd` で成功した。

- source archive SHA256: `7278f57649b2006d5db3dbf566bad190c789f3247a7d760beeec5b5cf9b634db`
- result archive SHA256: `60bfc68730819dcea2a24d9ee425a686496aaef4d87f32eaece38b4b18b0e56f`
- 回収manifestの35ファイルを再hash照合。`output/owner-window-fp32/final-results/` に保全。
- その他の部分結果は `attempt1/`、`attempt2/`、`matched-adam/` とdefault pool job原本に保全。
- 入力init SHA256: `b49e95e722dab4e42d935a9ac2b239212e93660af55bf6d6a9f12f437a9f2ed6`
- 基準学習後checkpoint SHA256: `ca52f8e5c7982de2ca26e6cf6f9528c1d4d81f8ec54f5dd0d503f0609f3d079b`

再現: 上記hashの2つのptをinputsに置き、以下を同じ依存関係のL4環境で実行する。

```bash
PYTHONPATH=src python -m benchmarks.training.owner_window_fp32 \
  --inputs /absolute/frozen-inputs --output output/fp32-numerical
CST_GRAPH_INPUTS=/absolute/frozen-inputs PYTHONPATH=src python -m pytest -q \
  tests/test_owner_window_fp32_probe.py tests/test_radius_owner_window.py tests/test_full_step_graph.py
```

main/既定dispatcherは変更せず、#128上の研究Draft [#129](https://github.com/takumiecd/torchcst/pull/129) に保全する。

## Dense対照の結果と判断の範囲

`l4job-c14b257d85b4482cbe3e176f73666133` は完了（source `2486e5810f69afc3ac4554ded257819cff5aa45e`）。
同じ型・runtimeの**別allocation**のL4 `GPU-66891846-9da7-9834-559f-3fccf3d4f82f`。
N1024/B4、同じrandom X/dY、凍結初期effective W（FP32に保存されたdenseのParameter）を使用。
各表現自身の独立CPU FP64参照と比較した。CSTのFP64実効WをdenseにFP64保存して
全演算を完全に同じ関数へ揃えた比較ではない。CSTのatom座標dPとdenseのdWも異なる。

| random入力、初期状態 | CST FP32 body | dense FP32、TF32 off |
| --- | ---: | ---: |
| Y relative L2 error | 4.132e-7 | 1.749e-7 |
| dX relative L2 error | 4.154e-7 | 2.058e-7 |
| cold Adam max delta-u | .020336628 | 1.454e-5 |

この通常入力では出力・dXの誤差は同じ10^-7の桁で、CSTだけが桁違いに壊れる結果ではなかった。
denseの通常6条件はY/dX/dW許容差と更新方向gateを通った。
constantではdenseのdWは4で誤差0なのに対し、CSTには数学上0の中心勾配があり、
そこでwindow位相の打ち消し残差がAdamへ入る。CSTにはこの不変量を安定に扱う余地がある。

一方、意図的に打ち消しを作ったK4の全24順序のうち16順序で、標準denseも
FP64参照のdW約1e-8がFP32では0となった。max delta-u=.5、RMS=.015625、
同じ学習率でのpost-W差max=5e-5となり、CSTに適用したgateに不合格だった。
全30条件でsame-gradientのP/全Adam状態対照はbyte一致。
これはdense通常学習の失敗率でも、CSTの方が優れる証拠でもない。
**最大更新差gateはFP32の実用性そのものを判定する基準としては不十分**と判断する。
当初gateの不合格という事実は変えず、FP32候補の研究全体を使用不能と結論しない。

次の研究をするなら、座標/spacingの全面変更より、証拠のあるwindow表現・縮約を先に狙う。
例えば定数成分を分離して非DC成分を差分から生成し、解析上ゼロの位相を整数情報で扱う。
局所座標は大きい絶対位置からの引き算を避けるが、今回すでにその減算はFP64だった。
spacingを一様に変え、中心/半径/周期を同じ倍率で変えるだけでは、無次元位相と
浮動小数点の相対的な精度は改善しない。optimizerの単位・eps・学習率まで含む契約にも注意が要る。
これらの別候補は未実装・未測定で、学習精度・速度改善をまだ主張しない。

- Dense source archive SHA256: `3c6bb756af0b6cdd909ff40a9e11440f590190cc3f5da774e3378e3ee24a3b7c`
- Dense result archive SHA256: `a4759a2d099f1d580aa379a19e203bfe847eead62d1f8fe8457be877b56258a2`
- 回収manifestの7ファイルを再hash照合。`output/owner-window-fp32/dense-results/`に保全。

## Phase II: ユーザー指定による実用評価への継続

ユーザーはdenseでも起こる程度のFP32丸め誤差を許容し、FP32という理由だけで
却下しない方針を明示した。Phase Iの厳しい互換性gateの結果は変更しない。
以降は同一候補をopt-in Plan/Caseへ登録し、全モデルの実学習と時間・メモリで評価する。
既定dispatcherとmainは変更しない。座標表現・spacing・CST数学・optimizer則の変更は行わない。

測定前にPhase II条件を固定する:
- 同じ凍結seed41初期値・初期effective W・MNIST全train/test・データ順序。
- N1024/A52428/真B4、3 weighted layersの中央だけCST、3 epochs/45,000 updates。
- baseline FP64 body / candidate FP32 body / denseのfresh processを同じL4 allocationで順次実行。
- TF32 off、CPU math threads1、native CSTAdamWの凍結係数、gather/loss/dX/全dP/全層更新込み。
- epoch境界同期の未計装throughput、初期評価/warmup込みのallocated/reserved peak。
- 全epochのloss/精度、最終P/全optimizer Tensorのfiniteを確認。
  これは全stepの全中間Tensorを検査するものではない。
- 継続目安: candidateの最終test精度がbaselineより.2 percentage points以上悪化せず、
  loss悪化が.005以下。これは事前の研究目安で、単一seedを本採用の一般化証明にしない。
- この学習比較に問題がなければ、同じGPUで初期/既存baseline学習後Pの2状態、
  256 step×6 balanced-order fresh processをeager/Graph、CST両精度とdenseに実施する。
  速度対照の初期状態は両精度とも同じFP64の8更新を経たP/Adam状態から開始し、
  candidateのnative certificate準備には更新を伴わないforward/backwardのみを使う。
- 同じ精度のeager/Graphは状態/最終勾配/集計がbyte一致することを要求する。
  異精度の長期状態byte一致は要求しない。guard/入力供給とcapture込みpeakを含める。

現在のスコープはMNISTの一つのモデル・seed41。別モデル・別seedの改善を主張しない。

### Phase II実学習の結果

`l4job-a8a266a00bcd4946bab40d3b9af0e345` / source `440b5696385b45b70c047b095a7fa0a3dbbe43cc`。
L4 `GPU-39e86ded-86de-361e-b9e9-7d1b8996f84d`、同じTorch2.11/CUDA13/Triton3.6環境。
公開Planを使う契約検査は **208 passed**。FP32 Planのsame-precision eager/Graph、
hooks/live settings/storage/callback/finite-check変更時のfallbackを含む。
CPU標準suiteは **1,708 passed / 2,149 skipped**、GitHub CPU CI run688も成功。

同じallocation内でbaseline→candidate→denseの順に、各fresh processで3 epochs/45,000更新を完了。

| 完全学習、seed41 | 平均ms/step | 最終test精度 | 最終test loss | peak allocated MiB | peak reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| CST FP64 body | 2.506987 | 97.37% | .082855619 | 246.089844 | 392 |
| CST FP32 body | 2.498065 | 97.37% | .083672397 | 246.089844 | 390 |
| dense FP32 | 1.317238 | 97.87% | .070017517 | 257.861328 | 384 |

全epochのloss/評価がfinite、最後の全Parameter勾配・保存したmodel/optimizer Tensorもfinite。
初期hashと全epochのdata order hashは一致した。
FP32のaccuracy差は0、loss差+.000816777で、事前のPhase II継続目安を満たした。
これは1モデル・1 seedの結果で、FP32全般の保証ではないが、Phase Iの局所gate不合格を
「FP32で学習できない」と解釈しないことを実験で裏付けた。

eagerでの名目短縮は約.36%で、大きな速度改善は観測していない。
allocatedの全モデルpeakも同じで、reservedのみ2 MiB差だった。
1つの長区間・固定順序の比較から、この小差を確実な高速化と断定しない。

このjobはその後のFP32 Graph比較の準備で停止した。candidate側の無更新preflightの後に
`zero_grad(set_to_none=True)` を呼び、PreparedStepが要求するcurrent gradient metadataを
自分で消したため、guardが正しく拒否した。学習3条件の結果は完了・保全済み。
`63aa8079972ee14bedd80743f0964728799c0a57` で準備勾配を保持するよう測定器だけを修正した。
PとAdam状態は変えず、次のbodyの冒頭で通常通りzero_gradする。
Graph部分だけを `l4job-71f276106b15410fb32b3050324d6b8f` で再測定する。

- 学習job source archive SHA256: `5f0300bd2e1ee689be7ccc9baf2b820ed962dc6105378ad9437c8879b79a1c93`
- 学習job result archive SHA256: `04af3ecec268d9a71aa5b44705c8e50c5593c8e23b9f37e07f96b9b3d05a3b80`
- 回収manifestの34ファイルを再hash照合。`output/owner-window-fp32/phase2-learning-results/`に保全。

### FP32とGraphの併用で大幅な短縮を確認

再測定 `l4job-71f276106b15410fb32b3050324d6b8f` は正常終了した。
source `63aa8079972ee14bedd80743f0964728799c0a57`、L4
`GPU-4a2c5f13-35c7-09db-aea8-a56883d034ac`、driver580.82.07、
Torch2.11.0+cu130/CUDA13/Triton3.6、CPU math threads1。
学習jobとは別allocationだが、**以下の全比較はこの1台・同じallocation内**で行った。

初期/既存baseline学習後の2状態、256 step、6 balanced-order fresh process、全72 cellが完了。
CSTの全精度条件で開始P/Adam状態はbyte一致。同じ精度・モデル・状態の各12 cellは、
開始/終了Tensor状態、最終全Parameter勾配、loss/正解数集計、data/order/input hashが一致した。
最初に候補の両状態について4-step eager/Graphも照合した。
これは全256 stepの全中間Tensor比較や、45,000 stepのGraph学習ではない。

| 状態 | CST FP64 eager | CST FP32 eager | CST FP64 Graph | CST FP32 Graph | dense eager | dense Graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 初期共通8更新後 | 2.595192 | 2.606781 | 2.134053 | **.864602** | 1.325697 | .203586 |
| baseline学習後 | 2.691627 | 2.712309 | 2.302282 | **.876375** | 1.303198 | .205140 |

単位は未計装完全stepのmean ms、gather/loss/dX/全dP/optimizer/metricsとGraph guard・入力index供給・
host version通知を含む。評価/データ配置/準備はstep時間外で、準備時間は別記した。

**同じGraph実行同士のFP64→FP32短縮は初期59.485%、学習後61.935%（約2.47/2.63倍）。**
両状態で6/6 paired反復が短縮した。#128と同じpaired log-time比・df5片側Student-t法による
探索的95%下限は59.443%/61.879%。独立性・log比の分布を仮定した同一allocation内の指標で、
独立GPUや別モデルへの保証ではない。

同じFP32経路のeager→Graph短縮は初期66.833%/学習後67.689%。一方eagerだけでは
候補は名目.447%/.768%遅く、各状態3/6勝、探索的下限は負で、安定した改善/回帰とは断定しない。
先の3 epochでの約.36%の差と合わせ、**FP32化だけではeagerの全モデル速度は改善せず、
今回の実行方式との併用で大きな利益が出た**と判断する。
この観測だけで律速CPU関数やGPU kernel別の費用まで確定はしない。

同等Graph条件のCST/dense時間比は、FP64で10.482/11.223から、FP32で4.247/4.272へ改善した。
denseもGraphで約84%短縮する条件なので、CSTだけGraphにしてdense eagerへ勝ったとは主張しない。

### Capture込みメモリ・準備費用

6反復で各条件のpeak値は一致した。共通FP64 warmup、optimizer生成/load、workspace、
capture/復元・live開始状態再照合も含み、capture後のpeak resetは行わない。

| 状態 | 条件 | allocated MiB | reserved MiB | mean setup秒 |
| --- | --- | ---: | ---: | ---: |
| 初期 | CST FP64 eager | 245.402344 | 418 | 3.017804 |
| 初期 | CST FP32 eager | 245.402344 | 418 | 3.015636 |
| 初期 | CST FP64 Graph | 251.534180 | 442 | 3.120873 |
| 初期 | CST FP32 Graph | 251.534180 | 440 | 3.138187 |
| 学習後 | CST FP64 eager | 250.714844 | 418 | 3.156142 |
| 学習後 | CST FP32 eager | 250.714844 | 418 | 3.169238 |
| 学習後 | CST FP64 Graph | 251.534180 | 442 | 3.273043 |
| 学習後 | CST FP32 Graph | 251.534180 | 440 | 3.276027 |

Graph同士でallocated減少はなく、reservedは2 MiB減。FP32 eager→Graphでは
allocatedが初期6.131836/学習後.819336 MiB、reservedが22 MiB増える。
全モデルのメモリ半減を示すものではない。プロセス全体のGPU使用量は未測定。
dense Graphはallocated264.597656/265.160156 MiB、reserved432 MiB。
このmemory scopeは前述3 epochのmemory scopeと異なるので、表を混ぜない。

setupにはデータ/model配置やprocess初回compileを含まず、共通準備後のoptimizer/warmup/capture等を含む。
GPUジョブ全体のcompile/preflight/起動・回収費用は別途pool logs/phase logsに保全している。
このPhase 2終了時点では、複数seed、別GPU・別モデル、45,000 stepのGraph学習は未実施。
その後の[FP32自身の45,000更新eager/Graph比較](owner-window-fp32-long-graph-20261103.ja.md)で、
評価・保存・train復帰と実moment診断を含めて完了した。準備込み61.64–62.08%短縮、
各epochの数値状態・勾配・指標のbyte一致を確認。独立した実験なので、以下の短区間の表とは分けて読む。

### 保全と現在の判断

FP32主体版はseed41のeager実学習で精度を保ち、限定した全step Graphと組み合わせて
大きな速度利益を確認できた。**FP32を使えないという結論ではなく、継続・レビューに値する成果**。
通常APIの既定Graph対応ではなく、モデル・状態を専有する既存研究用PreparedStepとの併用である。
ユーザーの追加指示なく既定dispatcherへの採用・main merge・別の最適化候補追加は行わない。
[#129](https://github.com/takumiecd/torchcst/pull/129) は#128上に積んだDraftとして保全する。

- Graph比較source archive SHA256: `3fc488b8f2d30b61fe935978756cca0d797a212f8afd0149c6279d82df888824`
- Graph比較result archive SHA256: `7b11a69268f831169b99dc666cbebecdca47bdf7941268c4c6f8d25abde55e3a`
- 回収manifestの **176 files** を再hash照合。`output/owner-window-fp32/phase2-matched-results/`に保全。
- 集計は `output/owner-window-fp32/phase2-summary.json`、生成script/driver/中間失敗も同directoryで保全。
- 共有pool全slotのstoppedとsupervisor正常終了を確認し、`final-pool-status.json`へ記録した。
- [公開用集計JSON](owner-window-fp32-results-20261103.json) はraw tensor/logではなく、上記結果からの集計。

Graph比較の再現は、#128同様の7 inputファイル（初期値、baseline/dense最終checkpoint、MNIST gzip4つ）を
hash照合して用意し、以下のprecision/mode/stateを上記条件で組み合わせる。

```bash
PYTHONPATH=src python -m benchmarks.training.full_step_graph_comparison \
  --inputs /absolute/frozen-inputs --output output/precision-cell.json \
  --kind cst --precision fp32 --mode graph --state final --repeat 0 --steps 256
```

実学習は `benchmarks.training.cell_mass_mnist` に `--radius-plan benchmarks/training/owner-window-fp32-plan.json`
を指定する。FP64対照は同directoryのfp64-plan、dense対照は`--radius-execution global-response4`。
初期値・data directory・batch4・epochs3・`--allow-radius-1024`は全条件で同じにする。
