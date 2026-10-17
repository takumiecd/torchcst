# 2次B-splineのGPU直接評価と共有累積集計

2026-10-17。branch `kernel/quadratic-bspline-prefix`。
[merge済みTorch参照](quadratic-bspline-profile.ja.md)と同じ形・離散L2・Polar更新を使う研究Algorithm。
公開Registryや既定dispatcherは変更しない。以下は初版の実装範囲と検証手順。

## 初版の範囲

- Euclidean RegularGrid、出力1座標×入力1座標。軸のsite数は2以上。
- `QuadraticBSplineSpec` revision 1のPolar profile-product。中心曲率幅sigma、支持半幅√6 sigma。
- X/sourceはFP32、chartのorigin/spacingはFP64。batch 1..64、autocast/TF32なし。
- 任意の中心位置とatomごとのlive sigma。支持は有限gridの観測siteで切る。
- 空支持、零振幅、norm floor、retained backward、forward後の中心変更を扱う。
- first-order VJP。高階微分、周期座標、入出力側に複数の座標を持つgridは初版の範囲外。

`research_cuda_quadratic_bspline`のRecipeで同じ演算の4経路を比較する。
`torch`は既存のTorch factored参照、`direct`は支持内だけの軸別GPU直接評価、
`hybrid`は各軸の支持site数により直接評価と共有prefixを選ぶ。
直接評価のsite演算枠は16/64、半幅の逆数のatom内共有も独立のRecipe設定として比較する。
初期候補の閾値32 siteは実装上の比較値で、最適な閾値やσの上限の結論ではない。
計算blockと数学的な支持／周期を混同せず、支持をblockへ丸めない。

## 共有する値

格子indexをz_j∈[-1,1]へ正規化する。B-splineの各pieceを
`F_a(j)=c_a0+c_a1*z_j+c_a2*z_j²`と展開し、入力ごとに3本の

\[
P_{bk}(t)=\sum_{j<t}z_j^k X_{bj},\qquad k=0,1,2
\]

を作る。各pieceのindex区間[l,r)から
`Σ_k c_ak(P_bk(r)-P_bk(l))`でHを取り出す。
中心微分は同じprefixを別の係数で読む。境界siteの直接走査を追加しない。
入力が変わったstepではprefixを作り直す。sigma・中心・norm・係数も毎forward更新する。
固定index基底だけを同じdevice/shapeで再利用できる。

書き戻しは随伴で、pieceの左端へ`+c_k*w`、右端へ`-c_k*w`を加え、
site方向の累積和と基底の内積で出力を復元する。
端点の衝突にatomic加算が残る。CSRやsite×atomのindex表は作らない。
小さい支持は同じ山を支持内で直接評価し、siteへatomic加算する。

初版のbank、端点加算、係数、normはFP64。H/H'とG/G'、Y、dX、source勾配はFP32。
FP64にするだけで任意のdomain/幅比の精度を保証するわけではない。
FP32 prefixの採用や局所blockへの変更には独立oracleによる再検証が必要。

## 離散L2とVJP

各pieceのsample数をK、sample中点をm、`t_j=(2j+1-K)/K`とする。
山を局所座標で`a0+a1*t+a2*t²`と表すと

\[
M_0=K,\quad M_2=K/3-1/(3K),\quad
M_4=K/5-2/(3K)+7/(15K^3),
\]
\[
Q^{piece}=a_0^2M_0+(a_1^2+2a_0a_2)M_2+a_2^2M_4.
\]

空区間はQ=0、K=1ではM2=M4=0を使う。3pieceのQを足す。
積分への置換ではなく、実際の端点位相を含む有限離散和の恒等式。
中心微分の`S=Σ F*dF/dmu`も同じ局所momentで計算する。
`D=max(sqrt(Qo*Qi),epsilon)`としてfloorは積全体へ一度だけ適用する。

生profileによるH=Fi X、G=Foᵀ dYとその中心微分Hp/Gpを使い、batchについて

\[
d\alpha=\langle H,G\rangle/D,
\]
\[
d\mu_o=(\alpha/D)\{\langle H,G_p\rangle-\gamma_o\langle H,G\rangle\},\quad
d\mu_i=(\alpha/D)\{\langle H_p,G\rangle-\gamma_i\langle H,G\rangle\}.
\]

floor非作動時はgamma=S/Q、作動時は0。振幅のPolar pullbackは元sourceの
特異原点guardを含めて行う。幅はtask VJPでstop-gradient、更新は既存のPolar則。
forwardのmetadata、origin、spacing、H/Hpを保存し、後のforwardや更新値と混ぜない。

## 検証と測定

`tests/test_quadratic_bspline_cuda.py`は符号付き3区間式と全軸の離散normによる
独立FP64 oracleで、Y/dX/全source勾配を照合する。
公開CSTOptimizerは現在Graph captureを拒否するため、Graph検証は既存research runnerの
capturable AdamW＋Polar updateを使い、公開CSTOptimizerのeager更新と比較する。

既存`benchmarks.cuda.linear.run`へfixture、Plan catalog、Caseを接続した。
Case内の既存`widths.rho`欄はこの研究fixtureでは物理sigmaの初期値を記録する。
実際のsigma/spacingはdecode結果とspacing=.1から求め、他fixtureのrhoと混同しない。
別の計測runnerは作らない。`bspline.py`がfixtureとoracleを持ち、
結果adapter revision 9で他profileと区別する。dispatcherへの既定採用は未実施。

```bash
PYTHONPATH=src:. python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-64-a204-small.json
python -m pytest tests/test_quadratic_bspline_cuda.py -q
python -m benchmarks.cuda.linear.run \
  --plans benchmarks/cuda/linear/plans-quadratic-bspline.json \
  --case benchmarks/cuda/linear/cases/bspline-64-a204-small.json \
  --polar-update fused --phase-diagnostics --output output/bspline-small.json
python -m benchmarks.submissions check output/bspline-small.json
```

完全stepの未計装Graph時間とcapture/replay込みのallocated/reserved peakで比較する。
診断phaseの合計を完全stepの代用にせず、独立run数と計測sample数を区別する。
原本はignored `output/bspline-gpu-20261017/`および共有poolのjob directoryに保全する。


## 小さい支持のmatrix経路

`matrix`は支持内のFo*FiをWへ集約し、Y=XWᵀ、dX=dY WをGEMMで計算する。
`dW=dYᵀX`を全atomで共有し、T=Σji dWji Foja Fiiaとすると
`dalpha=T/D`、`dmu_o=(alpha/D)(Σ dW Fo_prime Fi - gamma_o*T)`、入力中心も同様。
H/Gでbatchを先に縮約する式と、和の順序だけが異なる。
生profile・global floor・幅stop-gradientは同じ。支持が広がった場合も全区間を走査する。
初版は各軸1024 site以下だけを受け付け、大きいsigmaでの高速性は保証しない。
W/dWとGEMM workspaceを含めた完全step peakを測り、小さい支持向けの候補として扱う。

5万atomを超える測定では、全sigma配列をJSONへ列挙すると受付上限5MiBを超える。
このfixtureに限り、配列をhash付き`*.widths.pt`へ保存し、JSONにはsample indexと16例、
全体のmin/max、changed-atom数を記録する。値を打ち切って実行する変更ではなく、
計測後の証拠表現だけを小さくする。両方のファイルをpoolのmanifestで保全する。
