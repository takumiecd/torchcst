# 1次を主軸にしたoptimizerの再構成

2026-09-08。公開契約はREADME。この文書は今回選択した数式と実装上の限界を記す。

## 方針とAPI

- `CSTAdam`を1次専用の主経路とする。
- `CSTSecondOrderAdam`を任意に選ぶ2次専用経路として残す。
- モデルのparameter所有管理、backward観測、momentのexpand/compress、dense AdamWとの同時commitは共通。
- `approximation_order`と`first_moment_frame`の組合せは公開しない。
- 旧`CSTOptimizer`、`ImplicitAdamConfig`は置き換える。過去の実験runnerも2次APIへ移行する。

```python
optimizer = CSTAdam(model, lr=0.05, trust_radius=0.25)
optimizer = CSTAdam(model, second_moment="atom_block", lr=0.05)
optimizer = CSTAdam(model, second_moment="atom_diag", lr=0.05)
optimizer = CSTSecondOrderAdam(model, lr=0.05, quartic=FullQuartic())
```

設定値をまとめる場合は、各クラス専用の`FirstOrderAdamConfig`、
`SecondOrderAdamConfig`を`cst=`へ渡す。直接の設定keywordと`cst=`の混在は拒否する。
`dense=AdamWConfig(...)`は両方で使える。

1次ではfactor対応kernelに1階factor微分を用いる。非対応kernelはgeneric経路へ
fallbackする。1次経路から2階微分を呼ばないことをcustom/hooks両方でテストする。
現在の1次solverはCPU/CUDAのfloat32/64に対応するeager実装で、MPS・deferred
device executionは対象外。今回の実測はCPUのみ。

## first moment：共通の1次比較基準

パラメータを`[K,q]`、総数を$p=Kq$、represented weightを$W$、その勾配を$g_t$とする。
旧状態は$(\alpha_{t-1},\theta_{t-1})$で、旧履歴の代表は$J_{t-1}\alpha_{t-1}$。

$$
b_t=\beta_1J_t^\top J_{t-1}\alpha_{t-1}+(1-\beta_1)J_t^\top g_t,
\qquad \widehat b_t=b_t/(1-\beta_1^t).
$$

更新問題に$\widehat b_t$を渡し、保存状態は

$$
\alpha_t=(J_t^\top J_t+\lambda I)^\dagger b_t
$$

とする。保存フレームは**更新前**の$J_t$。次stepで旧フレームから現在へ測り直す。
ゼロdamping・数値rank cutoffの範囲では現在の可視空間へのEuclidean射影であり、
dampingを入れるとridgeによる縮小を伴う。捨てた履歴の完全復元は行わない。

3種類のsecond momentとも、このfirst momentを共用する。

## second moment A：既定のseparable

勾配二乗の行・列平均をEMAして$r,c$を保存する。bias correction後に

$$
\widetilde v_{oi}=\frac{\widehat r_o\widehat c_i}{\operatorname{mean}_o\widehat r_o},
\quad D=\operatorname{Diag}(\sqrt{\widetilde v}+\varepsilon)
$$

と再構成する。ゼロのnormalizerは数値的に保護する。更新は

$$
\min_{\|d\|\le\rho}\;\widehat b_t^\top d+\frac1{2\eta}d^\top J_t^\top D J_t d.
$$

これは従来の軽量な比較基準。$v$側のstateは可視空間への射影・輸送ではない。
全原子間の結合を残し、1回の固有分解と境界時のスカラー方程式で解く。

## second moment B/C：実験用atom_block / atom_diag

原子$a$について$G_a=J_a^\top J_a=E_a\Lambda_a E_a^\top$とし、
`tangent_rtol`を最大Gram固有値に対する相対cutoffとして用いる。
有効固有値にのみ逆平方根を取り、

$$
B_a=E_a\Lambda_a^{\dagger/2},\qquad U_a=J_aB_a
$$

とする。$B_a$は座標から可視基底を再構成する係数行列。無効な列をゼロにし、固定$q$列で保存する。
有効列の$U_a$は正規直交する。

$$
T_{a,t}=B_{a,t}^\top(J_{a,t}^\top J_{a,t-1})B_{a,t-1},
\qquad S_{a,t}=B_{a,t}^\top[J_{a,t}^\top\operatorname{Diag}(g_t^{\odot2})J_{a,t}]B_{a,t}.
$$

`atom_block`は

$$
C_{a,t}=\beta_2 T_{a,t}C_{a,t-1}T_{a,t}^\top+(1-\beta_2)S_{a,t}
$$

を保存する。`atom_diag`はこの右辺の対角だけを保存し、次回は対角行列として展開する。
対角化は**輸送と現在観測の後、毎step**実施する。

$$
R_a=B_a^\top G_a,\quad
Q_a=R_a^\top[\sqrt{C_{a,t}/(1-\beta_2^t)}+\varepsilon I]R_a.
$$

更新問題は

$$
\min_{\sum_a\|d_a\|^2\le\rho^2}
\;\widehat b_t^\top d+\frac1{2\eta}\sum_a d_a^\top Q_a d_a.
$$

原子内の小行列をbatchedで固有分解し、**全原子共通の**半径に対するスカラー方程式を解く。
原子ごとの独立clippingではない。大きなblock-diagonal行列を構築しない。

新しい観測は、同一siteの繰返し利用の勾配を合算してから二乗する。
factor経路ではoutput行と原子をtile化して局所$J_a^\top\operatorname{Diag}(g^2)J_a$を計算する。
fullな可視Jacobianを保持しない。

この方式は**対角Adamとは異なる新しい計量**である。平方根と射影は交換しない。
また、second momentの輸送を同じ原子内に限定し、更新の二次項から原子間の結合を省く。
対角版は原子内の結合も省く。Gram固有基底の回転、特に縮退付近で対角近似が変化する。
低振幅で消える位置方向もrank cutoffの影響を受ける。精度が維持される保証はない。

## メモリの境界

| 対象 | 現在の保存/一時メモリ |
| --- | --- |
| 1次first moment永続state | $O(Kq)$（係数と旧フレーム。共通frame型のゼロdisplacementも保存） |
| separable second moment永続state | $O(O+I)$ |
| atom_block second moment永続state | $Kq+2Kq^2$ scalars（旧点・基底係数・計量） |
| atom_diag second moment永続state | $2Kq+Kq^2$ scalars（対角化しても基底係数は必要） |
| first momentの輸送・再圧縮 | **現状は$O((Kq)^2)$のfull Gram/cross Gramが残る** |
| separable更新の求解 | $O((Kq)^2)$の行列と固有分解 |
| atom block/diag更新の求解 | $O(Kq^2)$の小行列、共通の半径求解 |
| factor微分cache | $O(Kq(I+O))$。幅と原子数を同時に増やすと線形とは限らない |

generic経路は可視Jacobianを作るreference実装。separable metricも現実装では
一時的な$O(OI)$の対角tableを持つ。したがって、この再構成はoptimizer全体の
線形メモリ化を完了したものではない。既存の遅いPCGを既定へ移す変更もしていない。

## checkpointと更新の契約

### Autogradからの観測

optimizer構築時にmomentの`AtomGradRequest`を合成し、各siteの
`atoms.grad`へ`ImplicitLinearAtomGrad`を取り付ける。
`atom_grad_mode="auto"`では対応するcustom backwardを使い、`"hooks"`も選択できる。

- 1次版は`jg`（$J^\top g$）を要求する。custom backwardでは既に計算された
  parameter gradientを再利用する。
- 2次版はさらに`gh`（$g$と表現Hessianの縮約）を要求する。
- separableは`row_square`、`column_square`を要求する。
- atom block/diagは`atom_square`（$J_a^\top\operatorname{Diag}(g^2)J_a$）を要求する。

`jg`と要求された`gh`はbackwardで蓄積する。二乗統計はbackward中に入力と
出力勾配を保存し、`step()`冒頭のcapture完了処理で行tileごとに確定する。
同一siteを複数回使う場合は$g=\sum_j\delta Y_j^\top X_j$を合算した後に二乗し、
cross termを保持する。統計を取得後に保存した入力・出力勾配を解放する。
EMAの展開とcommitはoptimizer側の責務で、backwardで永続momentを直接進めない。
この観測経路は稼働しているが、全計算が単一の融合backward kernelで完了する実装ではない。

両optimizerとも`zero_grad → forward → backward → step`。
同一backwardからCSTとdense AdamWを提案して同時commitする。
実損失の再評価・採否判定は行わない。半径はパラメータ空間のEuclidean半径。

checkpoint version 2はalgorithm名、second-moment方式、betas、eps、first-moment
damping、rank cutoff、parameter所有manifestを検証する。
1次/2次、block/diag/separable間の暗黙のstate変換はしない。旧checkpointは拒否する。
ロード時にcomponent型・tensor shape・有限性も確認する。

## 次の判断

比較実験は companion `cst-experiments` repository で同一初期値・batch列を用いて行う。
separable、atom_block、atom_diagを比較し、`--second-order`で2次の参照も追加する。
小規模の数値一致テストと、学習精度・保存量・時間の評価を分ける。
未調整の短期実験をもって対角版またはblock版を既定にしない。

今回取り除いた未コミット変更はstash
`20e43d33cadc113209d038cbb66f38444e41b784`に退避した。
過去の実験ログを含むignoredな`output/`は削除していない。
