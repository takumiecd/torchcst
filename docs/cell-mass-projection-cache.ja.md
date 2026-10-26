# H/G生成と微分projectionを共有する

prepare融合後の約1.37ms/stepから、H/GとParameter勾配の重複を減らす。
目標は[denseの約2倍のstep時間](cell-mass-dense-step.ja.md)。
中間計算と保存する微分値はdoubleを保ち、数式・誤差基準・profileの窓を変更しない。

## 保存する値と勾配

入力profileをV、出力profileをU、上流勾配をdYとする。
既存のHとGに加え、profileパラメータに関する二つずつのprojectionを保持する。

\[
H_{ba}=\sum_iX_{bi}V_{ai},\quad
H^\sigma_{ba}=\sum_iX_{bi}\partial_{\sigma_a}V_{ai},\quad
H^\mu_{ba}=\sum_iX_{bi}\partial_{\mu^{in}_a}V_{ai},
\]
\[
G_{ba}=\sum_odY_{bo}U_{ao},\quad
G^\sigma_{ba}=\sum_odY_{bo}\partial_{\sigma_a}U_{ao},\quad
G^\mu_{ba}=\sum_odY_{bo}\partial_{\mu^{out}_a}U_{ao}.
\]

H/Gを作る同じthreadで値と微分projectionを計算する。
Parameter勾配はprofileを再評価せず、次のbatch方向の和だけになる。

\[
\partial_{A_a}\mathcal L=\sum_bH_{ba}G_{ba},\qquad
\partial_{\sigma_a}\mathcal L=A_a\sum_b(G_{ba}H^\sigma_{ba}+H_{ba}G^\sigma_{ba}),
\]
\[
\partial_{\mu^{in}_a}\mathcal L=A_a\sum_bG_{ba}H^\mu_{ba},\qquad
\partial_{\mu^{out}_a}\mathcal L=A_a\sum_bH_{ba}G^\mu_{ba}.
\]

sigmaは両軸に共通なので両側の項を足す。振幅0でもH/Gを計算し、振幅勾配を残す。
H/Gの値は従来のinput dtype、四つの微分projectionは常にfloat64で保存する。

## 関数の流れ

```cuda
// forward: each thread owns one atom, including all required input Z nodes.
Zx = build_z(X);
(H, H_sigma, H_center) = compute_h_with_derivatives(Zx, X, atoms);
release(Zx);
Y = compute_y(H, atoms);
save_original_X_atoms_rows_and_H_projections();

// backward
Zy = build_z(dY);
(G, G_sigma, G_center) = compute_g_with_derivatives(Zy, dY, atoms);
release(Zy);
dX = compute_dx(G, atoms);
dAtoms = compute_parameter_grad_from_cache(H, G, H_sigma, H_center, G_sigma, G_center);
```

`project_atoms_kernel<true>`はH/Gと二つの微分を同じZ読取りから作る。
`cached_parameter_grad_kernel`は各threadが一つのatomの全batchを合計し、
元のatom行へ書く。atomicやblock間の勾配集計を追加しない。
forwardごとのctxに保存し、optimizer更新を跨ぐ共有cacheにはしない。
元のXとatomのTensorを保持して、保存後のinplace変更をautogradのversion検査で拒否する。

## メモリと設定

実行設定`cache_projection_derivatives=True`で切り替える。既定はFalseで、
従来通りZx/Hを保存し、backwardで微分projectionを再計算する。
TrueならZxをbackwardまで保存せず、ZyもG生成後に手放せる。

追加する値は四つの[batch,atom] float64配列で、上限は4*B*A*8byte。
B4/A819では104,832byte。これは追加配列の合計であり、実測peakの増分ではない。
H側はforwardからbackward、G側はbackward中だけ存在し、Zの寿命短縮も影響する。
atomのコピー、全site/modeの係数表、全Jacobianは作らない。
保存量はB*Aで増えるため、全shapeでcacheが有利とはしない。

## 同じ数式の計算整理

同じharmonicのsin/cos二成分を`coefficient_pair`で一緒に計算する。
raw係数・正規化・sincosを一度だけ求め、Zの隣り合う二つのmodeへ掛ける。
nodeごとにcos/sinを足すため加算順は変わるが、独立参照の基準は維持する。

bankの固定値（tail_bound、mode、周期から決まるradiusの下限比）はhostで一度計算する。
4乗根はr=sqrt(sqrt(base))とし、半径のsigma微分に現れるbase^(-3/4)は1/r³で求める。

polynomial係数ではc_k=Q_k/(2D²)、面積P*c_0なので、

\[
\widehat c_k=\frac{2Q_k}{P Q_0},\qquad
\widehat c'_k=\frac{2}{P}\frac{Q'_kQ_0-Q_kQ'_0}{Q_0^2}.
\]

共通のD=T_M(beta)とD'を省略できる。DFT側には従来のD/D'を残す。
これらはdoubleの数式整理で、FP32/TF32への切り替えではない。

## 検証・比較

cacheのTrue/False両方でY/dX/全atom勾配、retained backward、元atomの変更拒否を検査する。
さらに微分projection単体を独立profileのautogradと照合し、元Xの変更拒否と
誤ったcache dtypeの拒否を検査する。空atom/batch、端数chunk、M16も両方式で検証する。

比較元はprepare融合を含む`9f60fc7b`。dense・変更前・数式整理後の再計算・cacheの4方式を、
同じGPU fused AdamW込みstepとピークで比較する。runtime既定ではcacheを有効化しない。
