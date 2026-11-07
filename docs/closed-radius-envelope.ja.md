# 両端で合流する半径境界（研究用opt-in）

[English（正本）](closed-radius-envelope.md) | 日本語

振幅ゼロと振幅上限でL・Uを合流させる実験的なPolar半径写像です。
半径の契約を変更します。Polarの活動度更新と、タスク勾配で半径をdetachする
契約は維持します。推奨既定値ではありません。

`x = abs(a) / amplitude_max`、`b = clamp(1-x*x, 0, 1)`として、

$$L=r_{min}+(r_{max}-r_{min})b^{p_L},\qquad
U=r_{min}+(r_{max}-r_{min})b^{p_U},\quad p_L\ge p_U\ge1.$$

半径は引き続き`exp((1-q)*log(L)+q*log(U))`で、丸め誤差を`[L,U]`にclampします。
ゼロでは両境界が`r_max`、振幅上限では`r_min`です。中間ではqが半径を選び、
両端ではqの半径に対する自由度がなくなります。各指数を増やすと対応する境界が
早く下がります。探索範囲・初期q・学習の改善を実証したものではありません。

```python
coordinates = kernels.polar(
    amplitude_max=1.0, w_c=0.3,
    bounds=RadiusBounds(minimum=0.5, birth=50.0, maximum=50.0, upper_floor=0.5),
    radius_envelope="closed_polynomial",
    closed_lower_power=4.0, closed_upper_power=1.0,
)
```

両軸とも`RadiusBounds`で`birth=maximum`、`upper_floor=minimum`を指定します。
新しい式では有理式のw_c/kappa/減衰設定は使いません。既存のPolar表現・更新用の
宣言として残ります。既定の`radius_envelope="rational"`は従来の半径式とbuffer構成を
保持し、optimizerのcheckpoint契約も維持します。新しい指数は固定checkpoint buffer、境界の種類はcheckpoint契約で、
種類をまたぐloadは拒否します。

CUDAへ移す前にCPUで`init.variance_scaling_`を実行します。半径を変えたら振幅を
再調整してDenseの同じ二次モーメントに合わせ、誤差・clippingの報告を残します。
有界探索で目標を達成できない場合は拒否されます。未調整の代替値で学習を続けません。

native対応は明示的な`GLOBAL_ACTIVE_GRAD4_NATIVE`と
`GLOBAL_ACTIVE_GRAD4_NATIVE_RECTANGULAR_2048`に限定し、他のnative scheduleは
この宣言を拒否します。学習前に実際の条件でY・dX・全Atom勾配とoptimizer・checkpointを
検証します。CPUの数学的検証はGPU検証や学習結果を意味しません。

## ゼロから徐々に開く境界

別の明示的な研究用設定 `radius_envelope="rational_zero_opening"` は、
従来の有理式の上限 U0 を保ち、下限 L0 を次のように補間します。

$$g(a)=\frac{a^2}{a^2+\tau^2},\qquad L=U_0-g(a)(U_0-L_0),\qquad U=U_0.$$

`zero_opening_scale=tau` に正の振幅単位の値を指定します。ゼロで L=U、
|a|=tau で相対的な隙間が半分になり、離れるほど従来の区間へ徐々に戻ります。
amplitude_max 側の合流は強制しません。上の宣言で `birth=4`, `upper_floor=1`,
`radius_envelope="rational_zero_opening", zero_opening_scale=0.03` とし、
closed powers を省く例です。tau は保存される固定スカラーで、学習する q とは別です。
CPU/CUDA は数値的に安定な等価式と端点補間を使います。同じ native schedule 制限、
分散校正、検証規則を適用します。既存の rational / closed-polynomial の
optimizer checkpoint manifest は保持します。
