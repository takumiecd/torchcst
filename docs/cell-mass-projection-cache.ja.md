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

## 検証と初回比較

source commit `4c435bfd`。CPU全体2006 passed / 3057 skipped、既存警告18件（43.51秒）。
wheel/sdistの321 runtime source filesが現行sourceとbyte一致。
NVIDIA L4、Torch2.11.0+cu130、CUDA13.0で対象309 passed / 0 failed / 0 skipped。
CPU参照勾配と同じGPU fused AdamWによる更新検査も全方式で通過した。
主測定caseのCPU参照への最大絶対誤差はY5.96e-8/dX4.47e-8/全atom勾配5.96e-8。
Parameter更新deltaとstepは差0、moments差も変更前と同じ基準内だった。
FP64・DFT・M16・空入力・保存状態等は309件の独立検査に含む。

N128/A819/B4・FP32・mode8・sigma0.2〜12・seed120、profile tile16/32、周期48/96、
merge Z・16site/8split/16thread。TF32=False、Graphなし、GPU fused AdamWの条件は
[dense step比較](cell-mass-dense-step.ja.md)と同じ。各2fresh process、2巡目は逆順、
warmup10回後30サンプル。検査・prepare・partial合計・更新を含む同期wall時間。

| 方式 | AdamW込みstep process1 / process2 | allocated peak | reserved peak | 同じjobのCST/dense比 |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.469 / 0.444ms | 17,310,208byte | 22MiB | 1 |
| 変更前CST | 1.366 / 1.375ms | 115,712byte | 2MiB | 2.91 / 3.10 |
| 数式整理・再計算 | 0.987 / 0.975ms | 115,712byte | 2MiB | 2.11 / 2.20 |
| 数式整理・微分cache | **0.838 / 0.832ms** | **204,800byte** | **2MiB** | **1.79 / 1.87** |

cacheは再計算より約15%速く、allocatedが113→200KiBへ87KiB増えた。
追加配列104,832byteよりpeak増分89,088byteが小さいのはZ等の寿命も変わるため。
既定Falseは維持し、これは当該caseの明示設定として記録する。
denseには大きいBLAS等の常駐割当が含まれるため、この差を圧縮率とはしない。
メモリはdense以下を必須条件とする。追加した
[workspaceなしdense・batch16との比較](cell-mass-memory-frontier.ja.md)で、
B4は200<264.5 KiBだがB16の保存案は制約超過と確認した。
dense既定の大きな割当は、隔離プロセスでclearしたcuBLAS workspaceの16.25 MiBと特定した。

別のstage GPU event中央値は次の通り。和でstep wall時間を代用しない。

| stage | 変更前 | 数式整理・再計算 | 微分cache |
| --- | ---: | ---: | ---: |
| H（cacheでは二つの微分も生成） | 0.165ms | 0.0932ms | 0.134ms |
| G（同上） | 0.168ms | 0.0942ms | 0.134ms |
| Parameter勾配 | 0.412ms | 0.220ms | 0.0236ms |
| Y | 0.217ms | 0.182ms | 0.181ms |
| dX | 0.186ms | 0.160ms | 0.160ms |

数式整理の個別項目ごとの寄与は分離していない。再計算行はbank定数・sqrt・共通因子・
sin/cos対の共有をまとめた比較。cache行との差は微分保存によるもの。
コンパイル表示はH/Gの通常REG82、微分付きREG112、再計算勾配REG164、cache勾配REG48。
Y/dXはREG94・SHARED14,336byte、表示上LOCALは全て0。実trafficのcounterではない。

## 独立jobでの再確認

runtimeを固定し、別L4 jobで同じstepを再測定した。
短いCPU参照勾配・更新検査を再実行し、runtime不変のため309件とCPU/buildを重ねていない。
各2fresh process、warmup10回後30サンプル、2巡目は逆順。

| 方式 | process1 / process2 | CST/dense |
| --- | ---: | ---: |
| dense | 0.465 / 0.475ms | 1 |
| 再計算 | 0.991 / 0.980ms | 2.13 / 2.06 |
| 微分cache | **0.850 / 0.844ms** | **1.83 / 1.78** |

allocated/reservedは初回と同じ。元のdense既定workspace条件のこのB4 caseでは、
目標の約2倍以内を独立jobでも確認できた。小メモリdense・他batchは
[メモリfrontier](cell-mass-memory-frontier.ja.md)の別jobで検査した。

```python
y = physical_linear(
    spec, x, atoms, z_method="merge", sites_per_block=16,
    atom_splits=8, atom_threads_per_block=16, cache_projection_derivatives=True,
)
```

## 証拠

初回job `l4job-07162dad4db846b2a2c5f23c5f5a4d3a`。

- source SHA256: `59e969eb54ed87799d75ed59077db9419a7fbedb72e618f0261df267065d12ca`
- result SHA256: `48c2a800a4e9f46a126ddfc92a48716f25e13366380ca7e455e1ad85e2058a44`
- driver SHA256: `369bc433db20c12cd8015d1e7036050b57116452d0f79721002bef1213e186bc`
- binary SHA256: `d79a5b7409692617aed1312a62a9a6d34d59a37d293dd8a72739d64fff4d38f0`

確認job `l4job-2e653f97e561424a958c05c52eabb604`。

- source SHA256: `e4a09292159ca5a2f7be2bf6e1ecc874890e701d2b2dbed0c2267c632aa18a0d`
- result SHA256: `a647cdf524e2e24b54f29489068e88caf8d02d57d427c5f258e60698a4930f40`
- driver SHA256: `a915d3ae79dd490ec29f7ce4d97bd6f3450d97818c4fc42d0eed8914185c134f`
- binary SHA256: `779199fa0065302954308bf632bb7c87d4c423927eb5acb77fb44091f36c7911`

両jobのsource/result archive、32/17result manifest、321 runtime source filesを照合。
回収binaryと各jobの記録hashも一致した。raw evidenceはignored
`output/cell-mass-projection-cache-20261027/`とその`confirmation/`、原本はhost共通poolに保持。
両jobの回収と所有slot停止を確認した。再実行はそれぞれのsource archiveを展開して
`__pool_driver__.py`を共有poolへ提出する。初回timeout720、確認は480秒。
公開CSTLinear/CSTOptimizer、Graph、他shape/deviceへの統合・一般化は未実施。
