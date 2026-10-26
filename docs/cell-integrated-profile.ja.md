# 区間確率profileの設計と正規化宣言

2026-10-26。新方式の設計は[Issue #120](https://github.com/takumiecd/torchcst/issues/120)で追う。
このブランチはmainのa4ac5e507df4b766e8b5ab6f38769bbf51d8661cから作成した。
旧研究の整理・復元先は[Issue #119](https://github.com/takumiecd/torchcst/issues/119)。

現時点で実装したのはNormalizationSpecの宣言と検査まで。
対応するセル積分profile、KernelSpec、Torch参照、CUDA、dispatch登録は未追加である。
旧L2版を新しい意味へ読み替えず、新しい数学契約と実装として進める。

## NormalizationSpecでの指定

```python
from torchcst.kernels import NormalizationSpec

normalization = NormalizationSpec(
    kind="continuous_mass",
    domain="profile_support",
)
```

continuous_massは非負密度を支持全体の正の面積で割る規則。
profile_supportはprofileの支持範囲全体で、例えば中心から見た[-L,L]を表す。
chart_sites/operator_sitesによる離散L2とは異なる。floorはNoneのみを許す。
宣言だけでセルの定義・支持・周期境界の方式を選ぶことはない。

既存ProfileBindingは点で評価するため、この指定を渡すと明示的に拒否する。
既存polar_profile_productも従来のL2契約を維持する。
新しい指定をL2や無正規化と解釈して実行しない。

## 各siteの担当区間へ確率を割り当てる

1軸について、中心mu、支持半幅L、正規化前の非負密度g、site間隔d、位置x_iとする。
gは支持外で0とし、支持全体の面積Sは有限かつ正であることを要求する。

\[
 S=\int_{-L}^{L}g(t)\,dt,
 \qquad
 \phi(i)=\frac{1}{S}
 \int_{x_i-d/2}^{x_i+d/2}g(x-\mu)\,dx.
\]

積分は原始関数の両端の差で評価し、数値積分のループを通常実行へ入れない。
担当セルが全密度を取りこぼし・重複なく覆うとき、全siteのphiの合計は1になる。
各タイルだけの合計を1にするものではない。入力X、振幅、atom和のW/Yまで確率にする契約でもない。
浮動小数点では許容誤差を定める。

有限の非周期領域で支持をclipすると質量が失われるため、境界の規則を決める必要がある。
トーラスでは一周を覆うセルと真の周期化を用い、広い支持の折り返し分も含める。
最短周期差で1回評価するだけでは、一般にはこの質量を保証できない。
任意のPointsや部分抽出siteへ、等間隔セル分割の保証をそのまま拡張しない。

## Fourierの積分とZの共有

周期2Lの整数調波で生密度を表す場合、

\[
 g(t)=c_0+\sum_{k=1}^{K}
 [c_k^c\cos(k\pi t/L)+c_k^s\sin(k\pi t/L)],
 \qquad S=2Lc_0.
\]

これは面積の式であり、非負性の保証は別途必要になる。
異なる支持・周期の成分を混ぜる場合は、各成分の実際の面積を使う。
生密度を混ぜてから正規化するか、正規化した密度を混ぜるかも数学契約で選ぶ。

同じ階層・bankでは周波数omegaを共有する。完全なセルの内部では、

\[
 \int_{x_i-d/2}^{x_i+d/2}\cos(\omega(x-\mu))\,dx
 =J(\omega,d)[\cos(\omega\mu)\cos(\omega x_i)
              +\sin(\omega\mu)\sin(\omega x_i)],
\]

\[
 J(\omega,d)=2\sin(\omega d/2)/\omega,\qquad J(0,d)=d.
\]

中心・幅と面積正規化をatom側の係数へ、セル積分基底をsite側へ分離する。
完全な入力タイルでは`Z[b,t,k] = sum_i X[b,i] * B[t,k](i)`を共有できる。
親Zを子Zから作れる条件は、周波数・原点・spacing・積分係数まで含めて確認する。
atomごとに周波数を変える式を、そのまま共通Zで扱えるとはしない。
支持で切れたセルは正確な境界積分で扱い、内部の寄与と二重に数えない。

## forwardはprepareとcomputeの2起動

```cpp
prepare<<<...>>>(atoms, spec, layout);
compute<<<...>>>(X, layout, spec, Y);
```

prepareは階層分類、ソート、pack、blockのbegin/endを作る。
atomの順序は任意でよく、実行用コピーと元のParameter行への対応を保持する。
支持とセルが重なる候補範囲を作り、境界セルを取りこぼさない。

computeの中は次のように分ける。関数名は設計上の仮名である。

```cpp
init_block();
load_block_inputs();
sync_block();
build_z_hierarchy();       // 子Z → 親Zの依存箇所でも同期する。
compute_atom_h();
publish_h();
sync_block();
build_output_c();
sync_block();
evaluate_and_store_y();
```

面積Sと正規化係数はcompute内のdevice関数で利用時に評価する。
独立したnorm準備kernelや全atom分の係数表をglobalへ保存することは前提にしない。
H担当threadのregisterを別のC担当threadが直接読めるとはしない。
係数の再評価と小さなshared chunkでの再利用は、容量・同期・時間を見て選ぶ。
Z/H/Cの寿命を分け、全員の利用終了を同期してからshared領域を再利用する。

## backwardと残る設計事項

正規化済みの入力profileをv、出力profileをuとして、

\[
 H_{b,a}=\sum_i X_{b,i}v_a(i),\quad
 G_{b,a}=\sum_o dY_{b,o}u_a(o),\quad
 dX_{b,i}=\sum_a \mathrm{amp}_a v_a(i)G_{b,a}.
\]

dY側のZは新しく作る。atom勾配にはH/Gとprofileの微分を使う。
Hの保存・再計算・一部保存、H/Gを同じthreadで作るか別の集団で作るかは選択肢として残す。
Qをセルの生の積分とすれば、正規化の微分は

\[
 \partial_\theta(Q/S)=(\partial_\theta Q)/S
                       -Q(\partial_\theta S)/S^2.
\]

2起動の決定はforwardが対象で、backwardのatom勾配合算の起動数まで固定しない。

非負profile族、sigmaから共通基底の係数への写像、支持幅・L・最小幅・階層遷移、
有限/周期境界、幅のtask VJPと更新則はIssue #120で確定していく。
新Specとsemanticsに対応したAlgorithmだけを登録し、未完成の経路を旧L2へfallbackしない。
CPU直接参照と独立oracleでY/dX/全atom勾配を確認した後に、CUDAの完全stepとピークメモリを測る。

## この準備段階の検証

mainベースの本ブランチで、CPU suiteは1978 passed / 2776 skipped / 18件の既存JIT警告。
新規則のJSON往復、領域/floorの検査、旧binding/presetが新規則を誤実行しないこと、
既存L2の計算・勾配の回帰を含む。変更PythonファイルのRuff検査・整形検査、
wheel/sdist buildも通過した。GPUや新セル積分profileの数値検証を行った結果ではない。
