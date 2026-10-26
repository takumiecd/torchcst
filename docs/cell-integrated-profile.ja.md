# 区間確率profile：新しいCST数学契約の準備

2026-10-26。等間隔siteの担当区間へ確率を割り当てる、新しいprofileの設計記録。
本書は合意した目標と実装前の条件を定める。NormalizationSpecの正規化則を宣言として追加した。
対応するprofile/kernel宣言・Torch参照・CUDA実装は未追加。
旧版の離散L2正規化、既存checkpoint、既存Algorithmの意味は維持する。

関連する境界は[宣言と実行の配置規約](../src/torchcst/_backends/README.md)、
状態の所有は[AtomとAlgorithmの契約](atom-state.ja.md)に従う。
現行CUDAから何を変更するかと各関数の役割は
[区間確率128版の処理分割](research-history/cuda-linear/hierarchical-fourier-cell-mass-pipeline-20261026.ja.md)に整理する。

## 今回決めたこと

- 各atomの各座標軸について、非負のprofileをsite全体で合計すると1になることを目指す。
  入力X、振幅、atom和によるW/Yまで確率にする契約ではない。
- siteの値は点での高さではなく、siteの担当区間に入る確率質量とする。
- 正規化はprofileの支持全体の面積を使う。タイルやblockごとに正規化し直さない。
- atom側の係数とsite側の基底を分離し、入力に依存するZを複数atomで共有する。
- 積分は解析的に評価し、通常の実行で数値積分や全siteの分母reductionを要求しない。
- 新128版のforwardはprepare（ソート・配置）とcomputeの2起動とし、
  面積正規化はcompute内のdevice関数へ含める。独立した正規化用kernelを置かない。
- 新しい意味の版として宣言・保存・対応判定を区別する。package全体の版番号の確定とは別。

この目標の全体を満たすprofile族、幅の対応、境界の共有方法はまだ確定していない。
以下の式だけで、有限支持全体のZ共有が解決済みとはしない。

## NormalizationSpecでの指定

```python
from torchcst.kernels import NormalizationSpec

normalization = NormalizationSpec(
    kind="continuous_mass",
    domain="profile_support",
)
```

kindは支持全体の連続的な面積で割る規則、domainはその支持全体を表す。
入力・出力の各座標profileについて使う。atomや階層によって面積が異なってもよい。
floorはNoneのみを許す。chart_sites/operator_sitesやdiscrete_l2とは別の意味である。
この指定自体はセル評価や周期境界の方式を選ばず、対応するprofile/kernel宣言も必要になる。

現時点で作成・JSON往復できるのは正規化則の宣言までで、既存ProfileBindingや
polar_profile_productへ接続すると明示的に拒否される。点で評価する旧実装が
continuous_massをL2や無正規化と解釈して実行する経路は追加しない。

## 正規化定数とsiteへの割り当て

まず1軸について書く。中心からの相対位置をt、正規化前の密度をg_a(t)、
支持の半幅をL_a、site間隔をd、site位置をx_iとする。
g_aは支持の外で0、支持内で非負、面積が有限かつ正であることを要求する。

\[
 S_a=\int_{-L_a}^{L_a}g_a(t)\,dt>0,
 \qquad
 \phi_a(i)=\frac{1}{S_a}
   \int_{x_i-d/2}^{x_i+d/2}g_a(x-\mu_a)\,dx.
\]

g_aの原始関数を用意すれば、積分は端点の関数値の差で評価できる。
支持を跨ぐセルでは支持との交差区間を使うか、支持外で一定になる累積関数を使う。
高さにspacingを掛ける近似で置き換えない。

全セルが密度全体を覆い、取りこぼし・重複がなければ、実数演算で
sum_i phi_a(i)=1となる。浮動小数点では許容誤差を定める。

- トーラスでは空間周期P=N*dを持つN個のセルが一周を覆う。密度を真に周期化し、
  支持が一周より広い場合も折り返された質量を含めて一度ずつ数える。
  最短周期差で密度を1回評価するだけでは、この条件を一般には保証しない。
- 有限の非周期格子で支持を単にclipすると、S_aで割っても観測site上の合計は1にならない。
  端の質量の扱い、境界での再正規化、または対応範囲の制限を数学契約として選ぶ必要がある。
- 任意のPointsや部分抽出siteへ、このセル分割の保証をそのまま拡張しない。

## Fourier係数から面積を直接求める

周期2Lの整数調波で、支持内の生profileを次のように書ける場合を考える。

\[
 g_a(t)=c_{a,0}+\sum_{k=1}^{K}
 \left[c^c_{a,k}\cos(k\pi t/L)+c^s_{a,k}\sin(k\pi t/L)\right].
\]

\[
 \boxed{S_a=2L\,c_{a,0}}.
\]

非定数項の積分が0になるためで、係数から直接求められる。
S_aはatomの幅等に依存してよい。全atom・全階層で同一値とは限らない。
非負性は別途必要で、DC係数を正にするだけでは保証できない。
既存FourierPeakSpecにも非負性の一般保証はなく、単に面積で割るだけでは移行できない。

異なる周期の成分を混ぜる場合、共通の区間へこのDC式を無条件に適用しない。
各成分の実際の支持についてS_{a,r}を求め、
S_a=sum_r beta_{a,r} S_{a,r}とする。生密度を混ぜてから正規化する意味と、
各成分を先に正規化して混ぜる意味は異なり、Specで選ぶ。

S_a<=0や非有限値をepsilonで黙って置き換えない。正の面積を保証するprofile族と
パラメータ範囲を用意する。floorによる縮小は厳密な合計1の契約と区別する。

## Z共有のために保存する係数

同じ階層・bankで共有する周波数omega_{h,k}を固定する。
atomごとにomega=2*pi/w_aを変える式では、そのまま同じZを共有できない。
幅の違いを係数へどう表すかは、profile族を選ぶ際の未確定事項である。

セル全体に同じFourier式を使える場所では、次のように分離できる。

\[
 \int_{x_i-d/2}^{x_i+d/2}\cos(\omega(x-\mu_a))\,dx
 =J(\omega,d)
 [\cos(\omega\mu_a)\cos(\omega x_i)
  +\sin(\omega\mu_a)\sin(\omega x_i)],
\]

\[
 J(\omega,d)=2\sin(\omega d/2)/\omega,
 \qquad J(0,d)=d.
\]

セル積分の係数Jはatomに依存しない。site側の基底へ含めても、atom側の係数へ
含めても同じ演算になるが、実装契約で配置を固定して二重に掛けない。
正規化したatom係数をA_{a,k}として、内部の寄与は

\[
 \phi_a(i)=\sum_k A_{a,k}B_{h,k}(i),\quad
 Z_{b,h,t,k}=\sum_{i\in t}X_{b,i}B_{h,k}(i),\quad
 H_{b,a}=\sum_{t,k}A_{a,k}Z_{b,h,t,k}
\]

と構成する。式のtはatomに必要な入力タイルを合計する範囲。
有限支持の窓条件と、支持端がセルを横切る部分はこの式だけでは分離できない。
cosが端で0になることも窓条件を消さない。境界残差等の正確な処理を別に定める。

## prepare_normsに代わる準備

現行block128のprepare_normsは、各atomの入力・出力profileを全128 sitesで評価し、
各側に二乗normと中心・幅の微分用内積を保存している。
新方式では同じ配列の中身だけを変更せず、正規化処理と配列の意味を区別する。
以下はcompute内部で使うdevice関数の設計上の仮名で、独立したkernelや公開APIではない。
forwardはprepare（ソート・配置）→computeの2起動とする。

```cpp
make_atom_profile(atom, axis, spec)
{
    // 幅・支持・階層・bankと、その基底に対する生係数を求める。
    raw = build_raw_profile_coefficients(atom, axis, spec);

    // Xにもbatchにも依存しない、支持全体の面積。
    mass = integrate_full_support(raw);

    // DCを含む係数全体を同じ面積で割る。
    normalized = normalize_coefficients(raw, mass);

    // 支持端・周期境界は係数とは別の情報として保持する。
    return {normalized, support_descriptor};
}
```

| 情報 | 数学上の所有・寿命 | 保持候補 |
| --- | --- | --- |
| profile族、正規化則、セル定義、周期・支持則、幅・微分の解釈 | 新SpecとChart等の宣言。モデル復元に必要 | checkpointの宣言 |
| 正規化済み係数、必要ならその微分係数 | 現在のatom値と宣言から再構築。X/batchに非依存 | Algorithmのglobal、利用時register/shared |
| S_a、1/S_a、面積の微分 | 係数正規化へ融合できる。独立bufferは必須でない | register、再計算、必要な場合のみ小global配列 |
| 支持・ソート・境界descriptor | 現在の中心・幅等に依存する派生情報 | Algorithmのglobal等 |
| セル積分の基底、J | 階層・bank・spacing・dtype等に依存 | 共有表、利用時shared等 |
| Z/H/C | XまたはdYにも依存 | 呼び出しごとの保存・再計算方針に従う |

係数・分母の二重保存や、旧normと新massを常時両方持つことを前提にしない。
どの値をglobalへ残すかは実装ごとに示す。optimizer更新で係数・面積・支持が変われば
再準備する。同じlayout_versionという理由だけで数値cacheを再利用しない。

## backwardに必要な正規化の微分

Q_{a,i}をセル内の生の積分とする。任意の微分対象thetaについて、

\[
 \partial_\theta\phi_a(i)
 =\frac{\partial_\theta Q_{a,i}}{S_a}
  -\frac{Q_{a,i}\,\partial_\theta S_a}{S_a^2}.
\]

正規化係数A=c/Sの微分も同じ商の法則を使う。
S=2L*c0ならdS=2L*dc0+2c0*dL。
完全な支持を積分する移動だけならSは中心muに依存しないが、有限領域でclipして
その領域の質量で再正規化する設計では、一般に中心微分も必要になる。
セル・支持・周期境界と階層遷移の微分規則を明示する。
幅がtask VJPの微分対象か、既存Polarのstop-gradient則を維持するかは、
新しいparameterization/update契約として決める。旧版から暗黙に変更しない。

## 新しい版として実装する順序

1. profile族と係数、非負性、wとsigmaの関係、最小幅、支持・周期境界を確定する。
2. 新profile/kernel Spec/presetとセル評価の宣言を追加し、数学契約のID/revisionを区別する。
   NormalizationSpecにcontinuous_mass/profile_supportを追加済みだが、現行ProfileBindingは
   非対応として拒否し、KernelSpec.profile_productもoperator_sitesのL2とfloorを必須にする。
   正規化則の宣言追加だけで実行可能にしない。
3. Torch直接参照と、独立した積分oracleでphi/W/Y・dX・微分対象の全atom勾配を検証する。
4. 宣言・checkpoint往復と既存L2版の回帰を確認し、新契約に対応するAlgorithmだけを選ぶ。
5. その後、正規化係数の準備、Z/H/C、境界処理をCUDAへ接続して実機検証する。

保存では数学契約のID/revisionを残し、package版、Algorithm revision、配置形式の版と
混同しない。旧checkpointを新方式へ読み替えない。backend変更時も同じ新Specを復元する。
旧L2版と同じatom値でもWは一般に変わるため、同じ意味のfallback・精度比較とは扱わない。

受け入れ条件には、全siteの合計1・非負性、中心移動、最小/広い幅、周期を跨ぐ支持、
支持端のセル、階層遷移、多次元の完全直積、有限境界の規則、保存/再計算と勾配を含める。
直接のセル積分と階層版が同じ演算になることを、性能測定の前に検証する。

## 宣言追加時の検証

2026-10-26、kernel/hierarchical-fourier-tilesで確認。
test_kernel_declarations、test_profile_product、test_regular_grid、test_periodic_grid、
test_effective_fourier_profile、test_hierarchical_fourier_profileのCPU検証は
352 passed / 1 skipped。既存torch.jit由来の非推奨警告18件。
新規則のJSON往復、floor/領域の拒否、旧binding/presetへの誤接続の拒否と、
既存L2の参照計算・勾配を含む。変更PythonファイルのRuff検査・整形検査も通過した。
この結果は新しいセル積分profileの数値検証やGPU検証を意味しない。
