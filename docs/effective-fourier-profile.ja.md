# Iを直接指定するFourier profile

`kernel/hierarchical-fourier-tiles`のCPU数学参照。
既存の`DyadicFourierSpec`とは別の宣言で、既存checkpoint・sigmaの解釈は変更しない。
形状、周期の選択、支持、正規化を区別する。

## 宣言と実装の境界

- `FourierPeakSpec(modes=4 | 8)`は形状族と係数の定義だけを知る。
- `IntegerFourierPeriodSpec(tile_width=T, minimum_half_period=L_min,
  transition_fraction=eta)`は周期と接続の数学的規則を知る。
- `I`と中心`mu`は評価時のTensor。atomごと・軸ごと・stepごとに異なってよい。
- `T > 0`、`L_min > 0`、`I > 0`は独立。従属するのは整数`k`と`L=k L_min`。
  `T`は物理的な論理タイル幅で、CUDAのthread数ではない。
- 支持の選択と`NormalizationSpec`は評価時に明示する。

現時点では`_backends/torch/kernels/effective_fourier.py`を使う研究参照である。
CSTLinear preset、既存sigma parameterization、checkpoint、CUDA dispatcherへの
接続は実装していない。宣言の公開はCUDAへの採用を意味しない。

## Iと形状

Iを「母形状がゼロになる半幅」として直接指定する。
Fourier近似の外側がどれだけ小さいかは別に検証する。

\[
 P_I(r)=\begin{cases}(1-r^2/I^2)^4,&|r|<I,\\0,&|r|\ge I.\end{cases}
\]

これは既存の四乗多項式母形状をIで書き直したもの。sigmaは新profileの入力ではない。
母形状だけを旧幅と対応させるなら`I=sqrt(11)*sigma`だが、近似後の幅を
その関係で保証するものではない。

\[
 q=I/L,\quad\theta=\pi(x-\mu)/L,\qquad
 F_M(\theta,q)=\sum_{n=0}^{M}b_n(q)\cos(n\theta).
\]

基本周期は`2L`。Mは非零周波数の上限であり、DCを含めて係数はM+1個。
係数を次のように一意に定める。まず母形状の通常のcosine係数を求める。

\[
 c_n(q)=\frac{(2-\delta_{n0})q}{2}
 \int_{-1}^{1}(1-u^2)^4\cos(n\pi qu)\,du.
\]

`W=diag(2,1,...,1)`として、連続L2に対応する重み付き射影を行う。

\[
 b=\arg\min_{Ab=d}\frac12(b-c)^TW(b-c)
  =c-W^{-1}A^T(AW^{-1}A^T)^{-1}(Ac-d).
\]

制約は`F(0)=1`と、M=4では`F(pi)=F''(pi)=0`、
M=8ではさらに`F''''(pi)=0`。
具体的にはAの行を`(-1)^n(n/M)^(2j)`（M=4: j=0,1、M=8: j=0,1,2）と
最後の全1行、dを最後だけ1とする。
実装は解析的な積分と小引数級数を使い、係数と`db/dq`を返す。
テストは独立の192点Gauss積分と全KKT方程式のsolveを使う。

Fには負の裾もあり、非負性や`F(pi*q)=0`は制約していない。
したがって「Iより外がほぼゼロ」は全Iに自動成立する契約ではない。
Mと`q=I/L`ごとの裾の振幅・エネルギーで適用範囲を判断する。

## 周期の安全条件

有効区間`[mu-I,mu+I]`と交差するタイルを丸ごと選ぶと、端は

\[
 a=o+T\lfloor(\mu-I-o)/T\rfloor,\qquad
 b=o+T\lceil(\mu+I-o)/T\rceil
\]

である。周期的に隣接する山の有効区間が`[a,b)`へ入り込まないための、
中心の位相に依存しない条件は

\[
 \boxed{2L\ge 2I+T},\qquad
 \boxed{k_{\min}=\left\lceil\frac{I+T/2}{L_{\min}}\right\rceil}.
\]

これは幾何条件であり、Fourier近似の裾の小ささまで保証しない。
背景と反例は[周期条件の整理](research-history/cuda-linear/fourier-effective-width-period-20261021.ja.md)を参照。

## 最小周期の即時切替と滑らかな接続

`eta=0`は常に最小kを使う比較対象。ただしIを連続に変えても、
周期の切替でFの形が一般に変わる。

`0 < eta < 1`では、旧周期が安全なうちに次の周期へ移す。
`I_k=k L_min-T/2`、`Delta=eta L_min`として

\[
 z=\operatorname{clamp}\big((I-I_k+\Delta)/\Delta,0,1\big),\quad
 w=6z^5-15z^4+10z^3,
\]
\[
 f=(1-w)f_{kL_{\min}}+w f_{(k+1)L_{\min}}.
\]

移行区間は`[I_k-Delta,I_k]`。両成分とも幾何条件を満たす。
I=I_kで次の周期への移行を終えるので、旧周期を安全限界より先まで使わない。
接続点でwの1階・2階微分は0となり、支持とL2床の問題がないところではC2につながる。
I微分には**係数の変化とwの変化の両方**を含める。

これは最小kの即時切替とは異なる数学的関数である。
移行中は二つの周期を評価し、移行完了時には最小安全周期より長い成分を使う。
空間窓の追加ではない。混合で長いLが入る分、qが小さくなり形状誤差が増える場合がある。
混合中は単一周期のFourier級数とはせず、それぞれの周期・支持を持つ二成分として扱う。

## 支持は二つを混同しない

`support="centered"`は各成分を`|x-mu|<L`で評価し、その外側を0とする。
これが従来の「支持[-L,L]、周期2L」に対応する参照である。
端点制約により、0への接続はM=4でC3、M=8でC5。
中心の移動や周期混合も滑らかに評価できるが、GPUでの全タイル処理とは別の課題が残る。

`support="whole_tiles"`はCPU比較用の別関数。
有効区間と交差する**全タイル**`[a,b)`の中で周期関数を評価し、その外側を0にする。
`|x-mu|>L`の周期的な裾も含むことがあり、centeredと同じ関数ではない。
隣の有効な山が入らなくても、タイル追加・除外時に小さい裾が出入りする。
裾が小さくなければ段差も大きい。通常のautogradのbranch勾配はこの段差を微分しない。
**周期接続をC2にしても、全タイルの選択による段差は消えない。**

## 実際のsiteに対するL2

\[
 N=\sqrt{\sum_{s\in\mathrm{chart}}f_s^2},\quad
 \widehat f_s=f_s/\max(N,\varepsilon).
\]

ノルムが床より上なら

\[
 \partial_p\widehat f_s=\frac{\partial_p f_s}{N}
 -\frac{f_s}{N^3}\sum_t f_t\partial_p f_t,
 \quad p\in\{\mu,I\}.
\]

床より下では分母は固定epsilon。実装はTorchのclamp規約と一致させる。
有限領域・格子位相・裾・周期混合をすべて含むsite和で正規化する。
単一周期の連続積分`L*(2*b0**2+sum(bn**2))`は参考量であり、離散ノルムの代用ではない。
profile productでは全軸を掛け合わせたatom全体に一度だけ床を適用する。

```python
import torch
from torchcst import FourierPeakSpec, IntegerFourierPeriodSpec, NormalizationSpec
from torchcst._backends.torch.kernels.effective_fourier import values_and_derivatives

profile = FourierPeakSpec(modes=8)
period = IntegerFourierPeriodSpec(
    tile_width=0.4, minimum_half_period=0.15, transition_fraction=0.25,
)
x = torch.arange(-40, 41, dtype=torch.float64) * 0.1
mu = torch.tensor([0.013, 0.37], dtype=torch.float64, requires_grad=True)
I = torch.tensor([0.332, 0.61], dtype=torch.float64, requires_grad=True)
y, dmu, dI, raw_norm = values_and_derivatives(
    profile, period, x, mu, I, support="centered",
    normalization=NormalizationSpec(
        kind="discrete_l2", domain="chart_sites", floor=1e-12,
    ),
)
# y, dmu, dI: [sites, atoms]. autograd through y is also supported.
```

## 精度条件とタイル集約

裾の条件から許容するqの下限`q_*`を別途定めるなら、
幾何条件だけでなく`L <= I/q_*`も必要になる。
即時切替では

\[
 \left\lceil\frac{I+T/2}{L_{\min}}\right\rceil L_{\min}
 \le\frac I{q_*}
\]

を満たす必要がある。整数丸めを無視しても必要条件は
`I >= q_* T/(2*(1-q_*))`。
混合中は長い方の周期も確認する。任意の独立したI・T・L_minが、固定Mで
同じ精度を持つわけではない。小さいIには小さい論理T・小さいL_minが必要になり得る。

同じ周期とタイルを使うatomでは、`sum X`と周波数ごとの
`sum X*cos(n*pi*x/L)`、`sum X*sin(n*pi*x/L)`を共有できる。
必要な実数momentは軸ごとに`2M+1`個で、M=4なら9個、M=8なら17個。
中心は位相回転、Iはb(q)で反映するので、中心・Iを共有する必要はない。
移行中は二つの周期のmomentが必要。異なる周期間の完全な再利用はこの数式だけからは
保証されず、階層Algorithm側で別に検証する。

[CPU検証結果と適用限界](research-history/cuda-linear/effective-fourier-cpu-20261021.ja.md)を参照。

## I=0.1〜10の階層選択

`HierarchicalFourierPeriodSpec`を別の周期選択規則として追加した。
元の`IntegerFourierPeriodSpec`の最小kの規則は変えない。
共通の独立した`T_min`と`L_min`に対し、階層jを

\[
 T_j=2^jT_{\min},\quad r_j=2^j,\quad s_j=r_jL_{\min},\quad
 k_j=r_j\left\lceil\frac{I+T_j/2}{s_j}\right\rceil
\]

と定める。使用するLは常に整数kと共通のL_minの積である。
`k_stride="fixed"`はr_j=1の比較対象。
profileの形状にはTやr_jを埋め込まない。

周期内混合の幅を`eta*s_j`とすると、非零成分のLは
`L <= I+T_j/2+(1+eta)*s_j`となる。
そこで精度判定用のq下限を`q_*`とし、階層の使用開始Iを

\[
 A_j=\frac{T_j/2+(1+\eta)s_j}{1/q_*-1}
\]

と置く。I>=A_jなら、周期混合中を含めて全非零成分が
`2L>=2I+T_j`と`I/L>=q_*`を満たす。
I>=A_0を要求し、未満なら基底T_minまたはL_minを小さくする。
**q条件から実際の裾の小ささを保証するには、形状族の別検証が必要**である。

最も粗い使用可能階層jを選ぶが、I=A_jで即時に変更しない。
旧階層j-1からjへの混合を

\[
 I\in[A_j,A_j+\gamma(A_j-A_{j-1})]
\]

でquintic smoothstepにより行う。両階層が使用可能になってから開始する。
0<=gamma<1で接続区間は重ならない（A_jの差は減少しない）。
既定eta=0.25、gamma=0.3。
階層混合2成分×周期混合2成分で最大4成分。
支持を適用した成分をすべて足し合わせた**後**に離散L2を一度だけ適用する。
微分には内外の重みの変化を両方含める。

```python
from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec
from torchcst._backends.torch.kernels import hierarchical_fourier

period = HierarchicalFourierPeriodSpec(
    minimum_tile_width=0.1,
    minimum_half_period=0.025,
    relative_width_floor=0.54,
    k_stride="dyadic",
)
# x, mu, Iは上と同じTensor。
y = hierarchical_fourier.values(
    FourierPeakSpec(modes=8), period, x, mu, I,
    support="centered", normalization=NormalizationSpec(
        kind="discrete_l2", domain="chart_sites", floor=1e-12,
    ),
)
```

Tの変更も数学的な選択規則の一部である。特にwhole_tilesでは支持が変わるので、
単なるlaunch設定の変更とはしない。中心がタイル境界をまたぐ有限の裾の段差は残る。

小さい支持では、同じFを各siteで局所直接評価する計算方式も使える。
そのfをHと離散ノルムに使い回せば、momentとGramの構築を省ける。
計算方式の切替はF、支持、L2の意味を変えず、学習するI・中心を固定しない。
[全域の形状・勾配・容量・共有と直接評価の比較](research-history/cuda-linear/hierarchical-fourier-range10-20261021.ja.md)
を参照。


実行方式の幅閾値`I_cut`と、形状を決めるT_minは別に扱う。
`I<I_cut`は同じFの局所直接評価、以上は十分大きいタイルの集約候補とする。
T_min=0.2、spacing=0.1の2点にM8の17実成分を保存しない。
保存側は例えば64/128点以上、共有atom数と全周期の合計byteを検査し、
見合わない場合は直接評価する。これは研究CPU prototypeの実行規則であり、
公開specの形状・支持を変更せず、CSTLinear dispatcherにはまだ統合しない。
