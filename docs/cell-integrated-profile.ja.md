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

## 基準実装はH/Gを保存し、5つの計算kernelに分ける

2026-10-26の合意。まず処理の所有者と依存を明確にするため、H/Gをglobalへ保存する。
以前の「forwardはprepareとcomputeの2起動」という固定は、この基準実装では置き換える。
以下は設計であり、対応するCUDAコードと性能結果はまだない。
各kernelの担当・保持先と短い擬似コードは
[5つの計算kernelの流れ](cell-mass-kernel-flow.ja.md)にまとめる。

prepareは階層分類、ソート、pack、blockのbegin/endを作る。
atomの順序は任意でよく、実行用コピーと元のParameter行への対応を保持する。
支持とセルが重なる候補範囲を作り、境界セルを取りこぼさない。
layoutは再構築できる実行用情報であり、数学的なprofileの定義とは分ける。

| 計算kernel | 担当する処理 | globalへの出力 |
| --- | --- | --- |
| `compute_h` | 入力タイルの寄与を集約し、atomごとのHを完成させる | H |
| `compute_y` | Hから担当出力タイルのCを作り、Yへ展開・加算する | Y |
| `compute_g` | dY側のタイルの寄与を集約し、atomごとのGを完成させる | G |
| `compute_dx` | Gから担当入力タイルの係数を作り、dXへ展開・加算する | dX |
| `compute_parameter_grad` | H/Gとprofileの微分情報からatomの勾配を作る | Parameter勾配 |

prepareとX/dYの階層構築は、この5つに含めない。全体の起動数が5個で確定するわけではない。
階層構築をlevelごとのkernelにするかなど、実際の起動単位は依存と容量を見て決める。
関数名は仮名で、次の擬似コードも未実装である。

```cpp
// forward: 同じstreamへ依存順に投入する。
prepare_layout(atoms, spec, layout);
build_input_hierarchy(X, layout, Zx);
compute_h(Zx, X, atoms, layout, H);
compute_y(H, atoms, layout, Y);

// backward: forwardと同じatomの値・配置に対応する状態を使う。
build_output_hierarchy(dY, layout, Zy);
compute_g(Zy, dY, atoms, layout, G);
compute_dx(G, atoms, layout, dX);
compute_parameter_grad(Zx, Zy, X, dY, H, G, atoms, layout, grad);
```

X/dYは必要な支持境界の補正にも使う。Zxをbackwardまで保存しない場合は、
`compute_parameter_grad`の前に同じforward入力・状態から再構築する。
基準実装は同一stream上のkernel完了を依存境界にし、block間のspin待ちを導入しない。
`compute_dx`と`compute_parameter_grad`はG完成後の別の利用先であり、互いの出力には依存しない。
別streamへ分ける場合のeventと実際の同時実行は、後の検討事項とする。

### 各blockとthreadが完成させる値

H/Gは`[batch, atom]`の完成値であり、必要な全タイルの寄与を足してから保存する。
基準案ではblockがatom群とbatch範囲を担当し、各threadへ別々のatomを割り当てる。
各atom担当が必要なタイルを読み、担当batchのHまたはGを完成させて一度書く。
一つのatomを複数threadへ分割することは基準にしない。
一つのH/G要素を複数blockが書かないので、ここでblock間reductionは不要になる。
大きな支持で分割した方がよい場合は、部分和bufferと集約段階を明示した別候補とする。

`compute_y`ではblockがbatch範囲と出力タイルを所有する。
担当範囲へ寄与する全グループを読み、階層・基底bankごとにCを局所生成する。
異なる基底のCをそのまま足さず、それぞれをsite値へ展開してYの累積値へ足す。
担当Y要素を完成させて一度保存する。`compute_dx`も入力タイルを所有して同様に処理する。
この所有方法では、Y/dXへのblock間atomicや、全Cのglobal保存を前提にしない。

Cはregister/shared、展開後の累積値は容量に応じてregister/sharedへ置く。
sharedへ書いた値を別threadが読む箇所にはblock内同期を置く。
全threadの最後の利用を終えてから作業領域を上書きする。
thread数、atom群の大きさ、batch範囲、タイル寸法は調整可能にする。
128サイト・128atom・batch 4の試作は説明と検証の出発点であり、固定の最適値ではない。

## パラメータ勾配とZの再利用

正規化済みの入力profileをv、出力profileをu、振幅をalphaとして、

\[
 W_{o,i}=\sum_a\alpha_a u_a(o)v_a(i),\qquad
 H_{b,a}=\sum_i X_{b,i}v_a(i),\qquad
 G_{b,a}=\sum_o dY_{b,o}u_a(o).
\]

H/Gは振幅を掛ける前の値として固定する。

\[
 dH_{b,a}=\alpha_a G_{b,a},\qquad
 dX_{b,i}=\sum_a \alpha_a v_a(i)G_{b,a},\qquad
 \frac{\partial\mathcal L}{\partial\alpha_a}=\sum_b H_{b,a}G_{b,a}.
\]

Gを保存する基準では、dH全体を別途保存しない。dX計算で振幅を二度掛けない。
振幅が0でも振幅勾配は非0になり得るので、値が0という理由でH/Gを除外しない。

H/Gだけで位置・幅の勾配まで求まるわけではない。
入力側だけに効くパラメータtheta_inと、出力側だけに効くtheta_outなら、

\[
 \frac{\partial\mathcal L}{\partial\theta_{\rm in}}
 =\alpha_a\sum_b G_{b,a}\frac{\partial H_{b,a}}{\partial\theta_{\rm in}},
 \qquad
 \frac{\partial\mathcal L}{\partial\theta_{\rm out}}
 =\alpha_a\sum_b H_{b,a}\frac{\partial G_{b,a}}{\partial\theta_{\rm out}}.
\]

両側に効くパラメータには両方の項を足す。
共通基底で`H = sum_(t,k) c[a,t,k] * Zx[b,t,k]`と分離できる部分では、
`partial_H = sum_(t,k) partial_c[a,t,k] * Zx[b,t,k]`として同じZxを使う。
Gとその微分にはdYから作るZyを使う。微分専用のZ全体を新たに作ることは前提にしない。
支持端で切れたセルの補正と、支持・階層の境界における微分の扱いは別途確定する。

Qをセルの生の積分とすれば、係数・境界補正に使う正規化の微分は

\[
 \partial_\theta(Q/S)=(\partial_\theta Q)/S
                       -Q(\partial_\theta S)/S^2.
\]

`compute_parameter_grad`はatomの勾配の担当を一意にし、全batchの寄与を合計する。
試作のbatch 4では担当内で合計する。大batchで複数blockへ分ける場合は、
部分Parameter勾配の集約を追加するため、常に5つの計算kernelで済むとはしない。
勾配はlayoutの対応を使って元のParameter行へ戻し、optimizerの状態を並び替えない。

## 保存量とHの寿命を切り替える選択肢

| 値 | 基準での保持先 | 必要な期間 |
| --- | --- | --- |
| H | global、`[batch, atom]` | forwardの生成から、Y生成とbackwardの最後の利用まで |
| G | global、`[batch, atom]` | backwardの生成から、dXとParameter勾配の最後の利用まで |
| Zx | 複数blockで共有するならglobal | H生成、保存する場合はParameter勾配の最後の利用まで |
| Zy | 複数blockで共有するならglobal | G生成からParameter勾配の最後の利用まで |
| C・入力側の展開係数 | 担当blockのregister/shared | 対応するタイルの展開が終わるまで |
| layout・X・forward状態 | 呼び出しごとの状態 | backwardの必要な読み取り・再計算が終わるまで |

H/Gを両方保存する値だけの容量は、dtypeのbyte数s、batch数B、atom数Aに対し`2*s*B*A`。
FP32、B=4、A=128なら合計4096 byte（4 KiB）。
Zx/Zy、layout、X/dY、出力、勾配、optimizer、Graph/allocatorの容量はこの数字に含まない。
H/Gが小さいことだけで、完全stepのピークメモリが小さいとは判断しない。
保存状態はAlgorithmインスタンスではなく、forwardごとのautograd状態が所有する。
retained backwardに必要な状態はその契約に従って保持する。

Hの保持には、次の三つを区別する。

1. **backwardまで保存する基準**：Hの再計算を省く。GもdX・Parameter勾配で共有する。
2. **forward内だけ一時的にglobal保存する**：全Yの利用が終わってからHを破棄し、
   backwardのParameter勾配で必要ならHを再計算する。forward時の一時容量は残る。
   一部atomだけ保存し、残りを再計算する方法も比較対象にできる。
3. **register/shared内で生成から利用まで完結させる**：H生成とC/Y生成を同じblock内で
   融合する別候補。別kernelのsharedへHを引き渡すことはできない。
   複数出力blockが同じHを使う場合は、再計算するか所有範囲を変更するかを決める。
   backwardでもHを再計算するか、一部をglobal保存する必要がある。

推論などHをbackwardで使わない場合は、全Yの利用後に保持を終えられる。
再計算には更新後のParameterではなく、forward時点のX・atom・配置・正規化を使う。
必要な保存Tensorのversion検査やsnapshotも保存量に含める。
Zxを保存しているならHだけの再計算、Zxも破棄するなら階層の再構築からになる。

sharedだけにすれば必ず高速になるとはしない。Hの再計算、global読み書き、
register退避、shared容量による同時実行block数の低下を含めて測る。
面積Sと正規化係数はdevice関数で利用時に評価し、独立したnorm準備kernelや
全atom分の係数表のglobal保存は前提にしない。
H/Gを同じthreadで作る案、固定の担当表、局所経路の融合も後の比較候補として残す。

## 残る設計事項と検証順序

非負profile族、sigmaから共通基底の係数への写像、支持幅・L・最小幅・階層遷移、
有限/周期境界、幅のtask VJPと更新則はIssue #120で確定していく。
基準の担当と依存を実装した後に、保存・再計算・融合を同じ数学契約で比較する。
新Specとsemanticsに対応したAlgorithmだけを登録し、未完成の経路を旧L2へfallbackしない。
CPU直接参照と独立oracleでY/dX/全atom勾配を確認した後に、CUDAの完全stepとピークメモリを測る。
新方式の基準は本節とし、既存のH寿命文書の旧L2式・最初はH非保存という比較条件は移植しない。

## 正規化宣言を実装した段階の検証

mainベースの本ブランチで、CPU suiteは1978 passed / 2776 skipped / 18件の既存JIT警告。
新規則のJSON往復、領域/floorの検査、旧binding/presetが新規則を誤実行しないこと、
既存L2の計算・勾配の回帰を含む。変更PythonファイルのRuff検査・整形検査、
wheel/sdist buildも通過した。GPUや新セル積分profileの数値検証を行った結果ではない。
本書の5つの計算kernel・保存方針の追記は文書のみの変更であり、上記のテストを
新CUDA実装の検証結果として扱わない。文書更新では差分の空白検査とリンク・記法を確認した。
