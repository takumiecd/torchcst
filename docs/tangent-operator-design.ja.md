# Parameter値を受け取るCST専用1次作用の設計

2026-09-08。**準備済み作用・輸送・PCG再圧縮を実装済み。**
現在稼働するoptimizerは[first-order-rebuild.ja.md](first-order-rebuild.ja.md)を参照。
今回の対象は1次版の輸送・再圧縮。2次optimizerの更新式は変更しない。

実装：`AtomDerivatives.tangent_ops()`、kernel-owned analytic factor微分、
tile化cross/weighted Gram作用、block前処理PCG。`CSTAdam`の輸送は新作用を使い、
`recompression="pcg"`と正の`first_moment_damping`で再圧縮も切り替える。
既定はdamping=0のdirect圧縮を維持。詳細な使用例・設定はREADME参照。

現在の制約：factor配列は`O(Kq(I+O))`、atom間の計算量は依然二乗。
PyTorch eager実装でCUDA融合は未実装、PCG収束判定はhost同期を伴う。
更新方向を解くweighted Gramの直接求解は残っており、optimizer全体の
linear-memory化までは完了していない。以下はその先も含む設計契約である。

## 1. 選択する境界

作用に渡す点は通常の`Tensor`とし、現在の`nn.Parameter`もその入力として扱える。
atomのcontainerやoptimizer state全体を作用の引数にしない。

ただし、同じ`[K,q]`のTensorでもkernelにより座標の意味は違う。
**parameter値だけでは作用を特定できない。** kernel・chart・固定設定をあらかじめ
束縛したoperatorをsiteが提供し、そのoperatorにparameter値を渡す。
これは作用APIの変更であり、公開optimizerの`CSTAdam(model, ...)`を
`CSTAdam(model.parameters(), ...)`へ変更する提案ではない。

| 層 | 責務 |
| --- | --- |
| `optim/` | 現在点・旧点・alpha・EMAの所有、線形求解、誤差判定、同時commit |
| `nn/` | siteとkernel/chartを関連付け、微分operatorを提供 |
| `_derivatives/` | Gram/cross-Gram/weighted作用、factor縮約、tile実行、workspace管理 |
| `kernels/` | atom座標の解釈、kernel/profile固有のfactor JVP/VJP、optional専用backendの提供 |
| `atoms/` | opaqueな`[K,q]`parameter表の所有 |

optimizerと汎用縮約コードには`p[:, 0]`を振幅とみなす処理や、kernel型ごとの
座標slice分岐を置かない。現在の`Kernel.parameter_dim`とkernel内のsplit規約を正とする。

## 2. Parameter入力と準備済み作用

内部APIの提案：

```python
ops = site.cst_derivatives().tangent_ops(backend="auto")

current = ops.prepare(p_current)   # Tensor [K,q]
previous = ops.prepare(p_previous) # 保存済みの旧Tensor [K,q]

y = current.gram_matvec(x)                         # J_current.T J_current x
y = current.cross_gram_matvec(previous, x)         # J_current.T J_previous x
y = current.weighted_gram_matvec(metric, x)        # J_current.T D J_current x
blocks = current.gram_blocks()                     # 原子内 J_a.T J_a のみ
```

`x`は原子parameterと同じshapeの方向・係数。Gram/crossの結果も`[K,q]`。
crossでは**左側が測定先、引数previousが展開元**。両フレームの互換性を検証する。
今回の範囲は同じsite・同じatom数と対応。原子の追加・削除・並べ替えは含まない。

`prepare`は次の契約とする。

- 明示されたparameter値で評価し、作用中にliveな`site.atoms.p`を再読しない。
- 呼び出し元が後からparameterをin-place更新しても結果が変わらないよう、
  backendは独立したdetach済みsnapshotを確保する。
- current/previousそれぞれのfactor値・必要な微分素材を一度準備し、反復中に再利用する。
- snapshotは内部でoptimizer保存用と読み取り専用共有できるが、外部へ可変aliasを渡さない。
- 作用はparameter・勾配・EMA・入力方向を変更しない。
- optimizer用経路は高階autograd graphを保持しない。Hessianは要求しない。
- device/dtype/layout契約を準備時に確認する。数値検査による同期は明示した境界へ集める。

`prepare`のオブジェクトはstep内の一時workspaceであり、checkpointへ保存しない。
同じdata pointer・shapeであっても値は変わるため、pointerだけでstepをまたぐcacheを再利用しない。
最初はstep内だけのcacheとし、永続cache・CUDA graph化は別の最適化段階とする。

通常のJVP/VJPも数学上の基本作用として定義する。

```python
delta_w = current.jvp(x)       # J x、可視tensorを返すreference API
delta_p = current.vjp(force)   # J.T force
```

しかし最適化経路では`vjp(jvp(x))`の間に全可視tensorを確保する実装を必須としない。
Gram作用そのものに専用経路を用意し、factorまたはtileのまま合成する。

## 3. 旧parameterからの復元と保存契約

1次のfirst momentは概念上、次だけで旧履歴を再構成できる。

$$
(\alpha_{t-1},p_{t-1}),\qquad
\widetilde m_{t-1}=J(p_{t-1})\alpha_{t-1}.
$$

ただし$J$を決める固定情報が不変であることが必要：

- kernel/profileの種類と座標layoutのversion；
- chartの座標値・feature対応；
- kernelの固定buffer（sigma、tau、temperatureなど）；
- atomの数とrow対応、parameter/device/dtypeの契約。

shapeが同じだけでは十分でない。固定情報はモデル側に保存し、checkpoint読み込み時に
operator descriptorと照合する。外部から固定設定を変える場合は新しいoperator世代を作り、
旧stateは明示的resetまたは別途定義した変換を要求する。kernelはoptimizer履歴を所有しない。

このdescriptor検証は**現行checkpointより追加が必要な設計項目**。
現在のmanifestだけでkernel/chartの値まで検証できるとは扱わない。

stepの順番：

1. 更新前のparameterからcurrent workspaceを準備する。
2. optimizerの保存した旧点からprevious workspaceを準備する。
3. 旧alphaをcross作用で現在へ測り直し、現在の観測とEMAする。
4. 更新候補と再圧縮alphaを計算し、双方の検査が通った後にcommitする。
5. **更新前の点$p_t$**と新alphaを次の履歴として保存し、modelには$p_t+d_t$を適用する。
6. step用workspaceを解放する。

旧点をmodelへ一時的に代入する必要はない。1次の旧フレームにaccepted displacementは不要。
2次の$J(p)+H(p)[d,\cdot]$を再現する場合は$p$に加えて$d$も必要で、今回とは別契約。

## 4. 圧縮の数式は維持する

$$
b_t=\beta_1J(p_t)^\top J(p_{t-1})\alpha_{t-1}
 +(1-\beta_1)J(p_t)^\top g_t.
$$

第2項は既存の`AtomGrad`による`jg`観測を使う。Gram作用のために学習のbackwardを再実行しない。

$$
A_t x=J(p_t)^\top J(p_t)x+\lambda x,
\qquad A_t\alpha_t=b_t.
$$

再圧縮はEuclideanな可視射影なので、この$A_t$にAdamの$D$を混ぜない。
weighted作用は更新側の$J^\top D J$に使う別の能力とする。

概念コード：

```python
b = beta1 * current.cross_gram_matvec(previous, alpha_previous)
b = b + (1 - beta1) * observation.jg
apply = lambda x: current.gram_matvec(x) + damping * x
alpha_next = linear_solver.solve(apply, b, preconditioner=preconditioner)
```

raw alphaを保存し、更新側でのみbias correctionする現在の規約を保つ。
cross作用にも、原子間の情報を捨てるblock近似は入れない。

## 5. kernel依存性と共通factor経路

kernelはoptionalな`tangent_backend(input_chart, output_chart)`を提供できる設計とする。
optimizerはそれを直接判別せず、`_derivatives/`のresolverが選ぶ。

1. kernel固有の専用backendがあれば使う。
2. exact factorizationがあれば、factor微分＋共通縮約backendを使う。
3. それ以外はgeneric autogradのreference経路を残す。

`backend="specialized"`の明示指定で未対応kernelなら構築時にerror。
`"auto"`のfallback時も選ばれたbackendを診断へ記録し、reference経路に同じ省メモリ性能を約束しない。
新kernelを追加するときにoptimizerを編集する必要をなくす。

factor経路で$W=VU^\top$とする。$U\in\mathbb R^{I\times K}$、$V\in\mathbb R^{O\times K}$。
振幅もkernelがどちらかのfactorへ含める。kernel固有部分は

$$
\operatorname{factor\_jvp}(p,x)=(\dot U,\dot V),
\quad
\operatorname{factor\_vjp}(p,Z_U,Z_V)
=(\partial U/\partial p)^*Z_U+(\partial V/\partial p)^*Z_V
$$

というpairを満たす。factor/profileの座標sliceとchain ruleをここへ閉じ込める。
separableなrank-one atomでは、

$$
Jx=\dot V U^\top+V\dot U^\top.
$$

展開元を$s$、測定先を$t$とし、$F=J_s x$をmaterializeせず、

$$
FU_t=\dot V_s(U_s^\top U_t)+V_s(\dot U_s^\top U_t),
$$

$$
F^\top V_t=U_s(\dot V_s^\top V_t)+\dot U_s(V_s^\top V_t)
$$

を計算すれば、`factor_vjp`に$Z_U=F^\top V_t$、$Z_V=FU_t$を渡せる。
同一点ならGram、異なる点ならcross-Gramになる。

**sourceの方向$x$を先に縮約して$\dot U_s,\dot V_s$を作る**。
全source座標のJacobian列ペアを計算してから$x$を掛ける旧blocked実装をそのまま使う必要はない。

ただし上式の$U_s^\top U_t$などを全保存すると$K^2$が残る。
source/target atomのtileごとに計算・累積し、全$K\times K$、全$p\times p$、全$O\times I$を
確保しない実装を最適化経路の要件とする。factor値や方向factorの$O(K(I+O))$は別途残る。
演算量自体は一般に原子ペアを含み、原子数に対して線形とは主張しない。

## 6. 既定の行・列EMAとの接続

行・列EMAからbias correction後の$\widehat r,\widehat c$を作ると、

$$
D_{oi}=s_o t_i+\varepsilon,
\quad s_o=\sqrt{\widehat r_o/\operatorname{mean}(\widehat r)},
\quad t_i=\sqrt{\widehat c_i}.
$$

ゼロnormalizer保護は現実装と同じ規約にする。この対角作用は

$$
D\odot(AB^\top)
=\operatorname{Diag}(s)A[\operatorname{Diag}(t)B]^\top+\varepsilon AB^\top
$$

なので、factorを行・列方向にscaleして縮約できる。
`weighted_gram_matvec`のために全対角tableを復元する必要はない。
初期の専用weighted経路は、このseparable diagonal metricだけを正式対応にする。
atom RMSのparameter block metricは別表現であり、同じvisible metric APIへ偽装しない。

weighted作用を実装しても、現在の固有分解型更新solverが自動的にmatrix-freeになるわけではない。
まず輸送・再圧縮を接続し、更新側のtrust-region反復求解は別の検証段階にする。

## 7. 既存kernelで必要な微分

`Separable`はinput/output profileのsliceを担当し、`Amplitude`は自分の振幅sliceと
内部kernelへのchain ruleを担当する。`AmplitudeBandwidthSeparable`では振幅がoutput幅を
変えるため、その微分を落とさない。

例えば$v(a,s)=a\phi(s,\kappa(a))$なら、

$$
\dot v=\phi\dot a+a(\partial_s\phi\,\dot s+\partial_\kappa\phi\,\kappa'(a)\dot a).
$$

振幅で割る実装は使わず、$a=0$でもこの式で評価する。

現在のGaussianはL2正規化されている。$z_i=-\kappa\|\mu_i-s\|^2$、
$\phi_i=\exp(\tfrac12\operatorname{logsoftmax}(z)_i)$なので、

$$
\dot z_i=2\kappa(\mu_i-s)^\top\dot s-\|\mu_i-s\|^2\dot\kappa,
\qquad
\dot\phi_i=\tfrac12\phi_i[\dot z_i-\sum_j\phi_j^2\dot z_j].
$$

正規化項も微分する。隣接featureからの寄与を省略した局所Gaussian微分に置き換えない。
専用JVP/VJPはこの数式とgeneric autogradの両方で検証する。

## 8. 求解と前処理

第一候補は正のdampingを持つ系へのblock-preconditioned CG。
原子内$J_a^\top J_a+\lambda I$を小行列として準備する。
これは**前処理だけ**であり、各iterationの作用は全原子間の結合を保つ。

- 同じdamping・同じrhsで直接解法と比較する。
- 現在既定のzero dampingを、CGの都合で黙って正に変えない。
- zero dampingの特異系は、最小ノルム・rank cutoffの扱いが一致するまで直接解法を参照として残す。
  単に残差の小さい解を得ただけで現行pseudoinverseと同じとは扱わない。
- 返すdtypeに丸めたalphaへ作用を再適用し、真の残差を検査する。
- 未収束・非有限値・breakdownは診断として返し、未検証のalphaを永続stateへcommitしない。
- 反復数上限、許容誤差、前処理の準備時間、実行時間を記録する。
- 収束判定の同期頻度とGPU実行方式は計測で選ぶ。長いPython loopや固定予算の空回りを性能根拠にしない。

有限反復は近似であり、厳密な圧縮との違いを消したと主張しない。
悪条件ではalphaの誤差と可視履歴$J\alpha$の誤差を両方調べる。

## 9. 検証・実装の区切り

1. **作用の契約とreference oracle**：点・方向・device/dtype・固定descriptor、純粋性、
   旧点がlive parameterから独立することを検証。
2. **1次factor作用**：Gram/crossの専用経路。dense autogradとの一致、adjoint、対称性、
   PSD、原子間cross term、ゼロ振幅、幅gate、partial tileを検証。
3. **輸送の接続**：cross Gramの構築を禁止した回帰テスト。旧履歴のpullbackを比較。
4. **再圧縮の接続**：同じdampingの直接解法に対する残差・alpha・$J\alpha$・学習軌道を比較。
   更新solverはこの段階では変えない。
5. **separable weighted作用と更新側の接続**：epsilon項を含めた一致を確認し、
   trust-regionの境界処理まで別途検証する。
6. **専用kernel・融合・cacheの最適化**：正規化Gaussianと幅gateの微分を含めて測定。

各段階を検証できるcommitで区切る。小規模の直接解法を残し、原子数・幅・condition・
複数layerを振って、prepared単体と準備込み双方の時間、iteration数、peak allocated/reserved、
モデルと履歴を含む全体メモリを記録する。速さを確認する前に既定を変更しない。

## 10. 現行コードからの変更先

- `nn/linear.py:CSTLinear.cst_derivatives`：kernel/chartを束縛したproviderへの接続。
- `kernels/base.py`と各kernel/profile：optional tangent backendと固有の微分。
- `_derivatives/tangent.py`：full matrixを返す`cross`に依存する経路から、prepared作用へ移行。
- `optim/moments/tangent.py`：旧点とalphaの所有規約を保ってcross作用を呼ぶ。
- `optim/`の線形solver：直接Gram solveと反復solveの診断・commit条件を分離。
- `optim/moments/second.py`：行・列EMAは維持。metricのfactor作用は既存の`separable_weights`と接続。

publicな1次/2次optimizerの分離、`AtomGrad`による観測、dense AdamWとの同時commitを維持する。
