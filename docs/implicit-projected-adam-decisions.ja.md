# Implicit projected Adam: 実験知見と設計判断

Status: selected research direction; production implementation is not yet complete.

この文書は、implicit projected moment optimizer に関する議論と実験から、
現時点で採用する判断、棄却した解釈、未解決問題を分離して保存する decision
record である。以前の導出そのものは
[`implicit-projected-moment-transport.ja.md`](implicit-projected-moment-transport.ja.md)
に残すが、その文書の force-level $\gamma$ transport は最終候補ではない。

## 1. 選択する最終目的

二次 CST displacement

$$
\Delta W_t(d)=J_td+\frac12H_t[d,d]
$$

に対して、解きたい局所 objective は

$$
\boxed{
\Phi_t(d)
=m_t^\top\Delta W_t(d)
+\frac1{2\eta}
\Delta W_t(d)^\top
\operatorname{Diag}(\sqrt{v_t}+\varepsilon)
\Delta W_t(d)
}
$$

である。ここで

$$
m_t=\operatorname{EMA}_{\beta_1}(g_t),
\qquad
v_t=\operatorname{EMA}_{\beta_2}(g_t^{\odot2})
$$

は概念上の represented-weight moments である。production 実装では、これらを
dense persistent tensors として保存しない。

まずはこの quartic objective を近似せずに解く。solver の計算量を落とす研究は、
state compression の正しさから分離する。

## 2. 実験で確認できたこと

### 2.1 Compact first moment は有力である

accepted point の可視 frame で

$$
\widehat m_t=V_t^\star\alpha_t,
\qquad
V_t^\star=V_t(d_t^\star)
$$

とし、$\alpha_t$ だけを永続保存する。次 step では、旧点 metadata から
$V_{t-1}^\star$ の作用を再構成し、

$$
V_t(d)^\top V_{t-1}^\star\alpha_{t-1}
$$

を評価する。旧 coordinate を現在の coordinate として直接再利用してはいけない。

A100 MNIST 実験では、この accepted-frame $\alpha$ と圧縮された対角 second
moment を組み合わせても Adam 級の精度を保った。したがって、first moment を
ambient dense state として保持する必要性は大きく低下した。

これは $\alpha$ 単独の完全な因果分離実験ではない。厳密な寄与分離には
compact-$m$/dense-$v$ hybrid control が必要だが、少なくとも本命の完全圧縮構成が
成立するという primary question には肯定的な結果が得られた。

### 2.2 成功した second-moment backend

MNIST の weight table を $10\times784$ とし、勾配二乗の row/column EMA

$$
r_{t,o}
=\beta_2r_{t-1,o}
+(1-\beta_2)\operatorname{mean}_i g_{t,oi}^2,
$$

$$
c_{t,i}
=\beta_2c_{t-1,i}
+(1-\beta_2)\operatorname{mean}_o g_{t,oi}^2
$$

だけを保存した。bias correction 後、

$$
\widetilde v_{t,oi}
=\frac{\widehat r_{t,o}\widehat c_{t,i}}
{\operatorname{mean}_o\widehat r_{t,o}}
$$

を再構成し、

$$
D_t=\operatorname{Diag}(\sqrt{\widetilde v_t}+\varepsilon)
$$

として full cross-atom quartic を解いた。

rank one なのは $\widetilde v$ の row/column 表現であり、operator $D_t$ は対角で
ある。これは

$$
\frac{g_tg_t^\top}{\lVert g_t\rVert}+\varepsilon I
$$

という ambient rank-one operator とは異なる。

### 2.3 A100 MNIST primary result

K=64、256 CST parameters、128 steps、seeds 17/29/43 で得た test accuracy は
次の通りである。

| optimizer | mean accuracy |
| --- | ---: |
| Adam-target quartic | 81.50% |
| dense $m,v$ exact-diagonal implicit | 81.48% |
| **compact $\alpha$ + separable-diagonal $v$** | **81.12%** |

compact 構成と dense exact-diagonal control の差は平均 $-0.37$ point、最大 seed
deficit は $1.05$ point だった。

永続 moment state は dense Adam の 15,680 scalars に対して、

$$
256\;\text{($\alpha$)}+10\;\text{(row)}+784\;\text{(column)}=1,050
$$

scalars である。これは 93.3% reduction、14.9x smaller に相当する。

この結果の runner、raw result、figure、完全な protocol は `cst` repository の
commit `73ec524` と
`docs/experiments/mnist_wgate_compact_diag_ablation_a100.md` にあり、Arctx lane
`wgate-compact-adam-diagonal` の result step
`t_ba64d250280b4e678a596149c6c8d9e0` に登録されている。

## 3. 棄却する解釈

### 3.1 Ambient rank-one operator は diagonal Adam の代替ではない

$$
D_t^{\mathrm{rank1}}
=\frac{g_tg_t^\top}{\lVert g_t\rVert}+\varepsilon I
$$

は平方根を解析的に扱えるが、勾配方向一つにしか大きな曲率を与えない。異なる方向へ
step が回転すると十分な RMS 制御がなくなる。

同じ K=64 系で、Adam-target 81.50% に対して ambient rank-one force EMA は
52.73%、compact $\gamma$ rank-one は 66.95% だった。したがって、この失敗を
compact compression だけの責任にすることはできない。operator 自体が diagonal
Adam と異なることが主要因である。

### 3.2 Force-level $\gamma$ は raw second moment ではない

以前の案は

$$
u_s=D_s\Delta W_s(d_s^\star)
$$

を先に作り、その可視代表

$$
\widehat u_s=V_s^\star\gamma_s
$$

を EMA した。これは

$$
\operatorname{EMA}
\left[D_s\Delta W_s(d_s^\star)\right]
$$

を保存する方式であり、operator を蓄積して現在候補へ作用させる

$$
\left(\operatorname{EMA}[D_s]\right)\Delta W_t(d)
$$

とは一般に一致しない。過去の $D_s$ と過去の $\Delta W_s$ が、EMA 前に
entangle されているためである。

したがって、$\gamma\in\mathbb R^p$ は candidate-specific contracted force の
圧縮には使えるが、Adam の第二状態の基礎表現にはしない。

なお、diagonal force-level $\gamma$ の full end-to-end run は完了していない。
よって「diagonal を入れても $\gamma$ は必ず失敗する」という実験的主張までは
行わない。棄却理由は、Adam second-moment semantics と一致しないという数式上の
理由である。

## 4. $\Delta W$ を履歴から分離する

二次モデルでは

$$
\boxed{
\Delta W_t(d)=V_t(d/2)d
}
$$

が厳密に成り立つ。したがって、

$$
V_t(d)^\top D_t\Delta W_t(d)
=V_t(d)^\top D_tV_t(d/2)d
$$

である。

ここで

$$
S_t(d):=V_t(d)^\top D_tV_t(d/2)
$$

を先に構成すれば、

$$
G_t(d)=S_t(d)d
$$

となる。つまり、過去の $\Delta W$ を保存せず、visible operator を保存して
現在候補の $d$ を最後に作用させる。

展開すれば、$S_t(d)$ は

$$
J_t^\top D_tJ_t,
\qquad
J_t^\top D_tH_t,
\qquad
H_t^\top D_tH_t
$$

の contractions から成る。これらが quartic objective の二次、三次、四次係数に
対応する。

## 5. Lifted visible operator

二次 jet basis を

$$
F_t=[J_t,H_{t,:11},H_{t,:12},\ldots]
$$

とし、二次特徴 $z(d)$ とその Jacobian $L(d)$ を用いて

$$
\Delta W_t(d)=F_tz(d),
\qquad
V_t(d)=F_tL(d)
$$

と書く。すると D-side は

$$
\boxed{
K_t:=F_t^\top D_tF_t
}
$$

だけを通じて

$$
\boxed{
G_t(d)=L_t(d)^\top K_tz_t(d)
}
$$

と評価できる。$D_t\Delta W_t$ を force vector として保存する必要はない。

$K_t$ は $\gamma_t$ より高次元だが、現在候補を作用させる前の operator 情報を
保持している。$\gamma_t$ から $K_t$ を復元することは一般にできない。

## 6. Diagonal operator の visible space と transport

$D_t=\operatorname{Diag}(q_t)$ とする。$F_t$ の列を $f_{t,a}$ と書けば、

$$
(K_t)_{ab}
=\left\langle q_t,f_{t,a}\odot f_{t,b}\right\rangle.
$$

したがって diagonal state に対する正しい可視空間は

$$
\boxed{
\mathcal P_t
=\operatorname{span}
\{f_{t,a}\odot f_{t,b}\}_{a\le b}
}
$$

である。$\operatorname{col}(V_t)$ だけではない。

積基底を列に並べた行列を $\Psi_t$ とし、symmetric entries を一貫した scaling で

$$
k_t:=\operatorname{svec}(K_t)=\Psi_t^\top q_t
$$

とする。旧 visible diagonal representative は

$$
\widehat q_{t-1}
=\Psi_{t-1}
(\Psi_{t-1}^\top\Psi_{t-1})^\dagger
k_{t-1}
$$

であり、現在 frame への transport は

$$
\boxed{
k_{t-1\to t}
=\Psi_t^\top\Psi_{t-1}
(\Psi_{t-1}^\top\Psi_{t-1})^\dagger
k_{t-1}
}
$$

となる。

これは first-moment transport と同じ「旧 visible representative へ展開してから、
現在 frame で測り直す」という構造である。現在 product space のうち旧 product
space と直交する方向には、過去の state を勝手に補わない。current gradient が
その新方向を埋める。

一般 symmetric operator として

$$
K_{t-1\to t}=T^\top K_{t-1}T,
\qquad
T=(F_{t-1}^\top F_{t-1})^\dagger F_{t-1}^\top F_t
$$

と運ぶ方法もある。しかしこれは ambient では
$P_{F_{t-1}}D_{t-1}P_{F_{t-1}}$ という一般に非対角な operator を表す。
diagonal Adam の構造を守る本命は product-space transport である。

## 7. まだ解けていない二つの問題

### 7.1 $\operatorname{Diag}(\sqrt v)$ の compact representation

Adam は

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}
$$

を作った後に $\sqrt{v_t}$ を取る。projection、EMA、elementwise square root は
一般に交換しない。したがって、visible $v_t$ の線形 transport だけから

$$
F_t^\top\operatorname{Diag}(\sqrt{v_t})F_t
$$

を厳密に得られるとは限らない。

現時点の実用候補は、実験で成功した row/column separable $v$ を backend として
使うことである。product-space representation は理論上の本命だが、平方根を含めた
安定かつ低コストな作用の作り方は未解決である。

### 7.2 Full quartic solve の高速化

$$
J^\top DJ,
\qquad
J^\top DH,
\qquad
H^\top DH
$$

を完全に残すと cross-atom interaction が入り、明示 coefficient storage や solve
が高価になる。correctness phase では full objective を解き、次の段階でのみ
same-atom block、low-rank correction、operator matvec、反復 solve などを比較する。

近似は dense/full oracle に対する objective、stationarity、accepted loss、最終精度の
差で評価する。

## 8. 現時点の optimizer blueprint

1. current represented gradient から、compact first-side observation と second-side
   statistics を取得する。
2. first state は accepted-frame $\alpha_{t-1}$ を旧 frame から現在 frameへ
   operator contractionで運ぶ。
3. second state は過去の $D\Delta W$ を保存しない。対角 $v$ の compact backend
   から、現在 frame で必要な $K_t=F_t^\top D_tF_t$ の作用を作る。
4. $m$ side の affine term と、$K_t$ side の cubic stationarity termを合わせ、
   quartic surrogateを解く。
5. exact minibatch loss、trust region、line searchで候補をaccept/rejectする。
6. accepted $d_t^\star$ でのみ $\alpha_t$ を再圧縮し、旧点 metadata とともに保存する。
7. structural mutation時にはslot remap、basis rank、gauge、state reset policyを明示する。

論理的な永続stateの当面の実装候補は

$$
\boxed{
(\alpha_t,\;r_t,\;c_t,\;\theta_t,\;d_t^\star)
}
$$

である。これは成功実験の backend であり、将来 $r_t,c_t$ を product-visible $v$
stateに置き換えられるよう、second-state backendをoperator APIの背後に隔離する。

### 8.1 最初の実装は continuous-only とする

最初のproduction実装では、optimizer step中の離散構造操作を扱わない。

$$
\boxed{
\text{fixed atom count}
+\text{fixed slot identity}
+\text{fixed parameter dimension}
+\text{continuous updates only}
}
$$

具体的には、学習中の birth、death、merge、absorb、slot remapをoptimizerの責務から
外す。連続的なparameter update、retraction、gauge normalization、trust region、
line searchは残す。

この制限により、first-stateおよびsecond-state transportで旧atomと現atomの離散的な
対応を解く必要がなくなる。旧点 $\theta_{t-1}$ とaccepted step
$d_{t-1}^\star$ から旧frameを再構成できることを、最初の実装契約とする。

これはmergeやabsorbがすべての問題で不要だという主張ではない。tested MNIST条件と
現在選択したoptimizerの成立には離散操作が不要だった、というscope判断である。
将来、固定budget下の容量再配置に構造変更が必要になった場合も、optimizer内部へ
混ぜず、独立したstructural controllerとしてepoch境界などで実行する。その境界では
optimizer stateを明示的にresetするか、別途検証済みのstate変換を要求する。

### 8.2 後方互換性を要求しない破壊的置換

このrepositoryの現行experimental APIを利用して継続的な外部実験を行っている
利用者はいないため、このbranchでは旧dynamic architectureとの後方互換性を
要求しない。旧branchとgit historyを復旧手段とし、deprecation layer、checkpoint
migration、旧optimizerとの二重実行経路は作らない。

新しいfixed-shape CST moduleを正の実行経路として作り、forward、derivative
contractions、optimizer ownership、checkpoint schema、public exports、testsをその
契約へ置き換える。新経路が成立した段階で、そこから到達不能になるpolicy engine、
mutation planning、birth/death/merge/absorb、slot remap、structural optimizer-state
reconciliationを削除する。

破壊的置換は一つの巨大commitにはしない。少なくとも、(1) continuous contract、
(2) fixed-shape representation、(3) implicit optimizer、(4) legacy removal、
(5) documentation and examples の検証可能なcommit境界に分ける。各境界ではtestsを
通し、数値oracleとの一致を維持する。

### 8.3 Chartはfixed-cardinality、座標は既定でfreezeする

chartの点数、feature対応、tensor shapeは学習中固定する。chart座標そのものは
`trainable=True`という将来拡張をAPIに残すが、defaultと推奨値はともに
`trainable=False`とする。

現在の設計では、amplitudeに依存してbandwidthを連続調整することで、chart移動に
期待していたsupport適応の多くを担える。外部入力chartを動かすとdata上のgroundingも
弱くなるため、少なくとも最初のimplicit optimizerでchartを学習する積極的な理由は
ない。

一方、hidden chartのlatent geometryを学習する将来研究まで禁止する必要はない。
chartを変数に含めても二次モデル

$$
\Delta W(d)=Jd+\frac12H[d,d]
$$

自体は成立する。ただし $d=(d_{\mathrm{atom}},d_\mu)$ となり、$H$には
atom/chartおよびchart/chartのblockが加わる。shared hidden chartなら前後のlayerも
結合される。将来対応する場合はblock JVP/VJP/HVPとして評価し、巨大なfull Hessianは
materializeしない。

したがって、representation APIはtrainable chartの目を残すが、最初の
`ImplicitProjectedAdam`はfrozen chartのみを正式対応とし、trainable chartを
明示的に拒否する。

## 9. 主張の境界

現時点で主張してよいことは次である。

- dense Adam momentsを永続保存しなくても、tested MNIST条件ではAdam級精度を保てた。
- accepted-frame $\alpha$ は有力なfirst-moment compressionである。
- diagonal RMS actionが重要であり、ambient rank-one operatorは代替にならなかった。
- force-level $\gamma$ はoperatorとcandidate displacementをEMA前に結合するため、
  Adam second stateの基礎表現として不適切である。
- 履歴の $\Delta W$ は、lifted visible operator $K$ を保存することで数式から分離できる。

まだ主張できないことは次である。

- product-space transportを含む完全ambient-free optimizerが実装済みであること。
- visible $v$ から visible $\sqrt v$ を厳密かつ安価に計算できること。
- full quarticと同精度の高速solverが得られたこと。
- structural birth/deathを跨ぐmoment transportが解決済みであること。

この境界を保ち、成功したseparable backendを最初のproduction候補、積空間
second stateを理論的な発展先として扱う。
