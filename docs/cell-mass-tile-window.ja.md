# セル境界を切らない、1周期分のタイル窓

2026-10-26。支持外のFourierの裾を近似としてゼロ扱いする設計に合わせ、
atom中心の支持境界でセル積分をclipする旧試作を置き換える。
旧測定・旧oracleの成功は旧定義の記録であり、この近似の誤差検証とは区別する。

## 支持 I、周期 2L、タイル T

site間隔d、1タイルのsite数n_T、奇数Kとして

\[
T=n_Td,\quad I=\lfloor K/2\rfloor T,\quad P=2L=KT
 =\left(\left\lceil\frac{2I}{T}\right\rceil+1\right)T.
\]

初期設定はK=3、I=T、P=3T。中心を含むタイルと左右1タイルを選ぶ。
I以内の集中領域を覆い、残りの裾はゼロの余白として近似する。
計算時は選んだタイルの全セルを同じ積分基底で評価する。
選んだ窓の中に残る小さな裾をsiteごとに切らない。窓の外はタイル単位でゼロ。
K=5,7などにも対応し、chart内にKタイルが収まることを宣言で検査する。
K=1はchart全体が1タイルの場合だけを許す。

gridのセル境界はorigin+(i-1/2)d。したがってtile ownerは

\[
q_a=\left\lfloor\frac{(\mu_a-origin)\bmod Q+d/2}{T}\right\rfloor,
\quad s_a=(q_a-\lfloor K/2\rfloor)n_T.
\]

窓はunwrappedなsite番号s_aからK*n_T個。chartのsiteへはmod Nで戻す。
Q=N*dとローカル周期Pは別。chartの端で分割した部分は、そのunwrapped位置に
対応する位相を使う。PがQを割り切るとは仮定しない。

## 面積の正規化と共有基底

非負な周期密度gを一周期の面積S=P*c0で割る。

\[
\widehat\phi_{ai}=\begin{cases}
\displaystyle\frac1S\int_{(j-1/2)d}^{(j+1/2)d}g(x-\mu_a)\,dx,
&j\in[s_a,s_a+Kn_T),\ j\equiv i\pmod N,\\
0,&\text{それ以外}.
\end{cases}
\]

窓がちょうど1周期を覆うのでsum_i(phi_ai)=1。追加の離散normや全siteの準備passは不要。
セル積分はsin/cosの共通基底に吸収し、H/Gとその微分はZx/Zyのwhole tileだけを使う。
CからY/dXも同じタイル所属条件を使う。セルの端をmin/maxで切る経路は存在しない。

ownerの整数変化は微分しない。窓が変わる位置には裾の大きさに応じた小さな不連続があり、
中心・幅の勾配は選択中の枝の微分。厳密に滑らかなcompact関数だとは主張しない。

## 近似誤差と実装誤差を分ける

1. **実装誤差**: 選んだ窓のwhole cellを独立scalar Chebyshev密度のGauss積分で評価し、
   TorchのFourier積分、CUDAのY/dX/全atom勾配と比較する。
2. **近似誤差**: 意図したcore [-I,I]で密度を切って再正規化する別のoracleと比較する。
   このoracleの境界積分はテスト・分析専用で、runtimeへ入れない。

core外の一周期あたり質量比をetaとすれば、非負密度のprofile L1誤差は2*eta以下。
ChebyshevSquareのraw side-lobe bound tauについて
eta <= (P-2I)*tau/S も確認する。tauはpeak高さに対する密度の上界で、
そのままYや勾配の相対誤差とは解釈しない。Y/dX/Parameter勾配の差も別に記録する。

## メモリと処理順

- globalは元atom、sorted_rows、Zx/Zy、H/G、入出力・勾配のみ。atomや係数の全表を複製しない。
- Y/dXのblockは最大4leaf×4batchのCをregisterへ蓄積する。
- 32atom chunkを一度読み、現在chunkのraw harmonicをsharedへ生成する。
  sin/cos・4batch・担当leafで共通のraw係数を使い回す。
- 全chunkの後にCをsharedへ渡してY/dXへ展開する。出力atomicは不要。
- H/GとParameter勾配は引き続き各threadが異なるatomを担当する。

```cpp
Z = build_hierarchy(signal);
H = project_selected_whole_tiles(Z, atoms);
for (atom_chunk : atoms) {
    prepare_chunk_and_harmonics_in_shared();
    sync();
    accumulate_C_for_owned_tiles();
    sync();
}
publish_C_in_shared(); sync();
expand_C_and_store();
```

## 現在の研究用API

```python
profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
spec = CellMassLinearSpec(
    input=CellMassAxisSpec(profile=profile, site_count=128,
                          spacing=1., tile_sites=16, period_tiles=3),
    output=CellMassAxisSpec(profile=profile, site_count=128,
                           spacing=1., tile_sites=32, period_tiles=3),
)
y = physical_linear(spec, x, atoms, z_method="merge")
```

以前のsupport_half_widthを直接渡す研究APIは廃止。IとLの混同を避ける。
今回は1軸1bank、128以下の2のべき乗site数、1024以下のatom。
CUDAのtile_sitesはsite_countを割り切る2のべき乗。sigma<=min(Pin,Pout)/3。
複数のsigma bankへの分類やblockのbegin/endによる候補範囲の絞り込みは未実装。
公開CSTLinear/optimizer/Graph/二階微分への接続も未実施。
