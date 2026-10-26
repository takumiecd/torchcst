# セル境界を切らない、1周期分のタイル窓

2026-10-26。支持外のFourierの裾を近似としてゼロ扱いする設計に合わせ、
atom中心の支持境界でセル積分をclipする旧試作を置き換える。
旧測定・旧oracleの成功は旧定義の記録であり、この近似の誤差検証とは区別する。

現在のボトルネックと診断コピーによるatom分割・位相共有の実測は
[atom走査・位相・同期の切り分け](cell-mass-deep-analysis.ja.md)を参照。
同じ数学で約4.63→2.31msとなった診断結果であり、productionへの採用とは区別する。
その後の本体実装・API・scratch寿命は
[atom軸の分担とchunk内の位相共有](cell-mass-atom-splits.ja.md)を参照。
目標の「denseの約2倍のstep時間」との距離は
[AdamW込みのdense比較](cell-mass-dense-step.ja.md)で追跡する。
直近の時間改善は[投影微分の再利用](cell-mass-projection-cache.ja.md)、
dense以下というメモリ制約は[時間・メモリの同時計測](cell-mass-memory-frontier.ja.md)を参照。

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

- globalは元atom、sorted_rows、Zx/Zy、H/G、入出力・勾配。
  `cache_projection_derivatives=True`ならH/Gの幅・中心微分4本も保存し、
  Zx/Zyはそれぞれの投影後に解放する。既定のFalseではParameter勾配で再投影する。
  atom分担が複数なら、Y/dXの呼び出し中だけ部分出力scratchも持つ。atomや係数の全表を複製しない。
- Y/dXのblockは最大4profile tile×4batchの範囲を担当する。
  `sites_per_block`以下に幅を抑え、wide tileはthreadごとに複数siteを展開する。その範囲のprofileタイルごとに
  担当atom集合からのCの部分和をregisterへ蓄積し、同じprofileタイル内のsiteで共有する。
- 32atom chunkを一度読み、寄与するatomの係数だけをsharedへ生成する。
  profile tileが一つなら位相を掛けたsin/cosを全batchで共用する。
  複数tileならraw harmonicを同じshared領域に持ち、各tileのchart imageに応じて位相を計算する。
- 全担当chunkの後にCをsharedへ渡して部分出力へ展開する。
  複数splitなら別kernelで部分出力を合計する。出力atomicは不要。
- H/GとParameter勾配は各threadが異なるatomを担当する。
  そのblock幅はZ・Y/dXと独立に`atom_threads_per_block`で指定する。

```cpp
Z = build_hierarchy(signal);
H = project_selected_whole_tiles(Z, atoms);
for (atom_chunk : assigned_atom_range) {
    prepare_contributing_chunk_coefficients_in_shared();
    sync();
    accumulate_C_for_owned_tiles();
    sync();
}
publish_C_in_shared(); sync();
expand_C_and_store_partial();
// 複数splitのときだけ、同じstream上の次kernelで合計する。
sum_split_outputs();
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
y = physical_linear(
    spec, x, atoms, z_method="merge", sites_per_block=16,
    atom_splits=4, atom_threads_per_block=16,
)
```

Y/dXのblock担当範囲は別の実行設定`sites_per_block=8/16/32/64/128/256`で比較できる。
profileのtile_sitesや周期は変更しない。詳細は[担当site数の比較](cell-mass-block-sites.ja.md)。
`atom_splits=1/2/4/8/16/32/64`はY/dXのatom集合をblock間で分担する。
`atom_threads_per_block=16/32/64/128`はH/G・Parameter勾配のblock幅。
省略時は従来通り32site/1split/128thread・微分再計算で、数式は全設定で同じ。
`double_intermediates=True`はZx/Zyをdoubleで保存する。
微分を保存する経路ではH/Gもdoubleで保存し、保存しない経路では
Parameter勾配の再投影で得たdouble値を直接使う。
入出力・元atom・Parameter勾配のdtypeは変えず、既定値はFalse。
幅や中心の勾配の相殺が大きい場合は、この保存精度も検証対象にする。
通常forwardの準備はmetadata検査、nativeの値検査・key生成、error wordのhost確認1回、
stable argsortの順。atom本体のglobalコピーや係数全表は持たない。
scratch・同期・設定別の測定は[atom分担と位相共有](cell-mass-atom-splits.ja.md)に記録する。

以前のsupport_half_widthを直接渡す研究APIは廃止。IとLの混同を避ける。
今回は1軸1bank、1024以下の2のべき乗site数、65536以下のatom。
[1024向けの拡張](cell-mass-1024.ja.md)は128/256siteのC再利用と最大64splitを可能にする。
性能・メモリの採否はshapeごとの実測を必要とする。
CUDAのtile_sitesはsite_countを割り切る2のべき乗。sigma<=min(Pin,Pout)/3。
複数のsigma bankへの分類やblockのbegin/endによる候補範囲の絞り込みは未実装。
公開CSTLinear/optimizer/Graph/二階微分への接続も未実施。

## タイル窓初版の検証と128サイトでの実測

以下はC共有最適化前の記録。[profileタイル内のC共有](cell-mass-c-reuse.ja.md)に
同じ数学での変更前・変更後の比較を追記する。

source commit `b0ed1754`、NVIDIA L4、Torch2.11.0+cu130、CUDA13.0。
CPU全体は2006 passed / 2824 skipped / 既存JIT警告18件。
L4では対象76件（CUDA 48件、CPU/math 28件）がpassed、failed/skippedは0。
3/5タイル窓、chart全体の1タイル、1/2site、非ゼロ原点、周期端とtileセル境界の両側、
両Z方式、両係数方式、FP32/64、819atomと1024atom上限、全atom勾配を確認した。
ownerが切り替わる点そのものの微分は主張せず、枝内の微分を検査する。

同じ819atom workloadで、新方式のTorch参照に対するCUDAの最大絶対誤差は次の通り。

| dtype | Y | dX | 全Parameter勾配 |
| --- | ---: | ---: | ---: |
| FP32 | 5.96e-8 | 3.73e-8 | 5.96e-8 |
| FP64 | 1.80e-15 | 2.33e-15 | 1.50e-15 |

元の固定誤差基準を緩和していない。ただし、参照する数学が旧clip方式から変わった。
独立Gauss積分と全Parameter方向の差分は、新しいwhole-cell窓を対象にする。

別途、同じraw密度をcore [-I,I]に制限し、core面積で再正規化したcompact oracleに対する
近似誤差をFP64で測った。今回のsigma分布0.2〜12・819atomの結果であり、全条件の上界ではない。

| 対象 | 最大絶対誤差 | 相対L2誤差 |
| --- | ---: | ---: |
| Y | 0.001283 | 0.2224% |
| dX | 0.001283 | 0.2067% |
| 全Parameter勾配 | 0.002186 | 0.2234% |

入力/出力profileのL1誤差最大はそれぞれ0.00247250/0.00247253、
core外の質量比最大は0.00123627。全819atomでL1<=2*etaを確認した。
site確率の合計と1の差は入力最大8.88e-16、出力最大6.66e-16。
合計を1に保つことと、compact profileへ近似することは別の検査。
ここでのParameter相対L2は全要素のnormであり、各微小勾配の相対誤差保証ではない。

時間とTensor確保ピークはoracleとは別のfresh processで測定した。
128×128・819atom・batch4・FP32・mode8、spacing1、入力T=16/出力T=32、
K=3、ローカル周期48/96、sigmaは0.2〜12の線形分布、seed120。
Zはmerge・1要素4thread、係数は直接多項式。
各processでwarmup3回の後9サンプル、2つの独立processを使った。

| process | forward+backward中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 1 | 7.810 ms | 85,504 byte | 2 MiB |
| 2 | 7.765 ms | 85,504 byte | 2 MiB |

prepare・入力検査を含み、optimizer・Graph・公開CSTLinearは含まない。
peak allocatedは83.5KiB、元のX/dY/atomの17KiBを含む。GPU process全体のVRAMではない。
旧clip試作の約69msから小さくなったが、局所周期・選択窓・正規化対象も修正したため、
同じ数学の演算に対する高速化率とは呼ばない。denseとの速度比較も未実施。

別の段階別GPU event診断は以下。中央値の和で全体時間を置き換えない。

| 処理 | 時間 |
| --- | ---: |
| prepare | 0.395 ms |
| Zx | 0.0369 ms |
| Zy | 0.0369 ms |
| H | 0.197 ms |
| Y | 3.518 ms |
| G | 0.214 ms |
| dX | 2.458 ms |
| Parameter勾配 | 0.621 ms |

この測定時点のY/dXは各4blockで、候補atom範囲の絞り込みと並列度が課題だった。
当時のコンパイル時リソースはY/dXがregister90/thread・shared9,984byte/block（9.75KiB）・stack40、
H/Gはregister86、Parameter勾配はregister166。表示上のLOCALは0。
sharedは旧方式の6,176byteから増えたが、全atomのglobal係数表ではなく現在chunkのみ。
これをwarp占有率やspill trafficの実測とは扱わない。
現行の並列度・リソース・実測は[atom分担の記録](cell-mass-atom-splits.ja.md)を参照。

## 証拠と再現

L4 job `l4job-d08f27f4bc9648c4bd485d0513e62170`。
source archive SHA256
`791829e437236cd2a59b53d791d6a97637453e42f5edbd5619a132e2bfa222eb`、
result archive SHA256
`0549afe74ec77ed00f56c4c643214825c8acbe45d89e23856fb5932b494deab5`。
両hashと321 runtime source filesのsnapshotとのbyte一致を再確認した。
wheel/sdist buildと321 runtime source filesの両packageとのbyte一致も通過。

raw evidence、driver、CPU log、packageと照合記録は
ignored `output/cell-mass-tile-window-20261026/`、source/result archive原本はhost共通poolの
job directoryへ保持した。job成功と所有slotの停止を確認済み。

再実行: CPUは`python -m tools.kernel_dev test --suite cpu`。
CUDA/nvcc/Ninja環境では
`python -m pytest tests/test_cell_mass_coefficients.py tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q`。
この近似設定をpublic dispatcherへ採用する判断は含まない。
