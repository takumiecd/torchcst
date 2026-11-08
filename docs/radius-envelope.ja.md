# 振幅に依存する半径の包絡線

[English (canonical)](radius-envelope.md) | 日本語

`kernels.radius_envelope`は、比率qを独立に学習するcell-mass座標を宣言します。
`RadiusEnvelopeSpec`とkernel revision8を持つ明示的な追加方式です。
既存Polarの座標・activity更新・checkpoint・default selectorの契約は維持します。

## 数式と契約

各Atomの保存列は`[a, q, output_center, input_center]`です。aが符号付き振幅、
qは[0,1]の学習変数です。入出力の因子は次の物理半径を共有します。

\[
U=r_{\max},\qquad
L(a)=r_{\min}+(r_{\max}-r_{\min})e^{-(a/\tau)^2},\qquad
r=L(a)^{1-q}U^q.
\]

a=0ではL/Uが最大半径に合流します。0から離れるとLは最小半径へ近づき、
Uは最大半径を保ちます。tauは振幅aと同じ単位で、amplitude_maxでは割りません。
|a|=2tauなら選択可能区間は**半径の差**r_max-r_minの98.17%です。
対数軸での区間幅、有用な探索、学習精度の主張ではありません。

qは損失で学習する通常のAtom座標であり、座標移動量やactivityの時計ではありません。
端点の微分は内部点の式を片側へ連続させたものです。

\[
\partial_qr=r\log(U/L),\qquad
\partial_ar=r(1-q)L'(a)/L(a).
\]

両方をAtomの勾配へ通し、半径のdetachは行いません。振幅が厳密に0なら、
境界の傾きとAtomの幾何パラメータへの損失勾配はともに0です。
この設計条件を、普遍的に最適な半径則とは主張しません。
階層・窓のルーティングはphysical cell-mass同様に区分的に一定であり、
境界では選ばれた分岐の導関数を使います。

## CPUでの使用例とoptimizer

```python
import torch
from torchcst import CSTAdamW, CSTLinear, ChebyshevSpec, chart_presets, init, kernels

chart = chart_presets.regular_grid(128, geometry="flat_torus", spacing=1.0)
model = CSTLinear(
    chart, chart, atoms=1024, dtype=torch.float32, backend="factored",
    kernel=kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=8)),
        coordinates=kernels.radius_envelope(
            radius_min=0.5, radius_max=50.0, tau=0.03, q_init=0.0,
            amplitude_max=1.0, amplitude_std=0.1,
        ),
    ),
)
# 既存のDenseモーメントとの橋渡し。CPU上でoptimizer作成前に行う任意の初期化。
report = init.xavier_normal_(model, q=0.0)
optimizer = CSTAdamW(
    model.parameters(), model=model, lr=1e-3, weight_decay=0,
    field_lrs={"radius_ratio": 1e-3},
)
x = torch.randn(8, 128)
model(x).square().mean().backward()
optimizer.step()
q = model.atoms.p[:, 1]  # 学習するAtom Parameterのview。別の状態ではない。
r_input, r_output = model.support_radii()
```

標準の振幅初期値はNormal(0, amplitude_std)の提案を振幅上限でclipします。
`variance_scaling_`・`kaiming_normal_`・`xavier_normal_`の既存の制約付きエネルギー探索も
新しい境界を含めて計算します。許容差と判定規則は従来どおりで、到達できない場合は
modelを変更せず失敗します。中心を保ち、初期化関数へ明示したqで初期化するため、
このqがconstructorのq_initより優先します。

`CSTAdamW`/`CSTOptimizer`は更新提案を|a|<=amplitude_max、q∈[0,1]へ射影し、
中心には既存の幾何更新を適用します。列1の学習率フィールドは`radius_ratio`です。
`amplitude`・`output_center`・`input_center`も独立です。
Adamのモーメントとcounterは通常の意味を保ち、射影ではリセットしません。
通常のtorch.optim.AdamWだけでは座標の範囲を保証しません。不正な行は評価時に拒否します。

Tauと半径の端点はcheckpointへ保存する固定設定で、qはAtomごとの学習状態です。
固定定数はModule.float()/double()を経てもFP64精度を保ち、Atom Parameterは指定dtypeに従います。
境界写像の参照実装はFP64演算・autograd中間値を使い、その後Atomのdtypeへ物理行をcastします。
端点が表現できない場合は、範囲内に入る向きへ符号化半径を丸めます。
全演算FP32の高速kernelではありません。

階層境界では精度形式が影響します。FP32で厳密に0.5の半径でも、内部変数を持つmodelを
doubleへ変換すると0.5+epsilonとなり、別の局所周期を選ぶ場合があります。
CUDA比較ではFP32で符号化した物理行を保ち、physical参照の集計をFP64で行います。
model.double()だけでは、この境界で同じ条件の参照になりません。

## 明示的なCUDA FFN合成

最初のGPU経路は、微分可能な写像と既存のphysical LM Algorithmを合成します。
FP32、幅512・展開率4（512->2048または逆）、8192 Atom、平坦化した行数1..256、
M8/base1/merge2、origin0/spacing1、半径境界[0.5,50]内、amplitude_max<=1、
eager・autocastなしが条件です。physical Planの残りの対応条件・精度検査も適用します。
CUDA演算自体は変更しません。既存physical/Polar Planはrevision8を拒否し、
qの保存行をそのままnativeの半径入力へ渡すことを防ぎます。

```python
from torchcst import FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.radius_envelope import (
    RADIUS_ENVELOPE_LM, RadiusEnvelopeLMAlgorithm,
)

registry = make_algorithm_registry()
registry.register(RadiusEnvelopeLMAlgorithm())
model.selector = FixedSelector(RADIUS_ENVELOPE_LM, registry=registry)
# 検証対象のFFN形状・Atom数を持つmodelを、CPU初期化後にCUDAへ移す。
```

登録とselectorの指定は明示的に行います。上の128正方CPUモデルがFFN専用Planの
対象になるわけではありません。Graph captureと他のGPU形状は未対応です。
メタデータだけのPlanテストではGPUの正しさ・性能は分かりません。
実際に実施した検証は、対応する検証記録を参照してください。
