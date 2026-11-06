# Denseと二乗平均目標を揃える初期化

[English](initialization.md) | **日本語**

`torchcst.init`は`CSTLinear`用のHe/Kaiming・Xavier/Glorot初期化を明示的に
提供します。互換性は対応するDenseと同じ平均的な作用素の二乗平均目標を意味し、
独立なDense重み・同じスペクトル・各行のエネルギー・すべての入力の勾配・
モデル精度の一致を意味しません。constructorの初期化は保持し、明示opt-inとします。

## API

対応する層をCPU上で作り、初期化後にCUDAへ移動し、optimizerを作成します。

```python
import torch
from torchcst import init as cst_init

# layerは2-chart Polar radius cell-mass kernelを持つCPU CSTLinear。
report = cst_init.kaiming_normal_(
    layer,
    mode="fan_in",
    nonlinearity="relu",
    q=0.0,
    generator=torch.Generator().manual_seed(17),
)
print(report.relative_error, report.clipped_atoms, report.atom_count)
layer = layer.cuda()
# 初期化後にoptimizerを作成する。
```

別のpresetと拡張用の共通入口は次のとおりです。

```python
report = cst_init.xavier_normal_(layer, gain=1.0, q=0.25)
# 新presetの例：Dense重みの標準偏差0.02に対応する目標。
report = cst_init.variance_scaling_(layer, variance=0.02**2, q=0.0)
```

これらの呼び出しもCPU上・optimizer作成前に行います。毎回新たに初期化します。
`generator`は任意指定で、CPU乱数を生成するものを使います。`q`・`s`・`t`は小文字です。
`torch.nn.init`と異なり、戻り値は入力Tensor/Moduleではなく、不変の
`InitializationReport`です。scheme・q・Dense分散・目標/期待エネルギー・相対誤差・
名目振幅scale・物理振幅RMS・クリップ数・Atom数・評価回数を記録します。
実験metadataと一緒に保存してください。moduleのstate dictへは追加しません。

現時点の対象はCPU FP32/FP64の`CSTLinear`、2-chart Polar radius cell-mass productです。
非対応kernelとGPU上での初期化はエラーにします。Dense W・Denseモデル・TRAIN特徴・
ラベルは不要です。中心・宣言・Atom数・Parameter形状とidentityを保持します。
探索失敗時はmodule状態を変更しませんが、乱数generatorは進む場合があります。
backward/optimizer作成前に行い、再初期化後に古いoptimizer momentsを再利用しないでください。

## DenseからCSTへの橋渡し

W=sum_a a_a Phi_aとし、振幅を除く形状Phi_aを固定し、振幅を独立・ゼロ平均・
分散sigma_a²とします。E_K=sum_a ||Phi_a||_F²なら、

```math
\mathbb E\|W\|_F^2=\sigma_a^2 E_K,\quad
f_{in}=E_K/d_{out},\quad f_{out}=E_K/d_{in}.
```

Heはsigma_a²=gain²/f_mode、Xavierはsigma_a²=2*gain²/(f_in+f_out)です。
Denseの全要素を座標基底Atomで表すとE_K=d_in*d_outとなり、元のDenseの式へ
正確に戻ります。正方形でA個のAtomの単位振幅平均エネルギーがeなら、
He/ReLUはsigma_a=sqrt(2/e)/sqrt(A/N)です。

共通APIは**Dense重みの分散**vを受け取り、E||W||_F²=v*d_in*d_outを目標にします。
Heは`torch.nn.init.calculate_gain`を使い、v=gain²/d_inまたはgain²/d_outです。
ReLUのgainはsqrt(2)。Xavierはv=2*gain²/(d_in+d_out)です。
今後のpresetはDense分散を計算して`variance_scaling_`を呼べばよく、
作用素固有solverにpresetを追加する必要はありません。別の活性化関数や
残差深さのscaleには、それに対応する目標を使います。

## 半径の割合と実際のsolver

rho=sqrt(s²+t²)、振幅a=a_max*s/rho、q=clip((rho²-1)/3,0,1)に対し、
物理半径はR=L(a)^(1-q)*U(a)^qです。初期化時にqを0〜1で選び、
rho=sqrt(1+3q)、s=rho*a/a_max、t=rho*sqrt(1-(a/a_max)²)と符号化します。
q=0は下限、q=0.5は幾何平均、q=1は上限です。qはAtom密度alpha=A/(d_in*d_out)と
区別します。初期条件の指定であり、学習中のradial activityを固定する拘束ではありません。

Polarでは半径と因子エネルギーも振幅の絶対値に依存します。その依存を含めて
二乗平均目標を解きます。標準正規乱数zを生成し、scale*zを
a_max*(-1+1e-6,1-1e-6)へ対称クリップします。実際のmodule dtypeでAtomsを符号化し、
CPU FP64で因子エネルギーを評価します。絶対値と中心を固定した独立符号の期待値は、

```math
\mathbb E_{sign}\|W\|_F^2=
\sum_a a_a^2\|\Phi_a(|a_a|,q)\|_F^2.
```

形状が符号反転で不変なので、Atomが重なっていても交差項が消えます。
1回の実現Wはこの条件付き期待値と一致するとは限りません。
因子ベクトルの二乗ノルムの積を使い、Atom×出力×入力配列やDense重みを作りません。

相対scale 0・.025・.05・.1・.2・.4・.8・1.6・3.2・6.4・12.8・32をa_max倍して
試し、符号の異なる区間を最大32回二分探索します。全体の単調性を仮定せず、
実測の相対目標誤差0.5%以内のみ受け入れます。失敗時は履歴を含むエラーにし、
大域的到達不能性の証明にはしません。q・Atom数・範囲・kernel正規化を変えたり、
出力gainを追加したりしません。

`normal`はクリップ前の振幅乱数を意味します。多くのAtomがクリップされる場合があり、
物理振幅RMSは名目scaleと大きく異なり得るため、reportを確認してください。
振幅上限とkernel形状によっては目標に到達できません。
非負・総質量1のAtom行列で振幅上限a_maxなら、期待エネルギーはA*a_max²以下となり、
これが必要な実現可能性条件です。

[cst-researchの探索的HIGGS pilot](https://github.com/takumiecd/cst-research/pull/26)は
この二乗平均原理を検証します。固定consumerとGaussian乱数は本公開APIとは別です。
ここへの移行は過去の結果の再実行・再検証を意味しません。
初期の信号scaleと最終精度は別の主張です。
