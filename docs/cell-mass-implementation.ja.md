# 128サイトのcell-mass初版実装

> 旧定義の実装・測定記録。ここでのatom中心の支持clipと境界セル経路は廃止した。
> 現在の数学・API・測定は[1周期分のタイル窓](cell-mass-tile-window.ja.md)を参照。
> 以下の旧数値を、新しい近似方式の同値なbaselineとして扱わない。

2026-10-26。[全体設計](cell-integrated-profile.ja.md)と
[kernelの擬似コード](cell-mass-kernel-flow.ja.md)を、まず一つの明示的な数式で検証する実装。
本書は新しい既定profileや公開dispatcherへの採用を意味しない。

## この初版が計算するもの

最初の対応候補は既存の非負なChebyshevSquareSpec。生密度の関数自体は既存実装を使い、
点評価・離散L2から、支持全体の面積で正規化したセル積分へ変える。
旧Planや既存ProfileBindingの意味は変えず、別の低水準numeric entryで実行する。

各軸は明示的なCellMassAxisSpecを持つ。site iのセルは
`[origin+(i-0.5)*spacing, origin+(i+0.5)*spacing]`、chartの周期はsite_count*spacing。
支持半幅Lは宣言のsupport_half_width、profileの周期は2Lである。
Lは数学的な支持であり、CUDAのleaf寸法8や出力タイル32とは別。

生密度を`g(t)=c0+2*sum_(k=1..M) ck*cos(k*pi*t/L)`とすると、面積S=2L*c0。
セルと[-L,L]の重なりを原始関数で積分し、chartの周期だけずらした全ての必要な像を足す。
最短周期差で一回評価する近似は使わない。各軸のsite確率の和は1、W/Yの和は一般には1ではない。

初版のphysical_atomsは`[A,4]`の連続配列で、列は次の順序。

| 列 | 値 | 微分 |
| --- | --- | --- |
| 0 | 生の振幅alpha | そのままの振幅方向 |
| 1 | 正のsigma | 入力・出力両profileを通る微分を合算 |
| 2 | 出力中心mu_out | セル積分の中心微分 |
| 3 | 入力中心mu_in | セル積分の中心微分 |

これは既存PolarのParameter行ではない。sigmaと振幅を独立した物理量として渡す。
旧checkpointをそのままこの配列と解釈しない。既存CSTLinear/CSTOptimizerとの接続や
ParameterizationのVJP・更新則の選択は、別に確認する段階である。
既存native profileの安定な範囲として`0 < sigma <= min(2L_in,2L_out)/3`を検査する。

## CPU参照から読む

```python
import torch
from torchcst.kernels.cell_mass import CellMassAxisSpec, CellMassLinearSpec
from torchcst.kernels.profiles.fourier import ChebyshevSquareSpec
from torchcst._backends.torch.kernels.cell_mass import physical_linear_reference

profile = ChebyshevSquareSpec(modes=8, tail_bound=1e-3)
spec = CellMassLinearSpec(
    input=CellMassAxisSpec(profile=profile, site_count=128,
                          spacing=1.0, support_half_width=24.0),
    output=CellMassAxisSpec(profile=profile, site_count=128,
                           spacing=1.0, support_half_width=32.0),
)
x = torch.randn(4, 128, dtype=torch.float64, requires_grad=True)
physical_atoms = torch.tensor(
    [[0.7, 2.0, 0.1, 127.8], [-0.2, 5.0, 63.2, 31.4]],
    dtype=torch.float64, requires_grad=True,
)
y = physical_linear_reference(spec, x, physical_atoms)
dx, dparams = torch.autograd.grad(y.sum(), (x, physical_atoms))
```

参照のprofile_cell_massesは、native係数からセル積分と面積を直接計算する。
physical_linear_referenceはX→H→Yの因数分解で、W全体を作らない。
参照計算のFourier係数・profileの一時Tensorは独立検証用であり、CUDAの保存方針ではない。
Torch autogradで面積のsigma微分も含める。

## CUDA実装の入口と担当

```python
from torchcst._backends.cuda.algorithms.linear.cell_mass.executor import physical_linear

# 元のatom配列を渡す。行番号だけソートし、globalのatomコピーは作らない。
gpu_x = x.detach().cuda().requires_grad_()
gpu_atoms = physical_atoms.detach().cuda().requires_grad_()
y = physical_linear(spec, gpu_x, gpu_atoms,
                    z_method="direct", z_threads_per_element=4)
dx, dparams = torch.autograd.grad(y.sum(), (gpu_x, gpu_atoms))
```

executor.pyが起動順とautogradの保存状態を持ち、bindings.cppはPythonとの接続、
kernels.cuが実際のCUDA C++。コンパイルはCUDA実行時に遅延し、宣言のimportでは起動しない。

| 段階 | 実装の担当と保存 |
| --- | --- |
| prepare_layout | output/input中心のタイルキーで行番号だけをGPUソート |
| build_z_direct_kernel | node/batch/modeを並列化し、siteは可変個数のthreadで部分和を取る |
| merge_z_level_kernel | 同じ基底を持つ子二つを足し、親の各要素を一つのthreadが保存 |
| compute_h / compute_g | 各threadが別々のatom。必要なZを足し、残りのセルを直接積分して完成値を保存 |
| compute_y / compute_dx | blockが32site×4batchを所有。小タイルごとのCを作り展開し、境界の残りを加えて一度保存 |
| compute_parameter_grad | 各threadが一つのatomの全batchを合計し、元のParameter行へ4方向の勾配を保存 |

H/Gの二方向はproject_atoms_kernel、Y/dXはexpand_tiles_kernelの同じ処理を使い、
方向ごとに明示的なnative入口を設ける。Y/dXのblock間atomicは使わない。

CUDAでは必要な係数とsigma微分を利用時に計算する。初版の2M+1点のDFTは比較用に残し、
現在は[同じ係数を直接求める正の多項式](cell-mass-coefficient-optimization.ja.md)を既定にする。
いずれも密度の近似fitではなく、同じ多項式の係数変換である。
全atom分の係数表・係数微分表をglobalへ作らない。

Y/dXは必要なatomの32個ずつのchunkとH/Gをsharedへ集め、使用後の同期を挟んで上書きする。
H/G・Parameter勾配では元のatomをregisterへ読む。保存するZ/H/Gはglobalで、
forward状態はautograd呼び出しが所有する。元のParameterをbackward前に更新するとversion検査で拒否する。

## 初版の対応範囲と比較スイッチ

- 入出力とも1次元、各軸に明示的な一つのFourier bank。128以下の2のべき乗site数。
  初版のatom上限128は、[5%条件の最適化](cell-mass-coefficient-optimization.ja.md)で1024へ拡張した。
- ChebyshevSquareSpec、2〜16の偶数mode、float32/float64、連続した入力とphysical atom行。
- CUDAでは支持の全幅2Lがchartの周期以下。CPU参照はより広い支持の複数像も検証する。
- 初版の各bank内では全階層が同じ基底を持つ。leafは最大8site、階層の全ノードをZに保存する。
- z_method="direct"なら全ノードを直接並列生成。"merge"ならleafだけ直接生成し、親はlevel順にマージ。
- z_threads_per_elementは1〜128の2のべき乗。まず1・4・8を同じ数学で比較する。

CUDA Graph capture、二階微分、多次元chart、sigmaによる複数bank分類、group範囲による
atom走査の限定、公開Algorithm/Recipe登録は未対応。初版は各出力blockがatomを全走査する。
入力のfinite/sigma検査にはhostへ結果を読む同期があり、現在の時間にはその費用も含める。
未対応の入力を旧L2や別のprofileへfallbackしない。

既存native密度は支持端で厳密に0とは限らず、支持端がセル端へ一致する点では
中心微分が非滑らかになる場合がある。通常点の独立微分検証と、この境界の規則の検証を分ける。
このprofileを将来の既定へ固定したことにはしない。

## 検証と記録

CPUのtest_cell_mass_reference.pyはnative密度のscalar評価をGauss積分する独立oracleと比較する。
単一セルより狭い支持、周期端、chart一周を超える支持、128×128、0振幅、空atom集合、
Y/dX/全物理Parameter方向の微分、gradcheckを含む。
CUDAのtest_cell_mass_cuda.pyは両Z方式、float32/64、全atom勾配、batch/chunk末尾、
空atom、保存状態の再利用、Parameter更新の検出、mode上限とthread配分を確認する。

```bash
python -m pytest tests/test_cell_mass_reference.py tests/test_cell_mass_cuda.py -q
```

GPUのない環境ではCUDA部分はskipされる。skipをGPU検証の成功とは扱わない。
実装後のCPU suiteは1989 passed / 2795 skipped / 18件の既存JIT警告。
新しい対象テスト単独では11 passed / 19 CUDA skipsで、Ruff検査も通過した。
wheel/sdist buildを通過し、321個のruntime sourceが両artifact内でcheckoutとbyte一致した。

L4での初回correctnessは20 passed / 0 failed / 0 skipped。
両Z方式、float32/64、Y/dX/全atomの4方向勾配、保存状態、空atomと末尾要素を含む。
source commitは`94042a881f3fa8d5cb341ebc0ceda980bd1073ee`、jobは
`l4job-709806652fc24d7b862424f0977523e2`。
実機はNVIDIA L4、Torch 2.11.0+cu130、CUDA 13.0。
source archive SHA256は`e3ed3eb48fba9adebe15b5893ee069f7f9ae81950ad4396db16601588ebec64a`、
回収結果archive SHA256は`7d28d7396a3e33d080ab800f90d0219436781e1fb3b88e713008fd474797dc9a`。
回収archiveのhashと、テストされたCUDAソースのbyte一致も手元で再確認した。

最初の同一process内の診断ではforward+backwardは約206msで、速度改善の採用結果ではない。
検証に使ったTorch計算の常駐allocationが残る可能性があるので、そのprocessの8,554,496 byteの
peak allocatedを、新方式だけの必要容量とは解釈しない。fresh processの測定を別に記録する。
測定専用の追加job `l4job-56dc6048ea8a41c992540cd61a2ff34d`は、driverがNinjaの準備を
欠いてコンパイル前に停止した。正しさの失敗や有効な測定結果としては数えず、失敗ログも保持する。
raw logs・JSON・driverとpackage照合記録はignored `output/cell-mass-implementation-20261026/`、
元のsource/result archiveはhost共通poolの各job directoryに保持する。

### fresh processでの診断値

測定専用jobは`l4job-bd97107095ca4fef94e7e2df5f8d1053`。
同じruntime sourceで、方式ごとに新しいprocessを起動した。
128×128、128atom、batch 4、FP32、mode 8、1つのZに4thread。
warmup後の5サンプルの中央値で、JITコンパイルを除く。
forward+backwardはprepareと入力検査を含むwall時間である。

| Z方式 | forward+backward | peak allocated | peak reserved |
| --- | --- | --- | --- |
| 全nodeを直接生成 | 206.483 ms | 34,816 byte（34 KiB） | 2,097,152 byte（2 MiB） |
| leaf生成＋親マージ | 206.060 ms | 34,816 byte（34 KiB） | 2,097,152 byte（2 MiB） |

これはTorch allocatorの計測で、GPU process全体のVRAM使用量ではない。
shared/registerやドライバの容量をTensorのallocated値へ混ぜない。
このケースで同時に存在する主なTensorの理論容量は以下の通り。

| Tensor | 計算したbyte数 |
| --- | --- |
| X・dY・元のphysical_atoms | 6,144 |
| ソート済み行番号（int64） | 1,024 |
| Zx・Zy（各4×31node×17基底×FP32） | 16,864 |
| H・G | 4,096 |
| Y・dX・physical_atoms勾配 | 6,144 |
| 合計（確保単位の丸めなどを含まない） | 34,272 |

実測peakの34,816 byteと、Tensor値だけの34,272 byteは区別する。
元atomのglobalコピー、全係数表、全係数微分表、全Wは含まれていない。

別に実行したphase診断のGPU event中央値は次の通り。診断値の和を完全step時間の代用にはしない。

| phase | 全node直接 | leaf＋マージ |
| --- | --- | --- |
| prepare | 0.359 ms | 0.342 ms |
| Zx | 0.235 ms | 0.036 ms |
| Zy | 0.233 ms | 0.036 ms |
| H | 28.175 ms | 28.149 ms |
| Y | 49.405 ms | 49.390 ms |
| G | 18.204 ms | 18.193 ms |
| dX | 47.443 ms | 47.426 ms |
| Parameter勾配 | 61.887 ms | 61.872 ms |

この実行ではZ単体はマージが速いが、主要な時間は5つの計算段階側にある。
128atom・batch 4ではH/Gが各1block、Parameter勾配も1blockで、GPU全体を使い切れる仕事の
分け方にはなっていない。今後、係数のその場計算の頻度とblock/thread配分を調べる。
この1回の小規模診断で、一般の採用方式や速さの原因を断定しない。

### 標準atom数をN²の5%にする場合の容量

ユーザー指定の今後の標準は正方N×Nに対してatom数を約0.05*N²とする。
128×128では819.2、約819atomになる。端数処理の規則はまだ固定しない。
上の実測は128atom（約0.78%）の初版検証であり、5%ケースの測定値ではない。
初版の128atom上限を拡張した5%ケースの実測は、上記の最適化文書へ記録する。

同じFP32・batch 4・mode 8・leaf 8のまま819atomにした主要配列の理論容量は次の通り。
これは確保単位の丸め、prepare/sortの一時領域、optimizer、Graph、ライブラリworkspaceを
含まない配列予算で、実測peakの予測値やdenseとの実測比較ではない。

| 比較する配列 | CST：819atom | dense：128×128 |
| --- | --- | --- |
| パラメータ | 13,104 byte（約12.8 KiB） | W：65,536 byte（64 KiB） |
| パラメータ勾配 | 13,104 byte | dW：65,536 byte |
| H/G | 26,208 byte | なし |
| Zx/Zy | 16,864 byte | なし |
| ソート済み行番号 | 6,552 byte | なし |
| X/dY/Y/dX | 8,192 byte | 8,192 byte |
| 合計 | 84,024 byte（約82.1 KiB） | 139,264 byte（136 KiB） |

1atomが振幅・sigma・二つの中心の4値を持つので、5%のatom数でもパラメータ保存量は
denseの約20%になる。H/G等を加えた上の予算では約60%であり、5%の容量になるわけではない。
Adamの二つのmomentを含める場合は、理論値にCST約25.6 KiB、dense 128 KiBがさらに加わるが、
実際のoptimizer stepには方式ごとの一時領域があり、別の測定が必要である。

上記128atomのfresh-process測定のsource archive SHA256は
`08e2b1e0bd64a624afb8f853c26d7624bc31618dcf5d9a11c6e1981cc929fe82`、
回収archive SHA256は`d6cb27b7eb90fafa35829845596aceec74fd2dfcc4fcd3cb9ca9ca0cdc3d8cf7`。
両hashを再照合し、所有pool slotの停止を確認した。検証用jobと測定用jobを含め、旧研究の
停止中の実験は再開していない。

公式Linear runnerへの新fixture/semanticsの接続と完全optimizer step・Graph込みの比較は次の段階であり、
この低水準entryのforward/backward時間だけを公開dispatcherの採用根拠にはしない。
