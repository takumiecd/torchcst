# Kernel の宣言と状態

利用者は `KernelSpec` を `CSTLinear` / `CSTConv2d` に渡す。preset は Spec を組み立てる
関数で、演算するクラスではない。個別の Kernel / Profile Module、ABC、互換 alias、
旧 checkpoint の読み替えは削除した。

## 配置と責務

| 配置 | 責務 |
| --- | --- |
| `profiles/` | Gaussian、Triweight など関数の形の不変な宣言。幅・正規化・演算を持たない。 |
| `parameterizations/` | atom 座標の解釈。固定幅、振幅依存幅、Direct / Polar activity、log width。 |
| `normalization.py` | 正規化の規則、対象領域、ノルム床。 |
| `spec.py` | ProfileBinding、radial / separable / amplitude の合成、初期化・更新方針。 |
| `presets.py` | 宣言を組み合わせる利用者向けの関数。 |
| `options.py` | site / atom chunk と checkpoint の実行設定。数学的な Spec と分離。 |
| `state.py` | 共通 KernelState / ProfileState。固定 Tensor、device / dtype、checkpoint を管理。 |
| `../_backends/torch/profiles/` | 関数の評価と解析微分。 |
| `../_backends/torch/parameterizations/` | 振幅・activity・幅の計算。 |
| `../_backends/torch/kernels/` | atom の合成、重み生成、初期化、factor 微分。 |
| `../_backends/torch/updates/` | 勾配射影、座標更新、optimizer 状態輸送。 |
| `../_backends/cuda/algorithms/` | 数学的な契約に対応する CUDA の計算方式。 |

`KernelSpec` は Tensor を持たない。`KernelState` は演算メソッドを持たない。
学習する `p` は `Atoms` が所有する。Layer が Spec から共通 State を作り、
Operator / backend が State と現在の Chart / p を使って計算する。

## 宣言の作り方

```python
from torchcst import GaussianSpec, TriweightSpec, BandwidthBounds, presets

# 固定幅：入力・出力それぞれの形、幅、正規化を宣言する。
fixed = presets.amplitude(
    presets.separable(
        input_profile=presets.fixed_profile(GaussianSpec(), 0.4),
        output_profile=presets.fixed_profile(TriweightSpec(), 0.7),
    )
)

# 動的幅：ProfileBinding に幅を入れず、parameterization が幅を定める。
activity = presets.polar_activity(
    amplitude_max=1.0,
    input_bounds=BandwidthBounds(minimum=0.1, birth=2.0, maximum=2.0, upper_floor=0.1),
    w_c=0.1,
    profile=presets.profile(TriweightSpec()),
    radial_regularization=0.2,
)

# 単一 Chart：原則 radial。Direct activity は非正規化 profile を使う。
radial = presets.direct_activity(
    amplitude_max=1.0,
    input_bounds=BandwidthBounds(minimum=0.2, birth=0.5, maximum=0.8, upper_floor=0.2),
    w_c=0.05,
    profile=presets.profile(TriweightSpec(), normalize=False),
)
```

`direct_activity` の既定は radial。入力・出力を分ける場合は
`composition="separable"` とし、必要なら `output_bounds` を指定する。
`polar_activity` と `amplitude_width` は separable。固定幅 radial は
`presets.radial(presets.fixed_profile(...))` とする。amplitude は両方に合成できる。

幅を二重に宣言すると拒否する。`profile()` は形と正規化、`fixed_profile()` は
それに固定幅を追加する。profile の入力側と出力側を別々に宣言した場合、
backend はそれぞれの形を使う。`couple_bandwidth` も微分の契約で、実行設定ではない。

```python
from torchcst import CSTLinear, KernelOptions

layer = CSTLinear(
    chart=chart,
    atoms=128,
    kernel=radial,
    kernel_options=KernelOptions(site_chunk=2048, atom_chunk=64),
)
spec = layer.declaration()  # 現在の固定 Tensor 設定を明示的に snapshot
```

## 実行と checkpoint

`Layer -> Operator -> backend` が演算の入口。State の固定 Tensor は `.to()` と
checkpoint に従う。`declaration()` は設定境界の snapshot で、forward / backward /
CUDA Graph capture では呼ばない。初期化・更新・状態輸送も backend が実行する。
CSTOptimizer の checkpoint は、共通 State のクラス名だけでなく数学的な Spec を
記録する。model / optimizer の state_dict は `torch.load(weights_only=True)` で読める。

backend が未対応の形、版、parameterization、初期化・更新方針は拒否する。
custom subclass が組み込み ID を引き継ぐ旧拡張方法は廃止した。新しい意味は宣言と
対応する backend 実装を追加する。旧 API の利用例は Git revision `c21ea97` を参照する。

`NORMALIZED_RADIAL_TRIWEIGHT` は operator 全体での離散 L2 正規化と log width の
固定契約。`NormalizedStripLinear` の既存の CUDA full / window 実行に使う。
任意の宣言が CUDA Registry に自動登録されるわけではなく、対応・検証・承認・
既定 dispatch への採用は別の段階である。
