# CSTLinear と既存 atom の再利用（2026-10-02）

公開 Linear の入口を CSTLinear に統一した。NormalizedStripLinear とその公開 export、
独自 p / metadata / checkpoint を削除した。旧 checkpoint の互換 adapter は作らない。

## パラメーター所有

| atoms= | 動作 |
| --- | --- |
| 正の整数 | kernel の初期化方針で新規 Atoms を作る |
| Tensor | detach・複製して新規 Atoms を所有する。指定した device/dtype に変換可能 |
| nn.Parameter | 同じ Parameter を登録する。requires_grad・optimizer の参照と状態を維持 |
| Atoms | 共通 owner と Parameter をそのまま再利用する |

既存座標は選択した kernel の parameter_dim と一致する必要がある。
Parameter / Atoms の constructor 内での device/dtype 変換は拒否する。
その後の Module.to は PyTorch の通常の変換なので、optimizer 作成前に行う。
明示した空表はゼロ演算子として扱う。整数指定の初期化は正の atom 数に限る。
checkpoint の共通 key は atoms.p / chart.* / kernel.* と CSTModule の extra_state。

```python
import torch
from torchcst import CSTLinear, LinearOptions, geometry_presets as layout, presets

chart = layout.product(
    shape=(8, 4),
    axes=(layout.line_pattern(8, spacing=1.0),
          layout.grid_pattern((2, 2), spacing=0.5)),
)
p = torch.nn.Parameter(torch.tensor([[0.2, 0.0, 1.0, 0.2, 0.3]]))
optimizer = torch.optim.AdamW([p], lr=1e-3)
layer = CSTLinear(chart=chart, atoms=p, kernel=presets.NORMALIZED_RADIAL_TRIWEIGHT,
                  linear_options=LinearOptions(memory='window'))
assert layer.atoms.p is p
optimizer.zero_grad(set_to_none=True)
layer(torch.randn(2, 4)).square().mean().backward()
optimizer.step()
```

## 実行の分担

chart と kernel の宣言は数学を表し、LinearOptions は full/window の実行選好を表す。
共通 _backends/linear.py が固定 regular 3D Euclidean の normalized radial 契約を判定し、
_backends/normalized.py が局所支持の Torch または既存 CUDA registry に接続する。
Product と連続 Strip に対応する。対応しない宣言は一般の Torch 参照経路を使用する。

数学的な意味、Triweight、global L2 norm、floor 1e-6、clamped log width .03–3.25、
signed amplitude、joint 3D distance、全5パラメーターの勾配は維持する。
LogWidthSpec の幅端点は FP64 buffer で保存し、float()/double() でも丸め直さない。

metadata の snapshot は初期構築と設定変更時に限る。buffer version / 置換 / device /
dtype を検知して再構成する。通常の forward と Graph capture で host scalar を読み戻さず、
atom は常に現在の Parameter を使う。変更後は capture の外で forward して再構成する。
非連続 Parameter も共有したまま扱い、CUDA backend 内の contiguous コピーに autograd を通す。

通常 torch.optim と CSTOptimizer が利用できる。後者の Graph capture 制限と、同一表への
複数 CST policy owner の制限は既存どおりである。Graph 性能測定は通常の fused capturable
AdamW を使う。GPU 性能による新規 algorithm の承認・昇格はこの変更の対象外。

## 検証

CPU は source と配布 wheel の両方で 516 passed / 201 skipped。lint / format も通過。独立 oracle による出力・dX・全5勾配のテストを
共通 CSTLinear に移した。Parameter 同一性・既存 AdamW 状態・CSTOptimizer・Tensor 複製・
一般 kernel での Atoms 再利用・checkpoint 後の参照維持・metadata 再構成も検証する。

L4 の配布 wheel、CUDA Graph replay、時間・capture を含む allocated peak は
validate_linear_unification.py で測定する。測定結果は完了後に追記する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_linear_unification.py" --label linear-unification --timeout 1800 -- SOURCE_COMMIT
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```
