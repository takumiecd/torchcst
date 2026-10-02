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

L4 の配布 wheel は **291 passed / 0 failed / 0 skipped**。source `28cc041`、
Torch 2.11.0+cu128 / CUDA 12.8 / Triton 3.6.0 / NVIDIA L4 / driver 580.82.07。
pool の初期 probe は CUDA 13.0 の system 環境だが、実際の検証は job-local CUDA 12.8
環境で行った。system package は変更していない。probe と検証 runtime を混同しない。

独立 FP64 oracle で W / y / dX / 全5勾配を照合した。full / window の混合・鋭い支持、
clip・floor・512行境界・capture 後の中心と幅の変更、AdamW の Parameter と状態を含む
Graph replay が通過した。shared Parameter の非連続 stride と、対応しない dtype / 間隔の
一般参照への fallback も通過。全 CUDA テストの網羅実行ではなく対象291件である。
CPU は旧入口 `e20f0df` と8ケース・24 Tensor の y/dX/dP を比較し、bitwise 一致した。
ドキュメントの使用例も CPU で実行した。

N1024² / M128 / 52,429 atoms（約5%）/ FP32 / TF32無効 / fused capturable AdamW
lr1e-4・weight_decay .01。各ケースは別プロセス。時間は forward・backward・optimizer
を含む Graph replay の同期 wall time 中央値。peak は capture から測定する。

| 条件 | 経路 | Graph ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: |
| 通常 sigma3 | full | 0.613194 | 51.8511 | 106 |
| 通常 sigma3 | window | 1.503789 | 46.1655 | 86 |
| 通常条件の比較 | dense | 0.128718 | 50.5024 | 106 |
| 鋭い支持 | full | 0.537161 | 51.8511 | 106 |
| 鋭い支持 | window | 0.306918 | 46.1655 | 86 |
| 鋭い条件の比較 | dense | 0.127008 | 50.5024 | 106 |

全 GPU process usage は未測定。大きい fixture は完全 step の測定であり、全 atom の
独立勾配 oracle ではない。同じ profile の full/window 初期 atom SHA は一致する。
通常幅と鋭い支持を別に記録した。速度改善や性能による algorithm 昇格の根拠とはしない。

job は `l4job-0ddda738c14c4dd39d92b5019a25fb3d`。
source archive SHA256 は `95f6d0b9169d58b5b97eabcaba2ec1ab987c5b082e389c81e878ed3bee886a02`、
result archive は `a15bfa4463b584a3b6c7075ee57f8ae5276733c447128f5b8e607f1d2360c6db`、
remote wheel は `be0085485c1957ab8e34901bba6da89f047686562296be6e024fd101a722b006`。
result archive を receipt SHA と照合し、140 Python file を source / wheel / installed の間で
byte比較した。旧専用 module が含まれないことも確認した。owned GPU は停止済み。
raw source / logs / wheel / tensors は ignored evidence に保存している。
[機械記録](../benchmarks/cuda/linear/results/linear-unification-20261002.json) を参照。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_linear_unification.py" --label linear-unification --timeout 1800 -- SOURCE_COMMIT
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```
