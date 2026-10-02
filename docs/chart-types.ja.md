> 以下の source commit・コマンド・測定値は検証当時の記録。現在の実行入口は
> [benchmark README](../benchmarks/cuda/linear/README.md)を参照。

# Chart の型と座標パッケージの分離（2026-10-02）

一つの ChartSpec に kind と全種類の nullable field を詰める方式を廃止した。
ChartSpec / ChartState は ABC とし、宣言と Tensor owner を Explicit / Product / Strip の
具体型へ分けた。Geometry・Pattern・Chart は同じ階層の別パッケージに置く。

```text
torchcst/
  geometry/   GeometrySpec、GeometryState、空間の presets
  patterns/   Line / Grid / Points の Spec、PatternState、点列の presets
  charts/     共通 ABC、Explicit / Product / Strip の Spec と State、compile_chart、presets
  kernels/    Profile・parameterization・normalization・KernelSpec / KernelState
  operators/  Chart layout と Kernel の組、live Operator
  _backends/  geometry / patterns / charts / kernels の実際の演算と dispatch
```

ChartSpec の field は geometry / shape / revision のみ。kind は各具体型の固定 ID で、
constructor に渡さない。Explicit は coordinates / trainable / optional spacing を持つ。
Product は axes を持ち、Strip は axes / tile_shape / axis / tile_pitch を持つ。
Explicit の座標を PointsPatternSpec で包む方式も廃止した。

State も同じ分担にする。Product に tile_pitch=None / spacing=None を置かず、
Explicit に axes を置かない。tile_grid / tile_count は StripChartState にのみある。
共通基底の抽象メソッドは reference / declaration などの所有・snapshot の契約であり、
座標生成や距離の演算は戻さない。backend は exact な具体 State / Spec の組と版を照合する。

旧 ChartState(spec) の代わりに compile_chart(spec) または具体的な State を使う。
Layer は Spec のまま受け取り、必要な State を構築する。入力・出力 Chart の組と単一 Chart、
trainable な明示点、bounded な直積展開、Strip の部分端 tile と物理的な pitch を維持する。
explicit の shape は座標表の点数と一致する論理 shape とし、明示した行列配置も表せる。

geometry_presets は euclidean / sphere / torus のみ。
pattern_presets は line / grid / points、chart_presets は points / grid / linspace /
product / strip。旧 mixed presets や geometry からの Chart / Pattern export は残さない。
root の具体型 export と geometry_presets / pattern_presets / chart_presets が公開入口である。

共通の finite scalar / shape / point-table validation と Tensor snapshot helper は内部の
_validation.py / _coordinate_state.py に置く。これらに距離・GPU・演算の実装は置かない。
Geometry の半径、representation、ambient chord metric と更新規則は変更していない。

Chart checkpoint は format_version=2 と具体型・shape・その型の構造を照合する。
旧 tagged ChartState の loader / kind 変換 / 互換 alias は用意しない。
Pattern / Geometry の元の checkpoint 契約は維持する。

## 検証

source とローカル配布 wheel の CPU 全テストは 524 passed / 201 skipped。
lint / format と使用例も確認した。変更前 96c3f86 との比較では、Product / Strip の
Euclidean、Sphere、Torus と ambient / intrinsic、trainable explicit を含む11ケース、
112 Tensor の positions・距離・offset・更新・輸送・初期化・y・dX・atom / 座標勾配が
bitwise 一致した。

型ごとの field / Tensor の不存在、ABC の直接構築拒否、concrete compiler、未登録型・版の
拒否、checkpoint の種類・旧形式の拒否、snapshot と演算の境界を追加検証した。

L4 の配布 wheel は **367 passed / 0 failed / 0 skipped**。source は `76d0a75`。
Chart・Geometry・部分端 tile・Torus・local/exhaustive routing・既存 fused / normalized
dispatch と checkpoint を含む対象テストを実行した。全 CUDA テストの網羅実行ではない。
独立 FP64 oracle で W / y / dX / 全5 atom 勾配を照合し、鋭い支持と floor / clip、
中心・幅を変更した Graph replay、AdamW の Parameter と状態も確認した。

検証 runtime は Torch 2.11.0+cu128 / CUDA 12.8 / Triton 3.6.0 / NVIDIA L4 /
driver 580.82.07。pool の初期 probe は system の CUDA 13.0 だが、実際の検証は
job-local CUDA 12.8 環境で行った。system package は変更していない。
pip は共有 system 環境の未使用 RAPIDS/CUDA 13 package との依存警告を出した。
この環境で確認した範囲は Torch/Triton の対象テストと以下の測定である。

N1024² / M128 / 52,429 atoms（約5%）/ FP32 / TF32無効 / fused capturable AdamW
lr1e-4・weight_decay .01。各ケースは別プロセスで、forward・backward・optimizer を
含む Graph replay の同期 wall time 中央値を測定した。peak は capture から測る。

| 条件 | 経路 | Graph ms | peak allocated MiB | peak reserved MiB |
| --- | --- | ---: | ---: | ---: |
| 通常 sigma3 | full | 0.620308 | 51.8511 | 106 |
| 通常 sigma3 | window | 1.506952 | 46.1655 | 86 |
| 通常条件の比較 | dense | 0.127080 | 50.5024 | 106 |
| 鋭い支持 | full | 0.538673 | 51.8511 | 106 |
| 鋭い支持 | window | 0.306169 | 46.1655 | 86 |
| 鋭い条件の比較 | dense | 0.126887 | 50.5024 | 106 |

全 GPU process usage は未測定。大きい fixture は完全 step の測定であり、全 atom の
独立勾配 oracle ではない。同じ profile の full/window 初期 atom SHA は一致する。
通常幅と鋭い支持を分けて記録し、速度改善や algorithm 昇格の根拠とはしない。

成功 job は `l4job-16c752a2fcc6477abc0dd05be209520d`。
result archive SHA256 を receipt と照合し、152 Python file を source / remote wheel /
installed の間で byte 比較した。local wheel と remote wheel の SHA256 も一致する。
owned GPU は停止済み。最初の job `l4job-e14ae3b8bc8c4a04ad9ca78ceaa13275` は
完了応答を失い、セッションも失われたため測定結果は未確認。pool recovery で
server 上の active session 不在を確認後、同じ source を一度再実行した。
計算エラーとは分類せず、ログを ignored evidence に保存した。
[機械記録と hash](../benchmarks/cuda/linear/results/chart-types-20261002.json) を参照。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_linear_unification.py" --label chart-types --timeout 600 -- 76d0a756e4483cfb810551d0c869bc007c04db28 --chart-suite
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py serve --workers 1 --idle-seconds 0
```

[Chart の field と使用例](../src/torchcst/charts/README.md)、
[Geometry](../src/torchcst/geometry/README.md)、[Pattern](../src/torchcst/patterns/README.md) を参照。
