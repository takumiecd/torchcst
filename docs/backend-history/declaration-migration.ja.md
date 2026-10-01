# Kernel Geometry Chart の宣言分離

2026-10-01。`codex/cuda-dispatch-registry-v1` の段階的な研究変更。
公開クラスが所有する設定・Tensor 状態・checkpoint と、Torch の計算本体を分けた。
数学的な意味は frozen dataclass で参照でき、実行時の値・勾配・状態は従来の
Module が所有する。宣言だけから実行する Operator / factory は次の段階である。

## ソースの区切り

- `e27e3a4`: Kernel / Profile / parameterization / normalization の宣言、
  組み込み評価・微分・初期化・更新の Torch backend への分離。
- `0455e786edff630011a315ed8ef58127f7a5f00d`: Geometry / Pattern / Chart の宣言、
  座標生成・距離・center 変換・射影・retraction・輸送・Strip support の分離。
  検証 driver に配布 wheel の全テスト実行を追加し、bench の source hash を
  package 全体へ広げた。

現在の配置と移行の境界は[配置規約](../../src/torchcst/_backends/README.md)、
[Kernel](../../src/torchcst/kernels/README.md)、[Geometry と Chart](../../src/torchcst/geometry/README.md)を参照。

## CPU と移動前後の比較

ローカル Python 3.11.15 / Torch 2.13.0。移動前は 498 passed / 189 skipped、
宣言の境界テスト 36 件を追加した後は 534 passed / 189 skipped。
既存の Torch JIT deprecation warning 18 件は変わらない。

移動前 `18ec3fb` に対する Kernel 65 ケースで、値、一次・二次微分、座標更新、
状態輸送、checkpoint の keys / metadata を比較した。Geometry / Chart 15 ケースで、
Euclidean、Sphere / Torus の ambient / intrinsic 表現、Product / Strip / Explicit、
点の位置、距離、勾配、bounds、端の tile、retraction / transport と checkpoint を
比較した。有限値は `rtol=0, atol=0` で一致し、非有限値の位置も一致する。

正規化 WendlandC2 の coincident-center fixture には移動前から二次微分の非有限値が
4 ケースに計 6 要素ある。これを有限性の成功として数えず、元と同じ位置であることを
確認した。この変更ではその数学的・数値的な問題の修正は行っていない。

移した 192 メソッドの計算部分の AST も一致する。比較では docstring を除き、
移動による import の変更と、free function での明示的な `super(Class, self)` の
MRO anchor を正規化した。公開型名と checkpoint tag は既存のまま保つ。

ローカル wheel を `uv build --wheel` で作り、宣言・backend の 63 Python module が
ソースと同じ bytes で収録されていることを確認した。原始ログ・比較スクリプト・
移動前 Tensor は ignored `benchmarks/cuda/linear/evidence/kernel-declarations/` に保存。

```bash
python -m pytest -q
python benchmarks/cuda/linear/evidence/kernel-declarations/compare_kernel.py \
  --compare benchmarks/cuda/linear/evidence/kernel-declarations/before.pt
python benchmarks/cuda/linear/evidence/kernel-declarations/compare_geometry.py \
  --compare benchmarks/cuda/linear/evidence/kernel-declarations/geometry-before.pt
python benchmarks/cuda/linear/evidence/kernel-declarations/audit_bodies.py
uv build --wheel
```

## L4 配布 wheel 検証

測定 source は `0455e786edff630011a315ed8ef58127f7a5f00d`。
job ID は `l4job-94c7dc9c37a6478f8dd4ea8517f7407f`。
source archive SHA256 は
`ebd9bfd745ca046f0d89ebd111d373e36ed1f2454f4327058d6ef6c4167e6a6c`。
新しい原始 evidence が snapshot に含まれないことを確認して提出した。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source "$PWD" --script "$PWD/benchmarks/cuda/linear/validate_dispatch_registry.py" \
  --label declaration-migration-wheel --timeout 1200 \
  -- 0455e786edff630011a315ed8ef58127f7a5f00d --full-suite
```

全テストを配布 wheel から実行し、独立 FP64 oracle と Graph / optimizer の照合後、
N=1024 / M=128 の normalized full / window と dense の完全な AdamW 学習ステップを
測る。sigma-three と sharp は別 fixture とする。allocated peak は Graph capture
を含み、reserved と分ける。process 全体の使用量を allocated と混同しない。
この変更は方式の性能昇格や承認ポイントの導入ではない。

### 回収障害と再提出

最初の job は remote wrapper の `POOL_RESULT_READY` まで進んだが、Colab の
結果ダウンロードが HTTP 503 で失敗した。receipt / manifest / driver の出力を
回収できないため、正しさ・性能の成功とは扱わない。pool の `recover` を実行し、
所有 runtime の停止を確認した。ソース・提出設定・転送ログは ignored evidence に保存。

再提出 job は `l4job-b21c83263adc4ebd9265fa1819ca6111`。
source archive SHA256 は
`084b235b048ccd7686973d1bf596272faf64f4bd745033a860a44e40bc5c2901`。
README の追加により archive は異なるが、全 Python ファイルの SHA256 は
最初の job と一致する。測定実装は引き続き `0455e78` である。

### 回収済み結果

再提出は成功。driver returncode 0 / timeout false、配布 wheel の全テストは
**788 passed / 2 skipped / 1 warning**、505.33 秒。skip は macOS socket binding と
A100 専用 FP fusion の検証で、この L4 / Linux 環境では適用されない。warning は
cuBLAS primary context の設定である。

実機 NVIDIA L4（SM 8.9、58 SM、23034 MiB）、driver 580.82.07。
Python 3.13.15、Torch 2.11.0+cu128、CUDA 12.8、Triton 3.6.0。
結果 archive SHA256 は
`1a3b109e9855ed23b13b068aaf5129342ced036acf6c620c95353a283be2d926`。
receipt と結果 archive、source archive ID の一致を確認した。配布された全 Python
module の SHA256 も提出した source_files と一致する。pool の全 slot が stopped
になったことを確認し、source / receipt / results を ignored evidence に保存した。

小さい混合 fixture の独立 FP64 oracle で normalized full / window の y、dX、
全 atom gradient、weight、更新後の値・勾配を照合し、Graph / AdamW の検証も通った。
最大 absolute error は y 2.88e-8、dX 3.46e-8、atom gradient 1.98e-4
（relative L2 6.92e-5）で、既存の許容範囲を通る。更新後も既存の許容範囲を通る。

完全ステップは N=1024（1024×1024 weight）、batch=128、atoms=52429（約5%）、
FP32 / TF32 off、seed=21。AdamW lr=1e-4 / weight_decay=0.01、fused / capturable。
通常の broad は初期 sigma=3、sharp は初期 sigma=0.199 で支持点に近い center の
別 fixture。full と window の初期 atom hash と optimizer 契約は同じ。

| fixture | implementation | eager median ms | Graph median ms | allocated peak MiB | reserved peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| broad | normalized_window | 2.865 | 1.511 | 46.16 | 86.00 |
| broad | normalized_full | 1.641 | 0.625 | 51.85 | 106.00 |
| broad | dense_linear | 0.743 | 0.126 | 50.50 | 106.00 |
| sharp | normalized_window | 2.795 | 0.307 | 46.16 | 86.00 |
| sharp | normalized_full | 1.596 | 0.544 | 51.85 | 106.00 |
| sharp | dense_linear | 0.775 | 0.130 | 50.50 | 106.00 |

時間は warmup 後の forward / backward / AdamW の完全ステップであり、初回 compile
や準備時間は含めない。メモリ peak は warmup 済みの model・勾配・optimizer 状態を
含み、capture / replay の前に peak を reset して測る。GPU process 全体の使用量は
未測定。各方式は独立 process で順番に測っており、交互の paired timing ではない。

1024×1024 の全 atom gradient を独立 oracle で網羅したという主張はしない。dense は
別の weight parameterization の参考測定である。この記録は宣言分離の動作確認で、
方式の承認、既定 dispatch への昇格、移動前からの速度改善を認定するものではない。
