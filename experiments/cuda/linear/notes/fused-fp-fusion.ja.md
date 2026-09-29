# 局所合算GEMMのFP32積和融合

> この文書の `prototypes` 実行例は commit `190a1bb3fa6259fde493a30d901d5cab7c2ebb82` 時点の
> 歴史的な再現手順です。削除したコードの参照方法は
> [旧実験コード](../legacy-prototypes.ja.md)を参照してください。

2026-09-26。前段の[局所合算の再利用改善](fused-compute-reuse.ja.md)に続き、A100でFP32の積和融合を調べた。

## 結論

目標のatom密度5%、CSTタイル64×64、入力128行では、準備込みforwardが
4096²で27.44→23.66ms、8192²で99.44→85.28msとなった。同一run内の比較で、
それぞれ13.8%、14.2%短縮した。低密度では逆に遅くなるため、自動選択は
**A100・64×64タイル・入力128行以上・atom数がdense要素数の5%以上**に限定した。
他の条件では従来どおりFP融合を無効にし、`FusedConfig(fp_fusion=True)`で明示指定できる。

変更したのはforward専用`BlockStripLinear` prototypeのTriton起動設定である。
`enable_fp_fusion=True`はFP32演算の丸め順序を変えるが、TF32は無効のまま、
全重み行列のHBM保存も追加していない。production backendとGPU backwardは変更していない。

## 診断と判断

まず、既存の局所合算カーネルと同じ計算をする診断用fullを作り、出力の完全一致と
同等の速度・リソースを確認した。同じGPU起動形状の反実仮想として、重み生成のみ、
合成16×16重み断片を使うdotのみ、距離・profileを安価なデータ依存式に置き換えた
経路を測定した。8192²/M128/3,355,443 atomsの準備済み実行では、
現行full 88.08ms、生成のみ80.78ms、合成dotのみ7.90ms、代替式68.14msだった。
代替式は現行と同じ168 registers/thread、2 spills、10KiB sharedで、PTX上の
`ld.global`とdotのFMA命令数も同じだった。これは距離・profile周辺の算術を調べる
根拠になるが、反実仮想カーネルの時間を加減算して個々の処理時間とは扱わない。

次に同じfull計算でFP融合だけを切り替えた。8192²では準備済みで
88.08→73.90ms、PTXの`fma.rn.f32`は静的に256→496箇所、
compiler spillsは2→0となった。レジスター168/thread、shared10KiB、
`ld.global`46箇所は同じ。spillsの変化だけを速度差の原因と断定しない。

## 同条件の速度

A100 80GB PCIe MIG 3g.40gb、42 SM、Torch 2.6.0+cu126、Triton 3.2.0、
FP32/TF32無効、Strip + Torus/raw Triweight、seed21。
CUDA Graph、3ラウンド、`rep=20ms`、経路順を交互に反転した中央値。単位ms。

| 形状 / 入力行 / atoms | FP融合なし・準備済み | FP融合あり・準備済み | FP融合なし・準備込み | FP融合あり・準備込み |
| --- | ---: | ---: | ---: | ---: |
| 4096² / 128 / 838,861 | 24.633 | 20.807 | 27.437 | 23.661 |
| 8192² / 128 / 3,355,443 | 88.076 | 73.898 | 99.441 | 85.285 |
| 4096² / 512 / 838,861 | 82.864 | 69.013 | — | — |
| 4096² / 128 / 131,072 | 6.171 | 5.711 | — | — |
| 4096² / 128 / 65,536 | 4.632 | 4.437 | — | — |
| 4096² / 128 / 8,192 | **3.872** | 3.971 | — | — |

準備込みにはdecode・routing・分類・sort・packを含む。同じ形状のdense対照は
4096²で0.658ms、8192²で3.025msだった。CSTは改善後もdenseより大幅に遅い。
以前の測定ではA100の速度状態が異なり、8192²の計算本体が約197ms、denseが
約6.41msだった。その絶対時間と今回の値を直接比較せず、上表の同一run内で新旧を判断する。

## 正確さと適用範囲

5%密度の全出力を同じdense重みの出力と`atol=rtol=3e-5`で比較し、
4096²・8192²とも合格した。8192²のFP融合あり経路はdenseとの差の最大絶対値が
4.69e-7、従来経路との差が8.57e-8だった。dense対照の重みは、各ケースで
1,152点を全atom参加のcanonical式と独立照合した。全重み要素の独立再計算ではない。
低密度・中間密度の診断経路も全出力比較に合格した。

A100で境界、Torus継ぎ目、端数形状、非連続入力、zero amplitude、
CUDA Graph後のatom更新を含む57テストが通過した。新しい自動選択は
小さい5%密度のGPUケースで明示設定と完全一致し、デフォルト経路を
CUDA Graphで捕捉した後のatom更新もdense対照と一致した。
今回の大型測定は初期配置であり、学習後の偏りや他GPUでの性能は未評価。

## 再現情報

- 診断と準備済み密度比較のsource: `47c9349bf219b0caaae93e17b174f3e369086276`。
- 準備込み比較のsource: `be7c22e24e03b9ee5e371a0445aafd476da24ea5`。
- 準備込み測定後の自動選択source: `e2bceb0ee614ebae205ff1cf2848973d1fada4c9`。
- 最終GPUテストsource: `bbcef37981b569635cc9108ef328093ca6fe76fa`。
- 診断結果bundle SHA256: `1cbf34cde68b3e62aaee5cda27a21b18c330a3b6fd9615fe18039bbd8424cda4`。
- 密度比較bundle SHA256: `86374d31dce8521beb7288a42fc72c2a77024bd9f9ceba1cd94065b2bacbce6e`。
- 準備込み結果JSON SHA256: `8aa49d0eb4944ebbefec695e7122b02d57312f939e08498d1d97c1278688d09a`。
- ローカル結果: `output/triton-a100-20260926/fused-stages-47c9349/`、
  `output/triton-a100-20260926/fma-full-be7c22e.json`。

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_fused_compute_configs.py tests/test_block_strip_linear.py
PYTHONPATH=src:. python -u -m prototypes.benchmark_fused_compute \
  --output-dir full --source-commit be7c22e24e03b9ee5e371a0445aafd476da24ea5 \
  --full --configs 128,16,16,8,4,1 128,16,16,8,4,1,1 \
  --cases 4096:128:838861 8192:128:3355443
```

source archiveと取得結果のリモート・ローカルhash一致、測定完了flag、
全出力比較、各経路3ラウンド、最終57テスト通過を確認した。
