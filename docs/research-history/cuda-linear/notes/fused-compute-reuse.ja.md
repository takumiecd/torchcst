# 局所合算のatom方向の縮約と入力128行への再利用

> この文書の `prototypes` 実行例は commit `190a1bb3fa6259fde493a30d901d5cab7c2ebb82` 時点の
> 歴史的な再現手順です。削除したコードの参照方法は
> [旧実験コード](../legacy-prototypes.ja.md)を参照してください。

2026-09-26。[I/B分類と一時メモリの改善](../../../backend-history/batched-support-preparation.ja.md)に続き、計算本体を改善した。

## 結論

目標密度（atom数=dense要素数×5%）で、準備込みforwardは4096²が119.4→60.4ms、
8192²が470.7→214.7ms。入力512行でも改善した。
64×64のCST tileかつ入力128行以上で、局所合算prototypeの標準設定を新方式へ変更した。
重み行列全体を保存せず、16×16の重み断片をGPU内で生成・消費する構造は維持する。

## 変更内容

従来は8 atomsごとに寄与を評価し、毎回atom方向を縮約して重み断片へ加算していた。
新方式は8本のlaneごとに寄与を蓄積し、対象となるI/B bucketの全atomsを処理した最後に
atom方向を一度縮約する。同時に、生成した重み断片を入力64行から128行へ再利用する。

```text
partial[weight_site, lane] = 0
for bucket in B_previous, I_current, B_current:
    for atom_chunk in bucket, step 8:
        partial += amplitude * profile(squared_distance * precision)
weight_fragment = sum(partial, atom_lane)
output_fragment += input_128_rows @ weight_fragment.T
```

atom方向の加算順序が変わるが、距離式・profile式・TF32無効のIEEE FP32 dotは維持する。
CSTのtile形状や学習パラメーターは変更しない。各出力の書き手は一つで、output atomicは使わない。
forward-onlyの`BlockStripLinear` prototypeの変更であり、GPU backwardの新実装ではない。
productionの`_weight`は参照・他経路用に維持し、新しい評価器はprototype内に置いた。

実行設定は`FusedConfig`にまとめる。CST tile64×64・入力128行以上なら、
BM128/BN16/BK16/BA8/4 warps/late_reduce=Trueを選ぶ。
それ以外は従来設定を維持し、明示的な`batch_tile`は従来の縮約を使う指定として尊重する。
`fused_config`で比較用の詳細設定を渡すこともできる。
A100で選んだ既定値であり、全ハードウェアでの最適性を保証するものではない。

## 条件・手順

A100 80GB PCIe MIG3g.40gb、42 SM。Torch2.6.0+cu126、Triton3.2.0。
Strip + Torus/raw Triweight、CST tile64×64、FP32/TF32無効、seed21。
初期化時のatom配置を測定し、学習後の分布やモデル品質の比較ではない。

1. 4096²/M128/838,861 atomsで14構成を準備済みの状態で比較。
2. 最良構成を旧構成と同じモデル・入力で比較し、準備込みの速度を測定。
3. 入力512行、低密度8192 atomsでも確認。
4. 自動選択を追加し、端数形状と明示overrideの検証を行う。

時間はCUDA Graph、3rounds、rep=20ms、順序交互反転の中央値。JIT compileとdense W生成は計測外。
各ケースで同じWのdenseと全出力を比較。Wの1,152点を全atom参加のcanonical式と独立照合する。
全Wを独立にcanonical式で再計算する検証ではない。

## Pilot：準備済み計算の比較

構成はBM,BN,BK,BA,warps,late_reduce。時間ms、中央値［最小–最大］。
registers/spillsはcompiler metadataで、実際のstallや占有率の測定ではない。

| 構成 | 時間ms | registers/thread | compiler spills | shared bytes/CTA |
| --- | ---: | ---: | ---: | ---: |
| 64,16,16,8,4,0 | 114.781 [114.092–114.795] | 128 | 4 | 5120 |
| 128,16,16,8,4,0 | 70.418 [69.780–70.445] | 168 | 0 | 10240 |
| 64,16,16,32,4,0 | 141.494 [141.440–141.687] | 255 | 0 | 5120 |
| 128,16,16,32,4,0 | 114.135 [114.079–114.238] | 255 | 0 | 10240 |
| 64,16,16,8,4,1 | 91.633 [91.582–91.808] | 128 | 2 | 5120 |
| 128,16,16,8,4,1 | 51.582 [51.428–52.067] | 168 | 2 | 10240 |
| 64,16,16,32,4,1 | 111.163 [110.861–111.437] | 255 | 36 | 5120 |
| 128,16,16,32,4,1 | 60.837 [60.358–60.852] | 255 | 54 | 10240 |
| 128,16,32,16,4,0 | 69.191 [69.172–69.789] | 255 | 0 | 18432 |
| 128,16,32,16,4,1 | 51.741 [51.441–56.547] | 255 | 66 | 18432 |
| 128,32,16,16,4,0 | 82.880 [82.564–82.922] | 255 | 8 | 10240 |
| 128,32,16,16,4,1 | 65.291 [65.267–65.388] | 255 | 72 | 10240 |
| 128,16,64,8,4,1 | 68.627 [68.597–68.674] | 255 | 102 | 36864 |
| 128,16,64,8,8,1 | 73.677 [73.474–73.875] | 255 | 44 | 36864 |

BM128のみでは約70.4ms、BM64のまま縮約を後ろへ移すと約91.6ms、両方で約51.6ms。
BAを大きくした構成や断片を広げた構成にはregister/spill増加が見られ、今回の最良にはならなかった。
採用構成は168 registers/thread、compiler spills=2、shared10KiB/CTA。
旧構成は128 registers/thread、spills=4、shared5KiB/CTA。

## 準備込みforward

単位ms、中央値［最小–最大］。

| N / M / atoms | dense | 旧構成 | 新構成 | 速度比旧/新 |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 128 / 838,861 | 1.406 [1.382–1.407] | 119.364 [119.341–119.412] | 60.373 [59.791–60.390] | 1.98× |
| 8192 / 128 / 3,355,443 | 6.412 [6.281–6.417] | 470.715 [470.132–470.746] | 214.731 [214.447–219.685] | 2.19× |
| 4096 / 512 / 838,861 | 5.205 [5.205–5.209] | 435.604 [434.939–436.043] | 191.008 [188.037–192.035] | 2.28× |
| 4096 / 128 / 8,192 | 0.660 [0.659–0.661] | 5.974 [5.969–5.980] | 4.009 [4.009–4.011] | 1.49× |

低密度ケースではdenseも約0.66msとなり、高密度ケースのdense約1.4msと速度状態が異なる。
密度による絶対時間の変化とGPUの状態変化を混同せず、同じケース内で新旧を比較する。
目標密度の8192²/M128では、新構成でもdenseの約33.5倍の時間が残る。
Tritonの最高到達点とは判断しない。

## プロファイルとメモリ

通常実行3回のCUPTI kernel時間平均。CUDA Graphの上記時間とは別測定。
cache hit率、HBM帯域、stall、達成occupancyは未測定。

| N / M / atoms | 旧計算本体ms | 新計算本体ms | 旧/new追加allocationピークMiB |
| --- | ---: | ---: | ---: |
| 4096 / 128 / 838,861 | 114.863 | 54.847 | 56.001 |
| 8192 / 128 / 3,355,443 | 446.247 | 196.707 | 219.602 |
| 4096 / 512 / 838,861 | 432.754 | 186.264 | 56.001 |
| 4096 / 128 / 8,192 | 5.819 | 3.867 | 2.250 |

追加ピークは出力を含み、保持済みinput/model/prepared/比較用Wを除くPyTorch allocatorの値。
GPU内部shared/register/spill、CUDA context、allocator予約、総学習メモリを表すものではない。
sharedは上記のとおり5→10KiBに増えるが、全WのHBM保存は追加していない。

## 検証と採用

初期GPUテスト52件、標準選択追加後53件が通過した。
入力・出力・CST tileの端数、非連続入力、zero amplitude、I/B継ぎ目、CUDA Graph後のatom更新を含む。
入力127/129行で旧/new設定の自動選択と明示overrideを検証し、選択された明示経路と完全一致した。

大型比較は14pilot＋8scale=22件すべて合格。atol=rtol=3e-5を維持し、最大絶対誤差は
約4.92e-7以内。atom方向の加算順序変更による差は、この許容範囲内に収まった。
14構成45timingsと4scaleケース36timings、合計81サンプル、10profilesを保存した。

計算本体は改善したが、目標密度ではdenseとの差はまだ大きい。
今後は距離/profileの評価費用とGPU上での実行効率をさらに調べる必要がある。
今回の結果だけで帯域律速や特定命令の律速を断定しない。

## 再現・保存

- 測定source：`46de3304d8af90e7330482010acfb48c359dcd67`。
- そのarchive SHA256：`2fe104ae32aabe1950b7a3b5a8ffaaf3f09dbc4ff7b11bf9bd8fefc72c3d41af`。
- 標準選択追加source：`c4b5574b3ff497081d01a3aba1704aae861bff02`。
- そのarchive SHA256：`5c40e1a8c18794d366e96349d0d9695d5f3767bdae1514c311fffb8d83cde41e`。
- 結果bundle SHA256：`dc1279f402cf06d169d23ab6f4cddb6e4e9673056e2746cc99f15e72a7cf62bc`。
- Remote：`srv11/cst-lab/torchcst-fused-compute-46de330`と`torchcst-fused-compute-c4b5574`。
- Local：`output/triton-a100-20260926/fused-compute/`と`fused-compute-default-tests.log`。

```bash
PYTHONPATH=src:. python -u -m prototypes.benchmark_fused_compute \
  --output-dir pilot --source-commit 46de3304d8af90e7330482010acfb48c359dcd67
PYTHONPATH=src:. python -u -m prototypes.benchmark_fused_compute \
  --output-dir scale --source-commit 46de3304d8af90e7330482010acfb48c359dcd67 \
  --full --configs 64,16,16,8,4,0 128,16,16,8,4,1 \
  --cases 4096:128:838861 8192:128:3355443 4096:512:838861 4096:128:8192
PYTHONPATH=src:. python -m pytest -q -x --tb=short --color=no \
  tests/test_fused_compute_configs.py tests/test_block_strip_linear.py
```

source/結果のremote-local hash、完了flags、5ケース、A100/42 SM、22出力比較、81timings、
10profiles、最終53テスト通過をローカルで検証した。
