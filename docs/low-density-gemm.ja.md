# 5%未満のmapped GEMM基準測定（A100）

2026-09-27。CST tileを64×64、N=K=4096、入力行数を128に固定し、atom密度5%未満を主対象として測った。密度はatom数÷16,777,216であり、非ゼロ重みの割合ではない。dispatchはこの測定の対象外。

## 同一runでの経路比較

`benchmark_low_density.py`は各atom数について同じ層、入力、prepared routing、dense Wを共有する。dense Wをcanonicalな全atom和の1,152地点と照合し、全経路の出力をそのWの`F.linear`と`atol=rtol=3e-5`で照合した。以下の全ケース・経路は合格。A100 80GB PCIe MIG 3g.40gb、42 SM、FP32、TF32無効、seed 21、CUDA Graph 3ラウンド・各20msの中央値。単位はms。

| atoms | 密度 | dense | atom-dot 準備込み | fused 準備込み | fused 準備済み |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.049% | 0.660 | 5.355 | 4.003 | 3.874 |
| 32,768 | 0.195% | 0.660 | 19.307 | 4.065 | 3.874 |
| 65,536 | 0.391% | 0.658 | 38.435 | 4.912 | 4.626 |
| 131,072 | 0.781% | 0.658 | 76.045 | 6.678 | 6.168 |

ここでのfusedは`FusedConfig(batch_rows=128, late_reduce=True, fp_fusion=False)`、atom-dotは既定のBM64/BN16。古いfusedとの別run比較からatom-dot優位を推測しない。現行fusedは最も疎な点でも速く、密度が上がるほど差が開いた。ただしdense Wを事前に持つdense対照よりは遅い。

## fused実行形状の限定比較

同じケース内で形状を交互に測った。以下は準備込みの中央値。`BA4`はatomの処理幅4、`BN32`は出力行32、基準はBA8/BN16。全形状の数値照合は合格。

| atoms | 密度 | 基準 | BA4 | BN32 |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.049% | 4.002 | **3.734** | 3.965 |
| 32,768 | 0.195% | 4.070 | 4.171 | **4.034** |
| 167,772 | 1.000% | **8.378** | 8.480 | 9.671 |
| 335,544 | 2.000% | **13.100** | 13.583 | 15.312 |
| 671,088 | 4.000% | **22.694** | 24.318 | 26.705 |

一つの形状が5%未満全体で最適とは言えない。0.049%でのBA4の改善だけで自動選択は変更しない。

## FP fusionを5%未満で比較

同じ実行形状BM128/BN16/BK16/BA8/late-reduceのまま、Tritonの`enable_fp_fusion`だけを切り替えた。同じケース内で交互に3ラウンド測定し、全出力照合は合格。準備込み中央値、単位ms。

| atoms | 密度 | fusion無 | fusion有 | 変化 |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.049% | **4.004** | 4.092 | 2.2%遅い |
| 32,768 | 0.195% | **4.069** | 4.157 | 2.1%遅い |
| 65,536 | 0.391% | 4.916 | **4.721** | 4.0%速い |
| 131,072 | 0.781% | 6.683 | **6.213** | 7.0%速い |
| 167,772 | 1.000% | 8.393 | **7.626** | 9.1%速い |
| 335,544 | 2.000% | 13.100 | **11.545** | 11.9%速い |
| 671,088 | 4.000% | 22.696 | **19.657** | 13.4%速い |

少なくとも測った0.195%と0.391%の間に有利な設定の転換がある。連続する密度や別形状での閾値は未確定。`fp_fusion=True`は既に明示的な実験設定として使えるが、既定の自動選択は変更していない。

## 生成後の空重み判定

実験commit `ed8fd8fc45c23837f73b1da8081e923ce2b782f6`で、BN×BK重み小片を生成した後、全てゼロなら入力ロードとdotを飛ばした。全経路の出力照合は合格したが、同一runで全密度が遅くなった。特に8,192 atomsの準備込みは基準4.001→4.534ms、32,768では4.063→4.591ms。重み生成を終えてからの分岐では費用を回収できないため、commit `9c1489a`でコードをrevertした。NaN/Inf入力に対するゼロ重みとの積の意味も変わるので、この方式を採用しない。

## 生成前の外接箱判定

Grok 4.7との検討を経て、実験commit `ac8f532a3e4aca702be8c0bfd66de965fe86ef7c`で、各BN×BKサイト小片の4次元外接箱と3つの所属バケットにあるatomのサポートを比較し、届かない場合に重み生成を飛ばした。約0.049%のパイロットでは全出力がdenseと一致したが、準備込み4.000→6.423ms、準備済み3.871→6.283msと大幅に遅い。カーネル内での箱計算とatomの再走査を伴うこの方式は先へ進めず、`f78569e`でrevertした。最も疎な点で明確に不利なため、密度を増やした測定は行わなかった。

## 再現

基準ソースcommitは`483501cfcbfcdbb63ade9e4550df11a8355d9969`、archive SHA256は`a450bcba12bd65b6a1b4ae0c34939c5d23940f92a3c00112be7e0cad1ff2e40f`。A100上の展開前にhash一致を確認した。`tests/test_block_strip_linear.py`は47件合格。結果JSONのローカル/遠隔SHA256も照合済み。

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_low_density \
  --output-dir results --source-commit 483501cfcbfcdbb63ade9e4550df11a8355d9969
PYTHONPATH=src:. python -m prototypes.benchmark_fused_compute \
  --output-dir low-sweep --source-commit 483501cfcbfcdbb63ade9e4550df11a8355d9969 \
  --cases 4096:128:8192 4096:128:32768 \
  --configs 128,16,16,8,4,1 64,16,16,8,4,1 128,32,16,8,4,1 \
            128,16,32,8,4,1 128,16,64,8,4,1 128,16,16,4,4,1 --full
PYTHONPATH=src:. python -m prototypes.benchmark_fused_compute \
  --output-dir low-fp-sweep --source-commit 483501cfcbfcdbb63ade9e4550df11a8355d9969 \
  --cases 4096:128:8192 4096:128:32768 4096:128:65536 4096:128:131072 \
          4096:128:167772 4096:128:335544 4096:128:671088 \
  --configs 128,16,16,8,4,1,0 128,16,16,8,4,1,1 --full
```

遠隔ディレクトリは`srv11/cst-lab/torchcst-low-density-483501c`。結果はGit管理外の`output/triton-a100-20260927/low-density-baseline/`と`output/triton-a100-20260927/low-density-skip/`に保存した。基準、低密度形状比較、1〜4%形状比較、生成後スキップの各JSON SHA256は順に`535610fc6898a66529c1687f819185c39ec5c389bcc6b66daf279f76119e7f93`、`fd33d45a1dbec07e70347a822cb12b6a196b6020b2bfa2924cb126f9cee3009f`、`516421400ec914c407d81e32be976ff4e80501b3a394a90ba568108c45bfcb7f`、`4c82a82d58bcb0f9284d69db328fbc02047e89279fdd523044c2b70dcdccc41d`。

FP fusion sweepと外接箱パイロットの結果は、それぞれ`output/triton-a100-20260927/low-density-baseline/fp-sweep.json`、`output/triton-a100-20260927/low-density-cull/pilot.json`。各JSON SHA256は`d9de8f7192e9128bdf88ea1e4d93b638db75d6408c1867a09b88a4e24b4ec560`、`7b488b1526dc4980119791647676765c1748469329e1cc2a0d393715cd65d32a`で、ローカル/遠隔一致を確認した。
