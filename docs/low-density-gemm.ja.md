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

## BA1の実験

初期配置では8,192 atomsが4096 tileに均等に属し、各tileのinterior bucketは2 atoms、boundary bucketは0だった。BA8は6 laneを使わないため、Grok 4.7にBA1/BA2を実験設定として追加してもらった。既定BA8は維持した。A100で`tests/test_fused_compute_configs.py`と`tests/test_block_strip_linear.py`の計59件が合格し、境界・Torus seam・CUDA Graph再生中のatom更新を含む。全性能ケースも同一dense Wとの出力照合に合格。以下は準備込みで同じケース内のCUDA Graph 3ラウンド中央値、単位ms。

| atoms | 密度 | BA8 / fusion無 | BA1 / fusion無 | BA1 / fusion有 |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.049% | 4.001 | **2.622** | 別runで差が不安定 |
| 32,768 | 0.195% | 4.068 | **3.111** | 別runで差が不安定 |
| 65,536 | 0.391% | 4.916 | **3.921** | 別runで差が不安定 |
| 131,072 | 0.781% | 6.681 | 5.709 | **5.278** |
| 167,772 | 1.000% | 8.393 | 6.733 | **6.152** |
| 335,544 | 2.000% | 13.091 | 11.510 | **10.233** |
| 671,088 | 4.000% | 22.684 | 21.461 | **18.918** |

最も疎な点ではBA1が既定の4.001→2.622ms（34.5%短縮）。BA2は同じ点で3.595ms、BA4は3.744msだった。1〜4%のBA1/fusion有も同一runでBA8/fusion有より速い。0.049〜0.391%のfusion有無の追加runではGPU速度が約2倍の状態に移り、ラウンド内にも変動したため小差の採否には用いない。BA1/fusion無の対BA8/fusion無の大きな優位は両runで維持した。自動選択はまだ変えていない。

同じコード生成のcompiler resourceはBA1が96 registers/thread・spill 0、BA2が160・0、BA4が168・0、BA8が168・spill 2。shared memoryは全て10,240 bytes/block。BA1でレジスタ圧力が下がる点は速度差と整合するが、実効occupancyの直接測定ではない。

同じatom数/密度帯で入力行数と行列サイズも限定確認した。4096²/M512/8,192 atomsはBA8 13.315→BA1 **8.903ms**、8192²/M128/16,384 atomsは14.095→**9.181ms**。両ケースとも同一dense Wとの全出力照合に合格し、各ケース内の3ラウンドで差が安定した。tileは引き続き64×64。他の形状やGPUへの一般化は未検証。

### BA1の実行形状

Grok 4.7の提案を基に、BM128/BA1/late-reduceでBN（出力行幅）、BK（列幅）、warpsを同一run比較した。8,192〜65,536 atomsはFP fusion無、131,072〜671,088 atomsはfusion有。各ケース内の全形状は同じ層・入力・Wを共有し、全出力照合に合格。準備込みCUDA Graph 3ラウンド中央値、単位ms。

| atoms | 密度 | BA1 BN16/BK16 | 最速形状 | 最速ms | 短縮 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 8,192 | 0.049% | 2.624 | BN32/BK16 | **2.422** | 7.7% |
| 32,768 | 0.195% | 3.111 | BN16/BK64 | **2.815** | 9.5% |
| 65,536 | 0.391% | 3.916 | BN16/BK64 | **3.496** | 10.7% |
| 131,072 | 0.781% | 5.280 | BN16/BK64 | **4.583** | 13.2% |
| 167,772 | 1.000% | 6.153 | BN16/BK64 | **5.359** | 12.9% |
| 335,544 | 2.000% | 10.235 | BN16/BK32 | **9.014** | 11.9% |
| 671,088 | 4.000% | 18.919 | BN16/BK32 | **16.757** | 11.4% |

全て最速形状と基準の3ラウンドの範囲は重ならない。BN32/BK16は128 registers/thread・spill 0、BN16/BK32は96・0、BN16/BK64は168・0。8 warpsの候補はこの低密度runで大きく遅く、採用しない。M512・4096²/8,192 atomsではBN32/BK16が8.902→**7.467ms**、8192²/M128/16,384 atomsでは9.183→**7.964ms**で、両方とも同じケースのBN16/BK16より速かった。測った形状ごとに最適点が変わるため、これらは明示的な実験設定として記録し、dispatchは後で扱う。

初期配置の性能照合とは別に、Grok 4.7がcommit `899d60e61a48ff7bba3abf711f683ab2e3674532`でBN32/BK16、BN16/BK64、BN16/BK32+FP fusionのBA1ケースを境界・Torus seam・CUDA Graph再生テストへ追加した。A100で関連テスト62件が合格。ソースarchive SHA256は`0da1937fc3bbfc9e0ac762cf817bb82b5785a22b3299ca87b05be318e03c6807`で、遠隔展開前に一致を確認した。ローカルではruffとCPUを含む全テスト424件が合格し、CUDA専用169件はskipされた。

共有Wの占有率も`benchmark_low_density.py`に記録した。8,192 atoms（atom密度0.049%）でもWの**82.08%**が非ゼロで、**65,536/65,536個の16×16小片が非ゼロ**。32,768 atomsでは95.53%、65,536 atomsでは95.63%が非ゼロで、小片はどちらも全件非ゼロ。atom密度はWのスパース率ではない。空小片の間引きや疎行列パネルを主戦略にしない根拠になる。

### BA1の入力行幅を縮めた比較

Grok 4.7とBM（入力行幅）の縮小を検討した。M128/BN32ではBM128のgridは128 CTAで、このA100 MIGの42 SMに対して約3 CTA/SMになる。BM64/32ならCTA数は2/4倍になる一方、各CTAが生成する重みを入力行の別部分で再生成する。0.049%の主ケースだけで、BA1/BN32/BK16/4 warps/late-reduce/fusion無を固定して同一run比較した。準備込みCUDA Graph 3ラウンド、単位ms。

| atoms | BM128 | BM64 | BM32 |
| ---: | ---: | ---: | ---: |
| 8,192（0.049%） | **4.968** [4.966–4.969] | 6.967 [6.845–7.291] | 7.545 [7.541–7.546] |

全形状が同じdense Wとの出力照合に合格し、compiler spillは0。BM128/64/32のレジスタ数はそれぞれ128/96/80 per thread、shared memoryは10,240/9,216/4,096 bytes/block。GPU全体の速度帯が前回のBM128 2.422msから約2倍遅い状態だったため、別runの絶対値は比較しない。同一runではBM64/32の3ラウンド範囲がBM128と明確に離れて遅い。重み再生成を上回る占有率改善はこの形状では確認できず、Grokと決めた停止条件に従って0.195%側のBM追加比較は行わなかった。BMの自動選択は変更していない。

結果は`output/triton-a100-20260927/low-density-bm/8192.json`、SHA256は`1154758e9cda2f965152b8ec7116bda30be0fac5a06167734342b1ac5494d35e`で遠隔結果と一致。ソースcommitは`b1fd5de61a24b906351bc80125104ac67fd2be9c`、遠隔ディレクトリは`srv11/cst-lab/torchcst-ba2-b1fd5de`。

```bash
PYTHONPATH=src:. python -u -m prototypes.benchmark_fused_compute \
  --output-dir low-density-bm-8192 \
  --source-commit b1fd5de61a24b906351bc80125104ac67fd2be9c \
  --cases 4096:128:8192 \
  --configs 128,32,16,1,4,1,0 64,32,16,1,4,1,0 32,32,16,1,4,1,0 --full
```

### BNとBKを同時に広げた比較

Grok 4.7と既存のBN/BK個別比較の交差セルを検討し、BM128/BA1/4 warps/late-reduce/fusion無を固定した。0.049%でBN32/BK16とBN16/BK64を同一runの対照に、BN32/BK32とBN32/BK64を測った。準備込みCUDA Graph 3ラウンド中央値、単位ms。

| 設定 | 中央値 | 3ラウンド範囲 | registers/thread | compiler spill |
| --- | ---: | ---: | ---: | ---: |
| BN32/BK16 | **4.690** | 4.616–4.969 | 128 | 0 |
| BN16/BK64 | 5.013 | 5.013–5.024 | 168 | 0 |
| BN32/BK32 | 5.058 | 4.942–5.272 | 168 | 0 |
| BN32/BK64 | 4.942 | 4.941–4.946 | 255 | 0 |

全設定が同じdense Wとの出力照合に合格。BN32/BK64の中央値は対照より遅く、3ラウンドの範囲も対照と重なる。BN32/BK32も速くない。このrunも以前よりGPU全体が約2倍遅い速度帯であり、別runの絶対値とは比較しない。Grokと決めた停止条件に従い、0.195%以上には広げなかった。自動選択は変更していない。

結果は`output/triton-a100-20260927/low-density-bm/bn-bk-8192.json`、SHA256は`113559ff2c85b00d5946b5f6fea8a292f84f03496d263c48a06f7ea806aecf80`で遠隔結果と一致。ソースcommitは`b1fd5de61a24b906351bc80125104ac67fd2be9c`、遠隔ディレクトリは`srv11/cst-lab/torchcst-ba2-b1fd5de`。

```bash
PYTHONPATH=src:. python -u -m prototypes.benchmark_fused_compute \
  --output-dir low-density-bn-bk-8192 \
  --source-commit b1fd5de61a24b906351bc80125104ac67fd2be9c \
  --cases 4096:128:8192 \
  --configs 128,32,16,1,4,1,0 128,16,64,1,4,1,0 \
            128,32,32,1,4,1,0 128,32,64,1,4,1,0 --full
```

### BA1のatom軸を除く試作

Grok 4.7とBA1カーネルの生成コードを検討した。BM128/BN32/BK16/BA1のTriton中間表現には`tensor<512x1xf32>`のpartialと長さ1の`tt.reduce`が残る。PTXでは不要なprofile傾き計算は見つからなかった。Strip+Torusの元atomパラメータは5列だが、`prepare`後のpacked配列は6列で、カーネルに渡す埋め込み次元は`D=4`。`D=2`と想定してSection追加次元を無視してはならない。

実験commit `60f71a780c0a22667c61307094461ad266d3b69f`でGrok 4.7が既定経路を保ったまま、BA1/late-reduce専用の明示的な`scalar_ba1`を追加した。3バケット順、FP32式、無効サイトのマスク、IEEE dotを維持し、partialのatom軸だけを外した。A100の関連テスト65件が合格し、境界・Torus seam・CUDA Graph再生後のatom更新を含む旧経路との出力ビット一致を確認した。

0.049%の準備込み同一run、CUDA Graph 3ラウンドでは、旧経路4.956945ms [4.950323–4.958344]、scalar経路4.953600ms [4.950972–4.956979]。全出力は同じdense Wとの照合に合格。両方128 registers/thread、compiler spill 0、shared memory 10,240 bytes/blockで、性能範囲が重なった。以前のrunよりGPU全体が約2倍遅い速度帯にあるため、別runの絶対値は比較しない。Grokと決めた停止条件に従い、高密度側の追加測定は行わず、`8823531`で実験コードをrevertした。BA1の長さ1のatom軸をソースで除くだけでは、測った形状の性能改善を確認できなかった。

ソースarchive SHA256は`0dde5a33a0feeea550e467f608163e1909bc90243f795d28e5db88aa29a9af48`で遠隔展開前に一致を確認した。遠隔ディレクトリは`srv11/cst-lab/torchcst-ba1-scalar-60f71a7`。結果`output/triton-a100-20260927/low-density-bm/scalar-8192.json`のSHA256は`4eef3f2e01c03dae1eb7c9afc68ff29df48aa0348c53b42d4829db7d542e6380`、65件のテストlogは`75292d81558d2cc553772ba912a747239a5130b323f396b307cf162a68548433`で、いずれも遠隔と一致した。

```bash
PYTHONPATH=src:. python -m pytest -q --color=no \
  tests/test_fused_compute_configs.py tests/test_block_strip_linear.py
PYTHONPATH=src:. python -u -m prototypes.benchmark_fused_compute \
  --output-dir scalar-8192 \
  --source-commit 60f71a780c0a22667c61307094461ad266d3b69f \
  --cases 4096:128:8192 \
  --configs 128,32,16,1,4,1,0,0 128,32,16,1,4,1,0,1 --full
```

### 隣接するI/Bバケットのループ統合試作

Grok 4.7とsupport layoutのバケット順を確認した。`G>1`のstation `s>0`では、対象atomは前stationの境界`B[s-1]`、現在の内部`I[s]`、現在の境界`B[s]`の順に連続している。BA1なら、個別の3ループを`[Offsets[2*s-1], Offsets[2*s+2])`の1ループにしてもatomの訪問順を保てる。station 0は円周の継ぎ目で、最後の境界範囲の後に`[Offsets[0], Offsets[2])`を走査する必要がある。idle bucketを含めない終端にも注意した。

Grok 4.7が実験commit `d7b34404c0c0c9ed320b31c8d3b68206bd47fdfa`で、BA1/late-reduceだけに明示的な`merge_buckets`を追加した。既定経路とdispatchは変更していない。A100ではTriton 3.2が3条件をつなぐ`and`構文を拒否し、関連テスト13件がコンパイル失敗、52件が合格した。修正commit `596102fbb5563e283b6b5f910bd7e6c252700bc1`で構文を分けると64件が合格し、残り1件はG=2で特定境界バケットが非空というテスト側の誤った前提で失敗した。

テスト修正commit `2bd28dd5abc9b9966f335e1e879b6e746c9d189f`を別スナップショットで再実行すると、G=2の新経路のコンパイル中、Triton `make_ttgir`でPythonプロセスが`Aborted (core dumped)`となり、遠隔終了コード134だった。そこから追加のコンパイル試行と性能計測は行っていない。Triton停止の詳細原因は未特定であり、ループ統合の性能・正しさに結論は出ていない。実験コードは`a5f5712`、`4072d86`、`31ce9f0`の順にrevertした。

3つのsource archive SHA256は順に`0f4159cc66939fe49fdd050de8411a1756a5bdc8751f3e50e5d97d32cda0be83`、`c782080e3d2eb7fd0351c708d7bf307a9a615c250cf406ee54ce60b5caa2ad50`、`056f9f45a4bc9a1175cc625e841cee38c082853caa720f6877b3fe17da785a91`で、遠隔展開前に一致を確認した。最後の遠隔ディレクトリは`srv11/cst-lab/torchcst-merge-buckets-2bd28dd`。abort logは`output/triton-a100-20260927/low-density-bm/merge-tests-abort.log`、SHA256は`b731d1f7ca3cf772becc308b0247204c6c0472419d4c92af3c89f5696a4b4fec`で遠隔と一致した。

## 0.0015〜0.049%でのatom直接方式とBA1の交差

Grok 4.7がcommit `0b66d13940efb52a3f708e738e47d9d64f40a3ec`で`benchmark_low_density.py`へBA1の準備込み・準備済み経路、空station率、64×64重み小片の占有数を追加し、`c6cd6051272f5a029e6083fcc5398ab8bea9bc5a`で経路の出力照合失敗時にスイープを止めるよう修正した。カーネルとdispatchは変更していない。N=K=4096、M=128、CST tile64×64、seed21、A100 42 SM、FP32/TF32無効。BA1はBM128/BN32/BK16/4 warps/late-reduce/fusion無、atom直接方式は既定BM64/BN16。各atom数で同じ層・入力・prepared routing・dense Wを共有し、canonical全atom和と全7経路の出力照合は合格した。全Wはベンチマークの対照としてのみ生成し、各経路の計測外。CUDA Graph 3ラウンド・rep=20ms、単位ms。

| atoms | atom密度 | 空station | 非ゼロW | 非ゼロ64×64小片 | atom直接・準備込み | BA1・準備込み | 優勢 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8,192 | 0.04883% | 0% | 82.08% | 4,096/4,096 | 10.388 | **4.962** | BA1 |
| 4,096 | 0.02441% | 0% | 44.79% | 4,096/4,096 | 6.269 | **4.866** | BA1 |
| 3,584 | 0.02136% | 12.5% | 39.15% | 3,584/4,096 | 5.045 | **4.854** | BA1 |
| 3,072 | 0.01831% | 25% | 33.52% | 3,072/4,096 | **4.766** | 4.841 | atom直接 |
| 2,560 | 0.01526% | 37.5% | 27.95% | 2,560/4,096 | **3.841** | 4.828 | atom直接 |
| 2,048 | 0.01221% | 50% | 22.35% | 2,048/4,096 | **3.576** | 4.814 | atom直接 |
| 1,024 | 0.00610% | 75% | 11.26% | 1,024/4,096 | **2.121** | 4.789 | atom直接 |
| 512 | 0.00305% | 87.5% | 5.53% | 512/4,096 | **1.603** | 4.565 | atom直接 |
| 256 | 0.00153% | 93.75% | 2.76% | 256/4,096 | **1.324** | 4.540 | atom直接 |

最初の6点（8,192/4,096/2,048/1,024/512/256）と中間の3点（3,584/3,072/2,560）は別runであり、表の行をまたいだ絶対時間は比較しない。各行内の同一runで、atom直接とBA1の3ラウンド範囲は全て重ならなかった。交差付近は3,072 atomsでatom直接準備込み4.766ms [4.519–4.767]対BA1 4.841ms [4.841–4.842]、3,584 atomsでatom直接5.045ms [5.037–5.056]対BA1 4.854ms [4.662–4.855]。準備済みでも3,072ではatom直接4.673ms [4.429–4.682]対BA1 4.746ms [4.711–4.748]、3,584ではatom直接4.953ms [4.851–4.954]対BA1 4.754ms [4.540–4.757]で同じ順序だった。したがってこの入力・初期配置での交差は3,072〜3,584 atomsの間。異なる配置やM、GPUへの閾値の一般化やdispatch変更はしない。

今回のA100は8192 atomsでdense対照約1.35ms、BA1約4.96msの遅い速度帯だった。以前のrunの約0.66/2.42msとは絶対値を比較しない。256 atomsではatom直接準備込み1.324ms、事前生成Wを使うdense対照1.346msだが、3ラウンド範囲が重なる。さらに両経路の事前条件も異なるため、denseより速いと結論しない。

8,192 atomsでは16×16小片が全65,536枚、64×64小片が全4,096枚非ゼロだったが、256 atomsでは16×16小片が2,801/65,536枚、64×64小片が256/4,096枚。重みの空小片が多いのは超低密度域に限る。今回は占有を測っただけで小片スキップは実装していない。

測定source archive SHA256は`e8f5f59dc8df7b1ce80172716b33fe0dd7ca88a7638d5b28c98a0f4c7f8a3752`で遠隔展開前に一致を確認した。遠隔ディレクトリは`srv11/cst-lab/torchcst-ultralow-c6cd605`。ローカル結果は`output/triton-a100-20260927/ultra-low/`。最初の6点のJSON/log SHA256は`6b5dcc53845f68813421130f5be2c120f17aa32d159c9ee58dc67392b61e1251`/`8cdafa5c95c73697b2c3744f996f6c6c55a92e3e35e5de2c73774e6f731c5538`、中間3点は`7732e6e0e3a0c5b9266ecffaf5be6a4ee263366f90af39dbad0d2523d4110d09`/`89a6696cdc6455872e8ac749c72196a9f92b249816e7439a1ca69959eb33c0e7`で、全て遠隔と一致した。

```bash
PYTHONPATH=src:. python -u -m prototypes.benchmark_low_density \
  --output-dir ultra-low --source-commit c6cd6051272f5a029e6083fcc5398ab8bea9bc5a \
  --atoms 8192 4096 2048 1024 512 256
PYTHONPATH=src:. python -u -m prototypes.benchmark_low_density \
  --output-dir ultra-low-mid --source-commit c6cd6051272f5a029e6083fcc5398ab8bea9bc5a \
  --atoms 3584 3072 2560
```

## 生成後の空重み判定

実験commit `ed8fd8fc45c23837f73b1da8081e923ce2b782f6`で、BN×BK重み小片を生成した後、全てゼロなら入力ロードとdotを飛ばした。全経路の出力照合は合格したが、同一runで全密度が遅くなった。特に8,192 atomsの準備込みは基準4.001→4.534ms、32,768では4.063→4.591ms。重み生成を終えてからの分岐では費用を回収できないため、commit `9c1489a`でコードをrevertした。NaN/Inf入力に対するゼロ重みとの積の意味も変わるので、この方式を採用しない。

## 生成前の外接箱判定

Grok 4.7との検討を経て、実験commit `ac8f532a3e4aca702be8c0bfd66de965fe86ef7c`で、各BN×BKサイト小片の4次元外接箱と3つの所属バケットにあるatomのサポートを比較し、届かない場合に重み生成を飛ばした。約0.049%のパイロットでは全出力がdenseと一致したが、準備込み4.000→6.423ms、準備済み3.871→6.283msと大幅に遅い。カーネル内での箱計算とatomの再走査を伴うこの方式は先へ進めず、`f78569e`でrevertした。最も疎な点で明確に不利なため、密度を増やした測定は行わなかった。

## dot精度設定の試作

実験commit `c9138833b61c31581f95b1c738dc37bd9dff0e82`で、Grok 4.7に既定`ieee`を維持したまま明示的な`tf32x3`設定を追加してもらった。A100上で`tests/test_fused_compute_configs.py tests/test_block_strip_linear.py`を実行すると、追加したBA1/BN32/BK16の`tf32x3`ケースのTriton `make_llir`中にPythonプロセスが`Aborted (core dumped)`で終了し、遠隔終了コードは134だった。これ以上のコンパイル再試行や性能計測は行わず、`f3bee3e`で試作をrevertした。精度や速度の結論は出ていない。失敗原因の特定も未了であり、`tf32x3`一般がA100で使えないという主張ではない。archive SHA256は`388e61d30da7cfcf9e0713bac361b7256cfec249b6fe1b0cd1ddea8fa34c9a4c`で、遠隔展開前に一致を確認した。

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

BA1追加のソースcommitは`b1fd5de61a24b906351bc80125104ac67fd2be9c`、archive SHA256は`854f9815f41516af3574fd7712a8923fd999936c3d64756fb3ce54a5543709ac`。遠隔ディレクトリは`srv11/cst-lab/torchcst-ba2-b1fd5de`。同一runのBA幅比較、0.781〜4%のfusion比較、0.049〜0.391%のfusion比較、W占有率を`output/triton-a100-20260927/low-density-ba/`に保存し、結果JSON SHA256は順に`fd790c0de9d08993f4260d93f16cea4866f22c40ec53c8fa38fdde9609448fa4`、`bea6a6b90ab2bdb5fd96fa1febc21a28257ddb1b3207b12279924dfc0cf31b25`、`a9ef376acb31b886b5d07130af8e977af06fb9e97c60fafe9ced2c8dc33c62d4`、`3f4d9cb400bb82ae05d1d50afd94cc494ed29a2cf13d574bb24abd9793104e6c`。全てローカル/遠隔hash一致を確認した。

追加のM512/N8192確認結果は同ディレクトリの`shapes.json`、SHA256は`821e9379b2dcbc252b15aa4ef90389f90a8af0b91dac3bc5040b3a190adb04c0`で、遠隔と一致した。

BA1実行形状の結果は`output/triton-a100-20260927/low-density-ba1-shapes/`に保存した。8,192〜65,536 atoms、131,072〜671,088 atoms、M512/N8192の各`results.json`・`mid.json`・`wide.json`のSHA256は順に`75cd5547378421b62377e3cf57dba0340945ec9bfe193f53fb1597d587a1bde6`、`59a89b4f32c3985b7a269406157f5748178bbad85b13ba44a1dbbb61d0b644e1`、`3e6dc2d63d6d19f58d68e9901573a7ce6e5e70183768a3c70ad4037fa2781f84`。ローカル/遠隔一致を確認した。ソースはBA1追加の`b1fd5de61a24b906351bc80125104ac67fd2be9c`。
