# 複数出力特徴で入力を共有する直接計算

2026-09-26。[入力のatom間再利用](block-direct-reuse.ja.md)に続く実験。

## 結論

単純なbroadcast積＋sumによる共有は遅くなったが、atom一つ分の寄与をIEEE FP32 dotで
縮約する方式では、低atom密度の3条件で準備込み時間を約23〜39%短縮した。
CST tileとatom表は共通で、全Wを作らず、グローバルメモリの追加割当も増えていない。
今回の構成改善はTriton内で得られた。Triton全体の限界に達したという根拠にはならない。

| N=K | M | atoms | 現行reuse64 ms | atom dot ms | 短縮率 | dense ms | dot/dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 128 | 8,192 | 14.159 | 10.844 | 23.4% | 1.409 | 7.69× |
| 4096 | 512 | 8,192 | 55.963 | 42.481 | 24.1% | 5.202 | 8.17× |
| 8192 | 128 | 16,384 | 46.488 | 28.462 | 38.8% | 6.422 | 4.43× |
| 4096 | 128 | 131,072 | 171.370 | 178.267 | -4.0% | 1.407 | 126.66× |

32 atoms/tileの高密度条件では4.0%遅くなった。低密度向けの選択肢として残し、
全条件を自動で切り替えるdispatchは導入しない。denseとの差もまだ残っている。

選択した構成はBM64・BN16・4 warps。最終prototypeでは次の指定がその構成になる。

```python
with torch.no_grad():
    y = layer(x, backend="triton_atom_dot")
```

## 実装

`prototypes/block_shared_kernel.py`の`block_direct_shared`を追加した。
1ブロックがBM入力行×BN出力特徴を担当する。
CST tileは64×64を維持し、BNはその内側のCUDA実行分割。
同じCST stationに属する出力特徴で入力断片・atomの候補範囲を共有する。

各atomについてBN×BKの寄与を評価し、その場で入力との積を縮約して出力へ加算する。
atomを合算した重みタイルや全Wを作らない。broadcast版のBN×BM×BKの積はTriton内の一時値であり、
グローバルメモリに保存するテンソルではない。dot版はこの3次元の積を明示的に広げず縮約する。全出力の書き手は一つで、atomicや部分和バッファも不要。
ソース上の入力共有が実HBM転送量を何倍減らしたかは未測定。

端数stationではBNがCSTの境界を越えないように分割し、行・列・batchの端数をmaskする。
旧`triton_reuse`は比較用に保持する。broadcast版は`triton_shared`、
最終的に低密度で有効だったdot版は`triton_atom_dot`として選択する。
GPU forward限定のprototypeで、公開CSTLinear APIとGPU backwardは対象外。

## 測定

- A100 80GB PCIe MIG3g.40gb、42 SM。Torch2.6.0+cu126、Triton3.2.0。
- FP32、TF32無効、seed21、Strip + Torus/raw Triweight。
- 同じケース内で入力・atom・chart・dense Wを共通にし、準備込みで比較する。
- CUDA Graph、3ラウンド、rep20ms、中央値。Python壁時計時間とdense W生成を含めない。
- BM=16/32/64、BN=1/2/4/8、4/8 warpsから7構成を4096²/M128/8192 atomsで比較。
- 全初期atomはIに所属。BとTorus継ぎ目は別の数値テストで検証する。

A100への到達とGPU名を確認した。`ncu`/`nsys`はPATHと通常のCUDA/NVIDIA配置場所に
見つからず、今回も性能カウンタは取得していない。CUPTIのカーネル時間とcompilerの
register/spill/shared情報を使う。実帯域・cache hit率・stallや達成occupancyは未確定。

## 正しさ

新経路を既存のcanonical dense比較、非連続入力、空batch、I/B境界、Torus継ぎ目、
CUDA Graph後のatom更新のテストへ追加した。
さらにBM16/32/64、BN2/4/8、4/8 warpsで、CST高さ5・幅192、物理shape17×197、
入力65行と振幅0を検証する。CST境界を越える出力書き込みや複数列断片の誤りを対象にする。

大型ケースの比較用Wは既存generatorで作り、ランダム1024点と128行の最大絶対値要素を
候補絞り込みなしのcanonical全atom和と照合する。そのWを使ったdense出力と
全出力をatol=rtol=3e-5で比較する。独立oracleで全Wの全要素を評価した検証ではない。

## 再現

Source commit: `48763728a2c7b54b24ef0acee556e03f197e2994`。
Source archive SHA256: `16607beec339c23a7fcab6648669dd9231a2cc578504a4824bf0b65d26159175`。
ローカル/リモートhashを照合し、`srv11/cst-lab/torchcst-block-shared-4876372`へ展開した。

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_block_strip_linear.py
PYTHONPATH=src:. python -m prototypes.benchmark_block_shared \
  --output-dir pilot \
  --source-commit 48763728a2c7b54b24ef0acee556e03f197e2994
```

## 縮約方法の追加比較

最初の3次元broadcast積＋sumは現行reuse64より遅かった。
そのため、同じatom寄与を`tl.dot(X, contribution.T)`へ渡す版を追加した。
`input_precision="ieee"`、TF32無効を維持し、全atomを合算したWは作らない。
振幅をprofile側へ掛けてからdotに渡すため、旧版とはFP32の計算順序が異なる。
同じ数値許容値でcanonical/denseと検証する。

Dot版はBM16/32/64、BN16/32、4/8 warpsから5構成を比較する。
CST tileが小さくてもdotが要求する実行幅を満たすようBKを最低16へpaddingし、maskする。
CPU referenceと全モデルの幾何は変えない。

Dot版source: `44010fdcbff729e043ed4c33da46b7e0180e752f`。
Source archive SHA256: `d80386b91652b53df3e8db5b1ffacbbe8f645ecced807cee78b84698c061f41c`。
遠隔ディレクトリ: `srv11/cst-lab/torchcst-block-shared-44010fd`。

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_block_shared \
  --atom-dot --configs 16:16:4 32:16:4 64:16:4 32:32:4 64:16:8 \
  --output-dir pilot \
  --source-commit 44010fdcbff729e043ed4c33da46b7e0180e752f
```

## pilotの構成比較

構成表記はBM:BN:warps。各セルは中央値［最小–最大］ms。

### broadcast積＋sum

| 構成 | 時間ms | registers/thread | spills | shared bytes/block |
| --- | --- | ---: | ---: | ---: |
| dense | 0.663 [0.659–1.166] | — | — | — |
| reuse64 | 7.202 [7.202–16.099] | — | — | — |
| fused64 | 6.359 [6.357–13.932] | — | — | — |
| 16:1:4 | 25.293 [25.292–56.108] | 56 | 0 | 2176 |
| 16:2:4 | 21.381 [21.377–46.071] | 74 | 0 | 2176 |
| 16:4:4 | 17.666 [17.666–37.920] | 129 | 0 | 2176 |
| 16:8:4 | 16.174 [16.157–33.890] | 167 | 0 | 2176 |
| 32:4:4 | 15.570 [15.527–33.330] | 168 | 0 | 2176 |
| 32:4:8 | 17.953 [17.949–38.262] | 93 | 0 | 4352 |
| 64:4:8 | 19.501 [19.497–42.023] | 159 | 0 | 4352 |

### atomごとのIEEE dot

| 構成 | 時間ms | registers/thread | spills | shared bytes/block |
| --- | --- | ---: | ---: | ---: |
| dense | 0.658 [0.658–0.658] | — | — | — |
| reuse64 | 7.206 [7.202–7.206] | — | — | — |
| fused64 | 6.361 [6.358–6.361] | — | — | — |
| 16:16:4 | 9.931 [9.929–9.935] | 128 | 0 | 8192 |
| 32:16:4 | 9.356 [9.327–9.363] | 130 | 0 | 12288 |
| 64:16:4 | 5.727 [5.727–5.745] | 168 | 0 | 20480 |
| 32:32:4 | 6.501 [6.490–6.511] | 190 | 0 | 16384 |
| 64:16:8 | 9.420 [9.405–9.433] | 128 | 0 | 20480 |

broadcastのpilotでは第3ラウンドに複数の経路が約2倍へ変動した。
原因は未特定。異なる実行の絶対時間同士を比較せず、同じ実行の現行版との比較を使う。
Dot pilotでは候補中64:16:4が最速だったため、この構成に絞って大型条件を測った。

単純版の最速構成32:4:4は168 registers/thread、shared 2,176 bytes。
選択したdot版も168 registers/threadだが、sharedは20,480 bytesへ増える。
旧reuse64は113 registers/thread、shared256 bytes。全てcompiler spills=0。
「資源使用が減ったので高速化した」という結果ではない。
共有と縮約方法の変更による改善であり、帯域・命令・同期待ちのどれが主因かは未特定。

## 大型条件の範囲と融合版比較

中央値［最小–最大］ms。準備込み。全ケースCST tile64×64。

| N / M / atoms | dense | reuse64 | atom dot | 融合64 |
| --- | --- | --- | --- | --- |
| 4096 / 128 / 8,192 | 1.409 [1.348–1.455] | 14.159 [14.006–14.868] | 10.844 [10.010–10.867] | 13.953 [13.930–13.960] |
| 4096 / 512 / 8,192 | 5.202 [4.908–5.208] | 55.963 [55.656–56.917] | 42.481 [41.548–42.528] | 46.407 [45.154–46.541] |
| 8192 / 128 / 16,384 | 6.422 [6.126–6.425] | 46.488 [46.150–46.515] | 28.462 [28.388–28.520] | 54.222 [54.175–55.172] |
| 4096 / 128 / 131,072 | 1.407 [1.347–1.408] | 171.370 [170.927–172.494] | 178.267 [177.992–179.421] | 36.690 [36.644–36.950] |

## 計算本体とメモリ

大型比較の4096²/M128をCUPTIで3ステップ記録。
`cat=kernel, ph=X`だけを集計し、注釈範囲との二重計上を避けた。
CUDA Graph時間の表とは別計測。

| 経路 | 全kernel ms | 計算本体ms | 本体割合 |
| --- | ---: | ---: | ---: |
| reuse64 | 14.876 | 14.346 | 96.43% |
| 64_16_4 | 10.825 | 10.294 | 95.09% |

残る時間の約95%はatom計算と出力への縮約。準備だけを改善しても差は埋まらない。

出力込み追加割当ピークMiB。既存入力・モデル・prepared・比較用Wは除外する。

| N / M / atoms | reuse64 | atom dot | 比較用dense W本体（別枠） |
| --- | ---: | ---: | ---: |
| 4096 / 128 / 8,192 | 2.252 | 2.252 | 64 |
| 4096 / 512 / 8,192 | 8.252 | 8.252 | 64 |
| 8192 / 128 / 16,384 | 4.627 | 4.627 | 256 |
| 4096 / 128 / 131,072 | 10.113 | 10.113 | 64 |

## 最終検証と記録

- 最終sourceでA100 **42 tests passed**（21.58秒）。選択した64:16:4の端数station/batchを含む。
- 小タイル3×5 / CST4×8も追加し、dotのpaddingを確認した。
- 初期pilot28 tests、dot pilot41 testsも通過。
- pilot 10経路、dot pilot 8経路、大型4条件×4経路の計34出力比較が全て合格。
  最大絶対誤差は1.45e-6未満。許容値を緩和していない。
- 各ケースのcanonical点検、全bucket内atom数、全atomがI、3ラウンド、完了フラグ、
  source commit、A100/42 SM、結果bundleのローカル/リモートSHA256一致を確認した。

最終source commit: `5239a038daaf6729615c25e80327fdcb68310ae1`。
変更はatom dotの既定BM64と、同じ構成の端数テスト追加。
Source archive SHA256: `88a52ed4c6dc56d006d04d9bee0c83af8803c5a35c1b654a1053a48a363b3acb`。

```bash
PYTHONPATH=src:. python -m prototypes.benchmark_block_shared \
  --atom-dot --configs 64:16:4 \
  --cases 4096:128:8192 4096:512:8192 8192:128:16384 4096:128:131072 \
  --output-dir scale \
  --source-commit 5239a038daaf6729615c25e80327fdcb68310ae1
```

保存先は`output/triton-a100-20260926/`（Git管理外）。

| 結果bundle | SHA256 |
| --- | --- |
| `block-shared-broadcast-results.tar.gz` | `6c6594fb12df04cf76bfab7250acf0878ca174275af8b55da5065c224d788d81` |
| `block-shared-dot-pilot-results.tar.gz` | `ff45b8be71380e4a565f56a1652be64e6cf8d61c266bd199edcfd89686473195` |
| `block-shared-scale-results.tar.gz` | `000208e21fecb63fa2180945c9450339b563e36d5045d0af962f641de6696f93` |
