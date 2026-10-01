# I/B分類のまとめ処理と準備用整数配列の削減

> この文書の `prototypes` 実行例は commit `190a1bb3fa6259fde493a30d901d5cab7c2ebb82` 時点の
> 歴史的な再現手順です。削除したコードの参照方法は
> [旧実験コード](../../experiments/cuda/linear/legacy-prototypes.ja.md)を参照してください。

2026-09-26。[所属探索の局所化](local-strip-routing.ja.md)に続き、I/B分類と準備用メモリを改善した。

後続の[計算本体の改善](../../experiments/cuda/linear/notes/fused-compute-reuse.ja.md)で、atom方向の縮約と入力128行への再利用を変更した。
以下はその変更前の計算本体を使った測定である。

## 変更

- 1 atom/CTAのI/B分類を、8 atoms/CTAにまとめた。各atomについて最近傍rowの選択、
  sectionのsupport判定、I/B/idleへの符号化は旧式と同じ演算順序・同点規則を使う。
- 列方向の処理幅は、固定256から`min(256, next_power_of_two(K))`へ変更。
  現在のCST幅64では64列を処理し、幅256超は従来同様に分割する。
- 準備呼び出しはownerの返却を不要にし、所有先配列を分類キーへ上書きして再利用する。
  kernelの各atomは自分のownerだけを読んで最後に自分のkeyを保存するため、他atomとの競合はない。
- キーが収まる場合はint32を使用。stable sortの並び順と、int64のorder/offsetsは維持する。
- `route_and_layout`の既定はownerを保持・返却する。production `prepare`だけが
  `retain_owners=False`を渡す。幅に収まらないキーはint64に戻る。
- atomパラメーター・packed float配列・Pack backwardの保存形式・計算本体は変更しない。
  全Wを保存する処理をCST forwardに追加していない。

## 条件

A100 80GB PCIe MIG3g.40gb・42 SM。Torch2.6.0+cu126、Triton3.2.0。
Strip + Torus、raw Triweight、CST tile64×64、入力128行、FP32/TF32無効、seed21。
atom数はdense要素数の5%（4096²:838,861、8192²:3,355,443）。
初期配置は全atomがI区分。学習後の偏った分布・Bが多数の配置を代表する性能ではない。

同じモデル・atom・入力で旧方式、分類まとめのみ、整数配列も削減した版を交互に比較。
旧方式も前回導入した局所owner探索を使う。benchmark専用dispatchで分類kernelと
owner保持を切り替え、旧式のint64キーとownerの寿命を再現する。
CUDA Graph3rounds、rep=20ms。中央値と範囲を記録。dense W生成とJIT compileは計測外。

## 正確さ

A100で150テスト通過。10件の追加テストで、0 atoms、端数atom数、1/2/多数station、
端数の最終station、16/64/304列、support分類あり/なしを確認した。
旧方式とbatched版のowner/order/offsets、およびcompact版のorder/offsetsが整数完全一致。
compact版でもorderはint64で、全atomが一度ずつ現れることを確認する。
既存のI/B継ぎ目・idle・CUDA Graph更新・forward・勾配の検証も含む。

大型ケースでは新旧のprepared全配列が完全一致。1,152点/ケースを全atom参加の
canonical式と独立比較してdense対照を点検し、5経路/ケースの全出力を同じWのdenseと比較。
5経路にはdense自己比較を含む。許容誤差は従来のatol=rtol=3e-5のまま。

初回はbatchedループ状態のスカラー/ベクトル型不一致でコンパイルに失敗した。
状態を明示的な8要素vectorにして修正。0 atomsのテストも、正のatom数を要求する
モデルから空のparameter viewを渡す形に修正した。修正後の150テストは全通過。

## 時間

単位ms、中央値［最小–最大］。

| 経路 | 4096² | 8192² |
| --- | ---: | ---: |
| dense | 1.409 [1.408–1.409] | 6.309 [3.036–6.415] |
| 準備・旧方式 | 18.883 [18.874–18.887] | 75.159 [34.049–75.708] |
| 準備・分類まとめのみ | 7.368 [7.367–7.373] | 29.399 [14.164–29.413] |
| 準備・分類まとめ＋配列削減 | 6.199 [6.130–6.208] | 24.159 [11.394–24.165] |
| 局所合算forward・旧準備 | 132.722 [132.638–132.965] | 516.605 [232.451–522.195] |
| 局所合算forward・新準備 | 119.409 [119.248–119.960] | 465.957 [465.428–469.939] |
| 局所合算forward・準備済み | 113.932 [108.423–114.726] | 443.880 [443.195–444.098] |

8192²の初回途中で、dense/旧準備/旧forwardを含む複数経路に約2倍の速度状態の変化があった。
原因は未特定。旧forward初回232msと新forward初回465msをそのまま優劣比較に使わない。
後半2ラウンドの旧/new forward比較は522.2→466.0ms、516.6→469.9msで、約9〜11%短縮。
準備単体は初回34.0→11.4ms、後半75.7→24.2ms/75.2→24.2msと、どのroundでも改善した。
4096²の全体は中央値132.7→119.4ms、約10%短縮。

## 一時メモリ

PyTorch追加allocationピーク、MiB=2²⁰ bytes。既存input/model/prepared/比較用dense Wを除き、
forwardでは出力を含む。allocator予約量、CUDA context、Graph pool、勾配・optimizer stateまで
含めた総学習メモリではない。分類まとめのみでは割当量は変わらない。

| 形状 | 旧準備/forward | 新準備/forward | 削減量 | 削減率 |
| --- | ---: | ---: | ---: | ---: |
| 4096² | 65.894 | 56.001 | 9.893 | 15.0% |
| 8192² | 259.921 | 219.602 | 40.319 | 15.5% |

8192²では約40MiB削減したが、まだ約220MiBの作業領域がある。
atomパラメーター単体はdenseの約25%だが、総実行/学習メモリが25%になるという主張ではない。

## プロファイルと残る費用

CUPTIによる通常実行3回平均。CUDA Graphの時間とは別測定で、cache hit率やstallは未測定。

| 形状 | 旧I/B分類ms | 新I/B分類ms | 新forwardの計算本体ms |
| --- | ---: | ---: | ---: |
| 4096² | 15.318 | 4.532 | 114.970 |
| 8192² | 67.194 | 18.230 | 447.245 |

8192²の準備は約24msまで短縮し、計算本体は約447msとなった。
次の主対象は局所合算の計算本体で、atom寄与評価のまとめ方と再利用を改善する必要がある。
現状の全体時間はdenseとまだ大きく離れており、準備の改善だけで競争できるとは言えない。
backendの選択は変更していない。GPU準備は新分類とcompactキーを標準で利用する。

## 再現・保存

- Source commit：`fd1b7dc95a96594db1b793081b7e06fdb005ba82`。
- Source archive SHA256：`4b98d2d6b1fdfc582c17d3515f9e0cdeb423759a27c5e286448b1d8aa272b4d2`。
- リモート：`srv11/cst-lab/torchcst-batched-fd1b7dc`。
- 結果bundle SHA256：`13a9d101ae40174ffa6e7f57ceb978d8ee1c0262cc5684e9ec4e0b3eeb749871`。
- ローカル：`output/triton-a100-20260926/batched/`。

```bash
PYTHONPATH=src:. python -m pytest -q -x --tb=short --color=no \
  tests/test_batched_support_preparation.py tests/test_local_strip_routing.py \
  tests/test_large_strip_preparation.py tests/test_triton_linear.py \
  tests/test_support_layout.py tests/test_block_strip_linear.py
PYTHONPATH=src:. python -u -m prototypes.benchmark_batched_preparation \
  --output-dir study --source-commit fd1b7dc95a96594db1b793081b7e06fdb005ba82
```

source/結果のremote-local hash一致、150テスト通過、2ケース完了、A100/42 SM、
prepared完全一致、10出力比較、42 timing samples、8 profilesを確認した。
ローカルCPU対象は18 passed、73 skipped。変更ファイルのRuffも通過。
