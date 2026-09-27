# atom密度5%のmapped GEMM改善（A100）

続く測定で距離計算の列方向因数化と32行tileによるX再利用を導入した。現在の既定設定と比較値は[atom群とX再利用の記録](five-percent-atom-group-and-x-reuse.ja.md)を参照。以下は当時の16行tileの測定記録である。

2026-09-27。5%ちょうどを主対象に、既存のBA8局所合算からBA1/BN16/BK32、K方向8分割、Section座標のループ外読出しへ進めた。対象は`BlockStripLinear`のforward prototype。全重み行列`W`は標準経路で生成・保持しない。

## 条件と結論

- A100 80GB PCIe MIG 3g.40gb、42 SM、FP32、TF32無効。Strip＋Torus/raw Triweight、CST tile64×64、入力128行、seed21。
- atom数は`round(N*K*0.05)`。4096²で838,861、8192²で3,355,443。初期配置は全atomがI区分で、学習後の偏りやB多数を代表しない。
- 各ケースの同じ層・atom・入力・dense Wで経路を交互にCUDA Graph 3ラウンド、各rep=20ms測定。JITとdense W生成は計測外。準備込み経路はdecode、routing、I/B分類、sort、packを毎回行う。
- dense Wは1,152点を全atom参加のcanonical式と照合。全経路の出力は同じWによる`F.linear`と`atol=rtol=3e-5`で照合し、全て合格。

準備込み中央値、単位ms。旧経路は**明示的にFP融合を有効にした**BA8/BM128/BN16/BK16であり、8192²の旧自動選択はatom数の丸めによりFP融合が無効になる場合があった。したがって表は、旧実装に有利な比較である。

| N=K | dense・事前生成W | 旧BA8 | BA1/BN16/BK32・分割なし | 新既定・8分割＋座標hoist | 旧BA8からの時間短縮 | 新既定/dense |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 0.660 | 23.637 | 20.544 | **16.281** | **31.1%** | 24.7倍 |
| 8192 | 3.026 | 85.239 | 74.457 | **65.131** | **23.6%** | 21.5倍 |

4096²の3ラウンド範囲は旧23.626–23.637ms、新16.276–16.282ms。8192²は旧85.222–85.243ms、新65.112–65.153ms。改善は明確だが、保持済みdenseよりなお20倍以上遅い。

## 変更と選択範囲

`block_fused`のCG列グループを8つのCTA群に分け、各群が`[M,N]`部分出力を保存する。二つ目のkernelが8部分出力を合算する。4096²/M128の一時領域は16 MiB、8192²/M128は32 MiBであり、全Wの64/256 MiBではない。各有効な出力要素は全splitから毎回書かれる。分割によってK方向の浮動小数点加算順序は変わる。

BA1/BN16/BK32では、D=4のSection座標2列をatomループの外に読み出す。距離・Triweight式は維持する。分割のみとhoist込みの同一run比較は4096²で16.733→16.280ms、8192²で67.006→65.126ms。分割なしBA1との比較は上表。

自動選択はA100、tile64×64、論理Wが4096²または8192²、M=128、atom密度が丸めた5%以上かつ5.1%未満に限定する。それ以外の既定は従来どおり。明示的な`FusedConfig`で分割数とhoistの有無を指定できる。8192²の`round(N*K*0.05)`は浮動小数点の比率が0.05を僅かに下回るため、選択閾値を要素数に対する丸め後の比率に合わせた。

## 棄却した案と相談結果

`agy`のClaude/Geminiと`cursor-agent`のGrokから案を集め、既存実験と式を照合した。放射状Triweightを単純なrank-1外積とする案は厳密には成立しない。支持範囲内の多項式展開・prefix momentsは既存の小形状試作が遅く、5%の約84万〜336万atomへそのまま拡張する根拠がない。計算量削減の別設計としては残す。

全Wを一時生成してvendor GEMMを使う診断では、4096²で生成のみ13.246ms・準備込み16.574ms（融合16.737ms）、8192²で生成のみ52.725ms・準備込み67.199ms（融合66.944ms）。64/256 MiBのW領域を追加しても全体の明確な改善はなく、標準経路には採用しない。生成Wとdense対照の値・全出力は許容誤差内。

保守的なsupport boxで省けるatom×列小片は、16列で4096²が0.652%、8192²が0.637%、32列では0.317%/0.298%。全ゼロ小片の正確な割合は16列で約4.0%/3.9%で、atom寄与のゼロ要素は約37.5%/37.4%。この初期配置では、事前box indexによる大幅な評価削減という予想を支持しない。既存の低密度試作でもbox判定の費用が勝っており、今回は実装しない。

K方向8分割に加えてatom方向も2/4分割する試作は全出力が正しいが、4096²では分割なし16.549→2分割18.381/4分割21.500ms、8192²では66.019→73.572/85.779ms。入力読出しとdotをatom分割数だけ繰り返す費用が勝った。この試作は`656a0a5`でrevertし、最終kernel・設定は最終測定sourceと同じである。

## 検証と再現

最終kernel・既定選択のA100関連テストは67件合格。端数形状、I/B境界、Torus seam、CUDA Graph再生後のatom更新を含む。ローカル全体はRuff合格、pytest 426件合格・172件CUDA skip。GPU backwardと学習後の配置は未検証。

- 最終測定source commit：`ff16049478c1daa7bd77de0ba7d4f1cc1f3b31ea`。archive SHA256：`84d695ba17313f1d058ece073ded418364adb04f28f4e3de3b20efec5f0a3153`。
- A100最終測定JSON SHA256：4096² `0bd31e55b6b2bd498d659e949eb49780c8ba4674a0f3e868c17da6a28673687b`、8192² `e0d56b84eae3493a4e733af4a80ab24fa68470408e2a05667616ef82faa166b8`。
- A100 67テストlog SHA256：`50f5ff41a9389f31a424e23d6d83a003ff9572bde8f2e457e7421574000c0672`。remote：`srv11/cst-lab/torchcst-five-percent-final-hoist`。
- 診断source commit：全W `9bab7b97fd0a8d8354a547a3b8c83d8f2fc6ebf0`、support `6bc9a7a49d5d28d53b2fe3f83c431d79445eb071`。結果JSON SHA256はそれぞれ4096² `e26d80e72cd903d88d6f0e9929aa4cd4a7a7cd3ab776e23bd3df2de0c7879b92`、8192² `0892af2ef1486eda4aded3979d72eaaedb3cbf5035774d3ce161dc08b4a86e93`、support `ecf15d08756a6edb801aa435ae8d1106a57fdaa6bc6150a269f559a29e909604`。
- atom方向分割の実験commit：`58d235a0104dd62d9934ec4bfc628b3cbbc7b444`。archive SHA256：`2754ad9a82ba89098d40bb58fb51e4ed576d76c024c3340b9a16701daf159850`。結果JSON SHA256：4096² `7687617566ce98791fb8f46f35d9a8e69083300c92c801f9090a9487cc7026d2`、8192² `3e65f60f2db0e5abd509bb74ba41e1ec0f50fd06135510f228f95d27443fb434`。
- ローカル結果：`output/triton-a100-20260927/five-percent/`。各JSONでsource commit、A100/42 SM、完了flag、atom数、canonical照合、全経路照合、各経路3サンプル、remote/local hash一致を確認した。

```bash
PYTHONPATH=src:. python -m pytest -q --color=no tests/test_fused_compute_configs.py tests/test_block_strip_linear.py
PYTHONPATH=src:. python -u -m prototypes.benchmark_split_k_five_percent --output final-4096.json --source-commit ff16049478c1daa7bd77de0ba7d4f1cc1f3b31ea --size 4096 --batch 128 --full --splits 8
PYTHONPATH=src:. python -u -m prototypes.benchmark_split_k_five_percent --output final-8192.json --source-commit ff16049478c1daa7bd77de0ba7d4f1cc1f3b31ea --size 8192 --batch 128 --full --splits 8
```

次の優先課題は、D=4 Triweight寄与のatom×サイト評価そのものを減らす方法と、学習後の偏り・B区分が多い配置への一般化である。現時点の結果だけでdenseとの速度差を埋めたとは言えない。
