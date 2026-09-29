# Blackwellでの5% CST学習試作の再測定

2026-09-29。Colab Proの実機 **NVIDIA RTX PRO 6000 Blackwell Server Edition**（compute capability 12.0、PyTorch 2.11.0+cu128、Triton 3.6.0）で、コミット `dd10ecf6ae35b438331faa2e9fd1fed547e1bf14` をGit archiveから展開して測った。archive SHA256は`0ff90ca646809b14c82fd0762583947db181e96c2549cad374d8e5354b5d0f34`。結果JSONは[`data/blackwell-20260929/`](data/blackwell-20260929/)に保存した。測定後にColabセッションを停止し、割当0件を確認した。KyutechのAda profileは切り替えていない。

Triweight、64×64 tile、atom数`round(0.05*N*N)`、FP32、seed 21、単層AdamW `foreach=True`。CSTは`listed_bounded`の8 bit候補リスト、局所W生成`listed_unroll=4`、候補構築`listed_builder_ba=32, listed_builder_warps=1`を明示指定した。1024²では512行窓・保持0枚、8192²では1024行窓・保持2枚または4枚。M=2048のforwardと入力勾配はFP16x3、局所dWはIEEE FP32。M=128はIEEE FP32。CST学習ステップは全W・全dWを生成しない。

## 完全ステップ速度

dense、従来設定、改善設定のGraphを同一プロセスで24回ずつ交互再実行した。時間比は対応ペア比の中央値。比較用のdense重みとoptimizer状態は同じプロセスに共存するため、**この測定のメモリ値は採用しない**。

| W形状・入力行数 | W保持窓 | dense中央値 | CST中央値 | CST/dense対応ペア比 | builder BA32/1 対 BA8/4 | unroll4 対 unroll1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024²・M=128 | 0 | 0.135 ms | 0.712 ms | **5.222倍** | 0.9883倍 | 未測定 |
| 1024²・M=2048 | 0 | 0.372 ms | 0.928 ms | **2.487倍** | 0.9895倍 | **0.9207倍** |
| 8192²・M=2048 | 2 | 15.205 ms | 20.629 ms | **1.342倍** | 0.9812倍 | 未測定 |
| 8192²・M=2048 | 4 | 15.208 ms | 20.266 ms | **1.327倍** | **0.9822倍** | **0.9759倍** |

8192²・4窓のunroll比較は別の交互runで、dense 15.189 ms、CST 20.149 ms、対応ペア比1.3269だった。表の単独中央値は実行順や負荷で変動するため、改善率には同じrunの対応ペア比を使う。1024²・M=2048のunroll4効果は約7.9%、builder変更は約1.1%。8192²・4窓では順に約2.4%と約1.8%。それぞれ個別の比較であり、改善率を単純加算しない。全条件のforward出力はdense oracleへの`atol=rtol=3e-5`照合に通過した。

## メモリと更新の確認

ピーク割当は方式ごとに**別プロセス**を起動し、Graph capture中の`torch.cuda.max_memory_allocated()`を記録した。CUDA contextやallocator予約分は含まない。

| W形状・入力行数 | CSTピーク | denseピーク | CST削減率 |
| --- | ---: | ---: | ---: |
| 1024²・M=128 | 46.98 MB | 57.33 MB | 18.1% |
| 1024²・M=2048 | 62.70 MB | 85.25 MB | 26.5% |
| 8192²・M=2048、保持2窓 | 839.61 MB | 1,577.59 MB | 46.8% |
| 8192²・M=2048、保持4窓 | 884.78 MB | 1,577.59 MB | 43.9% |

全CST条件でGraph captureに成功し、最初の再実行でパラメータが変化した。Graphとeagerの3更新後までを照合した最大パラメータ差は、1024²・M=128で4.77e-7、1024²・M=2048で2.98e-8、8192²・4窓で5.96e-8。今回のBlackwell runでは、atom勾配全要素を独立dense oracleへ照合していない。1024²・M=128のGraphピークはBlackwellのdense Graphより低いが、Adaの以前の**eager** denseピークとの横断比較で採否は決めない。

前回のColab Blackwell記録では旧listed経路・eager同期wall測定の8192²・M=2048がdense比約1.60倍だった。今回は新コードのGraph交互測定なので、1.60→1.33を一つの変更による短縮率とは解釈しない。Adaでの最新の同設定・Graph交互測定はdense比約1.23～1.24倍だった。**絶対時間はBlackwellのほうが短いが、この条件のdenseに対する相対差はAdaより大きい。** GPU、PyTorch/Triton版、実行環境が異なるため、世代差の原因はこの測定だけでは分からない。

生データは`paired-*.json`、`peak-*.json`、機種と版は`inventory.json`。候補はまだ試作APIの明示指定であり、公開backendの既定dispatchへは入れていない。
