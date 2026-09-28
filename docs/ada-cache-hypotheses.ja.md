# RTX 6000 Adaのキャッシュを使う次の仮説

2026-09-28。8192²、atom密度5%、seed 21、RTX 6000 Ada。学習経路は全dense W・dWを持たず、局所W窓を使う。基準の学習ステップは[既存の記録](ada-precision-and-shared-gpu.ja.md)を参照。

## 構造を測った結果

候補一覧は65,536個の16×64タイルに対して空タイル0個、候補atom数は中央値125、平均147、最大176。先頭・中央の各1024行の局所Wを調べると、全16×64タイルが非ゼロで、非ゼロ要素率もほぼ100%だった。**5%のatom密度からブロック疎GEMMの効率は期待できない**。

64個の親タイルを均等間隔で採り、候補atomとサイトの距離を評価した。16×64では候補の11.1%がタイル全体で支持領域内、85.7%が境界を横切り、3.1%が全体で支持領域外だった。4×16まで小さくしても、全体内25.8%、境界54.0%、全体外20.2%。Triweightの多項式を「完全に内側」の候補だけ集約する案は、この分布では主経路を置き換えられない。

局所Wの16×64タイルを256個採ってSVDのtailを測ると、rank 4/8/12近似の相対Frobenius誤差中央値は1.40×10⁻³ / 3.67×10⁻⁴ / 1.40×10⁻⁴。rank 8の最大誤差は1.33×10⁻³。現在の全出力照合を満たす根拠はなく、低ランク近似を既定経路に入れない。これらはWタイルの診断値であり、学習出力・勾配の誤差そのものではない。

## L2保持ヒント

入力XはM=2048で64 MiB、局所Wの1024行窓は32 MiB、packed atomは約76.8 MiB。[NVIDIAのRTX 6000 Ada資料](https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf)にあるL2は96 MiBなので、三者を同時には保持できない。繰り返し読むXを`evict_last`、書き終えたYを`evict_first`にする仮説を試すため、`bounded_gemm_kernel`に既定値が空の任意ヒントを追加した。ヒントの各組み合わせは既定結果と完全一致した。

速度比較中に別のGPUプロセスが約4.7 GiBを使い始め、同じ条件の反復が約13–35 msに揺れた。記録した中央値は採否に使えない。別プロセスを止めず、速度実験を中断した。次に空いたときは、既定・X保持・W保持・Y早期退避を交互に複数回実行し、ステップ全体とピークメモリも確認する。[CUDAのL2保持方針](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)は優先度であり、ヒットを保証しない。

次に検討する大きな変更は、atomの移動を安全に検査したうえで候補リストとルーティングを複数ステップ再利用すること。毎ステップの構築とソートを減らせる可能性があるが、支持領域の変化を見逃すと正確性が崩れる。まず更新前後の候補集合変化率と検証コストを測る。

実験コードは`prototypes.profile_candidate_occupancy`、`prototypes.profile_weight_tile_rank`、`prototypes.profile_gemm_cache_hints`。生のJSONは`output/ada-20260928/cache-hypotheses/`に保存した。
