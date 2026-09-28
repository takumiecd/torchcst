# 5% CST学習：固定長の圧縮候補リストと窓の選択

2026-09-29。RTX 6000 Ada 48 GB、PyTorch 2.9.1+cu128、Triweight、64×64 tile、FP32、atom密度5%。主対象は8192²・入力行数M=2048の単層AdamW完全ステップ。CST側は全W・全dWを生成・保持しない。最も大きい試験窓は4096×8192で全Wの半分、採用候補の窓は1024×8192以下。速度は同一プロセスでGraphを交互再実行した結果を優先し、ピークは方式ごとに別プロセスで測ったPyTorch割当ピークとする。

## 実装した候補リスト

`materialize_mode="listed_bounded"`を試作経路に追加した。各16×64 site tileへ候補番号を192件だけ確保し、通常はbucket内の順位を8 bitで格納する。候補が192件を超えるか、元のbucket範囲が256件を超えるtileは`count=-1`とし、局所W生成とatom勾配の双方がそのtileのbucketを全件再走査する。したがってatomが移動してもリスト書き込みが範囲を超えず、CPUで最大件数を読み出す同期もない。全件再走査では候補箱による省略だけを失い、支持域の計算式は維持する。

初期配置では65,536 tileの候補数は中央値145.5、最大176、bucket全件数は205。192件・8 bitの通常経路で全tileを表せる。リスト領域は8192²で約12.6 MB。従来のGPU予約CSRは証明可能な上限`8×atom数`に対して約107 MBを確保する。圧縮が可能なのは**溢れたtileを安全な全走査へ戻す**ためであり、初期分布だけを信じた固定長リストではない。

同じ8192²初期配置で従来リストと8 bitリストの有効要素は完全一致し、先頭1024行の生成Wもビット一致した。局所dWからのatom勾配差は最大7.63e-6、相対L2が2.22e-8で、atomic加算順序による従来CSRとの差と同程度。候補構築wall中央値はCSR 0.723 ms、8 bit版0.692 msだった。

## 完全ステップの結果

| 経路 | 窓・保持数 | Graph capture割当ピーク | Graph単独再実行中央値 |
| --- | --- | ---: | ---: |
| dense AdamW | 全W | 1,577.59 MB | 35.91 ms |
| 旧`listed_csr` | 1024行・2窓 | 933.83 MB | 38.98 ms |
| 旧`listed_csr` | 1024行・4窓 | 979.00 MB | 37.90 ms |
| 新`listed_bounded`、8 bit | 1024行・4窓 | **884.78 MB** | 38.00 ms |
| 新`listed_bounded`、8 bit | 512行・8窓 | 868.00 MB | 37.77 ms |
| 新`listed_bounded`、8 bit | 512行・0窓 | **822.84 MB** | 39.39 ms |

単独再実行はGPUの時間変動を受けるため、行間の小差を速度の結論に使わない。1024行・4窓の8 bit版とCSR版、dense版を同じプロセスで32回ずつ交互にGraph再実行した。対応ペアのCST/dense比は2回で**1.262倍／1.272倍**、8 bit/CSR比は**0.995倍／0.986倍**。後者はoverflow経路の型変換修正後に再測定した値である。速度はCSRとほぼ同じまま、同じ4窓で割当ピークを979.00→884.78 MB（94.22 MB）下げた。元の2窓CSRと比べても約49 MB低い。dense Graphより約43.9%低い。密な全W・全dWへ戻す経路はない。

候補リストだけ16 bitにした中間版は同じ4窓で896.52 MB、同一実行内の8 bit版/CSR版に先立つ比較で比0.985だった。約1%の差はGPU変動を考えると断定しない。最終8 bit版はメモリ削減が明確で、速度の悪化は確認されていない。

## 窓幅とCUDA C++の反証

8 bit化前に窓幅を256、512、1024、2048、4096行へ振った。4096行まで広げても単独再実行の完全ステップ中央値は約38 msで横ばい、captureピークは最大約997 MBへ増えた。256行は保持16窓でも約44 msで遅かった。8 bit版では512行・保持0窓がピーク約823 MBのメモリ優先設定、1024行・保持4窓が旧CSRと同程度の速度を保つ設定となる。512行・保持8窓は単独再実行では37.77 msだったが、1024行・保持4窓と同一プロセスで交互に32回比べると**1.116倍遅い**。速度優先の推奨は1024行・4窓とする。

局所W生成だけCUDA C++へ移した単独試作も測った。atomパラメータのロードをsite間で共有し、32～512 threads/CTAを比較した後でも、1024×8192窓はTriton 0.537 msに対してCUDA最良64 threadsが0.689 ms。生成Wはビット一致するが、CUDA版の移植は速度改善にならない。Triton側も1 warpが最速で、2/4/8 warpは遅かった。CUDA C++試作は完全ステップへ接続していない。

## 正しさと残る範囲

境界をまたぐatomを含むGPUテスト、リスト容量1件での強制overflow、8 bitで表せないbucket順位のguardを通した。atomを一か所へ集める追加試験では、256件を超す順位を`uint8`のまま代入すると折り返すバグを発見し、**全走査用の順位を先にint32へ変換**して修正した。この状態でGraph再実行のforwardをdense参照へ照合し、同じoverflowに対するforward・入力勾配・atom勾配をCSRへ照合した。8192²でGraph版とeager版のAdamW更新を30回照合し、パラメータ最大絶対差は1.62e-5。atomic加算順序による非決定性を含む。dense Wとのforward照合は`atol=rtol=3e-5`で違反0。Graph capture成功と更新後パラメータ変化も確認した。

`listed_bounded`はまだ明示的に選ぶ試作モードで、公開backendの既定dispatchは変えていない。ピークはPyTorch allocatorの値であり、CUDA contextや他プロセスを含むカード全体のVRAMではない。1024²・M=128でGraphのピークがdenseより大きくなる問題も別途残る。次の速度改善対象は局所W生成とatom勾配の合計、およびMが小さい場合の別経路である。

小形状1024²・M=2048も8 bit版の512行・保持0窓を測った。Graph captureピーク62.70 MB、再実行中央値0.965 msで、5回のGraph/eager更新照合は最大絶対差2.98e-8。以前の同shape dense Graphは約85.25 MB、0.487 msであり、メモリは低いが速度差はなお約2倍ある。異なるrunの時間比は目安に留める。

再現コードは`prototypes.block_streamed_backward`、`prototypes.block_tile_atom_lists`、`prototypes.block_materialize_listed`、`prototypes.profile_candidate_csr`、`prototypes.probe_csr_graph_step`、`prototypes.profile_paired_dense_cst`。生JSONは`output/ada-20260929/overnight/`。CUDA C++反証は`prototypes/cuda/materialize_bounded.cu`と`prototypes.profile_cuda_materialize`。結果はRTX 6000 Adaが空いていることを確認してから測った。

CUDA C++試作の実行にはCUDA Toolkitと`ninja`が必要だった。Ada側では作業ディレクトリだけに`python -m pip install --target .probe-deps ninja`で入れ、`PATH="$PWD/.probe-deps/bin:$PATH" PYTHONPATH=src:. python -m prototypes.profile_cuda_materialize --size 8192 --output results/cuda-w.json`で測った。PyTorch環境全体への依存追加はしていない。

## 02:30 JSTの追加検証（採用なし）

同じ空き状態のRTX 6000 Adaで、最終8 bit版のGraphを再プロファイルした。GPU kernel合計34.93 ms中、FP16x3の局所GEMMが9.02 ms、局所W生成6.35 ms、atom勾配5.97 ms、IEEE FP32の局所dW GEMM5.82 ms、routing1.35 ms、候補一覧構築1.00 msだった。単一ステップのprofile値で、交互Graph再実行の中央値とは測定対象が異なる。

局所dWのTensor Core化では、従来の3項FP16分解へ残差同士の積を加えた4項版`fp16x4_dw`を試した。M=2048、seed 21でforwardと入力勾配は3e-5基準に合格したが、atom勾配は3e-4基準を **10,387 / 16,777,215要素**で超えた。3項版の10,421件からほぼ改善しておらず、省略した4項目だけが原因ではない。完全ステップへ採用しない。試作コードは`prototypes/bounded_gemm_fp16x3.py`、再現は`prototypes.compare_bounded_gemm_gradients --compare-mode fp16x4_dw`。

候補一覧と計算タイルの行数を16から8、32へ変更した。最初の1024行窓で、8行版のW生成は0.696→0.686 msとほぼ同じだが、候補構築は0.884→1.657 ms、一覧領域は12.58→25.17 MB、atom勾配は1.099→1.126 msとなった。32行版は候補構築0.689→0.376 ms、一覧領域12.58→6.29 MBだが、W生成0.541→0.631 ms、atom勾配0.909→0.971 msへ悪化した。各値は別々の交互測定runで、そのrun内の16行版と比較する。Wは両方とも16行版とビット一致した。一方atom勾配は3e-4基準で8行版216,147件、32行版177,508件の差があり、縮約順の違いを含むので採用しない。試作は`prototypes.profile_bounded_tile_rows`に残した。既定の16行経路は変更していない。

最後にPyTorch AdamWの`foreach=True`を`fused=True`へ変えて比較した。同一プロセスの32回交互Graph再実行でCST完全ステップの中央値は38.341→38.236 ms、対応ペアのfused/foreach比中央値は0.993。明確な速度改善とは言えない。別プロセスのfused Graph captureピークは884,781,056バイトで、従来foreachと同値だった。CSTパラメータの32回後の相対L2差は1.07e-6、最大絶対差は0.133で、atom勾配のatomic加算順やoptimizer実装差を含む。denseとfused CSTの交互比較ではCST/dense比が1.314だったが、foreach時の1.262–1.272と同一runではないため直接の優劣判断に使わない。`foreach`を維持する。測定用CLIに`--optimizer-mode`を追加した。

この追加検証後、RTX 6000 Adaで`tests/test_block_streamed_backward.py`と`tests/test_streamed_materialization.py`は61件通過した。結果JSONは`docs/data/ada-20260929/`にも保存した。次は局所W・atom縮約で実際に処理している候補atomとsiteの比率を測り、候補選別を強める余地を判断する。

その後、32 stationを標本にして実際の支持域とbox候補を照合した。16×64 tileの26,208個のatom–tile対のうち、boxが残すのは18,809件（71.77%）、真に1サイト以上を支持するのは18,229件（69.56%）。**box候補内の偽陽性は3.08%**だった。8×64でもbox候補33,935件に対する偽陽性は4.26%で、より細かく分ける費用を正当化しにくい。これは標本・初期配置についての割合であり、atom移動後や別の密度での上限ではない。64×64 station内のatom–site対全体では約44.93%が支持域内にある。候補をさらに削る余地より、候補のうち支持域外の個別site約37%に費やす演算と、実際に支持するsiteの計算を減らす方が有望と判断した。診断コードは`prototypes.profile_backward_support`、集計JSONは`docs/data/ada-20260929/overnight-support-ratio-box.json`。

## 03:30 JST：局所Wのループ展開

Adaが引き続き空いていることを前後に確認した。`materialize_listed`のatomループへTritonの`loop_unroll_factor`を付け、`listed_unroll=4`を明示的に選べるようにした。既定値1と公開backendは変更しない。最初の1024×8192窓では、1/2/4/8回展開のGraph中央値が0.538/0.504/0.490/0.494 ms。どの版の生成Wも基準と**ビット一致**した。4回展開を採用候補とする。

8192²・M=2048・5%の完全AdamWステップで、dense、従来unroll1、unroll4のGraphを同じプロセスで交互に32回、続いて48回実行した。前後ともRTX 6000 Ada profileで使用率0%、割当39 MiB、別ジョブは見えなかった。

| 交互Graph測定 | 32回 | 48回 |
| --- | ---: | ---: |
| unroll4 / unroll1 対応ペア比の中央値 | **0.95896** | **0.95899** |
| unroll4 / dense 対応ペア比の中央値 | 1.2256 | 1.2385 |
| unroll4単独中央値 | 38.37 ms | 39.45 ms |
| unroll1単独中央値 | 42.44 ms | 43.27 ms |

対応ペアの比では約**4.1%短縮**。単独中央値にはGPUの時間変動が混ざるため、差4 msをそのまま改善量とは扱わない。1ステップのGPU profilerでは局所W生成12呼び出し合計が6.351→5.293 ms、全kernel合計が34.929→33.580 msとなり、改善方向は交互Graph測定と一致した。なお異なるprofile実行間の小差に厳密な因果帰属はしない。

別プロセスのGraph capture割当ピークはunroll4 CST **884,781,056 bytes**、dense **1,577,585,152 bytes**。従来unroll1とCSTピークは同値で、denseより43.9%低い。全W・全dWは引き続き作らない。seed 21/22/23のIEEE対照ではforwardと入力勾配の3e-5基準違反0、atom勾配の3e-4基準違反0。Graphとeagerの12更新後の最大パラメータ差は7.63e-6。強制overflowとGraph再実行を含むGPUテストは63件通過した。

並行して二つの案を採否判定した。WをGEMM前に高位・低位のFP16へ分ける試作は局所forward 0.546→0.549 msで改善せず、入力勾配GEMM単体は0.578→0.547 msだったが、分割コストと2配列の生成・保持を含めていない。完全ステップへは接続しない。atom勾配で1 CTA当たりのatomレーンやwarp数を増やす試行は、最良のBA2・1 warpが局所0.897→0.874 msと小差に留まり、atom勾配の3e-4基準を228,724要素で外した。BA4以上はregister spillや低速化も発生したため採用しない。両方とも診断コードとJSONを保存した。

推奨する実験設定は`mapped_streamed_trainable(..., materialize_mode="listed_bounded", listed_unroll=4, window_rows=1024, cache_windows=4, gemm_mode="fp16x3_dx", forward_gemm_mode="fp16x3", atom_kernel="staged_listed")`。unroll4は大形状Adaで検証した明示的な試作オプションである。測定CLIは`prototypes.profile_paired_dense_cst --compare-unroll --listed-unroll 4 --graph`と`prototypes.probe_csr_graph_step --listed-unroll 4`。JSONは`docs/data/ada-20260929/overnight-unroll4-*.json`、単独カーネル比較は`overnight-w-unroll.json`、反証は`overnight-bounded-lanes.json`と`overnight-presplit-weight-gemm.json`。

同じ展開をatom勾配kernelへ移す案もboundedリストで別途測った。1段・展開1回の局所0.901 msに対し、2～4段のpipelineは1.010～1.042 ms。展開2回は0.949 ms、4回は1.007 msと遅く、両方で3e-4基準を約228,730要素で超えた。展開の有無がatomの縮約・atomic更新順を変えるため、局所W生成と同じ判断はできない。atom勾配側は現行1段・展開1回を維持する。JSONは`docs/data/ada-20260929/overnight-bounded-atom-pipeline.json`。
