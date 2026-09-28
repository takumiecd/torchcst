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
