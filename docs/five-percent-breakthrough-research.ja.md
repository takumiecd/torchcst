# 5% CST 学習：新しい計算経路の探索

続くRTX 6000 Adaでの圧縮候補リスト、窓幅、CUDA C++の実測は[固定長の圧縮候補リストと窓の選択](five-percent-bounded-lists-ada.ja.md)に記録した。

2026-09-29。対象は Triweight、64×64 tile、RTX 6000 Ada。全 W と全 dW を作らず、dense 学習より低いピークメモリを守る。1024²・M=128/2048 は小形状の反証試験、8192²・M=2048 は主目標。速度の基準は[対象形状と完全ステップの測定](five-percent-target-shapes.ja.md)。以下の時間は特記しない限り**局所窓の kernel 時間**で、AdamW 全ステップとは異なる。

## 今回分かったこと

1024²・5%（52,429 atom）で、候補atomは16×64 site tile当たり平均146.77、中央値125、最大176。候補atom×siteの62.48%が実際の支持域内だった。論理 W は先頭・後半とも非ゼロ率100%。したがって5%は W の疎度ではない。16×64 subtileを丸ごと省ける割合は4%だけで、境界 subtile が84.9%。候補の支持判定を細かくするだけでは、密 GEMM を省く経路にならない。測定は `prototypes.profile_candidate_occupancy`、JSON は `output/ada-20260928/cache-hypotheses/candidate-occupancy-1024.json`。

| 試作 | 条件 | 現行 | 試作 | 判断 |
| --- | --- | ---: | ---: | --- |
| site×atom の2次元 tile | 1024²、先頭512行の W 生成、最良 8×16 site×8 atom | 0.0797 ms | 0.0966 ms | 遅い |
| 同上 | 8192²、先頭1024行、最良 8×16 site×8 atom | 0.5423 ms | 1.1939 ms | 遅い |
| 局所 dW GEMM と atom 縮約の融合 | 1024²、M=128、先頭512行 | 0.1315 ms | 0.2136 ms | 遅く、勾配の要素別許容違反298 |

2次元 tile は atom 直列ループを減らすが CTA 数と候補読み込みが増え、利得がなかった。W の照合は最大差約4e-9、出力違反0。コードは `prototypes.block_materialize_subtile` と `prototypes.profile_parallel_weight_atoms`、JSON は `subtile-materialize-{1024,8192}.json`。融合案は `prototypes.profile_fused_listed_dw`、JSON は `fused-listed-dw-1024-m128.json`。勾配の相対 L2 は6.77e-6と小さいが、最大絶対差0.00757で要素別許容を超える。Tensor Coreを使う TF32x3 局所 dW はこの構成で採用できない。いずれの試作も全ステップの高速化を示していない。

## 残る候補：支持区間の多項式集約

Triweight は支持域内で `amp × (1 − precision × distance²)^3`。理想的な円座標なら各列・atomについて角度方向の3次三角多項式となり、定数、cos/sin の1～3倍角の7基底で表せる。支持域は角度の区間なので、atomごとの係数を区間端に加減し、行方向へ prefix scan すれば、W を各 site×atom で直接評価せずに作れる。dWからatom勾配への縮約も7本の重み付き区間和で表せる。これはGrok/Claudeとの相談で出た案で、**GPU速度は未測定**。

問題は実際の site が `float32(circle[row] * section[col])` として丸められていること。1024²/8192²の実際のprepared geometryからランダムに100 stationを採った試験では、理想的な角度幾何を使う式は8192²で単一atom W の最大差4.56e-4、支持域判定131箇所の不一致を生んだ。1024²でも支持域判定2箇所の不一致がある。既存の極座標chord経路も大形状で不正確かつ遅かった。7基底の速度見積りをそのまま採用根拠にできない。

各4行 sub-tileを**実際に丸められた FP32 site**に固定し、行内変位 `u,v,h=u²+v²` から立方式を展開すると20項になる。同じ実geometry試験で単一atom W 最大差は両サイズとも2.98e-8以下、支持域判定不一致0。205 atomを16×64 tileに足した10例では W 最大差が5.96e-7以下、M=128の局所積の最大差が7.63e-6以下。16行固定でも集約 W 最大差6.56e-7以下、局所積最大差8.59e-6以下。一方、64行固定の FP32 展開は集約 W 最大差1.41e-3、局所積最大差0.0107で不可。訓練後atom、勾配での検証も未実施。CPU pilot は `prototypes.probe_station_polynomial`、JSON は `output/station-polynomial-pilot-real-{4,16,64}.json`。

20項版の単純なprefix kernelはまだ書かない。16行×64列×平均147候補では直接評価が約15万 site×atom、20係数の候補×列更新だけで約19万となり、prefix本体を加える前から演算数で負ける。64行なら係数を再利用できるが、上記の数値試験では FP32 精度が破綻した。FP64による64行版は単一atomの W 差が3.52e-7まで改善したものの、AdaでのFP64演算コストと集約後誤差は未測定。この枝は、係数数を減らしつつ丸め誤差を補正する式か、64行を高速に安定化する方法が見つかってからGPU化する。

## 別の固定費

1024²では `candidate_counts.max().item()` が候補配列の幅を決めるため、毎ステップCPU/GPU同期が入る。prepared tensorsを固定し、事前に証明した幅を直接渡す診断では、候補生成のwall時間が1024²で0.106→0.035 ms、8192²で0.751→0.683 msになった。ただし固定幅は訓練中の候補移動で溢れ得るため、そのまま採用できない。

## GPU上で領域を予約する候補リスト

`listed_csr`試作では、各16×64 tileのプログラムが対象となるbucket内atomの**全件数**だけをGPU上のcursorから一度に予約する。支持域判定で残ったatomを、その予約範囲の先頭へ詰める。各atomは高々2 stationの近傍に現れ、各stationは4 tileなので、予約総数は高々 `8×atom数`。これをint32配列として事前確保すれば、atomが訓練中に移動しても範囲外書き込みは起きず、CPUへの件数読み出しも不要。1024²で約1.68 MB、8192²で約107 MBの予約上限が必要。全W・全dWは作らない。

同じ prepared tensorsで、従来の候補リストと有効要素が完全一致し、生成 W はビット一致した。atom勾配の差はatomic加算順序によるもので、1024²で最大3.82e-6／相対L2 1.11e-8、8192²で最大7.63e-6／相対L2 2.24e-8。候補リスト生成wall中央値は1024²で0.100→0.040 ms、8192²で0.766→0.724 ms。1024²の完全AdamWステップはM=128で従来1.310→CSR1.229 ms、M=2048で1.558→1.478 ms。8192²・M=2048の24回交互測定は従来43.139→CSR40.399 ms、対応ペア比中央値0.979。後者の差は約2%なので、負荷変動の範囲を考慮して評価する。

別プロセスで測った8192²・M=2048のPyTorch割当ピークは、dense 1,560.54 MB、従来listed 839.61 MB、CSR 933.41 MB。CSRは従来より93.80 MB増えるが、denseより40.2%低い。1024²・M=2048ではdense 68.21 MB、CSR 63.98 MBで、差は4.23 MBまで縮む。どちらも現時点の絶対条件は満たす。結果は`prototypes.profile_candidate_csr`、`prototypes.profile_paired_dense_cst`、`prototypes.benchmark_mapped_training_memory`と`output/ada-20260928/cache-hypotheses/{candidate-csr,paired-csr,memory-*}.json`にある。

## 完全ステップのCUDA Graph

GPU上の領域予約で候補幅のhost同期がなくなり、`listed_csr`の準備、forward、backward、AdamWを同じCUDA Graphに収められた。従来listedを同じ手順でcaptureするとCUDAエラーで失敗した。1024²・M=2048のCSRはeager約1.48 msからGraph再実行約0.99 msへ短縮。denseも同じGraph条件にすると、20回交互測定の中央値はdense 0.487 ms、CSR 0.994 ms、対応ペア比2.053。CSRのcapture時ピーク63.14 MBはeager denseの68.21 MBとGraph denseの85.25 MBの両方より低い。GraphとeagerのAdamW後パラメータを10ステップ照合した最大差は8.94e-8。

8192²・M=2048のGraph同士を24回交互測定するとdense 32.408 ms、CSR 42.273 ms、対応ペア比1.266。CSRのcapture時ピーク933.83 MB、Graph denseは1,577.59 MB。3ステップのGraph対eagerパラメータ最大差4.66e-8。別々のプロセスで得たGraph速度は負荷変動で比が大きく変わったため、**交互測定の1.266倍**を採る。8192²ではGraph化そのものの速度効果は小さく、GPU kernelが主な費用として残る。

1024²・M=128はGraph再実行約0.803 msだが、capture時ピーク47.89 MBがeager dense約40.29 MBを超える。メモリの絶対条件により、この形状ではGraph版を採用しない。Graphの初回capture時間、別shapeの再capture、複数層への拡張は測っていない。これらの結果は`prototypes.probe_csr_graph_step`と`prototypes.probe_dense_graph_step`、JSONは`output/ada-20260928/cache-hypotheses/graph-*.json`。

decode、pack、optimizerの小kernel融合はなお候補だが、Graphでhost launch費用が消えた後に速度が残る分だけを対象にする。

動いたatomと支持域境界を含むCUDAテストを追加し、Adaで対象16件が通過した。Graph対eagerの複数更新照合も上記のとおり通過。CSRは試作モード`materialize_mode="listed_csr"`で明示的に選ぶ。既定backendやdispatchは変更していない。

## 判定

**今回、同期を除いて完全ステップをGraph化する安全な経路は得られたが、dense同等の速度にはまだ届かない。** 主目標の8192²・M=2048では、同一Ada上のGraph交互測定でdenseの1.266倍、ピークは約41%低い。小形状1024²・M=2048では速度約2.05倍、ピークはdense未満。M=128のGraphはメモリ条件で失格。候補リストの107 MB予約領域を縮めてもhost同期を戻さない方法は、引き続き検討する。

## dispatch と CUDA C++ の順序

その後、8192²・M=2048の`listed_csr`完全AdamWステップをGraph再実行でプロファイルした。GPU kernel合計35.09 msのうち、局所W生成`materialize_listed`が7.52 ms（14回）、atom勾配`mapped_backward_atoms_listed`が5.69 ms（8回）、FP16x3 GEMMが8.92 ms（16回）、局所dW GEMMのCUTLASS kernelが5.85 ms（8回）。候補リスト生成は0.73 ms（1回）。1024²・M=2048でも局所W生成0.260 ms、atom勾配0.219 msがkernel時間約1.00 msの約48%を占めた。traceを含むJSONは`output/ada-20260928/cache-hypotheses/profile-graph-csr-{1024,8192}-m2048.json`、計測コードは`prototypes.profile_mapped_training_kernels`。

これに基づく実装順は、**小さな内部実行ポリシーを先に作り、CUDA C++は局所W生成の単独比較から始める**こと。ポリシーはshape、実効入力行数M、dtype、GPU、学習モード、ピークメモリ上限を入力にし、測定済みケースだけ`listed_csr`とGEMM精度、Graph適格性を選ぶ。Graph capture自体は学習ループの責務として分ける。1024²・M=128はGraphがメモリ条件に失敗したためeagerへ、未測定GPU・shapeは保守的な明示経路へ戻す。これはPyTorchのdevice/autograd dispatch key登録とは別の、CST内部の実行ポリシーである。

TritonからCUDA C++へ同じ計算式を移すだけで速くなる証拠はない。2次元site×atom tileとdW融合はTriton試作で遅く、7基底は数値誤差、20項は演算数が障害だった。CUDA C++を試す理由は、station単位の共有メモリ配置、warp間の仕事配分、atomデータ再利用を直接制御し、**局所W生成7.52 msを明確に削れるか**を反証するため。最初は1窓のW照合、時間、register/shared-memory使用量、ピーク増分を比べ、優位がなければ全経路の移植には進まない。全W・全dWを作らない条件は維持する。
