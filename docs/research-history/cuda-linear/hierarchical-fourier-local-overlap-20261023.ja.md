# 局所出力block数の制限とZ/H生成の重ね合わせ

2026-10-23。多段分類へ向けた最初の限定prototype。
既存階層Fourierと同じ有限支持・混合profile・離散L2/floor・live Iの数学を保つ。
旧Algorithm/Recipeのserialized schemaは保持し、新規
`research_cuda_hierarchical_fourier_local` / `LocalFourierRecipe`を研究registryへ追加。
GPU検証・測定の結果は回収後に追記する。

## forwardの処理

現行routing/demand、係数・正規化・sample cacheの共通準備後に、
I<=local_width_cutoff、両軸のcacheが全profile、各支持<=32 sites、
全成分のbankが負であるatomを局所候補とする。
所属keyを各atom最大2件生成し、出力tile順にsort/searchsortedする。
支持を切らず、幅・中心・profileは量子化しない。
幅閾値は初回実験用で、多段の最終分類規則ではない。

出力32 sites×batch4件を1つの仕事とし、worker数Kを固定する。
worker kは `k,k+K,k+2K,...` 番目の仕事を処理する。
atom4個ずつ同じ出力tileのリストを読み、Xとnorm計算時の全profile cacheから
局所縮約をon-chipで作り、出力をFP64で累積してFP32 Yへ書く。
局所縮約Hのglobal storeはない。空の出力リストもゼロを書き初期化する。

\[
Y^{local}_{bo}=\sum_{a\in L(o)}\frac{A_a}{D_a}u_a(o)
\sum_{i\in S^{in}_a}X_{bi}v_a(i).
\]

serialでは同一stream上に局所出力→Z→Hを並べる。
overlapでは局所出力をside streamへforkし、元streamでZ→Hを処理する。
Hでは局所atomをmaskし、計算とstoreを省く。
元streamがside streamをjoinした後、非局所のC集約・残差加算・Y展開を行う。
Yの途中値を別blockへ受け渡し、register/sharedのblock間保持は仮定しない。
出力への同時書込みはない。fork/joinもGraph capture/replayの対象。

## 初回の制約・未実装

- 全atomを1つのbounded chunk内に制限。初回A=1024。通常密度へはまだ適用しない。
- Graph用の固定形状H scratch容量は残る。局所slotのstore/readは省くが、
  allocation自体を選択数に応じて圧縮する実装ではない。
- backwardは既存の全atomを対象にした再計算経路。局所backward融合は未実装。
- 係数・norm・cache準備は共通prefixで逐次。この準備を隠す案ではない。
- 階層側のCは既存bank走査/中間buffer。出力owner内のC→Y融合は未実装。
- 実際のregister/shared配置はTriton生成物に依存。紙上の配置と実測値を混同しない。
- block数Kによる局所kernelの広がりの制限で、特定SMや帯域を予約しない。
  streamを分けただけで同時実行・高速化は保証されない。

## 比較と検証

独立FP64 oracleのY/dX/全atom勾配、retained backward、live変更後のGraph replayを
既存runnerで照合する。許容誤差はatol=rtol=4e-4。
既存direct、同一chunk/recomputeのcontrol、局所serial8、overlap8/32を比較する。
未計装のforward/backward/AdamW完全stepとGraph込みallocated/reservedピークを測る。
独立のforward profiler診断でlocal出力とZ/Hの時間的重なりを確認する。
profiler計測を主性能値にしたり、kernel時間を足して完全stepの代わりにしない。

Plan: `benchmarks/cuda/linear/plans-hierarchical-fourier-local.json`。
Case: `benchmarks/cuda/linear/cases/local-fourier-m8-mixed-1024-a1024.json`。
raw/driverはignored `output/hierarchical-fourier-local-20261023/`。
