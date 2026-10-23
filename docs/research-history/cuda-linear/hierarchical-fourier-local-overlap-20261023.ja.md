# 局所出力block数の制限とZ/H生成の重ね合わせ

2026-10-23。多段分類へ向けた最初の限定prototype。
既存階層Fourierと同じ有限支持・混合profile・離散L2/floor・live Iの数学を保つ。
旧Algorithm/Recipeのserialized schemaは保持し、新規
`research_cuda_hierarchical_fourier_local` / `LocalFourierRecipe`を研究registryへ追加。
L4の独立2 jobを回収・hash照合済み。局所計算とZ/Hの時間的重なりは確認したが、
今回の混在caseでは既存directの完全step時間を上回らず、採用はしない。

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

## 回収した結果

数値実装checkpoint `3bf1fb50`。`c47deac2`はserial32対照のcatalog追加のみで、
runtimeは同一。GPUは両jobともNVIDIA L4、driver 580.82.07、Python 3.13.15、
Torch 2.11.0+cu130、CUDA 13.0、Triton 3.6.0。GPU UUIDは異なる。
N1024/B32/A1024/M8/mixed/seed41、IEEE FP32、初期X/target/Pのhashは全planで一致。
各jobにcorrectness5 worker、性能5 worker、Dense性能1 workerがあり、全22 worker PASS。

| plan | job 1 Graph完全step ms | job 2 Graph完全step ms | allocated MB | reserved MB |
| --- | ---: | ---: | ---: | ---: |
| direct-samples-h | 7.817216 | 7.805952 | 14.638080 | 27.262976 |
| local-control | 7.818240 | 未実施 | 31.042048 | 56.623104 |
| local-serial8 | 8.208384 | 8.271872 | 31.042048 | 56.623104 |
| local-overlap8 | 8.032256 | 7.960576 | 31.042048 | 56.623104 |
| local-serial32 | 未実施 | 7.940096 | 31.042048 | 56.623104 |
| local-overlap32 | 7.902208 | 7.874560 | 31.042048 | 56.623104 |
| Dense | 0.079872 | 0.078848 | 51.382784 | 111.149056 |

MBは10^6 bytes。時間は各独立workerのGraph replay 7 samplesの中央値で、
7 samplesを7独立runと数えない。job 2はplanの実行順を変更し、serial32対照を追加した。
8 blockのoverlapは同数blockのserialに対し2.15%/3.76%短縮。
32 blockのmatched対照はjob 2の1回だけで0.83%短縮。一般的な改善とは断定しない。
overlap32はdirectより1.09%/0.88%遅く、Denseとの速度差も未解決。
全atomを1 chunkで準備するため、prototypeは既存directよりメモリが多い。

初回のforward profiler診断では、局所kernelとZ/H kernelの重なりは
serial8で0、overlap8で567.772 us、overlap32で488.000 usだった。
これは計装された1回のforwardのkernel区間の交差で、SM利用率の実測や主性能値ではない。
同じtraceのlaunch情報は128 threads/block、shared 4096 bytes/block。
register/threadはK8で56、K32で48。traceのestimated occupancyは実測稼働率として使わない。
共通prepare/norm prefixが残り、stream制御だけで全費用が消えるわけではない。

初回GPU検査は6 passed（M4/M8の全勾配とretained、所属リストと幅変更を含む）。
runnerの各correctness workerは独立quadrature/KKT FP64 oracleで3つのlive状態と
変更後Graph replayのY/dX/全atom勾配を照合した。atol=rtol=4e-4を維持。
CPU全体は2259 passed / 5738 skipped / 18 warnings。GPU skipを成功には含めない。
宣言検査とRuffも成功。全82 GPU regressionを実施したという記録ではない。

## job・hash・停止確認

- job 1: `l4job-231fc612c90741e3ba301ac5894c00aa`
  - source archive SHA256: `fe6bf24dd1ac0e104a7694190aa4ee2ac2b2b083438d7f59fa4a683f2ea7201a`
  - result archive SHA256: `f4c16ab44d95afc88aa1c57584ecabe06f950acd35c3b3d7c87f033e9099df52`
- job 2: `l4job-97cbfeb054d44dcba16f79d7d120a84a`
  - source archive SHA256: `fe13d91f25cec541c1c4f9ed305d1c435f291e6bad8fd642e9c4ecec995b3ce4`
  - result archive SHA256: `997bb23fa1604dba9950640d03387eda21142c3ea8828d2ab6e2a456324eddff`

両jobはreturncode0、timed_out=false、全worker PASS。
各archiveをreceiptと照合し、driverの全artifact hashと各workerのsource file hashを照合した。
source file照合は各jobで4466件。rawはhost共通poolの各jobの`results/`にも保持する。
collectorは現在のruntime/benchmark sourceとの一致もassertする。

```bash
python3 output/hierarchical-fourier-local-20261023/collect.py \
  l4job-231fc612c90741e3ba301ac5894c00aa \
  l4job-97cbfeb054d44dcba16f79d7d120a84a
```

collectorが`verified-summary.json`を再生成する。
`one-case.py`と`repeat.py`は既存runnerを呼ぶjob driverであり、新しい性能runnerではない。
終了後のpool全slot stopped、Colab serverのactive sessionsなしを確認済み。
停止状態の原本は同じoutput内の`final-pool-status.json`。
追加jobは投入していない。PR/push/mergeは行わず、研究branchに保持する。
