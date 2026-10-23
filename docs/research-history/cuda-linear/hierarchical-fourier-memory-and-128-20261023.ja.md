# メモリ保持の監査と128からの試作

追記：[128の実機実行・保持と受け渡しの確認結果](hierarchical-fourier-128-execution-20261023.ja.md)。
以下はGPU実行前に行ったshape/容量監査の記録。

2026-10-23。ユーザーの「局所Hを保存しないのにメモリが増える理由」と、
128程度から段階的に試作する方針に対応する。runtime/kernelは変更していない。
以下の配列容量は現行executorとCPU上のshape確認によるpayload bytesであり、
allocatorの丸め・sort workspace・Graph poolを含むGPUピークの実測内訳ではない。

## 1024実験で増えたもの

[既存の実測](hierarchical-fourier-local-overlap-20261023.ja.md)は
N1024/B32/A1024/M8。directのpeak allocatedは14,638,080 bytes、
局所prototypeは31,042,048 bytesだった。
局所経路を使わない同じshared/chunk/recompute設定のlocal-controlも
31,042,048 bytesである。したがって、この比較の増加を
「2 streamでYを二重保持したため」と説明することはできない。

最大の準備bufferはNativeで、shapeは `[chunk_atoms,2,3,cache_capacity]`、FP64。
2は出力/入力軸、3はprofile値・中心微分・幅微分である。
実支持長で圧縮せず固定容量を確保している。

\[
M_{Native}=A_{chunk}\times2\times3\times S_{cache}\times8\quad\text{bytes}.
\]

| buffer | direct | 局所prototype | 用途 |
| --- | ---: | ---: | --- |
| Native、一組 | 6,291,456 | 25,165,824 | chunk256→1024、cache512。準備時に値と微分を保存 |
| Meta/Co/Ci/Norm/Beta/Bounds、一組の合計 | 475,136 | 1,900,544 | geometry、係数、正規化、変換係数、支持境界 |
| H、一chunkの確保容量 | 98,304 | 393,216 | `[B,chunk_atoms,3]` FP32。局所slotをmaskしても容量は残る |
| ZまたはC、一個 | 0 | 504,832 | `[B,232,17]` FP32。bank/tileごとの縮約・出力係数 |
| 静的table、全Tensorの合計 | 3,120 | 1,011,592 | projection、lookup、Fourier basis等 |
| XまたはY、一個 | 131,072 | 131,072 | `[32,1024]` FP32。局所用の別Yは作らない |

容量はbytes。directの準備はchunk256で走査する。ただしPythonで次のprepを
作ってから前の変数を置き換えるため、chunk境界で準備bufferが重なる可能性もある。
表は「一組」の容量であり、directの同時保持量の上限とは扱わない。
Native一組の増分18,874,368 bytesとGPUピーク増分16,403,968 bytesは異なる。
生存期間・保存H・Graph allocatorの再利用条件が異なるため、表の差分を足して
実測ピークの厳密な分解だとは主張しない。peakがどのphaseで発生したかの実測は未実施。

「局所Hを保存しない」は、現行forwardの局所slotを計算/storeしないことを指す。
H配列の確保自体をなくしたという意味ではなかった。この区別を明確にする。
directのreuse=hは各chunkのHと微分をbackwardまで保存し、合計393,216 bytes。
prototypeのreuse=noneはそれを保存しないが、backwardで全atomのH/Gを
再計算する。局所backward融合も、非局所atomだけのH容量への圧縮も未実装である。

## 何をどこに保持し、いつ渡すか

1. Parameter PとXをglobal memoryで受け取る。W全体はこのCST経路では実体化しない。
   routing/demand、全atomのgeometry・係数・norm・Nativeをglobal memoryに準備する。
   この共通準備後に局所/階層の2経路を開始する。
2. 局所の所属情報はLocal `[A]` int32、sorted keys `[2A]` int32、
   order `[2A]` int64、begin `[ceil(N/32)+1]` int64。
   N=A=1024ではpayload合計28,936 bytes。sortの一時領域は別途必要。
   Meta由来のbank順order/beginも別に存在する。
3. 局所blockは所属リストとX/Nativeをglobalから読み、局所縮約と出力和を
   register/shared等のon-chip領域で作る。block終了時にこのon-chip状態は残らない。
   全出力tileへ直接Yをstoreする。空のリストもゼロを書いて初期化する。
4. 階層経路はglobal Z→Hを作る。局所slotのH storeは省くが、
   `[B,A,3]` の全容量を確保する。Z/H生成中も共通のpreparedは保持する。
5. 元streamが局所streamの完了を待つ。その後global Cを確保し、非局所残差をYへ加算、
   Cを集約して同じYへ展開する。register/sharedをblock間で渡すのではなく、
   global Yを途中結果の受け渡しに使う。別のY_local/Y_hierarchyはない。
6. forward終了時、autogradはX/P/demand/routesを保存する。X/Pは参照でありコピーではない。
   prepared/局所リスト/Z/H/Cはこの経路ではbackward保存対象ではない。
   ただしGraph poolやallocatorのreserved領域がその分だけ減るとは限らない。
7. backwardではZx/Zg、dx/dP、Cを用意し、preparedを再生成し、全atomのH/Gをglobalに
   作って勾配を計算する。optimizerのParameter勾配・AdamW momentsも完全step測定に含む。

ここでglobalは物理的なdevice tensor storageを指す。L2へのcache hit/residencyを
保証する配置指定ではない。register/sharedはcompilerの実配置にも依存する。

## 次は128から確認する

最初の条件はX `[32,128]`、論理W `[128,128]`、Y `[32,128]`、atom128、M8。
batch32・profile mixed・seed41・AdamW等は維持する。
全比較planのatom_chunk上限を256にそろえる。現行宣言の最小値が256だが、
実際の準備shapeはlen(p)=128なので、256 atom分をpadding確保するわけではない。
cache_capacityも全plan128にそろえ、512 slotを維持したままにしない。
これは1024測定の単なる縮尺換算ではなく、独立した小規模検証条件である。

| 一個/一組のpayload bytes | 128 direct | 128 shared/local |
| --- | ---: | ---: |
| Native `[128,2,3,128]` FP64 | 786,432 | 786,432 |
| H容量 `[32,128,3]` FP32 | 49,152 | 49,152 |
| ZまたはC `[32,28,17]` FP32 | 0 | 60,928 |
| 静的table全体 | 3,120 | 129,504 |
| XまたはY `[32,128]` FP32 | 16,384 | 16,384 |

128では候補12 banks/28 nodesが存在する。実際のrouting/demand選択は
live geometryに依存するため、候補数を実稼働数とは呼ばない。
この条件も局所global H容量の削減を実装したものではない。

比較はdirect、局所機能なしcontrol、局所serial8、局所overlap8、Dense。
順番は所属分類/読み書き/生存期間の確認→独立Y/dX/全atom勾配→
完全stepとcapture/replayメモリの測定とする。その後256へ進め、
1024でのSM利用や高速化を128の結果から推定しない。
小規模では仕事量が少なく、launch/fork/join費用が支配し得る。
最終的な多段分類は維持し、初回の局所/非局所という実装制約を最終設計にはしない。

Plan: `benchmarks/cuda/linear/plans-hierarchical-fourier-local-128.json`。
Case: `benchmarks/cuda/linear/cases/local-fourier-m8-mixed-128-a128.json`。

```bash
PYTHONPATH=src:. python -m tools.kernel_dev check \
  --plans benchmarks/cuda/linear/plans-hierarchical-fourier-local-128.json \
  --case benchmarks/cuda/linear/cases/local-fourier-m8-mixed-128-a128.json
```

宣言/JSON往復検査PASS。CPU上のtable形状・payload確認結果はignored
`output/hierarchical-fourier-local-20261023/memory-audit-cpu.json`、再現scriptは同名`.py`。
関連するcontract/localテストは40 passed / 5 skipped。GPUを要するskipは検証済みに含めない。
この監査時点では128条件のCUDA correctness、速度、peak allocated/reservedは未測定で、
新しいGPU jobは投入していなかった。後続の実機検証は冒頭リンクを参照。
