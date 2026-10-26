# atom軸の分担とchunk内の位相共有

2026-10-26。[診断コピーの比較](cell-mass-deep-analysis.ja.md)を元に、
Y/dXのatom走査を分担する実行設定と位相係数の共有を本体へ実装する。
数学のprofile・周期・tile窓・面積正規化・double内部演算は変更しない。

## 入口と分担

```python
y = physical_linear(
    spec, x, atoms, z_method="merge", sites_per_block=16, atom_splits=4
)
```

`sites_per_block`は出力siteの担当幅、`atom_splits`はその範囲へ寄与するatom走査の分担数。
どちらも実行上の設定であり、`CellMassAxisSpec`の宣言や係数の数学へ入れない。
既定は従来通り32site、1split。atom_splitsは1/2/4/8を明示指定でき、
空のproducerを作らないよう実際の分担数をchunk数以下へ制限する。
1splitでは部分出力bufferと合計kernelを作らず、直接Y/dXへ書く。

128site・819atom・batch4・32atom/chunkなら26chunk。
16site/block・4splitでは次の32blockを起動する。

```text
grid.x = 8個のsite範囲
grid.y = 1個のbatch範囲
grid.z = 4個のatom範囲

split0: sorted slot [  0,192)  6chunk
split1: sorted slot [192,416)  7chunk
split2: sorted slot [416,608)  6chunk
split3: sorted slot [608,819)  7chunk相当、最後は19atom
```

H/Gは担当範囲の全入力を集計済みの値を読む。
H/Gの生成・保存とParameter勾配の「各threadが別のatomを担当する」規則は維持する。
atomを複数threadで分割する変更ではなく、Cを作るときのatom集合をblock間で分担する。

atom集合を互いに素なA_sへ分割し、選択窓にtile tを含むatomをA(t)とする。
出力側係数uとセル積分基底Phiを使うと、

\[
C^{(s)}_{b,t,m}=\sum_{a\in A_s\cap A(t)} A_aH_{b,a}u_{a,t,m},\qquad
Y_{b,i}=\sum_s\sum_m C^{(s)}_{b,t(i),m}\Phi^{out}_{m,i}.
\]

線形な和なので、Cをsplit間で合計してから展開する代わりに、各splitで展開した出力を
合計できる。Hは入力全体の集計を完了した値であり、Hを途中の部分値へ置き換えていない。
dXもHをG、出力側profileを入力側へ交換した同じ分担で計算する。

## CUDAの関数とメモリ

`retained_tile_offset`は、atomの選択窓の先頭から対象tileまでの整数offsetを返す。
chart端を跨ぐ場合も同じmodulo規則で処理する。
`block_uses_atom`は、担当profile tileのいずれかが既存のtile窓へ入るかを判定する。
入らないatomにはprofile・harmonicの重い準備を行わない。
支持境界でのセルclipや、新しい近似の切り捨て判定は追加しない。

`cache_expansion_coefficients`は、1threadが1atomの現在chunkの係数を準備する。
担当profile tileが一つなら、harmonicごとにsincosを一度計算し、
位相を掛けたsin/cos両成分をsharedへ置く。全batchが同じ係数を使う。
複数profile tileなら、chart imageが異なる場合に備えてraw harmonicを同じ領域へ置き、
各tileの位相は消費時に計算する。二つの表を同時保持しない。

```cuda
// expand_tiles_kernel: 1block = site範囲 × batch範囲 × atom範囲
for (chunk : assigned_atom_chunks) {
    if (thread_id < chunk.atom_count) {
        read_original_atom_through_sorted_rows();
        if (block_uses_atom()) cache_expansion_coefficients();
    }
    load_H_or_G_chunk_to_shared();
    __syncthreads();
    accumulate_owned_C_in_registers();
    __syncthreads(); // chunkを上書きできる
}
publish_C_to_shared();
__syncthreads();
expand_C_to_partial_sites();

// sum_split_outputs_kernel: 同じstream上の次kernel
// 各threadが一つの(batch,site)を確定し、atomicを使わない。
output[batch,site] = sum_over_splits(partial[split,batch,site]);
```

| 場所 | 保持するもの | 寿命 |
| --- | --- | --- |
| 元のglobal | atom唯一のParameter配列、rows、Zx/Zy、H/G、X/dY | 従来のautograd契約通り |
| register | 自分が担当するCの積算 | 一つのexpand block |
| shared | 現在32atomのmetadata・H/G・係数と、展開用C | 一つのexpand block |
| 追加global | partial[split,batch,site] | YまたはdXの生成呼び出し中だけ |

4split・batch4・128site・FP32のpartialは8,192byte。
Y用はforwardの呼び出し後、dX用はbackwardの呼び出し後に解放でき、
ctxには保存しない。partialは入力dtypeで書き、合計はdoubleで積算する。
分担によって加算順と丸めは変わるので、独立参照で従来の許容誤差を検査する。
全atomのコピー、global係数表、全C表は追加しない。

sharedの静的宣言は最大14,336byte/block。
診断版の18,464byteから、同時に存在したraw/phase表と効果のなかったH paddingを除いた。
GPU binaryの表示はREG92/thread・SHARED14,336byte/block・STACK40・LOCAL0。
診断版のREG104/threadからも減った。実際のピークは下記の完全経路で測る。

## 検証と計測

CPU全体は2006 passed / 2888 skipped / 既存JIT警告18件、42.52秒。
Ruff・clang-format・差分検査を通過。wheel/sdistをbuildし、
321 runtime source filesが両packageとbyte一致した。
比較元は`d61fd6d3`のruntime、変更前のsites_per_block=16。
同一jobで入力・profile・精度を固定し、完全経路をfresh processで比較する。
既存CUDA全suiteに、1/4split、複数profile tile、端数chunk、mode16、
batch末尾・batch0、atom0、単一site、設定拒否の検査を追加した。
測定はJIT/buildと独立参照の検証を分離し、prepare・入力検査・partial合計を含める。
公開dispatcher/optimizer/Graph/dense比較はこの数値研究入口の対象外。

source commit `6cc20cfd`。NVIDIA L4、58SM、Torch2.11.0+cu130、CUDA13.0。
対象CUDA・数式suiteは140 passed / 0 failed / 0 skipped。
128site・819atom・batch4のFP32/FP64で、比較元と全設定のY/dX/全atom勾配を
独立Torch参照へ照合した。FP32の最大絶対誤差はY5.96e-8、dX5.96e-8、勾配5.96e-8。
FP64はY1.96e-15、dX2.33e-15、勾配1.50e-15。
変更前との差はFP32 Y5.96e-8/dX2.98e-8、FP64 Y6.66e-16/dX6.11e-16で、
全atom勾配の差は全設定で0だった。許容誤差を緩和していない。

同一jobで変更前と本体実装を測定した。N128、A819、B4、FP32、mode8、seed120、
sigma0.2〜12、入力profile tile16/出力tile32、周期48/96、merge Z・1要素4thread。
各設定2つのfresh process、2巡目は逆順、各process warmup5回後15サンプル。
下表はprepareと検査、必要なpartialの合計も含む同期wall時間。

| 本体・設定 | process1中央値 | process2中央値 | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 変更前、16site | 4.635 ms | 4.627 ms | 85,504 byte | 2 MiB |
| 位相共有・early、既定32site/1split | 4.237 ms | 4.245 ms | 同上 | 同上 |
| 位相共有・early、16site/1split | 3.343 ms | 3.374 ms | 同上 | 同上 |
| 16site/4split | 2.335 ms | 2.296 ms | 同上 | 同上 |
| 16site/8split | 2.156 ms | 2.167 ms | 88,576 byte | 2 MiB |
| 32site/4split | 2.576 ms | 2.563 ms | 85,504 byte | 2 MiB |

4splitは約49.6%/50.4%短縮し、診断コピーの改善を本体でも確認できた。
8splitはこのcaseでさらに短いが、allocatedが3KiB増える。
1回のpartialは4splitで8KiB、8splitで16KiBだが、他bufferとの寿命が異なるため、
全体ピークの差はscratchサイズの差と一致しない。
メモリ据え置きなら16site/4split、追加3KiBを許容するこのcaseの最短は16site/8split。
一般的な自動dispatchは追加せず、既定32site/1splitは維持する。

別のGPU eventによる各stage中央値。和でwall時間を置き換えない。

| stage | 変更前16site | 本体16site/4split | 本体16site/8split |
| --- | ---: | ---: | ---: |
| prepare | 0.385 ms | 0.378 ms | 0.364 ms |
| Zx | 0.0358 ms | 0.0348 ms | 0.0358 ms |
| Zy | 0.0348 ms | 0.0348 ms | 0.0348 ms |
| H | 0.195 ms | 0.196 ms | 0.195 ms |
| Y（合計kernelを含む） | 1.608 ms | 0.319 ms | 0.217 ms |
| G | 0.222 ms | 0.214 ms | 0.214 ms |
| dX（合計kernelを含む） | 1.262 ms | 0.215 ms | 0.186 ms |
| Parameter勾配 | 0.625 ms | 0.620 ms | 0.620 ms |

次に残る大きいstageはParameter勾配。現行は128thread/blockで7blockしか起動していない。
このkernelのNCU観測はREG166/thread、SHARED0、spilling requests0、
DRAM throughput0.04%、Compute (SM) throughput6.33%。
short scoreboard待ちは平均16.8cycle、instruction発行間隔22.3cycleの75.3%で、
wall時間の割合ではない。sharedの宣言がないためshared bank conflictとは説明しない。
counterだけで特定の数学命令へ帰属させず、block分担を変える診断を次に行う。
profiler下の1.55msは通常の約0.62msと区別し、速度比較へ含めない。

## 証拠

job `l4job-2546ab1a8bb6495381f7414b10435a5b`。

- source archive SHA256: `cd34c2d5a697b768acb05da2b589bed7f271192cce06425acd1334c99de301fa`
- result archive SHA256: `a56225d8d76d9ffb2a4906d1c48bea2143864482d5c205b6165e459091e14d7b`
- driver SHA256: `1b5653a44076359102a45da9080ac7ae3fc0bd41a8203838e002715abb78713a`
- 比較元binary SHA256: `4ac447003a0e8e1e2b18de66c09d3762295f8984c8ff1ac687630d4188fec22a`
- 本体binary SHA256: `27a25437c54173ab1c9dd7b9b010e080ce57830166ed4ecb04af3233aec8926d`

両archive hash、29result manifest、driverに埋めた変更前sourceとGPU回収物、
321 runtime source filesと現行・wheel/sdistのbyte一致を照合した。
native binaryのhashは各測定processで同じことを確認した。binary自体のコピーは保存していない。
source・driver・CPU/build log・package照合・NCU・JSON・receipt/specはignored
`output/cell-mass-atom-splits-20261026/`、原本はhost共通poolのjob directoryへ保持する。
このjobの回収と所有slot停止を確認した。続くthread分担の診断は別jobとして記録する。

再実行はsource archiveを新しいdirectoryへ展開し、そこに含まれる`__pool_driver__.py`を
共有poolへ提出する。driverが変更前のexecutor/CUDA/bindingsを独立extensionへ復元する。

```bash
python3 ~/.codex/skills/colab-l4-pool/scripts/pool.py submit \
  --source /absolute/extracted-snapshot \
  --script /absolute/extracted-snapshot/__pool_driver__.py \
  --timeout 720 --label cell-mass-atom-splits-repeat
```

## H/G・Parameter勾配のblock分担

H/GとParameter勾配にはblock内reductionやbarrierがなく、
`slot = blockIdx.x * blockDim.x + threadIdx.x`で各threadが別々のatomを担当する。
そのため、Z・Y/dXの128thread/blockとは別にblock幅を変えられる。

まず`6cc20cfd`のCUDAコピーへ診断設定を加え、同じ16site/4splitで比較した。
H/Gと勾配を128thread/blockのままにすると各7block、32threadなら26block、
16threadなら52blockになる。16threadでもCUDAのwarpは32laneで、半分のlaneを使う。
一つのatomを複数threadへ分解する変更ではない。

| H/Gのthread/block | 勾配のthread/block | process1中央値 | process2中央値 |
| ---: | ---: | ---: | ---: |
| 本体128 | 本体128 | 2.300 ms | 2.298 ms |
| 診断control128 | 診断control128 | 2.318 ms | 2.291 ms |
| 128 | 32 | 2.157 ms | 2.154 ms |
| 64 | 64 | 2.119 ms | 2.122 ms |
| 32 | 32 | 2.094 ms | 2.126 ms |
| 16 | 16 | 2.002 ms | 2.023 ms |

全てpeak allocated85,504byte/reserved2MiB。同じphysical forward/backwardの境界、
2つのfresh process、逆順の2巡、warmup5回後11サンプルで測定した。
本体128との比較で16threadは約13.0%/12.0%短縮した。
stage別ではH0.198→0.164ms、G0.214→0.168ms、勾配0.620→0.411ms。
SMへ分散するblock数に比例した速度向上にはならず、各threadの演算依存も残る。

6候補×4shape×FP32/64の48case、Y/dX/全atom勾配144比較で本体との差は全て0。
独立Torch参照への最大絶対誤差はFP32でY1.19e-7/dX5.96e-8/勾配2.38e-7、
FP64でY3.39e-15/dX3.11e-15/勾配9.88e-15。
32thread設定では既存の対象140件も全て通過した。
診断用Axisには追加設定があり、本体への反映・最終検証は別に行う。

本体には数学のAxisと分離した`atom_threads_per_block=16/32/64/128`を追加する。
forwardのH、backwardのG・Parameter勾配で同じ設定を使用し、ctxへ保存する。
既定は互換性のため128。Z・Y/dXは従来通り128thread/blockを使う。
この設定で値・メモリ配置・atomの所有を変更しない。

```python
y = physical_linear(
    spec, x, atoms, z_method="merge",
    sites_per_block=16, atom_splits=4, atom_threads_per_block=16,
)
```

thread診断job `l4job-9892464dc2ba4b03995f9c8d3115aab8`。
source SHA256 `3e20a4e428125a083372655b8af04cb309e67b4dcb3c341552279fce1082cb9c`、
result SHA256 `73c49e9c876e0d0f0717b27649fa90eb5584c44db34a07b7b0c67512522f6f15`、
diagnostic CUDA SHA256 `eaf85de9e442305ef651e04bf13f3b9ae889e4a04f70b529b0aadf34adc8fc0c`。
両archive hash、22result manifest、埋め込みCUDAと回収物、
321 runtime source filesと`6cc20cfd`のbyte一致を照合した。
同じignored evidence directoryの`thread-diagnosis/`へ保持した。
jobの回収と所有slot停止を確認した。

本体のatom_threads_per_block追加後はCPU全体2006 passed / 2929 skipped、
既存警告18件（44.23秒）。wheel/sdistの321 runtime source filesもbyte一致した。
GPUでの最終検証は以下へ追記する。
