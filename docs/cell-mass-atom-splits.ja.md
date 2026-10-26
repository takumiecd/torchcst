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
register数・実際のピーク・速度はGPUで別途確認する。

## 検証と計測

本体実装の検証・比較結果はこの節へ追記する。
CPU全体は2006 passed / 2888 skipped / 既存JIT警告18件、42.52秒。
Ruff・clang-format・差分検査を通過。wheel/sdistをbuildし、
321 runtime source filesが両packageとbyte一致した。
比較元は`d61fd6d3`のruntime、変更前のsites_per_block=16。
同一jobで入力・profile・精度を固定し、完全経路をfresh processで比較する。
既存CUDA全suiteに、1/4split、複数profile tile、端数chunk、mode16、
batch末尾・batch0、atom0、単一site、設定拒否の検査を追加した。
測定はJIT/buildと独立参照の検証を分離し、prepare・入力検査・partial合計を含める。
公開dispatcher/optimizer/Graph/dense比較はこの数値研究入口の対象外。
