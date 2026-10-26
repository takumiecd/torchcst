# 区間確率方式：prepareからbackwardまでのkernelの流れ

2026-10-26。[設計と数学契約](cell-integrated-profile.ja.md)に対応する、検討用の擬似コード。
関数名・引数は仮名であり、CUDA実装ではない。profile族、係数とその微分、支持境界の
数式が未確定なので、以下の補助関数を実装済みとは扱わない。

最初はH/GとZx/Zyをglobalへ保存し、Parameter勾配まで再利用する構成で考える。
この保存量を基準に、HやZxの再計算・局所融合を後から比較する。
入力・出力128サイト、atom数128、batch 4から始める。thread数などは後で変更できる。

## 前準備と順序

```cpp
// forward
prepare_layout(atoms, spec, layout);
packed_atoms = layout.packed_atoms;
build_input_hierarchy(X, layout, Zx);
compute_h(Zx, X, packed_atoms, layout, H);
compute_y(H, packed_atoms, layout, Y);

// backward
build_output_hierarchy(dY, layout, Zy);
compute_g(Zy, dY, packed_atoms, layout, G);
compute_dx(G, packed_atoms, layout, dX);
compute_parameter_grad(Zx, Zy, X, dY, H, G, packed_atoms, layout, grad);
```

同じstreamへ依存順に投入する。階層構築は主要な5つの計算kernelに含めない。
階層構築は「signalから直接作る要素」と「子Zから継承する要素」に分ける。
前者はlevelをまたいで並列に作り、後者だけlevel順に起動して子の完成を待つ。
CPU側で毎回同期するという意味ではない。
以下ではgeometry・正規化設定などの引数を省略し、補助関数が同じspecを参照するものとする。
compute系のatom番号はソート済みの行番号であり、H/Gもこの順序で保存する。
元のParameter行番号へ戻すのはParameter勾配の保存時である。

## 起動するものとkernel内で呼ぶものの区別

| 名前 | 種類 | 役割 |
| --- | --- | --- |
| `prepare_layout` | host側の起動関数 | 以下の分類・ソート・配置・範囲作成を順に起動 |
| `classify_atom_keys` | CUDA kernel | 元のatom行から階層とソートキーを作る |
| `device_sort_pairs` | GPUライブラリ呼び出し | キーと元の行番号を一緒にソート |
| `pack_sorted_atoms` | CUDA kernel | ソート順の実行用atomと元の行との対応を作る |
| `mark_group_heads` | CUDA kernel | 同じキーの連続区間の先頭を検出 |
| `device_inclusive_scan` | GPUライブラリ呼び出し | 先頭フラグの累積和からグループ番号を作る |
| `write_group_begins` | CUDA kernel | 各グループのキー・beginとグループ数を書く |
| `write_group_ends` | CUDA kernel | 次のbeginから各グループのendを書く |
| `build_input_hierarchy` | host側の起動関数 | Xからの直接生成と、level順のマージを起動 |
| `build_output_hierarchy` | host側の起動関数 | dYからの直接生成と、level順のマージを起動 |
| `build_z_direct` | CUDA kernel | 全levelの直接生成要素を並列に作る |
| `merge_z_level` | CUDA kernel | 指定levelの継承要素を子Zから並列に作る |
| `compute_h` / `compute_y` | CUDA kernel | 以下の第1・第2節 |
| `compute_g` / `compute_dx` / `compute_parameter_grad` | CUDA kernel | 以下の第3〜第5節 |

各kernel節のコードブロックはkernelの本体を示す。
`assigned_atom`や`contract_input_tile`などはkernel内のdevice補助関数で、追加のkernel起動ではない。
sort/scanの内部で使うkernel数はGPUライブラリに任せる。ライブラリ内部まで自作することは前提にしない。
host側でshapeから出力・scratch容量を確保し、同じstreamを全起動・ライブラリ呼び出しへ渡す。
以下の配列は明記しない限りglobalにある。

## prepare_layout：分類・並び替え・範囲の準備

目的はatomの意味を変えることではなく、実行用の順序と検索情報を作ること。
atom数A=128の例では、次の作業配列を最大A要素分用意できる。
sort/scan用scratchはライブラリが要求する容量を別途数える。

| 配列 | 内容 |
| --- | --- |
| `keys[A]`, `source_rows[A]` | ソート前のキーと元のParameter行番号 |
| `sorted_keys[A]`, `sorted_rows[A]` | ソート結果 |
| `packed_atoms[A]` | ソート順の実行用atomコピー |
| `head_flags[A]`, `group_numbers[A]` | グループ先頭とその累積和 |
| `group_key[A]`, `group_begin[A]`, `group_end[A]` | 有効なグループだけを先頭から格納 |
| `group_count[1]` | 有効グループ数。GPU上で後続kernelが読む |

ここでpacked_atoms、sorted_rows、group関連の出力はlayoutが参照するbufferであり、
擬似コード中の短い配列名はその別名とする。head_flagsやscan結果はprepareだけのscratch。

```cpp
// host側。同じstreamへ順に起動する。
classify_atom_keys<<<...>>>(atoms, spec, keys, source_rows);
device_sort_pairs(keys, source_rows, sorted_keys, sorted_rows);
pack_sorted_atoms<<<...>>>(atoms, sorted_rows, layout.packed_atoms);
mark_group_heads<<<...>>>(sorted_keys, head_flags);
device_inclusive_scan(head_flags, group_numbers);
write_group_begins<<<...>>>(sorted_keys, head_flags, group_numbers, layout);
write_group_ends<<<...>>>(layout);
```

A=0ならhost側でprepareの空経路を選び、group_count=0をstream順に設定する。
以下のprepare kernelはA>0を対象とする。空のatom集合でも後段はY/dXへ必要な0を書く。
その場合はH/G/Parameter勾配のatom用kernelを起動せず、Y/dXの所有blockで0を保存する。
keysなどのscratchは最後の利用が済んだ後に再利用できるが、packed_atoms、sorted_rows、
グループ範囲などbackwardで使う状態はforward呼び出しごとに保持する。

### classify_atom_keys：各atomの所属とキーを作る

各threadが元の一つのatomを読む。ここではprofile値やHを計算しない。

```cpp
a = blockIdx.x * blockDim.x + threadIdx.x;
if (a >= atom_count) return;

level = classify_hierarchy(atoms[a], hierarchy_spec);
tile_shape = hierarchy_tile_shape(level);
q_out = center_tile_coordinates(atoms[a].output_center, tile_shape.output);
q_in  = center_tile_coordinates(atoms[a].input_center,  tile_shape.input);

keys[a] = make_lexicographic_key(level, q_out, q_in);
source_rows[a] = a;
```

キーは`(level, q_out[0], ..., q_out[Dout-1], q_in[0], ..., q_in[Din-1])`の辞書順。
各軸のタイル座標は原点・spacingを合わせて`floor((center-origin)/physical_tile_width)`で求める。
周期軸の中心はchartが定める基本領域へ正規化してから求める。
多次元のキーの符号化はこの辞書順を保ち、packed整数のbit数超過を黙って切り捨てない。
同じキーのatomの順序は演算を変えない。元の行番号は対応付け用で、必須の数学的な分類軸ではない。

階層分類に使う幅・周期L・タイル寸法の関係はspec/実行設定で決める。
分類式そのものは未確定であり、sigmaだけで非対応のprofileを勝手に既存階層へ入れない。
中心の所属キーと、支持が重なる全タイルの集合は別である。

### device_sort_pairs：キーと元の行番号を一緒に並べ替える

```cpp
// GPU sortライブラリへ渡す意味。内部kernelはライブラリに任せる。
(sorted_keys, sorted_rows) = sort_by_key(keys, source_rows);
```

atomのParameter本体やoptimizerのmomentsを並べ替える操作ではない。
多wordキーを使う場合も辞書順を維持できるsort経路を選ぶ。

### pack_sorted_atoms：計算用の連続配列を作る

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

original_row = sorted_rows[s];
packed_atoms[s] = atoms[original_row];
```

一つのthreadが一つのソート済み行を書き、sorted_rowsをbackwardの勾配の戻し先に使う。
成分ごとの実配列をSoAにするかなどは後で決める。この代入はatom一つの全情報をコピーする意味。

### mark_group_heads：同じキーの区間の先頭に印を付ける

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

if (s == 0)
    head_flags[s] = 1;
else
    head_flags[s] = (sorted_keys[s] != sorted_keys[s - 1]);
```

その後にinclusive scanを行う。例えば`head_flags=[1,0,1,0]`なら
`group_numbers=[1,1,2,2]`となり、0始まりのgroup番号は`group_numbers[s]-1`。

```cpp
group_numbers = inclusive_sum(head_flags);  // GPU scanライブラリ呼び出し。
```

### write_group_begins：groupごとのbeginを記録する

```cpp
s = blockIdx.x * blockDim.x + threadIdx.x;
if (s >= atom_count) return;

if (head_flags[s]) {
    g = group_numbers[s] - 1;
    layout.group_key[g] = sorted_keys[s];
    layout.group_begin[g] = s;
}
if (s == atom_count - 1)
    layout.group_count = group_numbers[s];
```

beginとgroup_countの書き手はそれぞれ一意。全beginが完成するのをkernel境界で待つ。

### write_group_ends：次のgroupのbeginからendを決める

```cpp
g = blockIdx.x * blockDim.x + threadIdx.x;
if (g >= layout.group_count) return;

if (g + 1 < layout.group_count)
    layout.group_end[g] = layout.group_begin[g + 1];
else
    layout.group_end[g] = atom_count;
```

`[begin,end)`は同じキーのatomの連続区間。全区間を合わせるとソート済みatomを一度ずつ覆う。
gridは最大Aグループ分で起動でき、group_countをCPUへコピーして起動数を決める必要はない。

### blockが使うrangeはどう取得するか

出力タイルや入力タイルに寄与するatomが、一つの連続区間に全て収まるとは限らない。
各blockは必要なgroupの`[begin,end)`を順に読む。入力側の処理も同じpacked_atomsを使える。
最初の128atomの説明形では、groupを走査するdevice補助関数から始められる。

```cpp
for (g = 0; g < layout.group_count; ++g) {
    if (!group_can_contribute(layout.group_key[g], target_tile, side))
        continue;
    consume_range(layout.group_begin[g], layout.group_end[g]);
}
```

これは追加のkernelではない。`group_can_contribute`は階層ごとの最大支持幅を用いる
保守的な候補判定で、中心が対象タイルの外にあるgroupも必要なら含める。
支持幅の上限を保証できないgroupは、この判定で除外しない。
支持が周期境界をまたいでも、同じgroupの区間を二重に消費しない。
候補に含まれるが実際の寄与が0のatomを含めて処理してよい。範囲を狭めるためのatom別判定は必須にしない。
group検索の索引は後で改善できるが、タイルごとのatomパラメータ全体の複製は前提にしない。

## build_input_hierarchy：XからZxを作る起動関数

以前の`build_z_level`には、signalを読む処理と子Zを読む処理が一つに入っていた。
ここを分ける。二分マージを出発点にする。
nodesはタイルのsite範囲・bank・mode・子ノードを記述する実行設定で、atom値を含むZとは別。
最初は宣言済みのbankとlevelの全ノードを作る単純な構成とし、必要ノードだけへの限定は後で考える。

### 並列な部分と、依存で順序が必要な部分

| 処理 | 並列にできる範囲 | 待つ相手 |
| --- | --- | --- |
| signalからZを直接生成 | 異なるlevel・タイル・batch・mode | 入力XまたはdYが利用可能になることだけ |
| 一つのZへのsiteの加算 | siteを複数threadに分担し、部分和を作る | 部分和を合わせるときの担当thread |
| 子Zから親Zを継承 | 同じlevelの異なる親・batch・mode | その要素で使う子Zの完成 |

**親が子を使う依存は残るが、全levelの全処理が直列という意味ではない。**
直接生成するmodeは、子Zがなくても作れる。最初に全level分をまとめて起動できる。
マージのlevel単位の完了待ちは、必要な子ごとの待機より粗いが、安全で単純な基準になる。

基底設定から、各`(node,mode)`を直接生成か継承のどちらかへ一度だけ割り当てる。
`direct_jobs`は直接生成するmodeをnodeごとに束ねた記述、`merge_jobs[level]`は
継承する要素と子modeの対応を記述する。GPUで空きthreadを監視するjob管理ではなく、固定の担当表である。
同じZ要素を両方のkernelから書かない。これはatomの支持分類とは別の、基底間の対応である。

```cpp
// 並列部分：直接生成する要素は全level分をまとめて処理。
if (has_input_direct_jobs)
    build_z_direct<<<input_direct_grid, threads>>>(X, input_direct_jobs, Zx);

// 依存部分：子の完成後に次のlevelを起動。各起動の内部は並列。
for (level = 1; level < input_level_count; ++level) {
    if (has_input_merge_jobs(level))
        merge_z_level<<<input_merge_grid(level), threads>>>(
            input_merge_jobs[level], Zx);
}
```

forはCPU側から起動順序を記述するもの。タイルをCPUの一つのthreadで計算しているわけではない。
各起動を同じstreamへ投入し、CPU側で毎levelのGPU完了を待つ呼び出しは入れない。
例えば128サイト、最小タイル幅16なら、二分マージのノード数は8→4→2→1。
マージでは4親を並列→2親を並列→1親へ進む。どのmodeを直接生成するかは基底の継承条件による。
これは構造の例で、最小幅16や各levelの周波数・mode数を固定する決定ではない。

## build_output_hierarchy：dYからZyを作る起動関数

同じ処理にdYと出力側のタイル・基底設定を渡す。
入力側と出力側で次元、spacing、タイル寸法、bankが異なっていてよい。

```cpp
if (has_output_direct_jobs)
    build_z_direct<<<output_direct_grid, threads>>>(dY, output_direct_jobs, Zy);

for (level = 1; level < output_level_count; ++level) {
    if (has_output_merge_jobs(level))
        merge_z_level<<<output_merge_grid(level), threads>>>(
            output_merge_jobs[level], Zy);
}
```

Zyはbackwardで新しく作る。Zxを入力にしてZyを作るという意味ではない。

## build_z_direct：site方向も複数threadで並列に足す

以前のコードも、タイル・batch・modeは並列だった。ただし、各Z要素のsite方向の加算は
一つのthreadに任せていた。ここでは一つのZをP個のthreadで分担する案を示す。
これはatomを複数threadで処理する話ではなく、複数atomが再利用するZの生成である。

一つのblockは一つの直接生成nodeとbatch範囲を担当する。
Pは調整可能な2のべき乗で、blockDim.xを割り切る値とする。warp単位に固定しない。
例として128thread、P=4なら、32個のZ要素を同時に処理できる。

| thread | 担当 | 分担するsiteの番号 |
| --- | --- | --- |
| 0〜3 | 一つ目の`(batch,mode)` | 0,4,8,… / 1,5,9,… / 2,6,10,… / 3,7,11,… |
| 4〜7 | 二つ目の`(batch,mode)` | 同じ分割を次のZ要素に適用 |
| … | … | … |
| 124〜127 | 32個目の`(batch,mode)` | 同じ分割を次のZ要素に適用 |

batch 4・直接生成mode 8なら32個のZなので、この例では全128threadに担当がある。
要素が32個より多い場合は、次の波でまた32個を並列に処理する。
下のteamはP個のthreadをまとめるソフトウェア上の番号であり、新しいCUDAの実行単位ではない。

```cpp
job = direct_jobs[blockIdx.x];  // 異なるlevelのnodeも同じgridに含む。
team = threadIdx.x / P;
lane = threadIdx.x % P;
team_count = blockDim.x / P;
work_count = batch_size * job.direct_mode_count;
__shared__ Scalar partials[THREADS];

// 全threadが同じ回数繰り返す。各wave内の全teamは並列。
for (wave = 0; wave < ceil_div(work_count, team_count); ++wave) {
    q = wave * team_count + team;
    (b, mode) = decode_direct_job(q, job, blockIdx.y);
    valid = (q < work_count && b < total_batch_size);

    partial = 0;  // 各threadのregister。
    if (valid) {
        for (i = lane; i < job.site_count; i += P) {
            site = job.site_at(i);
            partial += signal[b, site] * integrated_basis(job.node, mode, site);
        }
    }
    partials[threadIdx.x] = partial;
    __syncthreads();

    // teamごとの部分和を並列reduction。P=4なら2段。
    for (offset = P / 2; offset > 0; offset /= 2) {
        if (lane < offset)
            partials[threadIdx.x] += partials[threadIdx.x + offset];
        __syncthreads();
    }
    if (valid && lane == 0)
        Z[job.node, b, mode] = partials[threadIdx.x];
    __syncthreads();  // 次のwaveでsharedを再利用する前の同期。
}
```

`decode_direct_job`は範囲外qでも不正な配列を読まず、valid=falseとなる番兵を返す。
担当のないthreadもpartial=0を書き、全barrierへ参加する。途中return/continueはしない。
siteのループはthreadごとにsite数/P程度の加算をするものであり、全siteを一人で走査しない。
足し算の依存は各thread内の部分和と、log2(P)段のreductionに残る。

`signal`は入力側ならX、出力側ならdY。signalはglobal、部分和はregister/shared、完成Zはglobal。
shared容量はこの例でTHREADS個のScalar分。128thread・FP32なら512 byteであり、
タイル全体や全levelのZをsharedに置くという意味ではない。
同じsignalをmode間で共有ロードする追加案もあるが、この擬似コードではまだ導入しない。
P=1なら以前と同様に一人で一つのZを作る割り当てになる。短いタイルやmode数が多い場合は
Pを増やさない方がよい可能性もあり、reductionと同期の費用を含めて比較する。

## merge_z_level：子Zの完成後、親の各要素を並列に作る

ここではsiteを走査しない。二つの子の完成値を足すだけなので、一つの親Z要素を
一つのthreadへ割り当てる。他の親・batch・modeは別threadで並列に進む。

```cpp
q = blockIdx.x * blockDim.x + threadIdx.x;
if (q >= merge_jobs.count * total_batch_size) return;
(job, b) = decode_merge_job(q, merge_jobs);

left = Z[job.left_child, b, job.left_mode];
right = job.has_right_child ? Z[job.right_child, b, job.right_mode] : 0;
Z[job.parent, b, job.parent_mode] = left + right;
```

子Zは先に起動したkernelで完成済みなので、ここにblock内barrierは不要。
親要素の書き手も一つ。次のlevelはこのkernelの完了後に読む。
rootが一つになってもbatch/modeは並列にできるが、上位ほど仕事が少なくなる点は残る。
この縮小した並列度や起動費用は、後でlevel融合や複数nodeの同一block処理と比較する。

## セル積分基底と継承条件

`integrated_basis`はsiteの担当セル全体を積分した基底値。例えば共通原点のcos基底なら、

```cpp
if (omega == 0) return spacing;
return 2 * sin(omega * spacing / 2) / omega
       * cos(omega * (site_position - common_origin));
```

sin基底も同じ積分係数にsinを掛ける。多次元の積基底なら軸ごとのセル積分を掛ける。
ここにはatomの中心・幅・振幅を混ぜない。atomごとの支持で切れたセルは後段の補正で扱う。

`can_inherit_same_basis`は、子タイルのsite集合が親を重複なく覆い、該当modeの周波数、
sin/cosの種類、座標原点、spacing、セル積分係数が一致するときだけtrueにする。
葉ノードや一方の子で該当modeがない場合はfalse。原点が異なる場合の位相変換は、この単純な継承には含めない。
親を二つの子から作ることだけでは、基底まで一致するとは限らない。
一致しないmodeはdirect_jobsへ、一致するmodeだけmerge_jobsへ入れる。
全levelの直接生成を先に行っても、継承できる要素までsignalから作り直すわけではない。
必要なZの総要素数も変えない。signalの再読込は残り、処理順によるcacheの効き方や
同時に扱う作業範囲は変わり得る。速さは未測定であり、元のlevel単位の混合kernelも比較候補として残す。

## 1. compute_h：入力側の全寄与をatomごとにまとめる

blockはatom群とbatch範囲を担当する。各threadが別々のatomを担当し、
そのatomに必要な全入力タイルを足す。最初の例では128thread、各threadに1atom、batch_size=4。
Hは振幅を掛ける前の値。Zx/Xはglobalから読み、Hの累積値はregisterへ置く。

```cpp
atom = assigned_atom(blockIdx.x, threadIdx.x);
if (atom >= atom_count) return;

for (b : assigned_batches(blockIdx.y, batch_size)) {
    h = 0;
    for (tile : full_input_tiles(atom)) {
        h += contract_input_tile(atom, tile, b, Zx);
    }
    h += input_boundary_contribution(atom, b, X);
    H[b, atom] = h;
}
```

`full_input_tiles`は支持内部を重複なく覆うタイル集合。親とその子を二重に含めない。
`contract_input_tile`は`sum_k input_coefficient(atom,tile,k) * Zx[b,tile,k]`。
係数には中心・幅と面積による正規化を含め、必要なmodeごとに評価する。
`input_boundary_contribution`は内部タイルに含めなかった残りのセルを、正確な積分で計算する。
この分割の具体式は未確定だが、二つを合わせて全profileの寄与を一度ずつ数える契約にする。
内部タイルがなくても境界側で必要な寄与を処理し、空支持なら0を書く。

この案では異なるthreadのHを足す必要はない。thread自身が必要な入力寄与を全て足す。
sharedへの共同ロードやblock内同期はまだ入れず、完成したHを一度globalへ書く。
後に共同ロードを導入する場合は、担当atomがないthreadも必要な同期へ参加させる。

## 2. compute_y：担当出力タイルのCを作り、Yまで完成させる

blockは出力タイルとbatch範囲を所有する。Yを0から累積し、最後に一度globalへ書く。
一つのY要素に対する全グループの寄与をこのblockが集める。

```cpp
output_tile = assigned_output_tile(blockIdx.x);
init_output_accumulator(y);  // 各threadの担当site/batchは重ならない。

for (bank : contributing_output_banks(output_tile)) {
    build_output_c_together(bank, output_tile, H, shared_c);
    __syncthreads();
    expand_output_c_into_y(bank, output_tile, shared_c, y);
    __syncthreads();  // 全員が読み終えてからshared_cを再利用する。
}
add_output_boundary_contributions(output_tile, H, y);
store_output_tile(output_tile, y, Y);
```

内部部分でのCは`C[b,k] = sum_a alpha[a] * H[b,a] * output_coefficient(a,tile,k)`。
`build_output_c_together`はCの各batch/modeに書き手を一つ決め、対応atomを走査して完成値を
sharedへ書く方式を出発点にする。atomごとの部分Cを並列生成してreductionする案は後で比較する。
係数を作るthread配置と、Yのsiteを展開するthread配置は異なってよい。

bankごとに基底が異なるため、C同士を混ぜず、展開したsite値をyへ足す。
境界補正は内部のCで表した寄与と重複しない残りを加える。
bankのループと二つのbarrierには全threadが参加し、担当要素がないthreadも途中returnしない。
globalのHは読み取り専用、Cはshared、yは各threadのregisterを基本とする。
各threadが持つyの個数は担当タイル寸法とbatch範囲で決まり、容量に応じて変更する。

## 3. compute_g：dY側でHと対称の集約を行う

atom群とbatch範囲を担当し、各threadが別々のatomを処理する。
Zy/dYを読み、振幅を掛ける前のGを完成させてglobalへ書く。

```cpp
atom = assigned_atom(blockIdx.x, threadIdx.x);
if (atom >= atom_count) return;

for (b : assigned_batches(blockIdx.y, batch_size)) {
    g = 0;
    for (tile : full_output_tiles(atom)) {
        g += contract_output_tile(atom, tile, b, Zy);
    }
    g += output_boundary_contribution(atom, b, dY);
    G[b, atom] = g;
}
```

`compute_h`と入出力の役割が逆になる。内部タイルと残りのセルの分割・係数の定義は、
forwardで使う同じprofileに対応させる。G全体のほかにdH全体を保存することは前提にしない。

## 4. compute_dx：担当入力タイルの係数を作り、dXまで完成させる

blockは入力タイルとbatch範囲を所有する。`compute_y`の入力側版として構成する。

```cpp
input_tile = assigned_input_tile(blockIdx.x);
init_input_gradient_accumulator(dx);

for (bank : contributing_input_banks(input_tile)) {
    build_input_c_together(bank, input_tile, G, shared_c);
    __syncthreads();
    expand_input_c_into_dx(bank, input_tile, shared_c, dx);
    __syncthreads();
}
add_input_boundary_contributions(input_tile, G, dx);
store_input_tile(input_tile, dx, dX);
```

内部部分の係数は`C_in[b,k] = sum_a alpha[a] * G[b,a] * input_coefficient(a,tile,k)`。
振幅は一度だけ掛ける。bankごとに展開し、境界補正を加え、担当dXを完成させる。
Gはglobal、係数はshared、dxはregisterを基本とする。barrierの規則は`compute_y`と同じ。
Y/dXの各要素を一つのblockが所有するため、block間の出力加算atomicは不要にできる。

## 5. compute_parameter_grad：H/Gと微分を組み合わせる

各threadが一つのatomの全batchを担当し、そのatomのParameter勾配を完成させる基準案。
batch分割で複数blockが同じ勾配を書く方式は、追加reductionを伴う別候補にする。

```cpp
atom = assigned_atom(blockIdx.x, threadIdx.x);
if (atom >= atom_count) return;

grad_amp = 0;
for (b : all_batches) {
    grad_amp += H[b, atom] * G[b, atom];
}
store_original_amp_grad(atom, layout, grad_amp, grad);

for (parameter : profile_parameters(atom)) {
    grad_parameter = 0;
    for (b : all_batches) {
        dh = input_parameter_derivative(atom, parameter, b, Zx, X);
        dg = output_parameter_derivative(atom, parameter, b, Zy, dY);
        grad_parameter += alpha[atom] *
            (G[b, atom] * dh + H[b, atom] * dg);
    }
    store_original_profile_grad(atom, parameter, layout, grad_parameter, grad);
}
```

ここで`dh/dg`はそれぞれ`partial H / partial parameter`と`partial G / partial parameter`。
逆伝播の中間勾配`dH = alpha * G`とは別である。
入力側だけに効くparameterならdg=0、出力側だけならdh=0、両側に効くなら両方を含める。

微分関数は、内部の`sum partial_coefficient * Z`と残りのセルの正確な微分を合計する。
面積正規化の微分も含める。H/Gだけからこの情報を復元できるとはしない。
Zx/Zyは読み取り専用のglobal、gradの累積値はregisterへ置く。
元のParameter行への対応はlayoutで戻し、optimizerによる更新はここでは行わない。

このコードはparameterごとにZを読む単純な説明形である。同じZを読んだときに複数parameterの
微分を同時に累積すれば読み取りを共有できるが、register使用量との交換になる。
幅のtask VJP・更新則、支持境界・階層遷移の微分は数学契約を確定してから実装する。

## 確認すること

各段階の保存値、配列のshape、書き手、寿命を先に確定し、独立oracleで
Y/dX/全atom勾配を検証してから速度を比較する。上記は実機での正しさ・性能の結果ではない。
H/Gの保存を省く案では、同じforward状態での再計算と、全利用先の終了を明示する。
