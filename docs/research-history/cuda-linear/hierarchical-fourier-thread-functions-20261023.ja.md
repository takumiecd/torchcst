# thread単位で割り当てる階層Fourier：関数別の擬似コード

2026-10-23。[全体の設計メモ](hierarchical-fourier-block-design-20261023.ja.md)の具体化。
**担当人数は最初からthread単位で指定する。warp単位への丸めはしない。**
以下はCUDA風の設計用擬似コードで、そのままコンパイルできる実装ではない。
この構成のGPU correctness・同期の進行・速度・実メモリ量は未検証。

## 1. 調整する人数を分ける

| 設定 | 意味 |
| --- | --- |
| `block_threads` | block全体の人数。最初は128 |
| `prepare_threads` | 最初にZ生成を担当する人数 |
| `z_group_threads` | 一枚のZを協力して作る集団の人数 |
| `work_group_threads` | H生成からC・出力部分和までを協力する集団の人数 |

配列の長さやループには、次の名前を使う。

```cpp
batch_size        = 4;   // このblockが担当する小バッチ。全体のbatch数とは別
input_site_count  = 128;
output_tile_sites = 32;
fourier_order     = 8;   // これまでMと呼んだFourierの次数
basis_count       = 2 * fourier_order + 1;
```

`batch_size`は仕事表に記録し、各blockが同じ意味で参照する。
全体のbatch数が割り切れないときは有効件数を別に持ち、範囲外を読まない。
この文書の基本例は4件ずつちょうど分割できる場合とする。

タイルはデータの範囲、thread集団はその処理担当、warpはGPUの実行単位である。
これらを一対一に対応させない。各集団を1人にすることも表現できるが、
その場合の部分和容量や実行効率まで良好になるとは仮定しない。

割り当ての例：`prepare_threads=48`、`z_group_threads=16`、`work_group_threads=8`。

```text
threads  0..47  → 16人ずつ3集団で、割り当てたZを作る
threads 48..127 →  8人ずつ10集団で、直接計算などを開始する

各Z作成集団が自分の準備を終えた後：
threads  0..47  →  8人ずつ6集団で、固定された計算担当へ移る
```

これは担当を示す例で、容量・速度の検証を通した設定ではない。
境界がwarpと一致しない場合も許す。32の倍数でない人数も指定でき、
最後の集団が小さければ、その実人数を同期の参加人数に使う。

## 2. kernel前に仕事表を作る：`make_thread_plan()`

共通の係数・正規化・支持分類は既存の数学に従って準備する。
blockごとに全atomを仕分け直すのではなく、その結果から担当表を作る。
以下の`split_threads()`は連続したthread番号範囲を指定人数ずつに分ける操作。

```cpp
make_thread_plan(atom_plan, config) {
    P = config.prepare_threads;
    T = config.block_threads;
    require(0 <= P && P <= T);
    require(config.z_group_threads >= 1);
    require(config.work_group_threads >= 1);

    plan.prepare_groups = split_threads([0, P), config.z_group_threads);

    // 準備の有無が違うthreadを、同じ計算集団へ混ぜない。
    plan.work_groups = concat(
        split_threads([0, P), config.work_group_threads),
        split_threads([P, T), config.work_group_threads));

    z_order = topological_order(必要なZと、その子依存);
    assign_z_jobs(plan.prepare_groups, z_order);  // 各Zの作成担当は一つ
    assign_atom_jobs(plan.work_groups, atom_plan);

    // 同じblock内でatomのHを重複生成しないよう、atom担当を一つにする。
    // 各計算集団では、Z不要の仕事を先にする。
    plan.C_members = group_output_parts_by_basis_level_tile(atom_plan);
    plan.direct_parts = remaining_output_parts(atom_plan);
    plan.shared_layout = allocate_bounded_slots(plan);

    return plan;
}
```

`assign_z_jobs()`は共通の子→親の順序を保ったまま、各固定集団へ仕事を配る。
集団内の実行順と子依存を合わせても循環しない計画にする。
`assign_atom_jobs()`は出力blockに寄与する全atomを拾い、計算集団へ重複なく分ける。

`allocate_bounded_slots()`はCの使用枠を集団別に数え、alignment・H scratch・通知・
barrier・出力部分和まで含める。launch時のshared上限はconfigとshapeから決める。
動的な所属を詰める場合も、その上限内でaddressを割り当てる。
容量を超える設定は計画段階で拒否するか、別の正確な計画へ変更する。
支持や寄与を切り捨てて収めない。

`prepare_threads=0`はZ需要がない場合だけ使える。全員を準備へ割り当てる設定は、
初期の直接計算との重なりがない比較用として表現できる。
準備の有無で分けた後の計算集団数は、空の範囲を0集団として
`ceil(P / work_group_threads) + ceil((T - P) / work_group_threads)`になる。

## 3. 全体の入口

```cpp
forward128(X, prepared_atoms, block_plans, Y) {
    plan = block_plans[blockIdx.x];
    ctx = init_block(X, prepared_atoms, plan);

    while (next_z_tile(ctx, tile)) {
        build_z_tile(ctx, tile);
        finish_z_tile(ctx, tile);
    }

    do_work(ctx);

    sync_finish();
    finish_output(ctx, Y);
}
```

各threadは自分の固定集団に従って進む。Z担当でないthreadの`next_z_tile()`はfalse。
準備担当も、自分のZ作成集団が一覧を終えた後に計算へ入る。
block全体の同期は初期化完了と最終合算前に置き、途中は関係する集団だけで同期する。

## 4. 初期化：`init_block()`

```cpp
init_block(X, prepared_atoms, plan) {
    ctx.tid = threadIdx.x;
    ctx.plan = plan;
    ctx.prepare = find_prepare_group(plan, ctx.tid);  // 無所属もある
    ctx.work = find_work_group(plan, ctx.tid);
    ctx.z_cursor = 0;                 // 同じ作成集団の全員が同じ値で進む
    ctx.shared = bind_shared_views(plan.shared_layout);

    if (ctx.tid == 0) {
        for (group : plan.prepare_groups)
            init(group.barrier, group.thread_count);
        for (group : plan.work_groups)
            init(group.barrier, group.thread_count);
    }

    for (i = ctx.tid; i < batch_size * input_site_count; i += blockDim.x)
        shared.X[i] = X[plan.batch_begin * input_site_count + i];

    for (i : block_strided_indices(plan.C_storage_size, ctx.tid))
        shared.C[i] = 0;
    for (i : block_strided_indices(plan.Ypart_storage_size, ctx.tid))
        shared.Ypart[i] = 0;
    for (tile : block_strided_indices(plan.Z_count, ctx.tid))
        atomic_store_relaxed(shared.Zready[tile], 0);

    __syncthreads();
    return ctx;
}
```

`bind_shared_views()`は、確保済みshared領域のoffsetを解釈する。threadごとの動的mallocではない。
Z・H scratch・H結果は、その値を使う前に全必要要素を書くので、ここでゼロにしない。
既存profileの微分配列などを、このforward用sharedへ無条件にコピーしない。

## 5. 次のZを取る：`next_z_tile()`

```cpp
next_z_tile(ctx, tile) {
    if (!ctx.prepare.exists)
        return false;

    jobs = plan.z_jobs[ctx.prepare.id];
    if (ctx.z_cursor == jobs.size)
        return false;

    tile = jobs[ctx.z_cursor];
    ctx.z_cursor += 1;
    return true;
}
```

集団内の各threadは同じ仕事表と同じcursorを持つ。
各threadが別のtileを取った後に、同じbarrierで同期する設計にはしない。

## 6. 完成を待つ：`wait_z_tiles()`

```cpp
wait_z_tiles(required_tiles) {
    for (tile : required_tiles)
        while (atomic_load_acquire(shared.Zready[tile]) == 0)
            wait_hint();
}
```

`wait_hint()`は待機中の実行を和らげる処理の設計上の名前で、メモリ同期の代わりではない。
通知はblock内scopeのatomicで扱い、全読者がacquire後にZを読む。
この文書では、一度作ったZを上書きしない一回限りのready通知を使う。

## 7. Z一枚の生成：`build_z_tile()`

```cpp
build_z_tile(ctx, tile) {
    team = ctx.prepare;
    wait_z_tiles(tile.required_children);

    // (batch, basis)という出力要素をthreadで分ける。
    for (q = team.rank; q < batch_size * basis_count; q += team.thread_count) {
        batch = q / basis_count;
        basis = q % basis_count;

        // basisは配列index。modeは波の次数。cosとsinを区別する。
        is_cosine = basis <= fourier_order;
        mode = is_cosine ? basis : basis - fourier_order;
        can_use_child_Z = tile.has_compatible_children && (mode % 2 == 0);

        if (can_use_child_Z) {
            child_mode = mode / 2;
            child_basis = is_cosine ? child_mode : fourier_order + child_mode;
            value = shared.Z[tile.left][batch][child_basis]
                  + shared.Z[tile.right][batch][child_basis];
        } else {
            value = 0;
            for (site : tile.input_sites)
                value += shared.X[batch][site] * phi(tile, basis, site);
        }

        shared.Z[tile.id][batch][basis] = value;
    }
}
```

旧名`can_inherit`は「すでに計算した子Zを使えるか」という意味だった。
その内容を隠さないため、条件を`can_use_child_Z`として上に展開した。
`has_compatible_children`は、左右の空間タイルが親を分割し、子側のFourier半周期が
親の半分である正しいbank対応が存在することを示す。子の完成確認は先頭で行う。
最小階層や対応する子bankがない場合はfalseとなり、Xから計算する。

### if側で行うこと：子で計算済みの積和を足す

例えば親が64サイト、左右の子が32サイトずつなら、親の和は二つの区間へ分かれる。

```text
親：[0, 64)  =  左の子：[0, 32)  +  右の子：[32, 64)
```

ただし、**同じphiに対する積和である場合だけ**、子Zをそのまま足せる。
対応する親の半周期をL、子の半周期をL/2とすると、cos基底では

\[
\phi^{\mathrm{parent}}_{2m}(x)
=\cos\!\left(\frac{\pi(2m)x}{L}\right)
=\cos\!\left(\frac{\pi m x}{L/2}\right)
=\phi^{\mathrm{child}}_m(x).
\]

sinでも同じ対応が成り立つ。したがって、親の偶数modeについて

\[
Z^{\mathrm{parent}}_{b,2m}
=\sum_{i\in\mathrm{left}}X_{b,i}\phi^{\mathrm{child}}_m(x_i)
 +\sum_{i\in\mathrm{right}}X_{b,i}\phi^{\mathrm{child}}_m(x_i)
=Z^{\mathrm{left}}_{b,m}+Z^{\mathrm{right}}_{b,m}.
\]

例えば「親のmode 2」は「子のmode 1」と同じ波形なので、
`Z_parent[mode 2] = Z_left[mode 1] + Z_right[mode 1]`で作れる。
親のmode 0も、左右の子のmode 0の和でよい。
親の奇数modeは子の半整数modeに相当し、保存済みの整数modeにないためelseで計算する。

`basis`は`cos(0)..cos(M), sin(1)..sin(M)`という配列のindexで、modeそのものとは限らない。
例えばM=8なら、親のsin mode 2のindexは10、子のsin mode 1のindexは9である。
この対応は現行の
[`demand_kernels.py::moments_demand`](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/demand_kernels.py)と
[`executor.py::tables`](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/executor.py)で確認した。
現行phiは共通の座標xで定義されている。タイルごとの原点へ定義を変える場合は、
位相も合わせる必要があり、そのまま子の和にしてはいけない。

親の全周波数を子の和へ置き換えない。この初期案では出力係数を分担し、
各係数内のsite和は一つのthreadが計算する。site方向も協調縮約する方式は後の候補。

## 8. Zの完成を公開：`finish_z_tile()`

```cpp
finish_z_tile(ctx, tile) {
    group_sync(ctx.prepare);       // 全作成者の書き込み完了

    if (ctx.prepare.rank == 0)
        atomic_store_release(shared.Zready[tile.id], 1);
}
```

`group_sync(group)`はその固定集団の`cuda::barrier::arrive_and_wait()`に相当する。
Zは一つの作成集団だけが書き、集団の代表が全書き込みの後に完成を公開する。
別の計算集団や親Zの作成者は、対応するacquire確認後に読む。

## 9. Hの生成：`compute_h()`

Hは振幅・正規化係数を掛ける前の \(H_{b,a}=\sum_i X_{b,i}v_a(i)\)。
入力支持は成分ごとに、Z経由の部分と直接部分へ重複なく分解済みとする。

```cpp
compute_h(ctx, atom) {
    team = ctx.work;
    wait_z_tiles(atom.input_plan.required_Z);
    h_partial[batch_size] = zeros(); // 自分の担当項だけの、バッチ別部分和

    for (term : strided(atom.input_plan.direct_terms, team.rank,
                        team.thread_count)) {
        for (b = 0; b < batch_size; ++b)
            h_partial[b] += shared.X[b][term.site]
                      * input_component(atom, term.component, term.site);
    }

    for (term : strided(atom.input_plan.Z_terms, team.rank,
                        team.thread_count)) {
        for (b = 0; b < batch_size; ++b)
            h_partial[b] += shared.Z[term.tile][b][term.basis] * term.beta;
    }

    shared.Hpart[ctx.tid] = h_partial;  // そのthread専用のbatch_size個の値
    group_sync(team);

    if (team.rank == 0) {
        h[batch_size] = zeros();
        for (tid : team.members)
            h += shared.Hpart[tid];
        shared.Hslot[team.id] = h;
    }

    group_sync(team);              // H結果の完成
    h = copy(shared.Hslot[team.id]); // 各threadのローカル値へ
    group_sync(team);              // 全員コピー済み。scratchを再利用できる
    return h;
}
```

これが`compute_H()`と`sync_compute_H()`をまとめた説明用の実装である。
任意人数に対応するため、まずshared経由の縮約と共有を示した。
一度に保持するのは集団が今処理するatomのHで、全atomのglobal H配列ではない。
直接経路でも入力縮約という数学上の値は存在するが、独立したglobal Tensorにはしない。

`input_component()`は指定成分だけを評価する。Z経由の寄与が別にある場所で
全混合profileを直接評価して足すと二重加算になるので、直接項の成分情報を保持する。
Hの入力支持を、担当する出力32サイトに切り詰めない。

## 10. 直接部分の出力：`add_direct_parts()`

```cpp
add_direct_parts(ctx, atom, h) {
    team = ctx.work;

    for (q = team.rank; q < batch_size * output_tile_sites; q += team.thread_count) {
        b = q / output_tile_sites;
        j = plan.output_begin + q % output_tile_sites;
        u = 0;

        for (part : atom.output_plan.direct_parts)
            if (part.contains(j))
                u += output_component(atom, part.component, j);

        shared.Ypart[team.id][b][q % output_tile_sites] += atom.alpha * h[b] * u;
    }
}
```

各集団の各出力要素は、集団内の一つのthreadだけが書く。
`atom.alpha`は既存の振幅・全支持の離散L2/floorから準備し、一度だけ掛ける。

## 11. 出力タイルへ集約：`add_c_parts()`

```cpp
add_c_parts(ctx, atom, h) {
    team = ctx.work;

    for (member : atom.output_plan.C_members) {
        slot = plan.C_slot[team.id][member.node];

        for (q = team.rank; q < batch_size * basis_count; q += team.thread_count) {
            b = q / basis_count;
            basis = q % basis_count;
            shared.C[team.id][slot][b][basis] += atom.alpha * h[b]
                * output_beta(atom, member.component, basis);
        }
    }
}
```

Cのnodeは「基底の種類・階層・空間タイル」で区別する。
`C_members`には、そのタイルの全サイトで有効な成分だけを登録する。
一つの成分を同じ出力位置の親・子タイルへ重複登録しない。
Cの同じ要素を複数threadが書かないよう、係数番号の分担を全atomで固定する。

## 12. Cを出力へ展開：`expand_c_parts()`

```cpp
expand_c_parts(ctx) {
    team = ctx.work;

    for (q = team.rank; q < batch_size * output_tile_sites; q += team.thread_count) {
        b = q / output_tile_sites;
        j = plan.output_begin + q % output_tile_sites;
        value = 0;

        for (node : plan.C_nodes[team.id]) {
            if (node.contains(j)) {
                slot = plan.C_slot[team.id][node.id];
                for (basis = 0; basis < basis_count; ++basis)
                    value += shared.C[team.id][slot][b][basis]
                           * phi(node, basis, j);
            }
        }

        shared.Ypart[team.id][b][q % output_tile_sites] += value;
    }
}
```

各階層のCを別々に評価し、出力値として足す。
支持外まで周期的に展開せず、nodeと担当出力blockの共通部分だけを評価する。

## 13. 計算担当の仕事：`do_work()`

```cpp
do_work(ctx) {
    for (atom : plan.atom_jobs[ctx.work.id]) {
        h = compute_h(ctx, atom);
        add_direct_parts(ctx, atom, h);
        add_c_parts(ctx, atom, h);
    }

    group_sync(ctx.work);        // この集団の全C係数が完成
    expand_c_parts(ctx);
}
```

同じ計算集団の全員が同じatom一覧を同じ順序で進む。
仕事が空でも最後の集団同期には全員が参加する。
担当別Cは他集団の完成を待たずに展開でき、最後に出力部分和を合算する。

## 14. 最後の同期・保存：`sync_finish()` / `finish_output()`

```cpp
sync_finish() {
    __syncthreads();             // 全threadが必ず参加する
}

finish_output(ctx, Y) {
    for (q = ctx.tid; q < batch_size * output_tile_sites; q += blockDim.x) {
        b = q / output_tile_sites;
        j = q % output_tile_sites;
        value = 0;

        for (group : plan.work_groups)
            value += shared.Ypart[group.id][b][j];

        Y[plan.batch_begin + b][plan.output_begin + j] = value;
    }
}
```

最終出力の各要素は一つのthreadが保存する。
block間で出力要素の所有範囲を重ねず、global Yへのatomic加算を不要にする。

## 15. メモリと調整時の確認

本書のsharedは次の領域からなる。Cのslot数は集団ごとに異なってよい。

| 領域 | 論理的な大きさ |
| --- | --- |
| 入力X | `batch_size × input_site_count` |
| Z | `必要node数 × batch_size × basis_count` |
| Hpart | `block_threads × batch_size` |
| Hslot | `計算集団数 × batch_size` |
| C | `各計算集団の使用node数の合計 × batch_size × basis_count` |
| Ypart | `計算集団数 × batch_size × output_tile_sites` |
| 通知・同期 | `Zready`、作成集団と計算集団のbarrier、必要なalignment |

thread単位の指定は、threadごとにC全体を複製するという意味ではない。
ただし、この部分和方式では計算集団を増やすほどC・Ypart・同期状態が増えるので、
集団数と使用slot数から実容量を確認する。4 warpsだからCも4組という固定計算はしない。
大量に増える場合の別案は、Cの所有者の分け方や保持するタイル数を変えることであり、
支持を切り詰めることではない。

最初に確認する事項は、threadの重複・抜け、固定集団の参加人数、Z作成順の依存、
支持の被覆、Cと直接部分の二重加算なし、書き手の一意性である。
部分warpを含む経路へfull-warpのshuffle maskを流用しない。
sharedのalignment、barrier/atomicの構築と寿命、対象GPUでの進行も実装時に確認する。

その後、独立参照でZ/H/C/Yを確認し、学習経路を接続した段階でdX・全atom勾配まで検証する。
性能は`prepare_threads`、`z_group_threads`、`work_group_threads`を独立に変え、
準備・所属構築を含む完全stepとallocated/reservedピークで判断する。
この文書作成ではGPUジョブを開始していない。

同期の参照：NVIDIA公式の[非同期barrier](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)、
[libcu++のメモリモデルとrelease/acquire](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html)。
