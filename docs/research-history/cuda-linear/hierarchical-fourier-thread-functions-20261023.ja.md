# thread単位で割り当てる階層Fourier：関数別の擬似コード

2026-10-23。[全体の設計メモ](hierarchical-fourier-block-design-20261023.ja.md)の具体化。
**担当人数は最初からthread単位で指定する。warp単位への丸めはしない。**
以下はCUDA風の設計用擬似コードで、そのままコンパイルできる実装ではない。
この構成のGPU correctness・同期の進行・速度・実メモリ量は未検証。

**現在の方針：同じ階層・参照タイルのatom群をjobとして、一括で並列処理する。**
一つのatomのHを複数threadの部分和へ分解する旧案は撤回した。
Hは各threadが別のatomを担当して入力方向を最後まで計算する。
空き・作業中は集団単位で管理し、実行可能なjobを取得する。

対応するbackwardの入口・G生成・入力側C・パラメータ勾配は
[`backward128`の設計文書](hierarchical-fourier-backward-128-20261023.ja.md)に分けた。

## 1. 調整する人数を分ける

| 設定 | 意味 |
| --- | --- |
| `block_threads` | block全体の人数。最初は128 |
| `worker_group_sizes` | 各作業集団の人数。一人刻みで指定し、合計をblock人数にする |
| `job_start_weights[level][policy_class]` | 階層・種類ごとに、実行可能なjobを選ぶ重み |
| `active_thread_budget[level][policy_class]` | 階層・種類ごとの同時担当人数の目安／上限。調整対象 |

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
これらを一対一に対応させない。実験ごとに集団の人数を変えられるようにする。
初期の動的割り当てでは、集団のメンバーとbarrier参加人数はblock開始時に固定し、
集団が取得するjobを動的に変える。実行中の集団へthreadを出入りさせる方式は別の検討事項。

```text
空き集団 → 実行可能なjobを取得 → Z生成、またはatom群の処理
         → 全員の作業・読み書きが完了 → 空き集団へ戻る
```

jobは`Z_TILE`と`ATOM_GROUP`を基本とする。後者にはその群の全atomを入れ、
`atoms_per_chunk=1`などで一つずつの協調H計算へ戻さない。
担当thread数を超えるatomがあれば、各threadがstrideで次のatomも担当する。
全atomが並列処理の対象だが、物理的に同時に実行する数は利用可能なthreadsに依存する。

境界がwarpと一致しない集団も表現する。人数・構成の実行可能性と性能は検証が必要で、
32の倍数に固定して解決済みとは扱わない。
発行policyでは、Z生成・Z不要のatom群・Zを使うatom群を`policy_class`で区別できる。
実行関数の分岐用`kind`と、割合を調整する分類を混同しない。

## 2. kernel前に仕事表を作る：`make_thread_plan()`

共通の係数・正規化・支持分類は既存の数学に従って準備する。
blockごとに全atomを仕分け直すのではなく、その結果から担当表を作る。
以下の`make_groups()`は指定した人数でthread番号を重複なく分割する操作。

```cpp
make_thread_plan(atom_plan, config) {
    require(all(config.worker_group_sizes >= 1));
    require(sum(config.worker_group_sizes) == config.block_threads);
    plan.worker_groups = make_groups(config.worker_group_sizes);

    plan.Z_jobs = make_Z_jobs(必要なZと、その子依存);
    plan.atom_jobs = group_all_atoms_by_level_and_references(atom_plan);
    // 出力blockに寄与する各atomは、一つのATOM_GROUPへ所属させる。
    // job.required_Zには、その群のHに必要な全Zを登録する。
    // 混合成分が複数階層にまたがるatomも、Hの寄与を欠かさない。

    plan.jobs = concat(plan.Z_jobs, plan.atom_jobs);
    plan.policy = config.job_policy;
    plan.C_members = group_output_parts_by_basis_level_tile(atom_plan);
    plan.direct_parts = remaining_output_parts(atom_plan);
    assign_disjoint_H_slots(plan.atom_jobs);
    plan.shared_layout = allocate_bounded_slots(plan);

    return plan;
}
```

jobの依存グラフは、子Z→親Z、必要なZ→atom群とする。
Z不要のatom群は初期状態から実行可能。Z一枚の実行担当はatomicな取得で一意にする。
実行可能になったatom群は、空いた適合集団が取得する。
階層ごとの所属は必要なbank・タイルの情報も含めて決め、sigmaだけで依存を省かない。

`allocate_bounded_slots()`はCの使用枠を集団別に数え、一時Hのjob別領域・alignment・通知・
barrier・出力部分和まで含める。動的取得が可能な全jobに対して、各集団のC枠を保証する。
launch時のshared上限はconfigとshapeから決める。
動的な所属を詰める場合も、その上限内でaddressを割り当てる。
容量を超える設定は計画段階で拒否するか、別の正確な計画へ変更する。
支持や寄与を切り捨てて収めない。

Z生成を選ぶ割合などはpolicyで変える。未完成のZを待つatom群を先に取得して、
全集団が待機する状態は作らない。必要な前提Zを発行できない重み・人数制限も避ける。

## 3. 全体の入口

**以下の入口の`while`は、配分を調べるための暫定的なjob探索・取得ループであり、
最終実装に必須とはしない。** 有効な人数配分と発行順が見つかり、必要なjobを
直接割り当てられる構成になれば、空き探索・再取得と入口の繰り返し自体をなくす候補を比較する。

```cpp
forward128(X, prepared_atoms, block_plans, Y) {
    plan = block_plans[blockIdx.x];
    ctx = init_block(X, prepared_atoms, plan);

    while (true) {                      // 調査用。最終的には除去する可能性がある
        job = acquire_ready_job(ctx);    // 集団で同じjobを受け取る
        if (job == ALL_DONE)
            break;
        if (job == NO_READY_JOB) {
            record_idle_and_retry(ctx);
            continue;
        }

        if (job.kind == Z_TILE) {
            build_z_tile(ctx, job.tile);
            finish_z_tile(ctx, job.tile);
        } else {
            do_work(ctx, job);           // atom群全体を処理する
        }
        complete_job(ctx, job);
    }

    expand_c_parts(ctx);                 // 自分の集団が蓄積したCを展開
    sync_finish();
    finish_output(ctx, Y);
}
```

各threadは自分の固定集団に従って進む。Z専任・H専任の役割へ固定せず、
集団の代表が次のjobを取得し、全員へ渡す。
実行可能jobが一時的になくても、未完了jobが残れば再試行する。
block全体の同期は初期化完了と最終合算前に置き、途中は関係する集団だけで同期する。

## 4. 初期化：`init_block()`

```cpp
init_block(X, prepared_atoms, plan) {
    ctx.tid = threadIdx.x;
    ctx.plan = plan;
    ctx.team = find_worker_group(plan, ctx.tid);
    ctx.shared = bind_shared_views(plan.shared_layout);

    if (ctx.tid == 0) {
        for (group : plan.worker_groups)
            init(group.barrier, group.thread_count);
        initialize_job_states_and_remaining_count(plan);
        initialize_policy_counters_and_group_states(plan);
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
Z・一時Hは、その値を使う前に全必要要素を書くので、ここでゼロにしない。
既存profileの微分配列などを、このforward用sharedへ無条件にコピーしない。

## 5. 空いた集団がjobを取る：`acquire_ready_job()`

```cpp
acquire_ready_job(ctx) {
    if (ctx.team.rank == 0) {
        job = select_and_atomically_claim_ready_job(plan.policy, ctx.team);
        // 候補がなければ、未完了件数からNO_READY_JOB / ALL_DONEを区別する。
        shared.dispatch[ctx.team.id] = job;
        record_group_assignment(ctx.team, job);
    }
    group_sync(ctx.team);
    job = copy(shared.dispatch[ctx.team.id]);
    group_sync(ctx.team);            // 全員コピー後にdispatchを再利用できる
    return job;
}
```

各集団の代表だけが取得を行う。jobは一つの集団だけが実行し、
同じ集団のthreadsは同じjobを共有する。各threadを毎回スキャンして空きを探さない。
選択処理は依存完了をacquireで確認し、取得をatomicに確定する。
jobの実行中に、取得用のlockや全体の集団管理を占有し続けない。
`record_idle_and_retry()`などの管理・記録も代表が担当し、全threadから重複更新しない。
取得に失敗しただけで終了せず、未完了件数が0のときだけ`ALL_DONE`を返す。

## 6. jobの完了と解放：`complete_job()`

```cpp
complete_job(ctx, job) {
    group_sync(ctx.team);            // 全員の読み書きが終わった
    if (ctx.team.rank == 0) {
        publish_job_done_release(job);
        decrement_remaining_jobs();
        record_job_completion_and_group_idle(ctx.team, job);
    }
    group_sync(ctx.team);
}
```

一人のthreadが先に終わっても、集団内の同期や一時Hの利用が残る間は空きへ戻さない。
完了件数・作業中件数の更新は、job全体の完了を公開してから行う。
残件数のatomicな更新と読み取り順序を定義し、取得競合を完了と誤認しない。

## 7. Z一枚の生成：`build_z_tile()`

```cpp
build_z_tile(ctx, tile) {
    team = ctx.team;
    assert(all_required_Z_are_ready(tile.required_children));

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
親の半分である正しいbank対応が存在することを示す。子の完成確認はjobの取得時に行う。
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
    group_sync(ctx.team);          // 全作成者の書き込み完了

    if (ctx.team.rank == 0)
        atomic_store_release(shared.Zready[tile.id], 1);
}
```

`group_sync(group)`はその固定集団の`cuda::barrier::arrive_and_wait()`に相当する。
Zは一つの作成集団だけが書き、集団の代表が全書き込みの後に完成を公開する。
別の計算集団や親Zの作成者は、対応するacquire確認後に読む。

## 9. atom群のHを一括生成：`compute_h_batch()`

Hは振幅・正規化係数を掛ける前の \(H_{b,a}=\sum_i X_{b,i}v_a(i)\)。
入力支持は成分ごとに、Z経由の部分と直接部分へ重複なく分解済みとする。

```cpp
compute_h_batch(ctx, job) {
    team = ctx.team;

    // 各threadが別のatomを担当。atom数が多ければstrideで次も担当する。
    for (a = team.rank; a < job.atom_count; a += team.thread_count) {
        atom = job.atoms[a];
        h[batch_size] = zeros();

        // このatomの入力縮約は、担当threadが最後まで行う。
        for (term : atom.input_plan.direct_terms)
            for (b = 0; b < batch_size; ++b)
                h[b] += shared.X[b][term.site]
                      * input_component(atom, term.component, term.site);

        for (term : atom.input_plan.Z_terms)
            for (b = 0; b < batch_size; ++b)
                h[b] += shared.Z[term.tile][b][term.basis] * term.beta;

        shared.Hjob[job.h_offset + a] = h;
    }
}
```

例えばjobに128 atoms、担当128 threadsなら、thread 0がatom 0、thread 1がatom 1、
…と分担する。32 threadsなら各threadが4 atomsを担当し、32 atomsずつの並列実行になる。
全員が同じ一つのatomへ移動する旧`for(atom)`とは異なる。

一つのatomのHを複数threadで分割・合算する`Hpart`、`Hslot`は使わない。
H生成後の同期は、**別々のatomのHが揃い、C・出力の集約へ進めることを確認する同期**である。
同一atomの入力部分和をthread間で合算する同期ではない。

`Hjob`はこのjobの全atomをC・出力集約で使うためのblock内shared領域。
jobごとに重ならないoffsetを決め、各atomの値は担当thread一人だけが書く。
初期案では全jobのatom総数を上限に枠を確保し、実行中の領域は重ならない。
全atomのglobal H Tensorを保存したり、backwardまで保持したりする方式ではない。
直接部分を含むjobも、まずこの一括処理で確認し、保存を省く融合は後の候補とする。

`input_component()`は指定成分だけを評価する。Z経由の寄与が別にある場所で
全混合profileを直接評価して足すと二重加算になるので、直接項の成分情報を保持する。
Hの入力支持を、担当する出力32サイトに切り詰めない。

## 10. atom群の直接部分を集約：`add_direct_parts_batch()`

```cpp
add_direct_parts_batch(ctx, job) {
    team = ctx.team;

    for (q = team.rank; q < batch_size * output_tile_sites; q += team.thread_count) {
        b = q / output_tile_sites;
        j = plan.output_begin + q % output_tile_sites;
        value = 0;
        for (a = 0; a < job.atom_count; ++a) {
            atom = job.atoms[a];
            u = 0;
            for (part : atom.output_plan.direct_parts)
                if (part.contains(j))
                    u += output_component(atom, part.component, j);
            value += atom.alpha * shared.Hjob[job.h_offset + a][b] * u;
        }
        shared.Ypart[team.id][b][q % output_tile_sites] += value;
    }
}
```

各集団の各出力要素は、集団内の一つのthreadだけが書く。
`atom.alpha`は既存の振幅・全支持の離散L2/floorから準備し、一度だけ掛ける。
このatom方向の和は、出来上がったHを出力へ縮約する処理である。
Hを一つずつ全員で生成する旧方式へ戻ったわけではない。
最初は出力要素を並列化した単純な和を示し、縮約内部の高速化は別に詰める。

## 11. atom群を出力タイルへ集約：`add_c_parts_batch()`

```cpp
add_c_parts_batch(ctx, job) {
    team = ctx.team;

    for (node : job.C_nodes) {
        slot = plan.C_slot[team.id][node.id];

        for (q = team.rank; q < batch_size * basis_count; q += team.thread_count) {
            b = q / basis_count;
            basis = q % basis_count;
            value = 0;
            for (member : job.C_members[node.id]) {
                atom = job.atoms[member.atom_index];
                h = shared.Hjob[job.h_offset + member.atom_index][b];
                value += atom.alpha * h
                       * output_beta(atom, member.component, basis);
            }
            shared.C[team.id][slot][b][basis] += value;
        }
    }
}
```

Cのnodeは「基底の種類・階層・空間タイル」で区別する。
`C_members`には、そのタイルの全サイトで有効な成分だけを登録する。
一つの成分を同じ出力位置の親・子タイルへ重複登録しない。
Cの同じ要素を複数threadが書かないよう、係数番号の分担を全jobで固定する。
群の全Hができた後に、`H_job × 出力係数`をまとめて縮約する。
このコードではC要素ごとの和を示した。atom方向の縮約を行列積などへ置き換える詳細は未決定。

## 12. Cを出力へ展開：`expand_c_parts()`

```cpp
expand_c_parts(ctx) {
    team = ctx.team;

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

## 13. 群全体を処理する：`do_work()`

```cpp
do_work(ctx, job) {
    compute_h_batch(ctx, job);           // 各threadが別々のatomを計算
    group_sync(ctx.team);                // job内の全Hが揃った

    add_direct_parts_batch(ctx, job);
    add_c_parts_batch(ctx, job);
    // 全員の読み書き完了は、呼出側のcomplete_job()で確認する。
}
```

jobに所属するatom全数を一括して渡す。各threadが担当するatomが違い、
全員が同じatomを順番に処理する外側のループは置かない。
atomを担当しないthreadがある場合も、H完成の集団同期には参加する。
H生成・出力集約・C集約でthreadの担当軸が変わってよい。
Zの一枚を複数threadで作る協調処理は残し、atom一個の協調H生成と区別する。

Cは集団ごとに蓄積し、jobが完了しても消さない。
全jobの完了を確認して取得ループを抜けた後、各集団が自分のCを一度展開する。
この初期構成では階層ごとのC展開を最終段へまとめる。
途中で完成nodeから展開する場合は、nodeへの全寄与の完了管理を追加する別案とする。

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

        for (group : plan.worker_groups)
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
| Hjob | `全ATOM_GROUPのatom総数 × batch_size`。job別に重ならない領域 |
| C | `各計算集団の使用node数の合計 × batch_size × basis_count` |
| Ypart | `計算集団数 × batch_size × output_tile_sites` |
| 通知・同期 | `Zready`、job状態・残件数、集団barrier・dispatch、policyカウンタとalignment |

thread単位の指定は、threadごとにC全体を複製するという意味ではない。
ただし、この部分和方式では計算集団を増やすほどC・Ypart・同期状態が増えるので、
集団数と使用slot数から実容量を確認する。4 warpsだからCも4組という固定計算はしない。
大量に増える場合の別案は、Cの所有者の分け方や保持するタイル数を変えることであり、
支持を切り詰めることではない。

最初に確認する事項は、threadの重複・抜け、固定集団の参加人数、Z作成順の依存、
支持の被覆、Cと直接部分の二重加算なし、書き手の一意性、job取得の一意性と終了判定である。
部分warpを含む経路へfull-warpのshuffle maskを流用しない。
sharedのalignment、barrier/atomicの構築と寿命、対象GPUでの進行も実装時に確認する。

その後、独立参照でZ/H/C/Yを確認し、学習経路を接続した段階でdX・全atom勾配まで検証する。
性能は集団ごとの人数と、階層・種類ごとのjob選択比率を変え、
準備・所属構築を含む完全stepとallocated/reservedピークで判断する。
この文書作成ではGPUジョブを開始していない。

## 16. 動的に配分を調べ、その後に規則化する

最初は、集団の空きと実行可能jobを見て発行する。
後で有効な配分が分かったら、仕事量から配分を計算し、空き探索・取得の費用を減らす。
同期対象や数学を変えず、jobの取得方針を差し替えられる境界にする。
割合を決める具体的なアルゴリズムはまだ固定しない。

最終形は動的なworkerループに限定しない。jobの直接割り当てや、
依存と発行順を事前に決めた実行構成によって、入口に`while`やjob取得の繰り返しを
持たない方式も候補とする。単にループを消すのではなく、全jobの被覆と依存の順序を
別の構成で保証できることを条件にする。並列化しただけで除去できると断定はしない。
ここで除去を検討するのは入口のjob管理ループであり、計算内部の積和・基底評価や
担当thread数を超えるatomの処理に必要な反復まで、すべてなくすという意味ではない。

```text
job：前提が未完成 → 実行可能 → 一つの集団が取得して実行 → 完了
集団：空き       → job取得   → 全員でjobを実行           → 空き
```

- jobの種類はZ生成とatom群処理、分類軸は階層・参照bank/タイルとする。
  Z不要／Zを使うatom群をpolicyで区別し、多階層の入力を使う群では依存集合も保持する。
- 階層・種類ごとの開始重みと、同時担当thread数を調整できるようにする。
  job一件の重さや人数が違うため、開始件数の比率だけで資源配分を判断しない。
- 初期の取得は集団の代表だけが行う。全threadの空き状態を繰り返し走査しない。
  同じ集団のメンバーはjobの途中で別のjobへ移らない。
- jobの取得と人数枠の予約は整合する手順で確定し、完了時にだけ解放する。
  各jobに適合集団が存在し、前提Zが枠や重みの制限で永久に発行されない状態を避ける。
- jobが未準備の時間、readyになってから取得されるまでの時間、処理中の時間、
  集団が仕事を持てなかった時間、取得・通知の管理費用を分ける。
  割り当て人数の記録は、GPUの実SM稼働率の実測ではない。
- jobのatom数、直接項数、Z項数、Cと直接出力への参照件数、割り当て人数を記録する。
  jobの重さを求める式と係数は、これらの実測から詰める。
- 各threadのatom分担には偏りが残り得る。集団の最も遅い担当が完了するまでは
  空きへ戻さない。atomの順序・割り当て人数を変える余地を残し、
  一つのatomのHを複数threadの入力部分和へ分ける旧案には戻さない。
- 有効な規則が得られた後は、同じjob量から人数・発行順を事前に決める方式と比較する。
  学習で幅や所属が変わるため、一つの固定比率を全条件へ当てはめると決めない。
  直接割り当てで全仕事を処理できれば、入口のjob取得ループを除去する可能性も含める。
- 空き探索を省いても、子Z→親Z、Z→atom群、H→C/出力という依存確認は残す。
  動的方式の計装時間を、管理費用のない計算時間や未計装の完全stepと混同しない。

同期の参照：NVIDIA公式の[非同期barrier](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)、
[libcu++のメモリモデルとrelease/acquire](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html)。
