# backward128：atom群を並列処理するbackwardの入口と関数

2026-10-23。[forwardの関数別擬似コード](hierarchical-fourier-thread-functions-20261023.ja.md)に対応する設計案。
**以下は未実装のCUDA風擬似コード。GPU correctness・同期・速度・総メモリは未検証。**
本書作成ではruntimeを変更せず、GPUジョブも開始していない。
2026-10-24追記：Hの保存・再計算・一部保存を選択できる設計にした。
同日追記：H/Gを同じthreadで計算する方式と、別集団で計算して受け渡す方式を切り替える。
同日追記：動的job管理をなくした固定担当表と、Gが軽い場合にH/Gをまとめる仮説を追加した。

## 1. 何を受け取り、何を返すか

| 名前 | 意味 |
| --- | --- |
| `X` | forwardで使った入力 |
| `dY` | 後段から渡される、出力Yに対する損失の勾配 |
| `forward_state` | forward時点のParameter・配置・支持・係数等を正しく参照／再生成するための情報 |
| `dX` | 入力Xに対する勾配 |
| `dP` | 各atomの振幅・出力中心・入力中心・幅logitに対する勾配 |

```text
forward ： X → 入力側Z → H → 出力側C → Y
backward：dY → 出力側Z → G → 入力側C → dX
                        │
          X → 入力側Z → H ──→ H/Gとprofile・正規化の微分 → dP
```

Hは入力Xを入力profileで縮約した値、GはdYを出力profileで縮約した値である。
Hをbackwardまで保存するかは別の選択で、以下の3方式を比較できるようにする。
Gは振幅を掛ける前の値とする。`dH`そのものは後述の`alpha * G`で、Gとは区別する。

**atom群の全atomを並列対象にし、各threadが別のatomのH/Gを最後まで計算する。**
一つのatomの入力部分和を複数threadで作って合算する方式には戻さない。
Z一枚の生成や、複数atomからC・dXへの集約は、集団で協力する別の処理である。

### Hの保持方針：`make_H_retention_plan()`

| 方針 | forwardから保持するもの | backwardですること |
| --- | --- | --- |
| `recompute` | Hは保持しない | 必要なHの項目を再計算する |
| `save` | 必要な全atomの、指定した項目 | 保存済みの項目を読み、不足分は再計算する |
| `hybrid` | 選んだ階層・atom群の、指定した項目だけ | 保存対象と再計算対象を組み合わせる |

保存する項目も独立に指定する。Hは以下の未正規化・振幅適用前の縮約値とする。

| 項目 | 数式 | 用途 |
| --- | --- | --- |
| `value` | \(H_{b,a}=\sum_i X_{b,i}v_a(i)\) | 振幅勾配など |
| `center` | \(\sum_i X_{b,i}\partial v_a(i)/\partial\mu_{in}\) | 入力中心の勾配 |
| `width` | \(\sum_i X_{b,i}\partial v_a(i)/\partial I\) | 幅の勾配 |

`save`でも`value`だけを指定すれば、中心・幅の縮約値はbackwardで再計算する。
全パラメータ勾配に必要な3項目を保存した場合は、H側の再計算が不要になる。
dXだけならHは不要で、Gから計算する。正規化分母とその微分は別途必要であり、
Hの保存だけでそれらが省けるわけではない。

```cpp
make_H_retention_plan(config, atom_groups) {
    for (group : atom_groups)
        saved_fields[group] = choose_fields(config.mode, group, config);
    return pack_saved_fields_and_assign_unique_writers(saved_fields);
}
```

`recompute`なら全群で空集合、`save`なら全群で指定項目、`hybrid`なら群ごとに指定する。
方針はforward前に決め、実際の保存先・atom ID対応・有効batch範囲とともに
`forward_state.H_retention`へ記録する。未保存の項目を保存済みとして扱わない。
階層ごとの再計算費用を見て保存対象を選ぶのは今後の調整対象で、sigmaだけの固定境界は置かない。

保存先はforwardのblock終了後も残るglobal領域で、sharedの`Hjob`とは別である。
同じatomを複数の出力blockで計算しても、保存する書き手は各atom・batch・項目につき一つ。
通常の出力jobが訪れない保存対象にも担当を用意し、保存範囲を欠落させない。
backwardはforward完了後に同じstream等の依存で読み、block内同期で代用しない。
保存値は同じforwardの状態に結び付け、retained backward中に破壊・再利用しない。
Xや元Parameterの保存・version検査は、Hの保存設定だけを理由に取り除かない。

## 2. 大きさと、勾配を書き込む担当

```cpp
input_site_count = 128;
output_site_count = 128;
atom_count = 128;
batch_size = 4;             // 一つのblockの小バッチ
input_tile_sites = 32;      // 一つのblockが保存するdXの範囲
basis_count = 2 * fourier_order + 1;
```

擬似コードは小バッチの全件が有効な基本例を示す。
全体batch数が割り切れない場合は、有効件数でload・縮約・storeをmaskする。

最初に具体化するため、次の所有方法を候補とする。速度の採用判断ではない。

| 対象 | 担当 |
| --- | --- |
| `dX[b,i]` | 小バッチ × 入力32サイトを所有する一つのblock |
| 小バッチ内の全atomの`dP_partial` | 同じ小バッチの入力tile 0を所有するblockだけ |
| 最終`dP` | 全小バッチの部分和を、全backward block完了後に合算する担当 |

バッチ4ならdX担当は4 blocks。block 0だけが全atomのパラメータ勾配も計算する。
他の3 blocksはGから自分の入力タイルのdXを作り、パラメータ勾配は書かない。

この所有方法では、パラメータ担当blockの仕事が重くなる可能性がある。
一方で、全blockが同じdPを重複して加算することを避け、書き手を明確にできる。
独立したパラメータ勾配用block群などとの比較は後の候補とする。

パラメータ担当blockは、**自分の入力32サイトに寄与しないatomも含め、全atomを扱う。**
それ以外のblockでは、入力側支持が担当dX範囲に重なるatomを拾えばよい。
いずれもGはそのatomの出力支持全体から計算する。
パラメータ勾配に使うHも入力支持全体から計算し、担当dXタイルへ切り詰めない。

### H/Gの担当方式：`config.hg_execution`

| 設定 | H/Gとパラメータ勾配の担当 | 中間値の受け渡し |
| --- | --- | --- |
| `same_thread` | 各threadが別atomのG・H・勾配を順に計算 | H/Gの微分項目はthread内。Gの値だけsharedへ |
| `split_groups` | H群・G群・勾配計算を別jobにする | H/Gの項目をsharedへ置き、完成後に勾配jobが読む |

これはHの保持方針とは独立の設定。例えば`hybrid × split_groups`も選べる。
方式はlaunch前に選び、途中で切り替えない。方式ごとに仕事表とsharedの容量を決める。
`same_thread`を最初の比較基準とし、速い方式を確定したとは扱わない。
`split_groups`でもHとGそれぞれの内部は、各threadが別atomの縮約を最後まで計算する。
H群とG群でthread数・thread番号が一致する必要はない。同じatomのH/Gを対応付けて読む。

別集団の初期案も**同じblock内**で完結させる。固定メンバーの作業集団が、
H job・G job・勾配job等を動的に取得する。H専用threadを実行中に移籍させる方式ではない。
同時に走らせるには少なくとも二つの適合集団が必要で、GPU上の実際の重なりは未検証。
集団が一つなら同じjobグラフを順次処理できるが、H/Gの並列実行にはならない。
担当人数は`worker_group_sizes`、発行比率は`job_start_weights[level][policy_class]`等で調整する。
backward用の分類にZ_X・Z_dY・H・G・勾配・dXを用意し、必要な作成jobの担当をなくさない。
人数・発行比率は一人刻みの設定を保ち、warp数を配分の単位に固定しない。

## 3. 全体の入口：`backward128()`

**入口の`while`は配分調査用の暫定構成。** 有効な配分と発行順が見つかれば、
jobの直接割り当てなどで取得ループを持たない方式も比較する。
依存の確認と、全jobを一度ずつ処理する保証は残す。

autograd側では、保存したX・元Parameter等のversion検査を先に行い、
forward時点の状態から係数・norm・job表を用意して、このdevice側の入口を呼ぶ。
この検査をGPU内の`init_backward_block()`が行うわけではない。

```cpp
backward128(X, dY, forward_state, block_plans, dX, dP_partial) {
    plan = block_plans[blockIdx.x];
    ctx = init_backward_block(X, dY, forward_state, plan);

    while (true) {                         // 調査用。後に除去する可能性がある
        job = acquire_ready_job(ctx);
        if (job == ALL_DONE)
            break;
        if (job == NO_READY_JOB) {
            record_idle_and_retry(ctx);
            continue;
        }

        if (job.kind == BUILD_Z_FROM_DY) {
            build_z_tile(shared.dY, shared.Z_dY, ctx.team, job.tile);
            publish_z_tile_ready(shared.Z_dY_ready, ctx.team, job.tile);
        } else if (job.kind == BUILD_Z_FROM_X) {
            build_z_tile(shared.X, shared.Z_X, ctx.team, job.tile);
            publish_z_tile_ready(shared.Z_X_ready, ctx.team, job.tile);
        } else {
            dispatch_backward_job(ctx, job, dP_partial);
        }

        complete_job(ctx, job);
    }

    expand_input_c_parts(ctx);             // 入力側C → 自分のdX部分和
    __syncthreads();                        // block全員の部分和が完成
    finish_dx(ctx, dX);
}
```

`acquire_ready_job()`と`complete_job()`はforwardと同じ集団単位の取得・完了の契約を使う。
必要なZが未完成のatom群を取得して、集団全体を待機させない。
queueはblock内で完結させ、別blockの未完了jobをこのループで待たない。

Zの計算はforwardの`build_z_tile()`を、入力sourceとZ保存先を指定できる形にしたもの。
親の偶数modeを対応する子から継承し、残りをsourceとphiの積和で作る数学は共通である。
`Z_X`と`Z_dY`の値・完成通知は別領域であり、同じnode番号でも取り違えない。

## 4. blockの仕事表と初期化

```cpp
make_backward_plan(forward_state, batch_group, input_tile, config) {
    plan.batch_group = batch_group;
    plan.input_tile = input_tile;
    plan.writes_param_grad = (input_tile.id == 0);
    plan.hg_execution = config.hg_execution;
    plan.worker_groups = make_groups(config.worker_group_sizes);

    if (plan.writes_param_grad)
        atoms = all_atoms();
    else
        atoms = atoms_overlapping_input_tile(input_tile);

    plan.atom_jobs = group_all_atoms_by_level_and_references(atoms);
    plan.Z_dY_jobs = required_output_moments(plan.atom_jobs);
    plan.H_missing = plan.writes_param_grad
                   ? missing_H_fields(forward_state.H_retention, atoms, batch_group,
                                      {value, center, width}) : empty;
    plan.Z_X_jobs = required_input_moments(plan.H_missing);
    plan.needs_X = has_input_contractions(plan.H_missing);

    // dXの直接部分・C所属は担当input_tileとの重なりだけを登録する。
    plan.dx_parts = partition_input_support(atoms, input_tile);
    plan.jobs = build_backward_jobs(plan); // 下の方式別依存から作る。子Zも含む
    plan.shared_layout = allocate_bounded_backward_slots(plan);
    return plan;
}
```

`same_thread`のatom jobは、Gに必要な`Z_dY`とHの不足項目に必要な`Z_X`を待つ。
`split_groups`ではH jobが`Z_X`、G jobが`Z_dY`を独立に待つ。
保存済みのH項目のために入力Zを作らない。子Zは不足項目の依存から再帰的に含める。
直接計算だけで済む場合は対応するZを作らない。入力側・出力側の経路を独立に判定する。

```cpp
init_backward_block(X, dY, forward_state, plan) {
    ctx = bind_thread_group_and_shared(plan);
    initialize_group_barriers_job_states_and_policy(ctx);

    load_dY_batch_to_shared(dY, plan);
    if (plan.needs_X)
        load_X_batch_to_shared(X, plan);

    clear_input_C_and_dx_partials(ctx);
    clear_Z_ready_flags(ctx);       // X由来とdY由来を分ける
    if (plan.hg_execution == split_groups)
        clear_HG_ready_flags(ctx);  // 群ごとのHready / Gready
    __syncthreads();
    return ctx;
}
```

初期化の分担・barrierの構築はforwardの`init_block()`と同じ約束とする。
Z・Gの一時値は使用前に全必要要素を書くので、無条件に全領域をゼロ初期化しない。
H側の再計算がないblockには、Xと入力Zのshared領域を用意しない。
直接項のみを再計算する場合はXだけを用意し、不要なZは確保しない。

## 5. atom群の仕事：`do_backward_work()`

入口からは次のdispatchを呼ぶ。`BUILD_Z_*`は入口で処理済みとする。

```cpp
dispatch_backward_job(ctx, job, dP_partial) {
    switch (job.kind) {
    case ATOM_HG:      do_backward_work(ctx, job, dP_partial); break;
    case H_GROUP:      produce_H_group(ctx, job); break;
    case G_GROUP:      produce_G_group(ctx, job); break;
    case PARAM_GROUP:  consume_HG_group(ctx, job, dP_partial); break;
    case APPLY_G:      apply_G_group(ctx, job); break;
    }
}
```

### 同じthreadで計算：`same_thread`

`ATOM_HG` jobを取得し、以下の処理を行う。

```cpp
do_backward_work(ctx, job, dP_partial) {
    compute_atom_backward_batch(ctx, job, dP_partial);
    group_sync(ctx.team);          // job内の全atomのGが完成

    add_dx_direct_parts_batch(ctx, job);
    add_input_c_parts_batch(ctx, job);
    // 全員の読み書き完了は、呼出側のcomplete_job()で確認する。
}
```

```cpp
compute_atom_backward_batch(ctx, job, dP_partial) {
    for (a = ctx.team.rank; a < job.atom_count; a += ctx.team.thread_count) {
        atom = job.atoms[a];
        need_derivatives = plan.writes_param_grad;

        g = compute_G_for_atom(shared.dY, shared.Z_dY,
                               atom.output_plan, need_derivatives);
        shared.Gjob[job.g_offset + a] = g.value;

        if (plan.writes_param_grad) {
            h = load_or_recompute_H_for_atom(ctx, atom);
            grad = atom_parameter_vjp(atom, h, g, atom.norm);
            original_id = forward_state.original_atom_id[atom.id];
            dP_partial[plan.batch_group][original_id] = grad;
        }
    }
}
```

`compute_G_for_atom()`とHの再計算は、**呼んだthread一人で**
そのatomの小バッチ全件を計算する。集団の他threadは別のatomを計算する。
128 atomsを128 threadsで担当するなら、一人一atom。人数が少なければstrideで次も担当する。

この方式のパラメータ担当では同じthreadがG・H・自分のatomのパラメータ勾配までを計算する。
Hと微分縮約値はまずローカル変数で使い切り、他threadへの受け渡し用にはGの値だけをsharedへ置く。
ローカル配列が実際にregisterへ収まるか、退避が起きるかはコンパイル後に確認する。

`original_atom_id`はforwardの物理的な並べ替えから元Parameterへの対応である。
並べ替えがなければ恒等対応とする。
最初から元IDへ書いた部分和を、最終合算時にもう一度逆並べ替えしない。
担当blockは全atomの全パラメータ欄を、ゼロの場合も含めて必ず書く。

### 別集団で計算：`split_groups`

atom群ごとに、H job・G job・勾配jobを対応付ける。各jobは群の全atomを並列対象にする。
ここでの「まとめる」は、**同じatomのHとGを取り出して勾配式へ渡すこと**であり、
異なるatomのH/Gを一つの値へ足す操作ではない。

```text
群qの不足項目に必要なZ_X → H_GROUP(q) → Hready[q] ─┐
                                                 ├→ PARAM_GROUP(q) → dP_partial
群qに必要なZ_dY         → G_GROUP(q) → Gready[q] ─┘
                                          └──────→ APPLY_G(q) → C_input / dXpart
```

| job | 開始条件 | 保存先・役割 |
| --- | --- | --- |
| `H_GROUP(q)` | H不足項目のZが完成。直接計算／全保存ならZ待ちはない | 保存Hを読み、不足分を計算し、`Htemp`へ書く |
| `G_GROUP(q)` | Gに必要な出力側Zが完成 | Gの必要項目を`Gtemp`へ書く |
| `PARAM_GROUP(q)` | HreadyとGreadyの両方が完成 | 同じatomのH/Gを読み、一意な担当がdP部分和を書く |
| `APPLY_G(q)` | Greadyが完成 | Gのvalueから、そのblockのC・dX部分和を作る |

dX専用blockにはH jobと勾配jobを作らず、Gにはvalueだけを用意する。
パラメータ担当blockのH/Gはそれぞれvalue・center・widthを揃える。
全項目をforwardで保存した場合も、初期のsplit案ではH jobが読み出して`Htemp`へ置く。
後にglobal保存値を直接読む案と比較できるが、ここでは受け渡し方法を統一する。

```cpp
produce_H_group(ctx, job) {
    for (atom : strided_atoms(job.atoms, ctx.team))
        shared.Htemp[slot(atom)] = load_or_recompute_H_for_atom(ctx, atom);
    publish_group_payload_ready(ctx.team, shared.Hready[job.group_id]);
}

produce_G_group(ctx, job) {
    for (atom : strided_atoms(job.atoms, ctx.team))
        shared.Gtemp[slot(atom)] = compute_G_for_atom(
            shared.dY, shared.Z_dY, atom.output_plan, plan.writes_param_grad);
    publish_group_payload_ready(ctx.team, shared.Gready[job.group_id]);
}

consume_HG_group(ctx, job, dP_partial) {
    // 取得時にHreadyとGreadyをacquire確認済み。ここでは待たない。
    for (atom : strided_atoms(job.atoms, ctx.team)) {
        h = shared.Htemp[slot(atom)];
        g = shared.Gtemp[slot(atom)];
        grad = atom_parameter_vjp(atom, h, g, atom.norm);
        original_id = forward_state.original_atom_id[atom.id];
        dP_partial[plan.batch_group][original_id] = grad;
    }
}

apply_G_group(ctx, job) {
    // shared.GjobをGtempのvalue項目のviewとして読む。追加コピーはしない。
    add_dx_direct_parts_batch(ctx, job);
    add_input_c_parts_batch(ctx, job);
}

publish_group_payload_ready(team, ready_flag) {
    group_sync(team);              // 作成者全員のpayload書き込み完了
    if (team.rank == 0)
        atomic_store_release(ready_flag, 1);
}
```

`slot(atom)`はこのblock・小バッチで一意なatomのslot。H/G双方で同じatom表を使い、
job内のthread順位で対応付けない。`strided_atoms()`はrankからthread数刻みで別atomを取るだけで、
一つのatomの縮約を複数threadへ分割しない。勾配jobにも別の人数を割り当ててよい。
勾配のbatch方向の和は担当thread内、atom間のdX集約は`APPLY_G`内で行う。

完成通知はZと同様、集団barrier後のrelease公開と、取得側のacquire確認を組にする。
単なるboolの書き換えや、分岐した一部threadだけの`__syncthreads()`で代用しない。
`PARAM_GROUP`は両フラグをacquire確認してからatomicに一度だけ取得し、
`APPLY_G`はGreadyだけで取得する。Hの完成を待つために計算集団を占有しない。
各jobの終了は引き続き`complete_job()`で集団全員が確認し、その後に次のjobを取る。

初期案の`Htemp`・`Gtemp`はatom別固定slotをblock終了まで保持し、読み手が複数でも上書きしない。
`Gtemp`には勾配jobとdX jobの二つの読み手がある。早期再利用を入れるなら両者の読了確認が必要。
保存用global Hとこの一時shared Hは別の寿命であり、保存値をscratchとして壊さない。
最後のblock同期は全job完了後にだけ置く。全job完了件数には勾配job・dX jobも含める。

`build_backward_jobs()`は方式に応じて`ATOM_HG`、または上記4種類へ分解し、
必要なZだけを含めた依存グラフを作る。全jobに実行可能な集団が存在することを計画段階で確認する。
readyな作成jobが取り続けられる発行規則とし、未完成のH/Gを待つjobだけで集団を埋めない。
`same_thread`の基準案ではH/G双方のZを待ってからatom jobを始めるため、
Gだけが先に準備できた際のdX先行は`split_groups`で比較する。

### 将来の選択肢：job取得をなくす固定担当表

発行方法を`config.scheduling = dynamic_jobs / static_table`として分ける案を残す。
現在の入口は`dynamic_jobs`の擬似コードであり、`static_table`は将来比較する構成。
Hの保持方針、H/Gの担当方式、発行方法は別の軸とする。
例えば`hybrid × same_thread × static_table`も候補となる。runtimeの設定追加はまだ行わない。

固定表は、各集団のthread範囲、担当atom群、必要なZと作成順、Hの書き手・読み手、
dX／dPの書き手をlaunch前に決める。空き探索・job選択・atomicな取得・残job件数の管理をしない。
支持・係数・H保持の情報は引き続き必要で、担当表を作る準備費用も完全stepへ含める。
「固定」は一回のlaunch中の担当が固定という意味で、学習による幅・所属の変化に応じて表を更新する。

`split_groups × static_table`の最初の例は、同じblockに入力側担当と出力側担当を一集団ずつ置く。
入力側担当はZ_XとH、出力側担当はZ_dY・G・dX側の処理・パラメータ勾配を受け持つ。
両者には対応するatom群qの表を渡す。集団内では各threadが別atomを処理する。

```cpp
backward128_static(ctx, table, dP_partial, dX) {
    init_static_block(ctx, table);       // 共通初期化。queue・jobカウンタは作らない

    if (ctx.team.role == INPUT_H) {
        build_input_Z_in_dependency_order(ctx, table);  // 不足項目に必要なZだけ
        for (q : table.H_groups)         // 担当表を進む。jobを取得しない
            produce_H_group(ctx, q);    // shared.Htempへ書き、Hreadyを公開
    } else {                            // OUTPUT_G_AND_GRAD
        build_output_Z_in_dependency_order(ctx, table);
        for (q : table.G_groups) {
            produce_G_group(ctx, q);
            apply_G_group(ctx, q);      // Hを待たず、C_input・dXpartを作る
            if (plan.writes_param_grad) {
                wait_for_H_group(ctx, q);
                consume_HG_group(ctx, q, dP_partial);
            }
        }
    }

    group_sync(ctx.team);                // 担当集団のCへの全書き込みが完了
    expand_input_c_parts(ctx);
    __syncthreads();                     // 分岐を抜けたblock全員が参加
    finish_dx(ctx, dX);
}
```

qは上の関数が必要とするatom表・slot・C所属を持つ固定レコードで、動的取得対象ではない。
各側のZはその側の集団が子から親へ作り、必要な箇所で集団同期する。
この例は各側の必要Zを先に揃える単純な構成で、群ごとのZ準備と処理を重ねる案も後に比較する。
初期化では入力担当のC・dX部分和も0にし、H/Gの固定slotと通知はblock終了まで保持する。
dX専用blockは入力H担当を置かず、出力側の処理だけを割り当てる。

`wait_for_H_group()`は代表がHreadyをacquire確認し、その確認を集団同期で全員へ伝える。
Gは同じ集団が生成済みなので、job取得時の完成確認をこの明示的な確認に置き換える。
H担当はGを待たずに進み、G担当が必要なHを待つ一方向の依存にして、循環待ちを作らない。
待機中にH担当の進行を妨げない同期方法は、集団のthread配置も含めて実装時に検証する。
固定表にしても依存の待機は残り、その内部に待機ループを使う可能性はある。
除去するのは入口のjob取得用`while`であり、担当群・階層・積和の`for`や同期ではない。

`same_thread × static_table`なら、必要な両側Zを依存順に準備してから、固定されたatom群ごとに
各threadがH/G・勾配を計算し、Gの値を集団内でdX側へ渡す。HtempやH/G間の完成通知は不要。
Z準備・Gからの集約・最終合算の同期は残す。
固定表は全atomと全勾配の被覆、一意な書き手、依存順、shared容量を検査してから使う。

### 検証する仮説：Gが軽ければH/Gを同じthreadへまとめる

ユーザーの見立ては「Gの計算がHより軽くなり、同じthreadでH/Gを計算する方式へ寄るかもしれない」。
Gが軽い条件では、別集団へ分けて重ねられる計算時間より、sharedへの受け渡し・通知や
人数の偏りの費用が大きくなる可能性がある。その場合は`same_thread`、さらに
`static_table`で管理を省く組み合わせを候補とする。現時点では未計測の仮説である。

HとGはsourceがX／dYというだけで本質的に軽重が決まるわけではない。
各側の支持・Z需要・必要な微分項目・Hの保存率で変わるので、
`Z_X準備 + H不足分の計算`と`Z_dY準備 + G計算`、Hの保存値読み出し、受け渡し・待ち・管理費用を分けて記録する。
保存率が高くH側が軽い場合や、register保持が増える場合も含め、
動的／固定と同一thread／別集団を、独立勾配検証後の完全step時間・ピークメモリで比較する。

## 6. GとHの中身

保存方針による分岐は、次の関数へまとめる。

```cpp
load_or_recompute_H_for_atom(ctx, atom) {
    required = {value, center, width};
    h = load_saved_H_fields(forward_state.H_retention, atom, plan.batch_group);
    missing = required - h.available_fields;
    if (!missing.empty())
        h.merge(recompute_H_fields(shared.X, shared.Z_X, atom.input_plan, missing));
    return h;
}
```

`recompute_H_fields()`は指定項目だけを縮約し、保存済み項目を上書きしない。
必要なZとXは仕事表で先に準備する。ここで未完成のZを待つ構成にはしない。
項目のmaskは少なくともjob計画で把握し、群を分けるか同じjob内で分岐するかは比較する。

両方とも「sourceと、その側のprofileの積和」である。

| 関数 | source | profile | 使う階層値 |
| --- | --- | --- | --- |
| `compute_G_for_atom()` | dY | 出力側u | Z_dY |
| `recompute_H_fields()` | X | 入力側vの不足項目 | Z_X |

値に加え、パラメータ勾配にはprofileの中心微分とlive幅I微分に対する積和が必要。
例えばGについて、`g.value[b] = Σ dY u`、`g.center[b] = Σ dY ∂u/∂μ_out`、
`g.width[b] = Σ dY ∂u/∂I`を返す。Hも入力側について同じ三つを返す。
`center`と`width`は損失勾配そのものではなく、縮約値をprofileのパラメータで微分した値。

```cpp
contract_one_atom(source, Z, side_plan, need_derivatives) {
    result = zeros(batch_size, need_derivatives ? 3 : 1);

    for (term : side_plan.direct_terms) {
        f = evaluate_component_with_derivatives(term, need_derivatives);
        for (b = 0; b < batch_size; ++b)
            result[b] += source[b][term.site] * f;
    }

    for (term : side_plan.Z_terms) {
        beta = coefficient_with_derivatives(term, need_derivatives);
        for (b = 0; b < batch_size; ++b)
            result[b] += Z[term.tile][b][term.basis] * beta;
    }
    return result;
}
```

上のレコードの加算は、値・中心微分・I微分の各fieldへそれぞれ加算する意味。
Hの再計算では`need_derivatives`の代わりに不足項目maskを使い、同じ積和の該当fieldだけを計算する。
パラメータ微分も既存の有限Fourier・混合・支持・離散正規化の数学を保つ。
直接項と階層項は成分ごとに重複なく分解し、全混合profileを直接項へ重ねて足さない。
GとHに正規化や振幅を先に混ぜず、後述のalphaと分母微分を一度だけ扱う。

## 7. dXの集約と保存

forwardで出力側へ行ったC集約・直接加算を、入力側へ行う。

```cpp
add_dx_direct_parts_batch(ctx, job) {
    for ((b, i) : owned_dx_elements(ctx.team, plan.input_tile)) {
        value = 0;
        for (a : job.atoms) {
            v_direct = input_direct_profile_at(a, i);
            value += a.alpha * shared.Gjob[job.slot(a)][b] * v_direct;
        }
        shared.dXpart[ctx.team.id][b][i - plan.input_begin] += value;
    }
}

add_input_c_parts_batch(ctx, job) {
    for ((node, b, r) : owned_C_elements(ctx.team, job.input_C_nodes)) {
        value = 0;
        for (member : job.input_C_members[node]) {
            atom = member.atom;
            value += atom.alpha * shared.Gjob[job.slot(atom)][b]
                   * input_beta(atom, member.component, r);
        }
        shared.C_input[ctx.team.id][node][b][r] += value;
    }
}
```

`owned_*`は、同じ集団の同じ出力要素を一つのthreadだけに割り当てる分担。
このatom方向の和は、完成した別々のGをC・dXへ縮約する処理であり、
一つのatomのGを全threadで作る方式ではない。縮約内部の高速化方法は別に詰める。

```cpp
expand_input_c_parts(ctx) {
    for ((b, i) : owned_dx_elements(ctx.team, plan.input_tile))
        for (node : plan.input_C_nodes[ctx.team.id])
            if (node.contains(i))
                shared.dXpart[ctx.team.id][b][i - plan.input_begin]
                    += sum_basis(shared.C_input[ctx.team.id][node][b], phi(node, i));
}

finish_dx(ctx, dX) {
    for ((b, i) : block_owned_dx_elements(threadIdx.x, plan)) {
        value = 0;
        for (group : plan.worker_groups)
            value += shared.dXpart[group.id][b][i - plan.input_begin];
        dX[plan.batch_begin + b][i] = value;
    }
}
```

入力側でも有限支持を厳密に扱う。Cにはnode全体で有効な成分だけを集約し、
支持の端は小さい領域か直接加算へ回す。担当入力タイルとの共通部分だけに展開する。
同じ成分・同じ入力位置を親と子、Cと直接部分で二重加算しない。

## 8. atomのパラメータ勾配：`atom_parameter_vjp()`

初めに読む際は、上の入口・atom分担を先に確認し、この節の数式は後から対応づけてよい。
ここでは現行`kernels.py::vjp`と同じ、生profileによる分母の配分を使う。
この添字aは一つのatomで、以下の式では読みやすさのため一部省略している。

\[
N_u=\sum_j u_a(j)^2,\qquad N_v=\sum_i v_a(i)^2,\qquad
D=\max(\sqrt{N_uN_v},\epsilon),\qquad\alpha=A/D,
\]

\[
H_b=\sum_i X_{b,i}v_a(i),\qquad
G_b=\sum_j dY_{b,j}u_a(j),\qquad
dX_{b,i}=\sum_a\alpha_aG_{b,a}v_a(i).
\]

分母の対数微分を \(\lambda_{u,\mu}=\partial_{\mu_u}\log D\) 等と表す。
例えばfloorが働かない場合は

\[
\lambda_{u,\mu}=\frac{\sum_j u_a(j)\,\partial_{\mu_u}u_a(j)}{N_u}.
\]

v側とI微分も同様。現行コードでは`raw = sqrt(Nu * Nv) > epsilon`かつ
該当normが正のときだけ有効とし、floor以下ではこれらの分母微分を0にする。
境界の扱いも現行の数学契約・独立参照と合わせる。

```cpp
atom_parameter_vjp(atom, h, g, norm) {
    D = max(sqrt(norm.output_squared * norm.input_squared), epsilon);
    alpha = atom.amplitude / D;
    lambda = norm_log_derivatives_with_floor(norm, epsilon);
    S = sum_batch(h.value * g.value);

    grad.amplitude = S / D;
    grad.output_center = alpha * (
        sum_batch(h.value * g.center) - S * lambda.output_center);
    grad.input_center = alpha * (
        sum_batch(h.center * g.value) - S * lambda.input_center);
    grad.width_I = alpha * (
        sum_batch(h.width * g.value + h.value * g.width)
        - S * (lambda.output_width + lambda.input_width));

    sig = sigmoid(atom.width_logit);
    grad.width_logit = grad.width_I * 9.9 * sig * (1 - sig);
    return [grad.amplitude, grad.output_center,
            grad.input_center, grad.width_logit];
}
```

`sum_batch()`はこのblockの有効な小バッチ全件の和であり、全体batchの和ではない。
幅の係数9.9は現行の`I = 0.1 + 9.9 * sigmoid(logit)`から来る。
Iを会話上のsigmaや成分支持半長Lへ無条件に読み替えない。
正規化は全観測site・混合後profileに対する離散L2とfloorを保つ。
atomやtileで切り分けた局所normへ置き換えない。

振幅0でも`grad.amplitude`は非ゼロになり得るため、振幅0を理由にjobを落とさない。
振幅で割って未スケールH/Gを復元する方法も使わない。

## 9. batch間のパラメータ勾配を合算する

各小バッチのパラメータ担当blockは`dP_partial[batch_group][original_atom][parameter]`を一度だけ書く。
小バッチが一つなら、その保存先を最終dPと兼用できる。
複数小バッチなら、全backward blockが完了した後に、別の縮約処理を実行する。

```cpp
reduce_parameter_gradients(dP_partial, dP) {
    for ((atom, parameter) : このthreadの担当要素) {
        value = 0;
        for (batch_group : all_batch_groups)
            value += dP_partial[batch_group][atom][parameter];
        dP[atom][parameter] = value;
    }
}
```

これはblock内の`__syncthreads()`では保証できないblock間の受け渡しである。
初期候補では、同じstream上でbackward128の後に縮約kernelを実行する等の依存を使う。
必要な別処理を入口の関数名やblock内同期の裏へ隠さない。
この合算はbatch方向の勾配の和であり、複製した全勾配を入力タイル数だけ足すものではない。

## 10. 保存・メモリ・同期の範囲

| 値 | 初期案の場所・寿命 |
| --- | --- |
| forward時点のXとParameter等 | autogradの保存・version検査に従い、同じ状態を再参照／再計算 |
| 小バッチのdY | 全blockのshared。G計算が完了するまで |
| 保存対象のH項目 | global。選択したatom・batch・項目だけをforwardからbackward利用終了まで保持 |
| 小バッチのX | Hの再計算があるパラメータ担当blockだけのshared |
| Z_dY / Z_X | 別々のsharedと完成通知。最初は上書きしない |
| atomのH/Gと中心・I微分縮約値 | same_threadでは担当threadのローカル値。split_groupsではsharedのHtemp/Gtempから担当threadが読む |
| Gjobの値 | same_threadではjob別shared。split_groupsではGtempのvalueへのview |
| Htemp / Gtemp / Hready / Gready | split_groupsのみ。block内の受け渡しに使用。固定slotをblock終了まで保持 |
| C_input / dXpart | 集団別shared。最後に一度展開・合算するまで |
| job状態・集団の空き・barrier | block内shared。集団代表が取得・完了を管理 |
| dX | 一つの所有blockからglobalへ最終値を保存 |
| dP_partial | global。全小バッチの勾配合算まで |

GjobはA128・batch_size4・FP64なら4,096 bytes/block。
split_groupsのパラメータ担当blockでは、Htemp・Gtempを各3項目とすると
`2 × 128 × 4 × 3 × 8 = 24,576 bytes/block`。
Gjob用の4,096 bytesはこのGtempのvalueと兼用するので、重ねて数えない。
same_threadのGjobだけの案との差は20,480 bytesで、通知・alignment・Z・C等は別途必要。
dX専用blockではGのvalueだけなので4,096 bytes以下（対象atom数による）となる。
split_groupsはregisterに保持する量を変える一方、sharedの使用量と読み書きが増える。
総shared上限を超える計画は拒否し、実機の進行・占有率と完全stepで比較する。
dP_partialはbatch32を4件ずつ、A128・4パラメータ・FP64なら32,768 bytes。
これらは部分配列の容量例であり、shared全体や完全stepのallocated/reservedピークではない。
保持Hの容量は`有効な(atom, batch, field)の件数 × dtypeのbytes`。
A128・全体batch4・FP64なら、全atomのvalueだけで4,096 bytes、3項目なら12,288 bytes。
64 atomsの3項目だけなら6,144 bytes。これは4出力block分に複製せず、一度だけ保持する。
保持Hには別途offset等の管理情報が必要で、forwardのshared Hjobも依然として存在する。
`recompute`ではこのglobal保持Hを作らない。W・dW全体はいずれの方式でも作らない。

同期は、Z作成者内、Z完成の公開と取得、job内全Gの完成、job全員の完了、
block内dXの最終合算前、batch間dPの合算前を区別する。
split_groupsでは、H/Gのpayload完成公開と依存jobの取得も別々に確認する。
jobの途中で一人だけ次のjobへ移らず、全員の読み書き終了後に集団を空きへ戻す。
Zの生成／atom群処理の割合、空き管理、将来の入口ループ除去はforwardと同様に検討する。

## 11. 確認する順番と未決定事項

1. Gの値・中心微分・I微分を、一つのatomについて独立参照と照合する。
   その後、各threadが別atomを担当する群全体の処理へ広げる。
2. Gから直接部分と入力側Cを作り、最終dXを照合する。
3. 再計算・全保存・一部保存（atom群と項目の両方）で、正規化微分を含む全dPの一致を確認する。
4. 入力tile 0のdP所有、batch間の合算、元atom IDへの対応に重複・抜けがないことを確認する。
5. strict支持境界・空／単一site・幅変化・振幅0・norm floor・retained backward・Graphを確認する。
6. 保存書き手の一意性、未訪問atom・端数batchの被覆、不足項目だけのZ依存を確認する。
   same_threadとsplit_groupsを各保持方針で比較し、H先行・G先行、空群・人数違い、
   通知前の読み取りなし、dP／dXの二重実行なし、全job終了とslotの寿命を確認する。
   固定担当表では全処理の被覆と一意な所有、待機の依存順、担当人数の違い、
   H/Gの軽重が入れ替わる条件を確認し、動的job管理の案と同じ結果になることを照合する。
   その後、パラメータ担当の偏り、block間再計算、別block群への分離、保存の3方式を比較する。
7. 独立Y/dX/全atom勾配を通した後、準備・更新まで含む未計装の完全stepと総ピークで判断する。

初期の所有方法や配分を最適と断定しない。数学とjobのインターフェースを固定して、
保存方針・担当人数・取得policy・実行構成は比較できるようにする。
再計算はforward時点の状態で行い、optimizer更新後のParameterを無検査で使わない。
Parameterの並べ替えやsnapshotを使う場合も、元Tensorのversion検査を保つ。

現行の数式・保存契約を確認した参照：
[executor.pyのbackward](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/executor.py)、
[kernels.pyのcontract / vjp](../../../src/torchcst/_backends/cuda/algorithms/linear/hierarchical_fourier/kernels.py)、
[H/Gの寿命を軸にした設計](../../h-lifecycle-kernel-design.ja.md)。
これら既存実装の検証結果を、本書の融合構成の検証結果として扱わない。
