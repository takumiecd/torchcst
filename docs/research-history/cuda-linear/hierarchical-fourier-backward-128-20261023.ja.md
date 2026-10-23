# backward128：atom群を並列処理するbackwardの入口と関数

2026-10-23。[forwardの関数別擬似コード](hierarchical-fourier-thread-functions-20261023.ja.md)に対応する設計案。
**以下は未実装のCUDA風擬似コード。GPU correctness・同期・速度・総メモリは未検証。**
本書作成ではruntimeを変更せず、GPUジョブも開始していない。

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
Hをbackwardまで保存するかは別の選択で、初期候補ではforwardの状態から再計算する。
Gは振幅を掛ける前の値とする。`dH`そのものは後述の`alpha * G`で、Gとは区別する。

**atom群の全atomを並列対象にし、各threadが別のatomのH/Gを最後まで計算する。**
一つのatomの入力部分和を複数threadで作って合算する方式には戻さない。
Z一枚の生成や、複数atomからC・dXへの集約は、集団で協力する別の処理である。

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
            do_backward_work(ctx, job, dP_partial);
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
    plan.worker_groups = make_groups(config.worker_group_sizes);

    if (plan.writes_param_grad)
        atoms = all_atoms();
    else
        atoms = atoms_overlapping_input_tile(input_tile);

    plan.atom_jobs = group_all_atoms_by_level_and_references(atoms);
    plan.Z_dY_jobs = required_output_moments(plan.atom_jobs);
    plan.Z_X_jobs = plan.writes_param_grad
                  ? required_input_moments(plan.atom_jobs) : empty;

    // dXの直接部分・C所属は担当input_tileとの重なりだけを登録する。
    plan.dx_parts = partition_input_support(atoms, input_tile);
    plan.required_Z = build_job_dependencies(plan);
    plan.shared_layout = allocate_bounded_backward_slots(plan);
    return plan;
}
```

Gだけを使うjobは`Z_dY`だけ、パラメータ勾配も作るjobは必要な`Z_X`と`Z_dY`の両方を待つ。
直接計算だけで済む場合は対応するZを作らない。入力側・出力側の経路を独立に判定する。

```cpp
init_backward_block(X, dY, forward_state, plan) {
    ctx = bind_thread_group_and_shared(plan);
    initialize_group_barriers_job_states_and_policy(ctx);

    load_dY_batch_to_shared(dY, plan);
    if (plan.writes_param_grad)
        load_X_batch_to_shared(X, plan);

    clear_input_C_and_dx_partials(ctx);
    clear_Z_ready_flags(ctx);       // X由来とdY由来を分ける
    __syncthreads();
    return ctx;
}
```

初期化の分担・barrierの構築はforwardの`init_block()`と同じ約束とする。
Z・Gの一時値は使用前に全必要要素を書くので、無条件に全領域をゼロ初期化しない。
Xを必要としないdX専用blockには、Xと入力Zのshared領域を用意しない。

## 5. atom群の仕事：`do_backward_work()`

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
            h = recompute_H_for_atom(shared.X, shared.Z_X,
                                     atom.input_plan, true);
            grad = atom_parameter_vjp(atom, h, g, atom.norm);
            original_id = forward_state.original_atom_id[atom.id];
            dP_partial[plan.batch_group][original_id] = grad;
        }
    }
}
```

`compute_G_for_atom()`と`recompute_H_for_atom()`は、**呼んだthread一人で**
そのatomの小バッチ全件を計算する。集団の他threadは別のatomを計算する。
128 atomsを128 threadsで担当するなら、一人一atom。人数が少なければstrideで次も担当する。

パラメータ担当では同じthreadがG・H・自分のatomのパラメータ勾配までを計算する。
Hと微分縮約値はまずローカル変数で使い切り、他threadへの受け渡し用にはGの値だけをsharedへ置く。
ローカル配列が実際にregisterへ収まるか、退避が起きるかはコンパイル後に確認する。

`original_atom_id`はforwardの物理的な並べ替えから元Parameterへの対応である。
並べ替えがなければ恒等対応とする。
最初から元IDへ書いた部分和を、最終合算時にもう一度逆並べ替えしない。
担当blockは全atomの全パラメータ欄を、ゼロの場合も含めて必ず書く。

## 6. GとHの中身

両方とも「sourceと、その側のprofileの積和」である。

| 関数 | source | profile | 使う階層値 |
| --- | --- | --- | --- |
| `compute_G_for_atom()` | dY | 出力側u | Z_dY |
| `recompute_H_for_atom()` | X | 入力側v | Z_X |

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
| 小バッチのX | パラメータ担当blockだけのshared |
| Z_dY / Z_X | 別々のsharedと完成通知。最初は上書きしない |
| atomのH/Gと中心・I微分縮約値 | 担当threadのローカル値。パラメータ勾配の計算まで |
| Gjobの値 | job別の重ならないshared領域。dX・Cへの集約まで |
| C_input / dXpart | 集団別shared。最後に一度展開・合算するまで |
| job状態・集団の空き・barrier | block内shared。集団代表が取得・完了を管理 |
| dX | 一つの所有blockからglobalへ最終値を保存 |
| dP_partial | global。全小バッチの勾配合算まで |

GjobはA128・batch_size4・FP64なら4,096 bytes/block。
dP_partialはbatch32を4件ずつ、A128・4パラメータ・FP64なら32,768 bytes。
どちらも部分配列の容量例であり、shared全体や完全stepのallocated/reservedピークではない。
H/Gの微分fieldを全atom分globalへ保存する配列や、W・dW全体は初期案で作らない。

同期は、Z作成者内、Z完成の公開と取得、job内全Gの完成、job全員の完了、
block内dXの最終合算前、batch間dPの合算前を区別する。
jobの途中で一人だけ次のjobへ移らず、全員の読み書き終了後に集団を空きへ戻す。
Zの生成／atom群処理の割合、空き管理、将来の入口ループ除去はforwardと同様に検討する。

## 11. 確認する順番と未決定事項

1. Gの値・中心微分・I微分を、一つのatomについて独立参照と照合する。
   その後、各threadが別atomを担当する群全体の処理へ広げる。
2. Gから直接部分と入力側Cを作り、最終dXを照合する。
3. Hの再計算と正規化分母の微分を含む全dPを確認する。
4. 入力tile 0のdP所有、batch間の合算、元atom IDへの対応に重複・抜けがないことを確認する。
5. strict支持境界・空／単一site・幅変化・振幅0・norm floor・retained backward・Graphを確認する。
6. パラメータ担当の偏り、H/G・Zのblock間再計算、別block群への分離、保存／再計算を比較する。
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
